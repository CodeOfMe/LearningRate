"""意见 2 补充：敏捷性（agility）——达到可接受精度所需的时间/步数。

验证：DALS 家族能否比 RAdam 等 baseline 用更少的 epoch / 时间达到一个可接受的精度。
指标：
  - 每 epoch 的 val acc 轨迹（train_with_trace 记录）；
  - 首次达到阈值（如 MRPC 80%）所需的 epoch 数；
  - 达到阈值所需的 wall-clock 时间（敏捷性 = 时间维度）。

DALS-Fast 的设计目标即「快速早期收敛」，预期它在 epoch 维度最早达到阈值。

用法：
  python stage2_statistics/agility.py --device cuda --datasets mrpc --threshold 80
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
CORE = os.path.join(AUDIT, 'core')
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(AUDIT, 'lib'))

BENCH = os.path.join(AUDIT, '..', 'revision_benchmark')
os.environ['MODEL_DIR'] = os.path.join(BENCH, 'models')
os.environ['DATA_DIR'] = os.path.join(BENCH, 'datasets')
os.environ['HF_DATASETS_CACHE'] = os.path.join(BENCH, 'datasets', 'huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import numpy as np
import data
import train as T
from run_final import build_opt, build_model, build_loaders, CONFIGS
from logger import ExperimentLogger, ResumableRunner  # noqa: E402

KEY_METHODS = ['DALS', 'DALS_Fast', 'DALS_Acc', 'RAdam', 'Lookahead', 'Adam', 'Discriminative']


def run_trajectory(ds, strategy, seed, device):
    """跑单个 (dataset, strategy, seed)，返回 per-epoch val_acc + 每 epoch 时间。"""
    cfg = next(c for c in CONFIGS if c['ds'] == ds)
    data.set_seed(seed)
    (tr, va, te), nl = build_loaders(cfg, seed)
    total_steps = len(tr) * cfg['ep']
    model = build_model(cfg, nl, device)
    o, sched, pcb = build_opt(strategy, model, cfg['lr_sgd'], cfg['lr_adam'], total_steps, ep=cfg['ep'])
    if pcb is None:
        pcb = lambda x: None
    import time
    import torch.nn as nn
    model = model.to(device)
    trace = []
    t0 = time.time()
    for ep in range(1, cfg['ep'] + 1):
        model.train()
        for batch in tr:
            if hasattr(batch, 'input_ids') or (isinstance(batch, dict) and 'input_ids' in batch):
                y = batch['labels'].to(device)
                out = model(input_ids=batch['input_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device))
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x, y = batch
                    x = x.to(device); y = y.to(device)
                    out = model(x)
                else:
                    *x, y = batch
                    x = [t.to(device) for t in x]; y = y.to(device)
                    out = model(*x)
            else:
                x, y = batch
                out = model(x.to(device))
            logits = out.logits if hasattr(out, 'logits') else out
            loss = nn.functional.cross_entropy(logits, y)
            o.zero_grad()
            loss.backward()
            if pcb is not None:
                pcb(loss.item())
            if sched is not None:
                sched.step()
            o.step()
        val_acc, _ = T.evaluate(model, va, device, n_labels=nl)
        trace.append({'epoch': ep, 'val_acc': round(val_acc, 4),
                      'cum_time_s': round(time.time() - t0, 2)})
    del model, o, sched, pcb
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return trace


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--datasets', nargs='+', default=['mrpc'])
    ap.add_argument('--methods', nargs='+', default=KEY_METHODS)
    ap.add_argument('--threshold', type=float, default=80.0)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    device = torch.device('cuda') if args.device == 'cuda' else torch.device('cpu')
    logger = ExperimentLogger(os.path.join(AUDIT, 'logs'), run_name='agility')
    runner = ResumableRunner(logger)

    tasks = []
    for ds in args.datasets:
        for m in args.methods:
            tasks.append({'task_id': f'agility|{ds}|{m}|{args.seed}',
                          'description': f'敏捷性轨迹 {ds}/{m} seed={args.seed}',
                          'dataset': ds, 'strategy': m, 'seed': args.seed})

    def run_task(task):
        trace = run_trajectory(task['dataset'], task['strategy'], task['seed'], device)
        # 计算达到阈值所需的 epoch 和时间
        reach_ep = None
        reach_time = None
        for t in trace:
            if t['val_acc'] >= args.threshold:
                reach_ep = t['epoch']
                reach_time = t['cum_time_s']
                break
        return {'trace': trace, 'reach_epoch': reach_ep, 'reach_time_s': reach_time}

    print(f'敏捷性测量：{args.methods} × {args.datasets}，阈值 {args.threshold}%')
    summary = runner.run_tasks(tasks, run_task)

    # 汇总
    results_dir = logger.results_dir
    rows = []
    for t in tasks:
        f = os.path.join(results_dir, f"{t['task_id']}.json")
        if not os.path.exists(f):
            continue
        rec = json.load(open(f, encoding='utf-8'))
        if rec.get('status') != 'success' or not rec.get('output'):
            continue
        out = rec['output']
        rows.append({'dataset': t['dataset'], 'strategy': t['strategy'],
                     'reach_epoch': out.get('reach_epoch'),
                     'reach_time_s': out.get('reach_time_s'),
                     'trace': out.get('trace', [])})

    with open(os.path.join(AUDIT, 'results', 'agility.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # 报告
    print(f'\n达到 {args.threshold}% 所需的 epoch / 时间：')
    for r in sorted(rows, key=lambda x: (x['reach_epoch'] is None, x['reach_epoch'] or 999)):
        tag = f"{r['reach_epoch']} ep / {r['reach_time_s']:.0f}s" if r['reach_epoch'] else "未达到"
        print(f"  {r['strategy']:<14} {tag}")
        for t in r['trace']:
            mark = ' <-- 达阈值' if t['val_acc'] >= args.threshold else ''
            print(f"      ep{t['epoch']}: {t['val_acc']}% ({t['cum_time_s']}s){mark}")


if __name__ == '__main__':
    main()
