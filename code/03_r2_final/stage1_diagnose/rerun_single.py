"""阶段一 / 步骤 2：死值项单项单跑（隔离变量定位根因）。

严格参照 benchmark 代码（core/run_final.py 的 build_opt / build_model / build_loaders，
CONFIGS 与 run_final.py 完全一致）。对 dead_value_manifest.json 里每一项，按
(dataset, strategy, seed) 逐条单独运行，并记录 per-epoch 训练轨迹，用于定位
坍缩发生在哪个阶段（早期/晚期），以及判别式衰减分组是否导致底层 lr≈0。

支持断点恢复：已完成 (dataset, strategy, seed) 会被跳过。

用法：
  python stage1_diagnose/rerun_single.py --device cuda --manifest ../results/dead_value_manifest.json
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
CORE = os.path.join(AUDIT, 'core')
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(AUDIT, 'lib'))

# 关键：将模型/数据路径指向 revision_benchmark 的真实位置（严格沿用 benchmark 数据）
ROOT = os.path.abspath(os.path.join(AUDIT, '..', '..'))
BENCH = ROOT
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

SEEDS = [42, 1337, 2024, 5555, 9999]


def run_one(ds, strategy, seed, device, record_trace=True):
    """对单个 (dataset, strategy, seed) 跑一次，返回诊断结果。"""
    cfg = next(c for c in CONFIGS if c['ds'] == ds)
    data.set_seed(seed)
    (tr, va, te), nl = build_loaders(cfg, seed)
    total_steps = len(tr) * cfg['ep']
    model = build_model(cfg, nl, device)
    opt_, sched, pcb = build_opt(strategy, model, cfg['lr_sgd'], cfg['lr_adam'],
                                 total_steps, ep=cfg['ep'])
    if pcb is None:
        pcb = lambda x: None

    if record_trace:
        res = train_with_trace(model, opt_, tr, va, te, device, n_labels=nl,
                               epochs=cfg['ep'], sched=sched, dals_phase_update=pcb)
    else:
        r = T.train_one(model, opt_, tr, va, te, device, n_labels=nl,
                        epochs=cfg['ep'], sched=sched, dals_phase_update=pcb,
                        select_by='val', verbose=0)
        res = {'test_acc_valsel': r['test_acc_valsel'], 'test_f1_valsel': r['test_f1_valsel'],
               'best_val_epoch': r['best_val_epoch']}
    del model, opt_, sched, pcb
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return res


def train_with_trace(model, optimizer, tr, va, te, device, n_labels=2, epochs=5,
                     sched=None, dals_phase_update=None):
    """参照 train.py/train_one 逻辑，但额外记录 per-epoch loss 与 val acc，
    用于定位坍缩阶段。"""
    import torch.nn as nn
    model = model.to(device)
    best_val = -1.0
    best_test = None
    best_ep = -1
    trace = []
    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        nb = 0
        for batch in tr:
            if hasattr(batch, 'input_ids') or (isinstance(batch, dict) and 'input_ids' in batch):
                y = batch['labels'].to(device)
                out = model(input_ids=batch['input_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device))
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x, y = batch
                    out = model(x.to(device))
                else:
                    *x, y = batch
                    out = model(*[t.to(device) for t in x])
            else:
                x, y = batch
                out = model(x.to(device))
            logits = out.logits if hasattr(out, 'logits') else out
            loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            if dals_phase_update is not None:
                dals_phase_update(loss.item())
            if sched is not None:
                sched.step()
            optimizer.step()
            ep_loss += loss.item()
            nb += 1
        val_acc, val_f1 = T.evaluate(model, va, device, n_labels)
        if val_acc > best_val:
            best_val = val_acc
            best_ep = ep
            best_test = T.evaluate(model, te, device, n_labels)
        trace.append({'epoch': ep, 'train_loss': round(ep_loss / max(1, nb), 6),
                      'val_acc': round(val_acc, 4), 'val_f1': round(val_f1, 4)})
    return {'test_acc_valsel': round(best_test[0], 4) if best_test else None,
            'test_f1_valsel': round(best_test[1], 4) if best_test else None,
            'best_val_epoch': best_ep, 'best_val_acc': round(best_val, 4),
            'trace': trace}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--manifest', default=os.path.join(AUDIT, 'results', 'dead_value_manifest.json'))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--no-trace', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda') if args.device == 'cuda' else torch.device('cpu')
    logger = ExperimentLogger(os.path.join(AUDIT, 'logs'), run_name='stage1_rerun')
    runner = ResumableRunner(logger)

    manifest = json.load(open(args.manifest, encoding='utf-8'))
    # 仅处理 dead_value 与 single_seed_collapse（severe_degradation 属判别式家族固有退化，
    # 由阶段三/统计另行处理，不在此重跑定位）
    items = [m for m in manifest if m['type'] in ('dead_value', 'single_seed_collapse')]

    tasks = []
    for it in items:
        for seed in args.seeds:
            tasks.append({
                'task_id': f"rerun|{it['dataset']}|{it['strategy']}|{seed}",
                'description': f"单跑 {it['dataset']}/{it['strategy']} seed={seed}",
                'dataset': it['dataset'], 'strategy': it['strategy'], 'seed': seed,
                'issue_type': it['type'],
            })

    def run_task(task):
        res = run_one(task['dataset'], task['strategy'], task['seed'], device,
                      record_trace=not args.no_trace)
        return res

    print(f'待跑 {len(tasks)} 项（死值/坍塌 × {len(args.seeds)} seed）')
    summary = runner.run_tasks(tasks, run_task)
    print(f'\n单跑完成：成功 {summary["success"]}，失败 {summary["failed"]}，跳过 {summary["skipped"]}')
    print(f'日志：{os.path.join(HERE, "logs", "stage1_rerun_log.jsonl")}')


if __name__ == '__main__':
    main()
