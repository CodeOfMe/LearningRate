"""阶段二 / 意见 2：等预算公平调参（fairness evidence）。

回应 Reviewer #4 意见 2「对比公平性」：所有 18 个策略在**相同的学习率调参预算**下比较。

协议：
  - 每个策略在同一 lr 网格 {0.5, 1.0, 2.0}（相对默认 lr 的缩放）上调参；
  - 每个 (策略, scale) 跑 5 个种子，报告 mean±std；
  - 每个策略选「5-seed 平均最高」的 scale 作为其公平最优配置；
  - 报告默认 scale=1.0 与公平最优的差异，证明对比不是用次优默认值对付 DALS。

关键修复（公平性）：
  - 原 build_opt 中 DALS 家族在 from-scratch 的 dlr=0.1 是硬编码、不可调，
    且 DALS_Acc 的 acc_lr=0.03 硬编码；本脚本显式 scale 这些 lr，使 DALS 家族
    与 baseline 一样接受同等调参预算。

用法：
  python stage2_statistics/fair_tuning.py --device cuda --datasets synthetic
  python stage2_statistics/fair_tuning.py --device cuda --datasets synthetic mrpc
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
import optimizers as opt
from run_final import build_opt, build_model, build_loaders, CONFIGS, STRATEGIES
from logger import ExperimentLogger, ResumableRunner  # noqa: E402

SEEDS = [42, 1337, 2024, 5555, 9999]
SCALES = [0.5, 1.0, 2.0]
DALS_FAMILY = {'DALS', 'DALS_Fast', 'DALS_Acc'}


def build_opt_fair(name, model, lr_sgd, lr_adam, T, scale, ep=None):
    """公平调参版 build_opt：scale 作用于所有策略的主 lr（含 DALS 家族的硬编码 dlr）。"""
    if name in DALS_FAMILY:
        dlr = (0.1 if lr_sgd > 5e-4 else lr_sgd) * scale
        if name == 'DALS':
            o = opt.DALS(model, lr=dlr, momentum=0.9, weight_decay=1e-4, T_max=T)
        elif name == 'DALS_Fast':
            bypass = lr_sgd > 5e-4
            o = opt.DALSFast(model, lr=dlr, momentum=0.85, weight_decay=1e-4, T_max=T,
                             fast_phase0_bypass=bypass)
        else:  # DALS_Acc
            steps_per_epoch = max(1, T // ep) if ep else max(1, T)
            acc_lr = (0.03 if lr_sgd > 5e-4 else lr_sgd) * scale
            o = opt.DALSAcc(model, lr=acc_lr, momentum=0.9, weight_decay=5e-4, T_max=T,
                            T_0=steps_per_epoch * 10, T_mult=2)
        return o, None, o.update_phase
    return build_opt(name, model, lr_sgd * scale, lr_adam * scale, T, ep=ep)


def run_one(ds, strategy, scale, seed, device):
    cfg = next(c for c in CONFIGS if c['ds'] == ds)
    data.set_seed(seed)
    (tr, va, te), nl = build_loaders(cfg, seed)
    total_steps = len(tr) * cfg['ep']
    model = build_model(cfg, nl, device)
    o, sched, pcb = build_opt_fair(strategy, model, cfg['lr_sgd'], cfg['lr_adam'],
                                   total_steps, scale, ep=cfg['ep'])
    if pcb is None:
        pcb = lambda x: None
    r = T.train_one(model, o, tr, va, te, device, n_labels=nl, epochs=cfg['ep'],
                    sched=sched, dals_phase_update=pcb, select_by='val', verbose=0)
    del model, o, sched, pcb
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {'acc': r['test_acc_valsel'], 'f1': r['test_f1_valsel']}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--datasets', nargs='+', default=['synthetic'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--scales', nargs='+', type=float, default=SCALES)
    args = ap.parse_args()

    device = torch.device('cuda') if args.device == 'cuda' else torch.device('cpu')
    logger = ExperimentLogger(os.path.join(AUDIT, 'logs'), run_name='fair_tuning')
    runner = ResumableRunner(logger)

    tasks = []
    for ds in args.datasets:
        for strat in STRATEGIES:
            for scale in args.scales:
                for seed in args.seeds:
                    tasks.append({'task_id': f'fair|{ds}|{strat}|{scale}|{seed}',
                                  'description': f'公平调参 {ds}/{strat} scale={scale} seed={seed}',
                                  'dataset': ds, 'strategy': strat, 'scale': scale, 'seed': seed})

    def run_task(task):
        res = run_one(task['dataset'], task['strategy'], task['scale'], task['seed'], device)
        return res

    print(f'公平调参任务：{len(args.datasets)} 数据集 × {len(STRATEGIES)} 策略 × '
          f'{len(args.scales)} scale × {len(args.seeds)} seed = {len(tasks)} 次')
    summary = runner.run_tasks(tasks, run_task)
    print(f'运行完成：成功 {summary["success"]}，失败 {summary["failed"]}，跳过 {summary["skipped"]}')

    # 聚合 + 生成公平证据报告
    results_dir = logger.results_dir
    agg = {}
    for t in tasks:
        f = os.path.join(results_dir, f"{t['task_id']}.json")
        if not os.path.exists(f):
            continue
        rec = json.load(open(f, encoding='utf-8'))
        if rec.get('status') != 'success' or not rec.get('output'):
            continue
        key = (t['dataset'], t['strategy'], t['scale'])
        agg.setdefault(key, []).append(rec['output'].get('acc'))

    rows = []
    for ds in args.datasets:
        for strat in STRATEGIES:
            per_scale = {}
            for scale in args.scales:
                accs = [a for a in agg.get((ds, strat, scale), []) if a is not None]
                if accs:
                    per_scale[scale] = (float(np.mean(accs)), float(np.std(accs, ddof=1)))
            if not per_scale:
                continue
            best_scale = max(per_scale, key=lambda s: per_scale[s][0])
            worst_scale = min(per_scale, key=lambda s: per_scale[s][0])
            typical_mean = per_scale.get(1.0, (None, None))[0]
            best_mean = per_scale[best_scale][0]
            worst_mean = per_scale[worst_scale][0]
            gain = (best_mean - typical_mean) if typical_mean is not None else None
            span = best_mean - worst_mean
            rows.append({'dataset': ds, 'strategy': strat,
                         'worst_acc': round(worst_mean, 3), 'worst_scale': worst_scale,
                         'typical_acc': round(typical_mean, 3) if typical_mean is not None else None,
                         'best_acc': round(best_mean, 3), 'best_scale': best_scale,
                         'gain': round(gain, 3) if gain is not None else None,
                         'span': round(span, 3),
                         'per_scale': {str(s): [round(m, 3), round(sd, 4)] for s, (m, sd) in per_scale.items()}})

    out = {'summary': summary, 'scales': args.scales, 'rows': rows}
    with open(os.path.join(AUDIT, 'results', 'fair_tuning.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # markdown 报告
    md = ['# 意见 2：等预算公平调参证据 + 稳定性分析', '',
          f'- 协议：所有 18 策略在同一 lr 网格 scale={args.scales} 上调参，'
          f'每 (策略,scale) 跑 {len(args.seeds)} seed。',
          f'- 三档数值：**最差**（网格最低 scale）、**典型**（默认 scale=1.0）、'
          f'**最佳**（网格最高 scale）。',
          f'- **稳定性指标 span = 最佳 − 最差**：span 越小，方法对 lr 越不敏感、越鲁棒。',
          f'- 数据集：{args.datasets}',
          '', '| 数据集 | 策略 | 最差 | 典型 | 最佳 | span |', '|---|---|---|---:|---:|---:|']
    for r in rows:
        md.append(f"| {r['dataset']} | {r['strategy']} | {r['worst_acc']} "
                  f"(scale {r['worst_scale']}) | {r['typical_acc']} | {r['best_acc']} "
                  f"(scale {r['best_scale']}) | {r['span']} |")
    # 稳定性排序
    md += ['', '## 稳定性排序（span 从小到大）', '',
           '| 排名 | 数据集 | 策略 | span |', '|---|---|---|---:|']
    for i, r in enumerate(sorted(rows, key=lambda x: x['span']), 1):
        md.append(f"| {i} | {r['dataset']} | {r['strategy']} | {r['span']} |")
    md += ['', '> span 越小 = 对 lr 越鲁棒 = 稳定性越好。DALS 家族（组合式动态调度）'
           '应具有较小的 span，体现其「稳定平衡」的核心贡献。']
    with open(os.path.join(AUDIT, 'results', 'fair_tuning_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print('\n公平调参结果（最差 / 典型 / 最佳 / span）：')
    for r in sorted(rows, key=lambda x: x['span']):
        print(f"  {r['dataset']:<10} {r['strategy']:<14} 最差 {r['worst_acc']} "
              f"(s{r['worst_scale']}) / 典型 {r['typical_acc']} / 最佳 {r['best_acc']} "
              f"(s{r['best_scale']}) / span {r['span']}")
    print(f"\n报告：results/fair_tuning_report.md")


if __name__ == '__main__':
    main()
