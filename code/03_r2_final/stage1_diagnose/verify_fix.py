"""阶段一 / 步骤 3+4：修复后验证（确认死值/坍塌已解决）。

用修复后的代码（core/optimizers.py、core/run_final.py）对 dead_value_manifest.json
里的每一项重跑 5 个种子，检查：
  - 原「死值」（5 种子 std=0）修复后 std 是否 > 0、且不再坍缩到多数类；
  - 原「单种子坍塌」修复后该种子是否回到正常水平。

支持断点恢复。输出 results/stage1_verify_report.md + JSON。

用法：
  python stage1_diagnose/verify_fix.py --device cuda --seeds 42 1337 2024 5555 9999
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
CORE = os.path.join(AUDIT, 'core')
sys.path.insert(0, HERE)
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(AUDIT, 'lib'))

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
from rerun_single import run_one
from logger import ExperimentLogger, ResumableRunner  # noqa: E402

SEEDS = [42, 1337, 2024, 5555, 9999]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--manifest', default=os.path.join(AUDIT, 'results', 'dead_value_manifest.json'))
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    args = ap.parse_args()

    device = torch.device('cuda') if args.device == 'cuda' else torch.device('cpu')
    logger = ExperimentLogger(os.path.join(AUDIT, 'logs'), run_name='stage1_verify')
    runner = ResumableRunner(logger)

    manifest = json.load(open(args.manifest, encoding='utf-8'))
    items = [m for m in manifest if m['type'] in ('dead_value', 'single_seed_collapse')]

    tasks = []
    for it in items:
        for seed in args.seeds:
            tasks.append({'task_id': f"verify|{it['dataset']}|{it['strategy']}|{seed}",
                          'description': f"验证 {it['dataset']}/{it['strategy']} seed={seed}",
                          'dataset': it['dataset'], 'strategy': it['strategy'],
                          'seed': seed, 'issue_type': it['type'],
                          'orig_value': it.get('value')})

    def run_task(task):
        res = run_one(task['dataset'], task['strategy'], task['seed'], device, record_trace=False)
        return res

    print(f'待验证 {len(tasks)} 项')
    summary = runner.run_tasks(tasks, run_task)

    # 聚合每项的 5-seed acc
    results_dir = logger.results_dir
    agg = {}
    for t in tasks:
        f = os.path.join(results_dir, f"{t['task_id']}.json")
        if not os.path.exists(f):
            continue
        rec = json.load(open(f, encoding='utf-8'))
        if rec.get('status') != 'success' or not rec.get('output'):
            continue
        key = (t['dataset'], t['strategy'])
        agg.setdefault(key, []).append(rec['output'].get('test_acc_valsel'))

    rows = []
    for it in items:
        key = (it['dataset'], it['strategy'])
        accs = [a for a in agg.get(key, []) if a is not None]
        if len(accs) < 2:
            status = 'insufficient'
        else:
            std = float(np.std(accs, ddof=1))
            mean = float(np.mean(accs))
            if it['type'] == 'dead_value':
                status = 'fixed' if std > 0.05 else 'still_dead'
            else:  # single_seed_collapse
                status = 'fixed' if min(accs) > mean - 15 else 'still_collapsed'
        rows.append({'dataset': it['dataset'], 'strategy': it['strategy'],
                     'type': it['type'], 'orig_value': it.get('value'),
                     'fixed_accs': [round(a, 4) for a in accs],
                     'mean': round(mean, 4) if accs else None,
                     'std': round(std, 4) if len(accs) > 1 else None,
                     'status': status})

    # 保存 + 报告
    out = {'summary': summary, 'rows': rows}
    with open(os.path.join(AUDIT, 'results', 'stage1_verify.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    md = ['# 阶段一：修复后验证报告', '',
          '| 数据集 | 策略 | 类型 | 原值 | 修复后 5-seed | mean | std | 结论 |',
          '|---|---|---|---|---|---|---|---|']
    for r in rows:
        accs = ', '.join(f'{a:.2f}' for a in r['fixed_accs'])
        verdict = {'fixed': '✅ 已修复', 'still_dead': '❌ 仍死值',
                   'still_collapsed': '❌ 仍坍缩', 'insufficient': '⚠️ 数据不足'}[r['status']]
        md.append(f"| {r['dataset']} | {r['strategy']} | {r['type']} | {r['orig_value']} | "
                  f"{accs} | {r['mean']} | {r['std']} | {verdict} |")
    with open(os.path.join(AUDIT, 'results', 'stage1_verify_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print('\n修复验证结果：')
    for r in rows:
        print(f"  [{r['status']:<16}] {r['dataset']:<10} {r['strategy']:<14} "
              f"修复后={r['fixed_accs']} mean={r['mean']} std={r['std']}")
    print(f"\n报告：results/stage1_verify_report.md")


if __name__ == '__main__':
    main()
