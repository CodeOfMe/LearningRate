"""阶段一 / 步骤 1：死值问题诊断。

遍历全部实验结果，识别：
  A. 死值 (dead value)：5 个种子结果完全相同 (std=0)，模型输出恒定；
  B. 单种子坍塌 (single-seed collapse)：个别 seed 掉到多数类/随机基线，其余正常；
  C. 严重退化 (severe degradation)：mean 显著低于该数据集正常水平（低于中位数 ≥15 点）。

输出：
  - logs/.../run_log.jsonl（过程记录，断点恢复用）
  - results/stage1_detect_*.json（诊断详情）
  - results/dead_value_manifest.json（死值/坍塌清单，供阶段一步骤 2「单项单跑」消费）
  - results/stage1_detect_report.md（人类可读报告）

数据源：默认读取 <repo>/final data.jsonl（最终提交数据的 5-seed per-seed 准确率），
也可用 --input 指定其它结果 JSON。
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)          # experiment_audit/
REPO = os.path.abspath(os.path.join(AUDIT, '..', '..'))  # repo root
sys.path.insert(0, os.path.join(AUDIT, 'lib'))
from logger import ExperimentLogger, ResumableRunner  # noqa: E402
RESULT_DIR = os.path.join(AUDIT, 'results')

# 多数类 / 随机基线（accuracy %）
BASELINE = {
    'synthetic': ('10类随机', 10.0),
    'cifar10': ('10类随机', 10.0),
    'cifar100': ('100类随机', 1.0),
    'rte': ('2类多数类', 53.0),
    'trec6': ('6类随机', 16.7),
    'imdb': ('2类多数类', 50.0),
    'sst2': ('2类多数类', 51.0),
    'sst2_b': ('2类多数类', 51.0),
    'mrpc': ('2类多数类', 68.0),
    'mrpc_b': ('2类多数类', 68.0),
}
SEEDS = [42, 1337, 2024, 5555, 9999]


def load_data(input_path):
    if input_path.endswith('.jsonl'):
        return [json.loads(l) for l in open(input_path, encoding='utf-8') if l.strip()]
    if input_path.endswith('.json'):
        d = json.load(open(input_path, encoding='utf-8'))
        # 若是 {strategy: {acc_mean, per_seed...}} 结构，扁平化处理
        rows = []
        for k, v in d.items():
            if isinstance(v, dict) and 'per_seed' in v:
                rows.append({'dataset': 'unknown', 'strategy': k, **v})
        return rows
    raise ValueError(f'无法识别的输入格式: {input_path}')


def detect(row):
    """返回该行的问题描述列表。"""
    ds = row['dataset']
    strat = row['strategy']
    accs = row.get('per_seed_acc') or row.get('per_seed') or []
    # 归一化：per_seed_acc 是纯数值列表；per_seed 是 dict 列表
    if accs and isinstance(accs[0], dict):
        accs = [a.get('acc') for a in accs]
    accs = [a for a in accs if a is not None]
    issues = []

    # A. 死值：5 种子完全相同
    if len(accs) >= 5 and len(set(round(a, 6) for a in accs)) == 1:
        issues.append({'type': 'dead_value', 'value': round(accs[0], 4),
                       'detail': f'5 种子完全相同 = {accs[0]:.4f} (std=0)'})

    # B. 单种子坍塌：某 seed 显著低于其余均值，且接近基线
    label, base = BASELINE.get(ds, ('?', 0))
    if len(accs) == 5 and len(set(round(a, 6) for a in accs)) > 1:
        for i, a in enumerate(accs):
            others = [accs[j] for j in range(5) if j != i]
            om = sum(others) / len(others)
            if (om - a) > 15 and a < base + 10:
                issues.append({'type': 'single_seed_collapse', 'seed': SEEDS[i],
                               'value': round(a, 4), 'others_mean': round(om, 4),
                               'detail': f'seed{SEEDS[i]}={a:.2f} 其余均值 {om:.2f}，接近 {label}基线 {base}%'})

    # C. 严重退化：mean 低于数据集所有策略中位数 15 点以上（需全局信息，单独处理）
    return issues


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=os.path.join(REPO, 'tables', 'full_tables', 'final_data_master.jsonl'))
    ap.add_argument('--log-dir', default=os.path.join(AUDIT, 'logs'))
    args = ap.parse_args()

    logger = ExperimentLogger(args.log_dir, run_name='stage1')
    runner = ResumableRunner(logger)

    rows = load_data(args.input)
    print(f'载入 {len(rows)} 条实验记录（{args.input}）')

    # 按数据集聚合，用于 C 类「严重退化」判断
    from collections import defaultdict
    by_ds = defaultdict(list)
    for r in rows:
        by_ds[r['dataset']].append(r)

    dead_manifest = []
    # 逐行检测 A/B 类问题（每行一个任务，支持断点恢复）
    tasks = []
    for r in rows:
        tasks.append({'task_id': f"detect|{r['dataset']}|{r['strategy']}",
                      'description': f"诊断 {r['dataset']}/{r['strategy']}",
                      'dataset': r['dataset'], 'strategy': r['strategy']})

    def detect_one(task):
        ds = task['dataset']; strat = task['strategy']
        r = next(x for x in rows if x['dataset'] == ds and x['strategy'] == strat)
        issues = detect(r)
        return {'issues': issues}

    summary = runner.run_tasks(tasks, detect_one)
    print(f'\n诊断完成: 成功 {summary["success"]}, 失败 {summary["failed"]}, 跳过 {summary["skipped"]}')

    # C 类：严重退化（跨策略比较）
    degrades = []
    for ds, rs in by_ds.items():
        med = sorted(r.get('acc_mean') or 0 for r in rs)[len(rs) // 2]
        for r in rs:
            mean = r.get('acc_mean')
            if mean is not None and mean < med - 15:
                degrades.append({'dataset': ds, 'strategy': r['strategy'],
                                 'mean': round(mean, 2), 'median': round(med, 2)})

    # 汇总死值清单（A + B + C 去重）
    manifest = []
    for r in rows:
        issues = detect(r)
        for it in issues:
            manifest.append({'dataset': r['dataset'], 'strategy': r['strategy'],
                             **it, 'acc_mean': r.get('acc_mean'), 'issue_flag': r.get('issue')})
    for d in degrades:
        manifest.append({'dataset': d['dataset'], 'strategy': d['strategy'],
                         'type': 'severe_degradation', 'acc_mean': d['mean'],
                         'detail': f"mean {d['mean']} 低于数据集中位数 {d['median']} 15+ 点"})

    # 保存诊断结果
    out = {'summary': summary, 'manifest': manifest, 'severe_degradation': degrades}
    with open(os.path.join(RESULT_DIR, 'stage1_detect.json'), 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULT_DIR, 'dead_value_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # 生成 markdown 报告
    lines = ['# 阶段一：死值问题诊断报告', '', f'数据源：`{args.input}`（{len(rows)} 条）', '',
             f"诊断结果：成功 {summary['success']} / 失败 {summary['failed']} / 跳过 {summary['skipped']}", '']
    for t in ['dead_value', 'single_seed_collapse', 'severe_degradation']:
        items = [m for m in manifest if m['type'] == t]
        title = {'dead_value': '## A. 死值（5 种子完全相同，std=0）',
                 'single_seed_collapse': '## B. 单种子坍塌',
                 'severe_degradation': '## C. 严重退化（低于中位数 15+ 点）'}[t]
        lines += [title, '', '| 数据集 | 策略 | 详情 |', '|---|---|---|']
        for it in items:
            lines.append(f"| {it['dataset']} | {it['strategy']} | {it['detail']} |")
        lines.append('')
    with open(os.path.join(RESULT_DIR, 'stage1_detect_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n死值/坍塌清单（{len(manifest)} 项）：")
    for it in manifest:
        print(f"  [{it['type']:<20}] {it['dataset']:<10} {it['strategy']:<14} {it['detail']}")
    print(f"\n报告已保存：results/stage1_detect_report.md")
    print(f"清单已保存：results/dead_value_manifest.json")


if __name__ == '__main__':
    main()
