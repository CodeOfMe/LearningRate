"""阶段一 / 步骤 3：根因分析与修复决策。

对 dead_value_manifest.json 里的每一项，给出：
  - category：死值类别（论据型坍缩 / 论文方法 bug / 优化器 bug / 单种子坍缩）
  - root_cause：根因解释
  - decision：处理决策（keep_and_annotate=保留+如实标注 / fix=修复）
  - fix_status：修复状态（fixed / not_needed / pending）

【关键决策（经确认）】
  判别式家族（Discriminative / STLR_Disc / SAM_Disc）在 MRPC/TREC-6 的坍缩是
  论文论证的「现象」——判别式衰减把底层 lr 压到 ≈0，模型坍缩到多数类。
  论文正文正是用这一坍缩论证「DALS 避免了这种退化」，因此【保留坍缩 + 如实标注
  std=0】，而非修复。这是数据诚实性的正确做法，也不削弱论据。

输出：
  - results/stage1_root_cause.json
  - results/stage1_root_cause_report.md
"""
import os
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(AUDIT, 'lib'))
from logger import ExperimentLogger  # noqa: E402

# 判别式家族：其坍缩是论文论据
DISCRIMINATIVE_FAMILY = {'Discriminative', 'STLR_Disc', 'SAM_Disc'}


def classify(it):
    ds = it['dataset']
    strat = it['strategy']
    t = it['type']

    if t == 'single_seed_collapse':
        if strat == 'DALS_Acc':
            return {
                'category': 'optimizer_bug',
                'root_cause': 'DALS_Acc 的 SGDR 重启无 warmup，重启瞬间 lr 从 ~0 跳回 base，'
                              '偶发导致微调坍缩到多数类（seed9999=50.92）。',
                'decision': 'fix',
                'fix_status': 'fixed',
                'fix_note': 'core/optimizers.py DALSAcc：每个 cycle 开头加 warmup_steps 线性爬升。',
            }
    if t == 'dead_value':
        if strat in DISCRIMINATIVE_FAMILY:
            return {
                'category': 'discriminative_collapse',
                'root_cause': '判别式衰减 lr=base_lr/2.6^(depth) 把 DistilBERT 底层 lr 压到 ≈0，'
                              '底层表征完全不更新，模型从 ep1 起恒定预测多数类，5 种子 std=0。',
                'decision': 'keep_and_annotate',
                'fix_status': 'not_needed',
                'fix_note': '论文论据：正文已述「坍缩到多数类/接近随机」，补充说明「5 种子 std=0」即可。',
            }
        if strat == 'DALS_Fast':
            return {
                'category': 'dals_fast_collapse',
                'root_cause': 'DALS-Fast（fast_phase0_bypass + momentum=0.85 + warmup_frac=0.02）'
                              '在 MRPC 上坍缩到恒定多数类预测。对照实验显示 DALS 家族在 MRPC 整体弱'
                              '（67~69，接近多数类 68.38），DALS-Fast 的激进配置使 std=0。'
                              '关闭 fast_phase0_bypass 未彻底解决，根因涉及 momentum/trust-ratio/phase 交互。',
                'decision': 'keep_and_annotate',
                'fix_status': 'attempted_ineffective',
                'fix_note': '建议论文如实说明 DALS-Fast 以快速收敛为目标，在 MRPC 上会坍缩到多数类，'
                            '并收敛「DALS 家族避免坍缩」的绝对表述。',
            }
        if strat == 'AdaBound':
            return {
                'category': 'optimizer_bug',
                'root_cause': 'AdaBound final_lr=0.1 硬编码，与 fine-tuning lr=5e-5 相差 2000 倍，'
                              '后期 SGD 界被 clamp 到 ~0.1，BERT-base 微调发散坍缩（50.92 多数类）。',
                'decision': 'fix',
                'fix_status': 'fixed',
                'fix_note': 'core/run_final.py：AdaBound final_lr 改为 lr_sgd（与 SGD 学习率同量级）。',
            }
    if t == 'severe_degradation':
        if strat in DISCRIMINATIVE_FAMILY:
            return {
                'category': 'discriminative_collapse',
                'root_cause': '判别式衰减过度抑制底层 lr，模型在 TREC-6 上崩到接近随机（6 类随机=16.7%），'
                              '5 种子在 28~45 间波动（非死值，但训练已崩坏）。',
                'decision': 'keep_and_annotate',
                'fix_status': 'not_needed',
                'fix_note': '论文论据：正文已述「接近随机水平」，是判别式敏感性的证据。',
            }
        if strat == 'AdaBound':
            # 与 dead_value 重复（sst2_b/AdaBound 既是死值又严重退化），返回 None 交由 main 去重
            return None
    return {'category': 'unknown', 'root_cause': '', 'decision': 'pending', 'fix_status': 'pending',
            'fix_note': ''}


def main():
    logger = ExperimentLogger(os.path.join(AUDIT, 'logs'), run_name='stage1_rootcause')
    manifest_path = os.path.join(AUDIT, 'results', 'dead_value_manifest.json')
    manifest = json.load(open(manifest_path, encoding='utf-8'))

    rows = []
    for it in manifest:
        c = classify(it)
        if c is None:
            continue  # 去重：dead_value 已覆盖的 severe_degradation
        rows.append({'dataset': it['dataset'], 'strategy': it['strategy'],
                     'type': it['type'], 'detail': it['detail'], **c})
        logger.log(f"rootcause|{it['dataset']}|{it['strategy']}",
                   f"根因分析 {it['dataset']}/{it['strategy']}",
                   {'type': it['type']}, c, 'success')

    with open(os.path.join(AUDIT, 'results', 'stage1_root_cause.json'), 'w', encoding='utf-8') as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    # markdown 报告
    md = ['# 阶段一：根因分析与修复决策', '',
          '## 决策总览', '',
          '| 数据集 | 策略 | 类别 | 决策 | 修复状态 |',
          '|---|---|---|---|---|']
    for r in rows:
        md.append(f"| {r['dataset']} | {r['strategy']} | {r['category']} | {r['decision']} | {r['fix_status']} |")
    md += ['', '## 逐项根因', '']
    for r in rows:
        md.append(f"### {r['dataset']} / {r['strategy']}（{r['category']}）")
        md.append(f"- 现象：{r['detail']}")
        md.append(f"- 根因：{r['root_cause']}")
        md.append(f"- 决策：{r['decision']}（{r['fix_status']}）")
        if r['fix_note']:
            md.append(f"- 处理：{r['fix_note']}")
        md.append('')
    with open(os.path.join(AUDIT, 'results', 'stage1_root_cause_report.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print('根因分析完成：')
    for r in rows:
        print(f"  [{r['category']:<22}] {r['dataset']:<10} {r['strategy']:<14} -> {r['decision']} ({r['fix_status']})")
    print(f"\n报告：results/stage1_root_cause_report.md")


if __name__ == '__main__':
    main()
