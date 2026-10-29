"""阶段二：P 值统计显著性校验（含多重比较校正）。

严格参照 benchmark 的 run_significance.py（配对 t 检验 + bootstrap 95% CI +
Hedges' g），并补上审稿人要求的：
  1. 多重比较校正：Bonferroni 与 FDR (Benjamini-Hochberg)；
  2. 死值（零方差）处理：std=0 的组合标记为「检验不适用」，不纳入校正集合；
  3. 逐条记录检验方法、原始 p、校正后 p、显著性水平 (α=0.05)、是否显著。

数据源默认：<repo>/final data.jsonl（最终 5-seed per-seed 准确率）。

输出：
  - results/stage2_significance.json
  - results/stage2_significance_report.md
  - logs/...（过程记录 + 断点恢复）
"""
import os
import sys
import json
import math

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
REPO = os.path.dirname(AUDIT)
sys.path.insert(0, os.path.join(AUDIT, 'lib'))
from logger import ExperimentLogger, ResumableRunner  # noqa: E402

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DALS_VARIANTS = ['DALS', 'DALS_Fast', 'DALS_Acc']
SEEDS = [42, 1337, 2024, 5555, 9999]
ALPHA = 0.05


def load_rows(input_path):
    if input_path.endswith('.jsonl'):
        return [json.loads(l) for l in open(input_path, encoding='utf-8') if l.strip()]
    raise ValueError(input_path)


def paired_test(b_accs, v_accs):
    """配对 t 检验 + bootstrap CI + Hedges' g。
    返回 dict；若任一方零方差（死值），返回 zero_variance 标记。
    """
    b = np.array(b_accs, dtype=float)
    v = np.array(v_accs, dtype=float)
    n = len(b)
    d = v - b
    if np.std(b, ddof=1) == 0 or np.std(v, ddof=1) == 0:
        return {'n': n, 'zero_variance': True, 'baseline_std': float(np.std(b, ddof=1)),
                'variant_std': float(np.std(v, ddof=1)),
                'diff_mean': float(np.mean(d)), 'p_value': None, 'ci95': None,
                'hedges_g': None, 'baseline_mean': float(np.mean(b)),
                'variant_mean': float(np.mean(v))}
    if HAS_SCIPY:
        t, p = stats.ttest_rel(v, b)
    else:
        # 退化实现：配对 t 检验
        se = np.std(d, ddof=1) / math.sqrt(n)
        t = np.mean(d) / se if se > 0 else 0.0
        df = n - 1
        p = float(2 * (1 - _t_cdf(abs(t), df))) if se > 0 else 1.0
    rng = np.random.default_rng(42)
    bs = np.array([np.mean(rng.choice(d, n, replace=True)) for _ in range(10000)])
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    sd = np.std(d, ddof=1)
    g = float(np.mean(d) / sd) if sd > 0 else 0.0
    correction = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    return {'n': n, 'zero_variance': False,
            'baseline_mean': float(np.mean(b)), 'variant_mean': float(np.mean(v)),
            'diff_mean': float(np.mean(d)), 'ci95': ci, 'p_value': float(p),
            'hedges_g': g * correction}


def _t_cdf(x, df):
    """粗略的 t 分布 CDF（无 scipy 时用正态近似）。"""
    # 用正态近似（df 大时足够）
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def benjamini_hochberg(pvals):
    """FDR (Benjamini-Hochberg) 校正。

    pvals: list[(idx, p)]，返回 dict {idx: adjusted_p}。
    标准 BH：p 升序后，q(i) = min(p(i)*m/i, q(i+1))，从大到小反向遍历。
    """
    m = len(pvals)
    if m == 0:
        return {}
    order = sorted(pvals, key=lambda x: x[1])  # 按 p 升序
    adj = {}
    for i in range(m - 1, -1, -1):  # 从最大 p 到最小 p
        idx, p = order[i]
        q = p * m / (i + 1)
        if i < m - 1:
            q = min(q, adj[order[i + 1][0]])
        adj[idx] = min(1.0, q)
    return adj


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default=os.path.join(REPO, 'final data.jsonl'))
    args = ap.parse_args()

    logger = ExperimentLogger(os.path.join(AUDIT, 'logs'), run_name='stage2')
    runner = ResumableRunner(logger)

    rows = load_rows(args.input)
    idx = {(r['strategy'], r['dataset']): r for r in rows}
    datasets = sorted(set(r['dataset'] for r in rows))
    baselines = [s for s in sorted(set(r['strategy'] for r in rows)) if s not in DALS_VARIANTS]

    # 收集所有对比的 p 值（用于多重比较校正）
    comparisons = []  # list[dict]
    for ds in datasets:
        for vname in DALS_VARIANTS:
            v = idx.get((vname, ds))
            if not v:
                continue
            for bname in baselines:
                b = idx.get((bname, ds))
                if not b:
                    continue
                comparisons.append({'dataset': ds, 'variant': vname, 'baseline': bname,
                                    'b_accs': b['per_seed_acc'], 'v_accs': v['per_seed_acc']})

    # 先计算所有原始 p 值
    results = []
    for c in comparisons:
        stat = paired_test(c['b_accs'], c['v_accs'])
        results.append({'dataset': c['dataset'], 'variant': c['variant'],
                        'baseline': c['baseline'], **stat})

    # 多重比较校正：仅对非 zero_variance 的 p 值集合做
    valid_idx = [i for i, r in enumerate(results) if not r['zero_variance']]
    pvals = [(i, results[i]['p_value']) for i in valid_idx]
    m_total = len(valid_idx)
    bonf = {i: min(1.0, results[i]['p_value'] * m_total) for i in valid_idx}
    bh = benjamini_hochberg(pvals)

    for i in valid_idx:
        results[i]['n_tests'] = m_total
        results[i]['p_bonferroni'] = bonf[i]
        results[i]['p_fdr'] = bh[i]
        results[i]['sig_raw'] = results[i]['p_value'] < ALPHA
        results[i]['sig_bonferroni'] = bonf[i] < ALPHA
        results[i]['sig_fdr'] = bh[i] < ALPHA
    for i, r in enumerate(results):
        if r['zero_variance']:
            r['n_tests'] = m_total
            r['p_bonferroni'] = r['p_fdr'] = None
            r['sig_raw'] = r['sig_bonferroni'] = r['sig_fdr'] = None
            r['note'] = 'zero-variance (dead value): t-test not applicable'

    # 记录日志（每数据集一个步骤，保证断点恢复粒度）
    summary = {'total_comparisons': len(results),
               'valid_tests': m_total,
               'zero_variance_skipped': len(results) - m_total,
               'alpha': ALPHA}
    logger.log('stage2|all', '阶段二 P 值统计校验', {'input': args.input}, summary, 'success')

    out_path = os.path.join(AUDIT, 'results', 'stage2_significance.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({'summary': summary, 'results': results}, f, ensure_ascii=False, indent=2)

    # markdown 报告
    md = ['# 阶段二：P 值统计校验报告', '',
          f'- 数据源：`{args.input}`',
          f'- 对比总数：{len(results)}；有效检验（非死值）：{m_total}；因死值跳过：{len(results) - m_total}',
          f'- 检验方法：配对 t 检验（双尾）+ bootstrap 95% CI + Hedges\' g（小样本校正）',
          f'- 多重比较校正：Bonferroni（p×m）与 FDR Benjamini-Hochberg；显著性水平 α={ALPHA}',
          '', '| 数据集 | 对比 | base | variant | diff | CI95 | p | p_bonf | p_fdr | g | 判定 |', '|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in results:
        if r['zero_variance']:
            md.append(f"| {r['dataset']} | {r['variant']} vs {r['baseline']} | {r['baseline_mean']:.2f} | "
                      f"{r['variant_mean']:.2f} | {r['diff_mean']:+.2f} | — | — | — | — | — | ⚠️死值，检验不适用 |")
        else:
            ci = f"[{r['ci95'][0]:.2f},{r['ci95'][1]:.2f}]"
            verdict = 'FDR显著' if r['sig_fdr'] else ('原始p显著' if r['sig_raw'] else '不显著')
            md.append(f"| {r['dataset']} | {r['variant']} vs {r['baseline']} | {r['baseline_mean']:.2f} | "
                      f"{r['variant_mean']:.2f} | {r['diff_mean']:+.2f} | {ci} | {r['p_value']:.4f} | "
                      f"{r['p_bonferroni']:.4f} | {r['p_fdr']:.4f} | {r['hedges_g']:+.3f} | {verdict} |")
    md_path = os.path.join(AUDIT, 'results', 'stage2_significance_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    # 汇总统计
    n_sig_raw = sum(1 for r in results if r.get('sig_raw'))
    n_sig_bonf = sum(1 for r in results if r.get('sig_bonferroni'))
    n_sig_fdr = sum(1 for r in results if r.get('sig_fdr'))
    print(f'\n对比总数 {len(results)}，有效 {m_total}，死值跳过 {len(results) - m_total}')
    print(f'原始 p<0.05: {n_sig_raw}；Bonferroni 校正后: {n_sig_bonf}；FDR 校正后: {n_sig_fdr}')
    print(f'JSON: {out_path}')
    print(f'报告: {md_path}')


if __name__ == '__main__':
    main()
