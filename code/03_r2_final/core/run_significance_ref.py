"""Statistical significance analysis for the revised manuscript.

Reads the 5-seed benchmark results (results_comp/*_cuda.json) produced by
run_final.py and reports, per dataset and for each DALS variant vs. each
baseline:

  - paired t-test (scipy.stats.ttest_rel) on the 5 per-seed accuracies
  - bootstrap 95% CI of the mean difference (paired, 10k resamples)
  - mean difference + Hedges' g effect size (optional)

This addresses reviewers R1-10 and R4-4 ("statistical significance:
multi-seed mean±std, CI, and statistical tests").

Usage:
  python run_significance.py --results results_comp --datasets synthetic cifar10 rte trec6 imdb sst2 mrpc
  python run_significance.py --results results_comp --datasets all        # all datasets found
  python run_significance.py --results results_comp --only dals --out significance_table.md
"""
import argparse
import json
import glob
import os
import numpy as np
from scipy import stats

DALS_VARIANTS = ['DALS', 'DALS_Fast', 'DALS_Acc']


def load_dataset_results(results_dir, ds):
    for pattern in (f'{ds}_cuda.json', f'{ds}_dml0.json', f'{ds}_cpu.json'):
        p = os.path.join(results_dir, pattern)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


def paired_test(name, baseline, variant):
    """Paired t-test + bootstrap CI between a baseline and a DALS variant.

    Returns dict with p_value, diff_mean, ci95, hedges_g, n.
    Uses per-seed accuracies (matched seeds).
    """
    seeds_b = {r.get('seed') for r in baseline.get('per_seed', []) if r.get('seed') is not None}
    seeds_v = {r.get('seed') for r in variant.get('per_seed', []) if r.get('seed') is not None}
    seeds = sorted(seeds_b & seeds_v)
    if len(seeds) < 2:
        return None
    b_by_seed = {r.get('seed'): r.get('acc') for r in baseline.get('per_seed', [])}
    v_by_seed = {r.get('seed'): r.get('acc') for r in variant.get('per_seed', [])}
    b = np.array([b_by_seed[s] for s in seeds], dtype=float)
    v = np.array([v_by_seed[s] for s in seeds], dtype=float)
    if np.any(np.isnan(b)) or np.any(np.isnan(v)):
        return None
    d = v - b
    n = len(d)
    if np.std(d, ddof=1) == 0:
        p = 1.0 if np.allclose(d, 0) else 0.0
        ci = (float(np.mean(d)),) * 2
    else:
        t, p = stats.ttest_rel(v, b)
        # bootstrap 95% CI of mean difference (paired resample)
        rng = np.random.default_rng(42)
        bs = np.array([np.mean(rng.choice(d, n, replace=True)) for _ in range(10000)])
        ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    # Hedges' g (corrected Cohen's d) on the differences
    sd = np.std(d, ddof=1)
    g = float(np.mean(d) / sd) if sd > 0 else 0.0
    correction = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    return {
        'n': n,
        'baseline_mean': float(np.mean(b)),
        'variant_mean': float(np.mean(v)),
        'diff_mean': float(np.mean(d)),
        'ci95': ci,
        'p_value': float(p),
        'hedges_g': g * correction,
    }


def run(results_dir, datasets, only=None):
    all_files = sorted(glob.glob(os.path.join(results_dir, '*_cuda.json')))
    if datasets == ['all']:
        datasets = sorted({os.path.basename(p).replace('_cuda.json', '') for p in all_files})
    out = {}
    for ds in datasets:
        data = load_dataset_results(results_dir, ds)
        if data is None:
            print(f'[skip] {ds}: no results file')
            continue
        variants = DALS_VARIANTS if only is None or only == 'dals' else [
            k for k in (only,) if k in data]
        out[ds] = {}
        for vname in variants:
            v = data.get(vname)
            if not v or not v.get('per_seed'):
                continue
            for bname, b in data.items():
                if bname == vname or not b.get('per_seed'):
                    continue
                stat = paired_test(bname, b, v)
                if stat is not None:
                    out[ds][f'{vname} vs {bname}'] = stat
    return out


def format_markdown(out):
    lines = ['# Statistical significance (paired t-test + bootstrap 95% CI)',
             '',
             '> DALS variant vs. each baseline, matched 5 seeds, accuracy (%). '
             '`diff` = variant - baseline; `CI95` = bootstrap interval of the mean '
             'difference; `p` = paired t-test; `g` = Hedges\' g.',
             '']
    for ds in sorted(out):
        lines.append(f'## {ds}')
        lines.append('')
        lines.append('| Comparison | n | base | variant | diff | CI95 | p | g |')
        lines.append('|---|---:|---:|---:|---:|---:|---:|---:|')
        for cmp_name, s in sorted(out[ds].items()):
            ci = f'[{s["ci95"][0]:.2f}, {s["ci95"][1]:.2f}]'
            lines.append(f'| {cmp_name} | {s["n"]} | {s["baseline_mean"]:.2f} | '
                         f'{s["variant_mean"]:.2f} | {s["diff_mean"]:+.2f} | {ci} | '
                         f'{s["p_value"]:.4f} | {s["hedges_g"]:+.3f} |')
        lines.append('')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results_comp')
    ap.add_argument('--datasets', nargs='+', default=['all'])
    ap.add_argument('--only', default=None,
                    help='Only compare a specific variant (default: the 3 DALS variants)')
    ap.add_argument('--out', default='significance_table.md')
    args = ap.parse_args()
    res = run(args.results, args.datasets, args.only)
    md = format_markdown(res)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(md)
        print(f'Saved {args.out}')
    else:
        print(md)


if __name__ == '__main__':
    main()
