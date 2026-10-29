"""Efficiency analysis for the revised manuscript (reviewer R1-15, R4-17).

Aggregates wall-clock time, peak GPU memory, and estimated throughput from the
5-seed benchmark results produced by run_final.py (results_comp/*_cuda.json),
and writes a markdown table per dataset.

Note: exact per-sample throughput requires training-loop counters; here we
estimate throughput from the recorded wall-clock and dataset size where known.
If the result file lacks a dataset size map, only time/mem are reported.

Usage:
  python run_efficiency.py --results results_comp
  python run_efficiency.py --results results_comp --datasets synthetic cifar10 imdb
"""
import argparse
import glob
import json
import os
import numpy as np

# (train samples, epochs) per dataset config used in run_final.py
# epochs here match CONFIGS in run_final.py.
DATASET_SIZES = {
    'synthetic': (7200, 80),
    'cifar10':   (45000, 50),
    'cifar100':  (45000, 50),
    'rte':       (2241, 5),
    'trec6':     (4907, 5),
    'imdb':      (22500, 3),
    'sst2':      (60614, 5),
    'mrpc':      (3297, 5),
    'sst2_b':    (60614, 5),
    'mrpc_b':    (3297, 5),
    'sst2_r':    (60614, 5),
    'mrpc_r':    (3297, 5),
}


def load(results_dir, ds):
    for pat in (f'{ds}_cuda.json', f'{ds}_dml0.json', f'{ds}_cpu.json'):
        p = os.path.join(results_dir, pat)
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default='results_comp')
    ap.add_argument('--datasets', nargs='+', default=['all'])
    ap.add_argument('--out', default='efficiency_table.md')
    args = ap.parse_args()

    if args.datasets == ['all']:
        files = sorted(glob.glob(os.path.join(args.results, '*_cuda.json')))
        datasets = sorted({os.path.basename(f).replace('_cuda.json', '') for f in files})
    else:
        datasets = args.datasets

    lines = ['# Efficiency analysis (wall-clock / GPU memory / throughput)',
             '',
             '> Aggregated over 5 seeds. `time` = mean wall-clock per run (s); '
             '`mem` = mean peak GPU memory (MB); `throughput` = samples/s '
             '(estimated = train_samples * epochs / mean_time).',
             '']
    for ds in datasets:
        data = load(args.results, ds)
        if data is None:
            print(f'[skip] {ds}: no results')
            continue
        lines.append(f'## {ds}')
        lines.append('')
        lines.append('| Strategy | time_mean(s) | mem_mean(MB) | throughput(sps) |')
        lines.append('|---:|---:|---:|---:|')
        n_tr, ep = DATASET_SIZES.get(ds, (None, None))
        for name in sorted(data, key=lambda k: -(data[k].get('acc_mean') or 0)):
            v = data[name]
            seeds = v.get('per_seed', [])
            times = [r.get('time') for r in seeds if r.get('time') is not None]
            mems = [r.get('mem') for r in seeds if r.get('mem') is not None]
            if not times:
                continue
            t_mean = float(np.mean(times))
            m_mean = float(np.mean(mems)) if mems else float('nan')
            if n_tr and ep and t_mean > 0:
                sps = n_tr * ep / t_mean
            else:
                sps = float('nan')
            lines.append(f'| {name} | {t_mean:.1f} | {m_mean:.0f} | {sps:.1f} |')
        lines.append('')
    md = '\n'.join(lines)
    if args.out:
        with open(args.out, 'w') as f:
            f.write(md)
        print(f'Saved {args.out}')
    else:
        print(md)


if __name__ == '__main__':
    main()
