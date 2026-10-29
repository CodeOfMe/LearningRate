"""Ablation study runner for DALS (reviewer R1-8, R4-9).

Compares 9 DALS variants on synthetic + CIFAR-10:
  full, cosine_only, trust_only, filter_only, phase_only, depth_only,
  cosine_trust, cosine_filter, filter_trust.
"""
import argparse
import json
import os
import time
import torch
import numpy as np

import optimizers as opt
import data
import train as T

VARIANTS = ['full', 'cosine_only', 'trust_only', 'filter_only', 'phase_only',
            'depth_only', 'cosine_trust', 'cosine_filter', 'filter_trust']


def run(device, seeds, out_dir, datasets=('synthetic', 'cifar10')):
    os.makedirs(out_dir, exist_ok=True)
    res = {}
    for ds in datasets:
        res[ds] = {}
        for variant in VARIANTS:
            runs = []
            for seed in seeds:
                data.set_seed(seed)
                if ds == 'synthetic':
                    tr, va, te = data.synthetic_loaders(seed=seed, batch=64)
                    ep = 80; n_labels = 10; model = data.MLP()
                else:
                    tr, va, te = data.cifar10_loaders(seed=seed, batch=128)
                    ep = 50; n_labels = 10; model = data.ConvNet()
                total_steps = len(tr) * ep
                lr = 0.1 if ds == 'synthetic' else 0.01
                o = opt.DALS(model, lr=lr, momentum=0.9, weight_decay=1e-4,
                             T_max=total_steps, variant=variant)
                pcb = o.update_phase
                print(f'[ablation {ds} seed={seed} {variant}] ...', flush=True)
                try:
                    r = T.train_one(model, o, tr, va, te, device, n_labels=n_labels,
                                    epochs=ep, dals_phase_update=pcb, select_by='val', verbose=0)
                    print(f'  -> acc={r["test_acc_valsel"]:.2f} f1={r["test_f1_valsel"]:.2f} t={r["wall_clock_s"]:.1f}s', flush=True)
                    runs.append({'acc': r['test_acc_valsel'], 'f1': r['test_f1_valsel'],
                                 'time': r['wall_clock_s']})
                except Exception as e:
                    print(f'  [fail] {variant}: {e}', flush=True)
                del model, o
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            accs = [r['acc'] for r in runs if r['acc'] is not None]
            f1s = [r['f1'] for r in runs if r['f1'] is not None]
            res[ds][variant] = {
                'n_seeds': len(runs),
                'acc_mean': float(np.mean(accs)) if accs else None,
                'acc_std': float(np.std(accs)) if len(accs) > 1 else 0.0,
                'f1_mean': float(np.mean(f1s)) if f1s else None,
                'per_seed': runs,
            }
    out = os.path.join(out_dir, 'ablation.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'Saved {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    ap.add_argument('--datasets', nargs='+', default=['synthetic', 'cifar10'])
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    dev = torch.device('cuda') if args.device == 'cuda' else torch.device('cpu')
    run(dev, args.seeds, args.out, tuple(args.datasets))


if __name__ == '__main__':
    main()