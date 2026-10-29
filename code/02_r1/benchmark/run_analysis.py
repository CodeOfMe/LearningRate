"""Sensitivity analysis, multi-seed and efficiency aggregation for DALS.

- Sensitivity: vary phase thresholds, loss-EMA coeff, depth-blend coeffs,
  trust coef, warmup frac on synthetic (reviewer R4-14).
- Multi-seed: run DALS + key baselines on 5 seeds, report mean±std and 95% CI
  (reviewers R1-10, R4-4).
- Efficiency: aggregate wall-clock, throughput, peak GPU memory from benchmark
  JSONs (reviewers R1-15, R4-17).
"""
import argparse
import json
import os
import math
import time
import numpy as np
import torch

import optimizers as opt
import data
import train as T


def run_sensitivity(device, seeds, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    base = dict(lr=0.1, momentum=0.9, weight_decay=1e-4, T_max=80 * 90, variant='full')
    grids = {
        'phase1_thresh': [0.005, 0.01, 0.02, 0.05],
        'phase2_thresh': [0.001, 0.002, 0.005, 0.01],
        'grokfast_alpha': [0.4, 0.5, 0.6, 0.7, 0.8],
        'trust_coef': [0.01, 0.02, 0.05, 0.1],
        'warmup_frac': [0.0, 0.02, 0.05, 0.1],
    }
    res = {}
    for param, vals in grids.items():
        res[param] = {}
        for v in vals:
            runs = []
            for seed in seeds:
                data.set_seed(seed)
                tr, va, te = data.synthetic_loaders(seed=seed, batch=64)
                total_steps = 80 * len(tr)
                kw = dict(base); kw['T_max'] = total_steps; kw[param] = v
                model = data.MLP().to(device)
                o = opt.DALS(model, **kw)
                pcb = o.update_phase
                try:
                    r = T.train_one(model, o, tr, va, te, device, n_labels=10,
                                    epochs=80, dals_phase_update=pcb, select_by='val', verbose=0)
                    runs.append(r['test_acc_valsel'])
                    print(f'[sens {param}={v} seed={seed}] acc={r["test_acc_valsel"]:.2f}', flush=True)
                except Exception as e:
                    print(f'  [fail] {e}', flush=True)
                del model, o
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            res[param][str(v)] = {'mean': float(np.mean(runs)) if runs else None,
                                  'std': float(np.std(runs)) if len(runs) > 1 else 0.0,
                                  'per_seed': runs}
    out = os.path.join(out_dir, 'sensitivity.json')
    with open(out, 'w') as f:
        json.dump(res, f, indent=2)
    print(f'Saved {out}')


def run_multiseed(device, seeds, out_dir, strategies, datasets):
    from run_benchmark import build_optimizer, STRATEGIES
    os.makedirs(out_dir, exist_ok=True)
    all_res = {}
    for ds in datasets:
        all_res[ds] = {}
        for name in strategies:
            runs = []
            for seed in seeds:
                data.set_seed(seed)
                if ds == 'synthetic':
                    tr, va, te = data.synthetic_loaders(seed=seed, batch=64); ep = 80; nl = 10; model = data.MLP()
                elif ds == 'cifar10':
                    tr, va, te = data.cifar10_loaders(seed=seed, batch=128); ep = 50; nl = 10; model = data.ConvNet()
                elif ds == 'rte':
                    tr, va, te, nl = data.nlp_loaders('rte', seed=seed, batch=32); ep = 5
                    from transformers import AutoModelForSequenceClassification
                    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=nl)
                else:
                    continue
                total_steps = len(tr) * ep
                lr_sgd = 0.05 if ds == 'synthetic' else (5e-4 if ds == 'rte' else 0.01)
                lr_adam = 3e-4 if ds != 'rte' else 5e-5
                try:
                    o, sched, pcb = build_optimizer(name, model, lr_sgd, lr_adam, total_steps=total_steps)
                    r = T.train_one(model, o, tr, va, te, device, n_labels=nl, epochs=ep,
                                     sched=sched, dals_phase_update=pcb, select_by='val', verbose=0)
                    runs.append({'acc': r['test_acc_valsel'], 'f1': r['test_f1_valsel'],
                                 'time': r['wall_clock_s'], 'mem': r['peak_gpu_mem_mb']})
                    print(f'[multiseed {ds} {name} seed={seed}] acc={r["test_acc_valsel"]:.2f}±', flush=True)
                except Exception as e:
                    print(f'  [fail] {name}: {e}', flush=True)
                del model, o
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
            accs = [r['acc'] for r in runs if r['acc'] is not None]
            f1s = [r['f1'] for r in runs if r['f1'] is not None]
            n = len(accs)
            mean = float(np.mean(accs)) if accs else None
            std = float(np.std(accs)) if n > 1 else 0.0
            # 95% CI (t approx for small n)
            ci = 0.0
            if n > 1:
                t = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(n, 1.96)
                ci = t * std / math.sqrt(n)
            all_res[ds][name] = {'n': n, 'acc_mean': mean, 'acc_std': std,
                                'acc_ci95': ci,
                                'f1_mean': float(np.mean(f1s)) if f1s else None,
                                'per_seed': runs}
    out = os.path.join(out_dir, 'multiseed.json')
    with open(out, 'w') as f:
        json.dump(all_res, f, indent=2)
    print(f'Saved {out}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['sensitivity', 'multiseed', 'efficiency'], default='sensitivity')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 1337, 2024])
    ap.add_argument('--datasets', nargs='+', default=['synthetic'])
    ap.add_argument('--strategies', nargs='+', default=['DALS (Ours)', 'Fixed SGD', 'AdamW', 'RAdam'])
    ap.add_argument('--out', default='results')
    args = ap.parse_args()
    dev = torch.device('cuda') if args.device == 'cuda' else torch.device('cpu')
    if args.mode == 'sensitivity':
        run_sensitivity(dev, args.seeds, args.out)
    elif args.mode == 'multiseed':
        run_multiseed(dev, args.seeds, args.out, args.strategies, args.datasets)
    elif args.mode == 'efficiency':
        # aggregate existing benchmark jsons
        files = [f for f in os.listdir(args.out) if f.startswith('benchmark_') and f.endswith('.json')]
        agg = {}
        for f in files:
            ds = f.replace('benchmark_', '').replace('_cuda.json', '').replace('_dml0.json', '').replace('_cpu.json', '')
            with open(os.path.join(args.out, f)) as fh:
                d = json.load(fh)
            agg[ds] = {}
            for name, v in d.items():
                agg[ds][name] = {'time_mean_s': v.get('time_mean_s'), 'per_seed': v.get('per_seed', [])}
        out = os.path.join(args.out, 'efficiency.json')
        with open(out, 'w') as fh:
            json.dump(agg, fh, indent=2)
        print(f'Saved {out}')


if __name__ == '__main__':
    main()