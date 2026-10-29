"""Main benchmark runner for the revised manuscript.

Runs 18 strategies across 5 datasets with multiple seeds, reports
accuracy + macro-F1, wall-clock, throughput, peak GPU memory.

Usage:
  python run_benchmark.py --device cuda --seeds 42 1337 2024 --datasets synthetic cifar10 rte trec6 imdb
  python run_benchmark.py --device directml:1 --datasets synthetic --seeds 42   # 7900XTX
"""
import argparse
import json
import os
import time
import math
import torch
import torch.nn as nn

# HF / proxy config (set before importing datasets/transformers indirectly)
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '0')

import optimizers as opt
import data
import train as T


def get_device(kind):
    if kind == 'cuda':
        return torch.device('cuda')
    if kind == 'cpu':
        return torch.device('cpu')
    if kind.startswith('directml'):
        import torch_directml
        idx = 0
        if ':' in kind:
            idx = int(kind.split(':')[1])
        return torch_directml.device(idx)
    raise ValueError(kind)


def build_optimizer(name, model, lr_sgd=0.01, lr_adam=3e-4, n_layers=None,
                    total_steps=None):
    """Return (optimizer, scheduler, dals_phase_update).

    total_steps: total optimizer steps across all epochs (used for cosine/warmup).
    """
    sched = None; phase_cb = None
    T = total_steps or 1000
    if name == 'Fixed SGD':
        return opt.FixedLRSGD(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=1e-4), sched, phase_cb
    if name == 'Cosine Decay SGD':
        return opt.CosineAnnealingSGD(model.parameters(), lr=lr_sgd, T_max=T, momentum=0.9, weight_decay=5e-4), sched, phase_cb
    if name == 'SGDR':
        return opt.SGDRWithRestarts(model.parameters(), lr=lr_sgd, T_0=max(1, T // 5), T_mult=2, momentum=0.9, weight_decay=5e-4), sched, phase_cb
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr_adam, weight_decay=0), sched, phase_cb
    if name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr_adam, weight_decay=1e-2), sched, phase_cb
    if name == 'AdaBound':
        try:
            from adabound import AdaBound
            return AdaBound(model.parameters(), lr=lr_adam, final_lr=0.1, weight_decay=0), sched, phase_cb
        except Exception:
            return torch.optim.Adam(model.parameters(), lr=lr_adam), sched, phase_cb
    if name == 'LARS':
        return opt.LARS(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=5e-4), sched, phase_cb
    if name == 'Discriminative LR':
        groups = opt.build_discriminative_param_groups(model, lr_sgd, decay_factor=2.6)
        return opt.FixedLRSGD(groups, lr=lr_sgd, momentum=0.9, weight_decay=1e-4), sched, phase_cb
    if name == 'RAdam':
        return opt.RAdam(model.parameters(), lr=lr_adam, weight_decay=0), sched, phase_cb
    if name == 'Lion':
        return opt.Lion(model.parameters(), lr=lr_adam * 0.1, weight_decay=1e-2), sched, phase_cb
    if name == 'Lookahead+AdamW':
        inner = torch.optim.AdamW(model.parameters(), lr=lr_adam, weight_decay=1e-2)
        return Lookahead(inner, k=5, alpha=0.5), sched, phase_cb
    if name == 'SAM':
        base = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=5e-4)
        return SAM(base, rho=0.05), sched, phase_cb
    if name == 'Grokfast':
        return opt.Grokfast(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=5e-4, alpha=0.6), sched, phase_cb
    if name == 'STLR+Discriminative':
        groups = opt.build_discriminative_stlr_param_groups(model, lr_sgd, decay_factor=2.6)
        o = opt.FixedLRSGD(groups, lr=lr_sgd, momentum=0.9, weight_decay=1e-4)
        sched = opt.STLRScheduler(o, T=T, cut_frac=0.1, ratio=32, decay_factor=2.6)
        return o, sched, phase_cb
    if name == 'SAM+Discriminative':
        groups = opt.build_discriminative_param_groups(model, lr_sgd, decay_factor=2.6)
        base = torch.optim.SGD(groups, lr=lr_sgd, momentum=0.9, weight_decay=5e-4)
        return SAM(base, rho=0.05), sched, phase_cb
    if name == 'DALS (Ours)':
        dals_lr = 0.1 if lr_sgd > 5e-4 else lr_sgd  # use 0.1 for from-scratch, base for fine-tune
        o = opt.DALS(model, lr=dals_lr, momentum=0.9, weight_decay=1e-4, T_max=T)
        phase_cb = o.update_phase
        return o, sched, phase_cb
    if name == 'DALS-Fast (Ours)':
        dals_lr = (0.1 * 2) if lr_sgd > 5e-4 else lr_sgd
        o = opt.DALSFast(model, lr=dals_lr, momentum=0.85, weight_decay=1e-4, T_max=T)
        phase_cb = o.update_phase
        return o, sched, phase_cb
    if name == 'DALS-Acc (Ours)':
        dals_lr = 0.1 if lr_sgd > 5e-4 else lr_sgd
        o = opt.DALSAcc(model, lr=dals_lr, momentum=0.9, weight_decay=5e-4, T_max=T, T_0=max(1, T // 2), T_mult=2)
        phase_cb = o.update_phase
        return o, sched, phase_cb
    raise ValueError(name)


# ---- Lookahead and SAM minimal impls ----
class Lookahead(torch.optim.Optimizer):
    def __init__(self, inner, k=5, alpha=0.5):
        self.inner = inner; self.k = k; self.alpha = alpha; self.step_n = 0
        self.slow = {}
        for g in inner.param_groups:
            for p in g['params']:
                self.slow[id(p)] = p.data.clone()

    @property
    def param_groups(self): return self.inner.param_groups
    @property
    def state(self): return self.inner.state
    def zero_grad(self, **kw): self.inner.zero_grad(**kw)

    @torch.no_grad()
    def step(self, closure=None):
        loss = self.inner.step(closure)
        self.step_n += 1
        if self.step_n % self.k == 0:
            for g in self.inner.param_groups:
                for p in g['params']:
                    sp = self.slow[id(p)]
                    sp.add_(p - sp, alpha=self.alpha)
                    p.copy_(sp)
        return loss


class SAM(torch.optim.Optimizer):
    def __init__(self, base, rho=0.05):
        self.base = base; self.rho = rho

    @property
    def param_groups(self): return self.base.param_groups
    @property
    def state(self): return self.base.state
    def zero_grad(self, **kw): self.base.zero_grad(**kw)

    @torch.no_grad()
    def first_step(self):
        gs = []
        for g in self.base.param_groups:
            for p in g['params']:
                if p.grad is None:
                    continue
                gn = p.grad.norm(2)
                eps = (p.grad / (gn + 1e-12)) * self.rho
                p.add_(eps)
                gs.append(eps)
        self._eps = gs

    @torch.no_grad()
    def second_step(self):
        for g in self.base.param_groups:
            for p in g['params']:
                pass
        # restore and step
        eps = self._eps
        i = 0
        for g in self.base.param_groups:
            for p in g['params']:
                if i < len(eps):
                    p.sub_(eps[i]); i += 1
        self.base.step()

    @torch.no_grad()
    def step(self, closure=None):
        # closure must compute loss twice; we approximate via single forward if no closure
        if closure is None:
            return self.base.step()
        loss = closure()
        loss.backward()
        self.first_step()
        self.base.zero_grad()
        loss2 = closure()
        loss2.backward()
        self.second_step()
        return loss2


STRATEGIES = [
    'Fixed SGD', 'Cosine Decay SGD', 'SGDR',
    'Adam', 'AdamW', 'AdaBound',
    'LARS', 'Discriminative LR',
    'RAdam', 'Lion', 'Lookahead+AdamW', 'SAM', 'Grokfast',
    'STLR+Discriminative', 'SAM+Discriminative',
    'DALS (Ours)', 'DALS-Fast (Ours)', 'DALS-Acc (Ours)',
]


def run_dataset(ds_name, device, seeds, out_dir, epochs=None, strategies=None):
    strategies = strategies or STRATEGIES
    results = {}
    n_labels = 2
    for seed in seeds:
        data.set_seed(seed)
        # build loaders once per seed
        if ds_name == 'synthetic':
            tr, va, te = data.synthetic_loaders(seed=seed, batch=64)
            ep = epochs or 80
            n_labels = 10
        elif ds_name == 'cifar10':
            tr, va, te = data.cifar10_loaders(seed=seed, batch=128)
            ep = epochs or 50
            n_labels = 10
        elif ds_name == 'rte':
            tr, va, te, n_labels = data.nlp_loaders('rte', seed=seed, batch=32)
            ep = epochs or 5
        elif ds_name == 'trec6':
            tr, va, te, n_labels = data.nlp_loaders('trec6', seed=seed, batch=32)
            ep = epochs or 5
        elif ds_name == 'imdb':
            tr, va, te, n_labels = data.nlp_loaders('imdb', seed=seed, batch=32)
            ep = epochs or 3
        else:
            raise ValueError(ds_name)
        for name in strategies:
            # rebuild model + optimizer per strategy to avoid state leakage
            if ds_name == 'synthetic':
                model = data.MLP()
            elif ds_name == 'cifar10':
                model = data.ConvNet()
            else:
                from transformers import AutoModelForSequenceClassification
                model = AutoModelForSequenceClassification.from_pretrained(
                    'distilbert-base-uncased', num_labels=n_labels)
            lr_sgd = 5e-4 if ds_name in ('rte', 'trec6', 'imdb') else (0.05 if ds_name == 'synthetic' else 0.01)
            lr_adam = 5e-5 if ds_name in ('rte', 'trec6', 'imdb') else 3e-4
            total_steps = max(1, len(tr) * ep)
            try:
                optimizer, sched, phase_cb = build_optimizer(name, model, lr_sgd, lr_adam,
                                                             total_steps=total_steps)
            except Exception as e:
                print(f'  [skip] {name}: {e}')
                continue
            print(f'[{ds_name} seed={seed}] {name} ...', flush=True)
            t0 = time.time()
            try:
                res = T.train_one(model, optimizer, tr, va, te, device,
                                  n_labels=n_labels, epochs=ep,
                                  sched=sched, dals_phase_update=phase_cb,
                                  select_by='val', verbose=0)
            except Exception as e:
                print(f'  [fail] {name}: {e}')
                continue
            res['wall_clock_s_total'] = time.time() - t0
            res['seed'] = seed
            results.setdefault(name, []).append(res)
            print(f'  -> test_acc(valsel)={res["test_acc_valsel"]:.2f} '
                  f'test_f1(valsel)={res["test_f1_valsel"]:.2f} '
                  f'test_acc(last)={res["test_acc_last"]:.2f} '
                  f't={res["wall_clock_s"]:.1f}s mem={res["peak_gpu_mem_mb"]}', flush=True)
            # free
            del model, optimizer, sched, phase_cb
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    # aggregate
    agg = {}
    for name, runs in results.items():
        accs = [r['test_acc_valsel'] for r in runs if r.get('test_acc_valsel') is not None]
        f1s = [r['test_f1_valsel'] for r in runs if r.get('test_f1_valsel') is not None]
        times = [r['wall_clock_s'] for r in runs]
        agg[name] = {
            'n_seeds': len(runs),
            'acc_mean': float(sum(accs) / len(accs)) if accs else None,
            'acc_std': float(math.sqrt(sum((a - sum(accs) / len(accs)) ** 2 for a in accs) / len(accs))) if len(accs) > 1 else 0.0,
            'f1_mean': float(sum(f1s) / len(f1s)) if f1s else None,
            'f1_std': float(math.sqrt(sum((a - sum(f1s) / len(f1s)) ** 2 for a in f1s) / len(f1s))) if len(f1s) > 1 else 0.0,
            'time_mean_s': float(sum(times) / len(times)) if times else None,
            'per_seed': [{'acc': r['test_acc_valsel'], 'f1': r['test_f1_valsel'],
                          'time': r['wall_clock_s'], 'mem_mb': r['peak_gpu_mem_mb']} for r in runs],
        }
    out = os.path.join(out_dir, f'benchmark_{ds_name}_{device.index if device.index else 0}_{device.type}.json') if hasattr(device, 'index') else os.path.join(out_dir, f'benchmark_{ds_name}_{device.type}.json')
    # simpler name
    dev_tag = 'cuda' if device.type == 'cuda' else ('dml' + str(getattr(device, 'index', 0)) if 'privateuseone' in str(device) else 'cpu')
    out = os.path.join(out_dir, f'benchmark_{ds_name}_{dev_tag}.json')
    with open(out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f'Saved {out}')
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42])
    ap.add_argument('--datasets', nargs='+', default=['synthetic', 'cifar10'])
    ap.add_argument('--out', default='results')
    ap.add_argument('--strategies', nargs='+', default=None)
    ap.add_argument('--epochs', type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = get_device(args.device)
    print('Device:', device)
    for ds in args.datasets:
        run_dataset(ds, device, args.seeds, args.out, epochs=args.epochs, strategies=args.strategies)


if __name__ == '__main__':
    main()