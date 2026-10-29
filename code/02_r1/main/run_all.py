"""Comprehensive benchmark: all reviewer-requested datasets, both GPUs.
REAL-TIME progress. Incremental saving. Robust error handling.

Datasets: synthetic, CIFAR-10, CIFAR-100, RTE, TREC-6, IMDb, SST-2, MRPC
Models: MLP, ConvNet, ConvNet100, DistilBERT
GPUs: 4070 (CUDA), 7900XTX (DirectML)
Seeds: 5 (42, 1337, 2024, 5555, 9999)
Metrics: accuracy, macro-F1, wall-clock, GPU memory
"""
import argparse, json, os, sys, time, math, traceback
import numpy as np, torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimizers as opt
import data
import train as T

STRATEGIES = [
    'Fixed SGD', 'Cosine Decay SGD', 'Adam', 'AdamW',
    'LARS', 'Discriminative LR',
    'RAdam', 'Lion', 'SAM', 'Grokfast',
    'STLR+Discriminative', 'SAM+Discriminative',
    'DALS', 'DALS-Fast', 'DALS-Acc',
]

SEEDS = [42, 1337, 2024, 5555, 9999]
EPOCHS_FS = 50   # from-scratch epochs (reduced for speed)
EPOCHS_FT = 3    # fine-tune epochs


def get_device(kind):
    if kind == 'cuda':
        return torch.device('cuda'), 'cuda'
    if kind.startswith('dml'):
        import torch_directml
        idx = int(kind.split(':')[1]) if ':' in kind else 0
        return torch_directml.device(idx), f'dml{idx}'
    return torch.device('cpu'), 'cpu'


def build_model(cfg, n_labels, device):
    m = cfg['model']
    if m == 'MLP':
        return data.MLP().to(device)
    if m == 'ConvNet':
        return data.ConvNet(n_classes=n_labels).to(device)
    if m == 'ConvNet_large':
        return data.ConvNet100(n_classes=n_labels).to(device)
    if m == 'DistilBERT':
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=n_labels).to(device)
    raise ValueError(m)


def build_loaders(cfg, seed):
    ds = cfg['dataset']
    if ds == 'synthetic':
        return data.synthetic_loaders(seed=seed, batch=64), 10
    if ds in ('cifar10', 'cifar100'):
        return data.cifar_loaders(ds, seed=seed, batch=128), (100 if ds == 'cifar100' else 10)
    return data.nlp_loaders(cfg['nlp_task'], seed=seed, batch=32, max_len=128), cfg['nl']


def build_opt(name, model, lr_sgd, lr_adam, total_steps):
    T = total_steps
    sched = None; pcb = None
    def nop(x): pass

    if name == 'Fixed SGD':
        return opt.FixedLRSGD(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=1e-4), sched, nop
    if name == 'Cosine Decay SGD':
        return opt.CosineAnnealingSGD(model.parameters(), lr=lr_sgd, T_max=T, momentum=0.9, weight_decay=5e-4), sched, nop
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr_adam, weight_decay=0), sched, nop
    if name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr_adam, weight_decay=1e-2), sched, nop
    if name == 'LARS':
        return opt.LARS(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=5e-4), sched, nop
    if name == 'Discriminative LR':
        groups = opt.build_discriminative_param_groups(model, lr_sgd, decay_factor=2.6)
        return opt.FixedLRSGD(groups, lr=lr_sgd, momentum=0.9, weight_decay=1e-4), sched, nop
    if name == 'RAdam':
        return opt.RAdam(model.parameters(), lr=lr_adam, weight_decay=0), sched, nop
    if name == 'Lion':
        return opt.Lion(model.parameters(), lr=lr_adam*0.1, weight_decay=1e-2), sched, nop
    if name == 'SAM':
        base = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=5e-4)
        return SAMW(base, rho=0.05), sched, nop
    if name == 'Grokfast':
        return opt.Grokfast(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=5e-4, alpha=0.6), sched, nop
    if name == 'STLR+Discriminative':
        groups = opt.build_discriminative_stlr_param_groups(model, lr_sgd, decay_factor=2.6)
        o = opt.FixedLRSGD(groups, lr=lr_sgd, momentum=0.9, weight_decay=1e-4)
        sched = opt.STLRScheduler(o, T=T, cut_frac=0.1, ratio=32, decay_factor=2.6)
        return o, sched, nop
    if name == 'SAM+Discriminative':
        groups = opt.build_discriminative_param_groups(model, lr_sgd, decay_factor=2.6)
        base = torch.optim.SGD(groups, lr=lr_sgd, momentum=0.9, weight_decay=5e-4)
        return SAMW(base, rho=0.05), sched, nop
    if name == 'DALS':
        dlr = 0.1 if lr_sgd > 5e-4 else lr_sgd
        o = opt.DALS(model, lr=dlr, momentum=0.9, weight_decay=1e-4, T_max=T)
        return o, sched, o.update_phase
    if name == 'DALS-Fast':
        dlr = 0.1 if lr_sgd > 5e-4 else lr_sgd
        o = opt.DALSFast(model, lr=dlr, momentum=0.85, weight_decay=1e-4, T_max=T)
        return o, sched, o.update_phase
    if name == 'DALS-Acc':
        dlr = 0.1 if lr_sgd > 5e-4 else lr_sgd
        o = opt.DALSAcc(model, lr=dlr, momentum=0.9, weight_decay=5e-4, T_max=T, T_0=max(1,T//2), T_mult=2)
        return o, sched, o.update_phase
    raise ValueError(name)


class SAMW(torch.optim.Optimizer):
    def __init__(self, base, rho=0.05):
        self.base = base; self.rho = rho; self._eps_list = []
    @property
    def param_groups(self): return self.base.param_groups
    def zero_grad(self, **kw): self.base.zero_grad(**kw)
    @torch.no_grad()
    def step(self, closure=None):
        if closure is None: return self.base.step()
        loss = closure()
        loss.backward()
        self._eps_list.clear()
        for g in self.base.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    gn = p.grad.norm(2) + 1e-12
                    eps = (p.grad / gn) * self.rho
                    p.add_(eps); self._eps_list.append(eps)
        self.base.zero_grad()
        loss2 = closure()
        loss2.backward()
        i = 0
        for g in self.base.param_groups:
            for p in g['params']:
                if i < len(self._eps_list):
                    p.sub_(self._eps_list[i]); i += 1
        self.base.step()
        return loss2


def load_pbar(cfg, device):
    """Progress bar that updates from train_one callback."""
    pbar = None
    def cb(epoch, total, loss):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc=f'{cfg["dataset"]:8s}', unit='ep', leave=False)
        pbar.update(1)
        pbar.set_postfix({'loss': f'{loss:.4f}'})
    return pbar, cb


CONFIGS = [
    {'dataset': 'synthetic',  'model': 'MLP', 'ep': EPOCHS_FS, 'nl': 10, 'type': 'fs', 'lr_sgd': 0.05, 'lr_adam': 3e-4},
    {'dataset': 'cifar10',    'model': 'ConvNet', 'ep': EPOCHS_FS, 'nl': 10, 'type': 'fs', 'lr_sgd': 0.01, 'lr_adam': 3e-4},
    {'dataset': 'cifar100',   'model': 'ConvNet_large', 'ep': EPOCHS_FS, 'nl': 100, 'type': 'fs', 'lr_sgd': 0.01, 'lr_adam': 3e-4},
    {'dataset': 'rte',        'model': 'DistilBERT', 'ep': EPOCHS_FT, 'nl': 2, 'type': 'ft', 'lr_sgd': 5e-4, 'lr_adam': 5e-5, 'nlp_task': 'rte'},
    {'dataset': 'trec6',      'model': 'DistilBERT', 'ep': EPOCHS_FT, 'nl': 6, 'type': 'ft', 'lr_sgd': 5e-4, 'lr_adam': 5e-5, 'nlp_task': 'trec6'},
    {'dataset': 'imdb',       'model': 'DistilBERT', 'ep': EPOCHS_FT, 'nl': 2, 'type': 'ft', 'lr_sgd': 5e-4, 'lr_adam': 5e-5, 'nlp_task': 'imdb'},
    {'dataset': 'sst2',       'model': 'DistilBERT', 'ep': EPOCHS_FT, 'nl': 2, 'type': 'ft', 'lr_sgd': 5e-4, 'lr_adam': 5e-5, 'nlp_task': 'sst2'},
    {'dataset': 'mrpc',       'model': 'DistilBERT', 'ep': EPOCHS_FT, 'nl': 2, 'type': 'ft', 'lr_sgd': 5e-4, 'lr_adam': 5e-5, 'nlp_task': 'mrpc'},
]


def run_one(cfg, name, seed, device, dev_tag, out_dir):
    """Run one strategy on one dataset with one seed. Returns dict or None."""
    data.set_seed(seed)
    try:
        (tr, va, te), n_labels = build_loaders(cfg, seed)
    except Exception as e:
        print(f'  [loader fail] {e}', flush=True); return None
    total_steps = len(tr) * cfg['ep']
    try:
        model = build_model(cfg, n_labels, device)
        o, sched, pcb = build_opt(name, model, cfg['lr_sgd'], cfg['lr_adam'], total_steps)
    except Exception as e:
        print(f'  [opt fail] {e}', flush=True)
        if 'model' in dir(): del model
        return None
    try:
        r = T.train_one(model, o, tr, va, te, device, n_labels=n_labels,
                         epochs=cfg['ep'], sched=sched, dals_phase_update=pcb,
                         select_by='val', verbose=0)
        acc = r['test_acc_valsel']
        f1 = r['test_f1_valsel']
        t = r['wall_clock_s']
        mem = r.get('peak_gpu_mem_mb')
        print(f'[{cfg["dataset"]:8s} {dev_tag} {name:20s} s={seed}] acc={acc:.2f} f1={f1:.2f} t={t:.1f}s mem={mem}', flush=True)
        return {'acc': acc, 'f1': f1, 'time': t, 'mem': mem}
    except Exception as e:
        print(f'  [train fail] {e}', flush=True)
        return None
    finally:
        del model, o, sched, pcb
        if 'cuda' in dev_tag:
            torch.cuda.empty_cache()


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--datasets', nargs='+', default=['synthetic'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--strategies', nargs='+', default=None)
    ap.add_argument('--out', default='results_comp')
    args = ap.parse_args()

    device, dev_tag = get_device(args.device)
    os.makedirs(args.out, exist_ok=True)
    strats = args.strategies or STRATEGIES

    configs = [c for c in CONFIGS if c['dataset'] in args.datasets]

    print(f'Device: {device} ({dev_tag})')
    print(f'Datasets: {[c["dataset"] for c in configs]}')
    print(f'Strategies: {len(strats)}, Seeds: {len(args.seeds)}')
    print(f'Total experiments: {len(configs)} × {len(strats)} × {len(args.seeds)} = {len(configs)*len(strats)*len(args.seeds)}')
    print('='*60, flush=True)

    all_results = {}
    for cfg in configs:
        ds = cfg['dataset']
        all_results[ds] = {}
        out_json = os.path.join(args.out, f'{ds}_{dev_tag}.json')

        for name in tqdm(strats, desc=f'{ds:8s}'):
            runs = []
            for seed in args.seeds:
                r = run_one(cfg, name, seed, device, dev_tag, args.out)
                if r: runs.append(r)
            accs = [x['acc'] for x in runs if x.get('acc')]
            f1s = [x['f1'] for x in runs if x.get('f1')]
            n = len(accs)
            mean_a = float(np.mean(accs)) if accs else None
            std_a = float(np.std(accs)) if n > 1 else 0.0
            mean_f = float(np.mean(f1s)) if f1s else None
            ci = (0.0 if n <= 1 else ({2:12.706,3:4.303,4:3.182,5:2.776}.get(n,1.96)*std_a/math.sqrt(n)))
            all_results[ds][name] = {'n': n, 'acc_mean': mean_a, 'acc_std': std_a,
                                      'acc_ci95': ci, 'f1_mean': mean_f, 'per_seed': runs}
        save_json(all_results[ds], out_json)
        # Print summary
        best = max(all_results[ds].items(), key=lambda x: x[1].get('acc_mean', 0) or 0)
        print(f'\n  [{ds}] Best: {best[0]} = {best[1]["acc_mean"]:.2f}%', flush=True)

    print(f'\nDone. Results in {args.out}/', flush=True)


if __name__ == '__main__':
    main()