"""Comprehensive benchmark: all reviewer-requested datasets on CUDA + DirectML.

Datasets (reviewers R1-4, R4-16):
  From-scratch: synthetic, CIFAR-10, CIFAR-100
  NLP fine-tune (DistilBERT): RTE, TREC-6, IMDb, SST-2, MRPC
  NLP fine-tune (BERT-base): SST-2, MRPC, MNLI (subset), QQP (subset)

Models: MLP, ConvNet, DistilBERT, BERT-base
Metrics: accuracy, macro-F1, wall-clock, peak GPU memory
Multi-seed: 5 seeds (42, 1337, 2024, 5555, 9999)
"""
import argparse, json, os, time, math, numpy as np, torch

import optimizers as opt
import data
import train as T

STRATEGIES_COMPREHENSIVE = [
    'Fixed SGD', 'Cosine Decay SGD', 'SGDR',
    'Adam', 'AdamW',
    'LARS', 'Discriminative LR',
    'RAdam', 'Lion', 'SAM', 'Grokfast',
    'STLR+Discriminative', 'SAM+Discriminative',
    'DALS (Ours)', 'DALS-Fast (Ours)', 'DALS-Acc (Ours)',
]

DATASET_CONFIGS = {
    'synthetic':   {'fn': 'synthetic_loaders', 'model': 'MLP', 'ep': 80, 'nl': 10, 'lr_sgd': 0.05, 'lr_adam': 3e-4},
    'cifar10':     {'fn': 'cifar10_loaders',   'model': 'ConvNet', 'ep': 50, 'nl': 10, 'lr_sgd': 0.01, 'lr_adam': 3e-4},
    'cifar100':    {'fn': 'cifar100_loaders',  'model': 'ConvNet100', 'ep': 50, 'nl': 100, 'lr_sgd': 0.01, 'lr_adam': 3e-4},
    'rte':         {'fn': 'nlp_loaders', 'model': 'DistilBERT', 'ep': 5, 'nl': 2, 'lr_sgd': 5e-4, 'lr_adam': 5e-5, 'nlp_task': 'rte'},
    'trec6':       {'fn': 'nlp_loaders', 'model': 'DistilBERT', 'ep': 5, 'nl': 6, 'lr_sgd': 5e-4, 'lr_adam': 5e-5, 'nlp_task': 'trec6'},
    'imdb':        {'fn': 'nlp_loaders', 'model': 'DistilBERT', 'ep': 3, 'nl': 2, 'lr_sgd': 5e-4, 'lr_adam': 5e-5, 'nlp_task': 'imdb'},
    'sst2':        {'fn': 'nlp_loaders', 'model': 'DistilBERT', 'ep': 5, 'nl': 2, 'lr_sgd': 5e-4, 'lr_adam': 5e-5, 'nlp_task': 'sst2'},
    'mrpc':        {'fn': 'nlp_loaders', 'model': 'DistilBERT', 'ep': 5, 'nl': 2, 'lr_sgd': 5e-4, 'lr_adam': 5e-5, 'nlp_task': 'mrpc'},
}

def get_device(kind):
    if kind == 'cuda':
        return torch.device('cuda')
    if kind.startswith('dml') or kind.startswith('directml'):
        import torch_directml
        idx = 0
        if ':' in kind:
            idx = int(kind.split(':')[1])
        return torch_directml.device(idx)
    return torch.device('cpu')


def build_model(ds_name, n_labels, device):
    if ds_name == 'synthetic':
        return data.MLP().to(device)
    if ds_name in ('cifar10', 'cifar100'):
        nc = 100 if ds_name == 'cifar100' else 10
        return data.ConvNet(n_classes=nc).to(device)
    if ds_name in ('rte','trec6','imdb','sst2','mrpc'):
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained(
            'distilbert-base-uncased', num_labels=n_labels).to(device)
    raise ValueError(ds_name)


def build_optimizer(name, model, lr_sgd, lr_adam, total_steps):
    T_steps = total_steps or 1000
    sched = None; phase_cb = None
    
    if name == 'Fixed SGD':
        return opt.FixedLRSGD(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=1e-4), sched, phase_cb
    if name == 'Cosine Decay SGD':
        return opt.CosineAnnealingSGD(model.parameters(), lr=lr_sgd, T_max=T_steps, momentum=0.9, weight_decay=5e-4), sched, phase_cb
    if name == 'SGDR':
        return opt.SGDRWithRestarts(model.parameters(), lr=lr_sgd, T_0=max(1, T_steps//5), T_mult=2, momentum=0.9, weight_decay=5e-4), sched, phase_cb
    if name == 'Adam':
        return torch.optim.Adam(model.parameters(), lr=lr_adam, weight_decay=0), sched, phase_cb
    if name == 'AdamW':
        return torch.optim.AdamW(model.parameters(), lr=lr_adam, weight_decay=1e-2), sched, phase_cb
    if name == 'LARS':
        return opt.LARS(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=5e-4), sched, phase_cb
    if name == 'Discriminative LR':
        groups = opt.build_discriminative_param_groups(model, lr_sgd, decay_factor=2.6)
        return opt.FixedLRSGD(groups, lr=lr_sgd, momentum=0.9, weight_decay=1e-4), sched, phase_cb
    if name == 'RAdam':
        return opt.RAdam(model.parameters(), lr=lr_adam, weight_decay=0), sched, phase_cb
    if name == 'Lion':
        return opt.Lion(model.parameters(), lr=lr_adam*0.1, weight_decay=1e-2), sched, phase_cb
    if name == 'SAM':
        base = torch.optim.SGD(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=5e-4)
        return SAMWrapper(base, rho=0.05), sched, phase_cb
    if name == 'Grokfast':
        return opt.Grokfast(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=5e-4, alpha=0.6), sched, phase_cb
    if name == 'STLR+Discriminative':
        groups = opt.build_discriminative_stlr_param_groups(model, lr_sgd, decay_factor=2.6)
        o = opt.FixedLRSGD(groups, lr=lr_sgd, momentum=0.9, weight_decay=1e-4)
        sched = opt.STLRScheduler(o, T=T_steps, cut_frac=0.1, ratio=32, decay_factor=2.6)
        return o, sched, phase_cb
    if name == 'SAM+Discriminative':
        groups = opt.build_discriminative_param_groups(model, lr_sgd, decay_factor=2.6)
        base = torch.optim.SGD(groups, lr=lr_sgd, momentum=0.9, weight_decay=5e-4)
        return SAMWrapper(base, rho=0.05), sched, phase_cb
    if name == 'DALS (Ours)':
        dals_lr = 0.1 if lr_sgd > 5e-4 else lr_sgd
        o = opt.DALS(model, lr=dals_lr, momentum=0.9, weight_decay=1e-4, T_max=T_steps)
        return o, sched, o.update_phase
    if name == 'DALS-Fast (Ours)':
        dals_lr = 0.1 if lr_sgd > 5e-4 else lr_sgd
        o = opt.DALSFast(model, lr=dals_lr, momentum=0.85, weight_decay=1e-4, T_max=T_steps)
        return o, sched, o.update_phase
    if name == 'DALS-Acc (Ours)':
        dals_lr = 0.1 if lr_sgd > 5e-4 else lr_sgd
        o = opt.DALSAcc(model, lr=dals_lr, momentum=0.9, weight_decay=5e-4, T_max=T_steps, T_0=max(1,T_steps//2), T_mult=2)
        return o, sched, o.update_phase
    raise ValueError(name)


class SAMWrapper(torch.optim.Optimizer):
    def __init__(self, base, rho=0.05):
        self.base = base; self.rho = rho
    @property
    def param_groups(self): return self.base.param_groups
    @property
    def state(self): return self.base.state
    def zero_grad(self): self.base.zero_grad()
    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            return self.base.step()
        loss = closure()
        loss.backward()
        # First step: perturb
        grads = []
        for g in self.base.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    gn = p.grad.norm(2) + 1e-12
                    eps = (p.grad / gn) * self.rho
                    p.add_(eps)
                    grads.append(eps)
        self.base.zero_grad()
        loss2 = closure()
        loss2.backward()
        i = 0
        for g in self.base.param_groups:
            for p in g['params']:
                if i < len(grads):
                    p.sub_(grads[i]); i += 1
        self.base.step()
        return loss2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--seeds', nargs='+', type=int, default=[42,1337,2024,5555,9999])
    ap.add_argument('--datasets', nargs='+',
                    default=['synthetic','cifar10','cifar100','rte','trec6','imdb','sst2','mrpc'])
    ap.add_argument('--strategies', nargs='+', default=None)
    ap.add_argument('--out', default='results_comprehensive')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    device = get_device(args.device)
    dev_tag = 'cuda' if device.type == 'cuda' else ('dml' + str(getattr(device,'index',0) if 'privateuseone' in str(device) else 0))
    print(f'Device: {device} ({dev_tag})', flush=True)

    strategies = args.strategies or STRATEGIES_COMPREHENSIVE

    for ds_name in args.datasets:
        cfg = DATASET_CONFIGS[ds_name]
        res = {}
        for name in strategies:
            runs = []
            for seed in args.seeds:
                data.set_seed(seed)
                if ds_name == 'synthetic':
                    tr,va,te = data.synthetic_loaders(seed=seed, batch=64)
                elif ds_name in ('cifar10','cifar100'):
                    tr,va,te = data.cifar_loaders(ds_name, seed=seed, batch=128)
                else:
                    nlp_task = cfg.get('nlp_task', ds_name)
                    tr,va,te,n_labels = data.nlp_loaders(nlp_task, seed=seed, batch=32)
                n_labels = cfg['nl']
                ep = cfg['ep']
                total_steps = len(tr) * ep
                try:
                    model = build_model(ds_name, n_labels, device)
                    o, sched, pcb = build_optimizer(name, model, cfg['lr_sgd'], cfg['lr_adam'], total_steps)
                    r = T.train_one(model, o, tr, va, te, device, n_labels=n_labels, epochs=ep,
                                     sched=sched, dals_phase_update=pcb, select_by='val', verbose=0)
                    print(f'[{ds_name} {dev_tag} {name} seed={seed}] acc={r["test_acc_valsel"]:.2f} f1={r["test_f1_valsel"]:.2f} t={r["wall_clock_s"]:.1f}s', flush=True)
                    runs.append({'acc': r['test_acc_valsel'], 'f1': r['test_f1_valsel'],
                                 'time': r['wall_clock_s'], 'mem': r.get('peak_gpu_mem_mb')})
                except Exception as e:
                    print(f'  [fail] {name} seed={seed}: {e}', flush=True)
                del model, o, sched, pcb
                if str(device) == 'cuda' or device.type == 'cuda':
                    torch.cuda.empty_cache()

            accs = [r['acc'] for r in runs if r['acc'] is not None]
            f1s = [r['f1'] for r in runs if r['f1'] is not None]
            n = len(accs)
            mean_a = float(np.mean(accs)) if accs else None
            std_a = float(np.std(accs)) if n > 1 else 0.0
            mean_f = float(np.mean(f1s)) if f1s else None
            t = {2:12.706,3:4.303,4:3.182,5:2.776}.get(n, 1.96)
            ci = t * std_a / math.sqrt(n) if n > 1 else 0.0
            res[name] = {'n': n, 'acc_mean': mean_a, 'acc_std': std_a, 'acc_ci95': ci,
                         'f1_mean': mean_f, 'per_seed': runs}
            print(f'  => mean={mean_a:.2f}±{ci:.2f} (n={n})', flush=True)

        out = os.path.join(args.out, f'comp_{ds_name}_{dev_tag}.json')
        with open(out, 'w') as f:
            json.dump(res, f, indent=2)
        print(f'Saved {out}', flush=True)


if __name__ == '__main__':
    main()