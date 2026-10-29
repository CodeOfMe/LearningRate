"""FULL benchmark with resume capability. Saves after EVERY experiment.
Progress: prints to stdout AND writes to progress.log.
Resume: loads existing JSON, skips completed (seed,strategy) pairs.
"""
import os, sys
# --- Force fully-offline mode BEFORE any heavy import. The environment has
# docling (installed but unused by this benchmark) which auto-downloads its
# internal HF models on import; block all network/HF access for this run.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
os.environ['HF_HUB_DISABLE_PROGRESS_BARS'] = '1'
import argparse, json, time, math
import numpy as np, torch

CODE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CODE_DIR, '..', '..', '..'))  # repo root
MODEL_DIR = os.environ.get('MODEL_DIR', os.path.join(ROOT_DIR, 'models'))
DATA_DIR  = os.environ.get('DATA_DIR',  os.path.join(ROOT_DIR, 'datasets'))
# IMPORTANT: propagate into the environment so data.py / data loaders see the
# same paths (os.environ.get in cifar_loaders would otherwise fall back to the
# train_code dir and trigger a CIFAR re-download).
os.environ['MODEL_DIR'] = MODEL_DIR
os.environ['DATA_DIR']  = DATA_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(DATA_DIR, 'huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.makedirs(os.path.join(DATA_DIR, 'huggingface'), exist_ok=True)

sys.path.insert(0, CODE_DIR)
import optimizers as opt
import data
import train as T

STRATEGIES = [
    'Fixed_SGD', 'Cosine_SGD', 'SGDR',
    'Adam', 'AdamW', 'AdaBound',
    'LARS', 'Discriminative',
    'RAdam', 'Lion', 'Lookahead', 'SAM', 'Grokfast',
    'STLR_Disc', 'SAM_Disc',
    'DALS', 'DALS_Fast', 'DALS_Acc',
]

SEEDS = [42, 1337, 2024, 5555, 9999]

CONFIGS = [
    {'ds':'synthetic', 'model':'MLP',         'ep':80, 'nl':10,'lr_sgd':0.05,'lr_adam':3e-4},
    {'ds':'cifar10',   'model':'ConvNet',     'ep':50, 'nl':10,'lr_sgd':0.01,'lr_adam':3e-4},
    {'ds':'cifar100',  'model':'ConvNet_lg',  'ep':50, 'nl':100,'lr_sgd':0.01,'lr_adam':3e-4},
    {'ds':'rte',       'model':'DistilBERT',  'ep':5,  'nl':2, 'lr_sgd':5e-4,'lr_adam':5e-5,'nlp':'rte'},
    {'ds':'trec6',     'model':'DistilBERT',  'ep':5,  'nl':6, 'lr_sgd':5e-4,'lr_adam':5e-5,'nlp':'trec6'},
    {'ds':'imdb',      'model':'DistilBERT',  'ep':3,  'nl':2, 'lr_sgd':5e-4,'lr_adam':5e-5,'nlp':'imdb'},
    {'ds':'sst2',      'model':'DistilBERT',  'ep':5,  'nl':2, 'lr_sgd':5e-4,'lr_adam':5e-5,'nlp':'sst2'},
    {'ds':'mrpc',      'model':'DistilBERT',  'ep':5,  'nl':2, 'lr_sgd':5e-4,'lr_adam':5e-5,'nlp':'mrpc'},
    {'ds':'sst2_b',    'model':'BERT_base',   'ep':5,  'nl':2, 'lr_sgd':5e-4,'lr_adam':5e-5,'nlp':'sst2'},
    {'ds':'mrpc_b',    'model':'BERT_base',   'ep':5,  'nl':2, 'lr_sgd':5e-4,'lr_adam':5e-5,'nlp':'mrpc'},
    {'ds':'sst2_r',    'model':'RoBERTa',     'ep':5,  'nl':2, 'lr_sgd':5e-4,'lr_adam':5e-5,'nlp':'sst2'},
    {'ds':'mrpc_r',    'model':'RoBERTa',     'ep':5,  'nl':2, 'lr_sgd':5e-4,'lr_adam':5e-5,'nlp':'mrpc'},
]

# model type -> (hub id, local dir name)
MODEL_REGISTRY = {
    'DistilBERT': ('distilbert-base-uncased', 'distilbert-base-uncased'),
    'BERT_base':  ('google-bert/bert-base-uncased', 'bert-base-uncased'),
    'RoBERTa':    ('FacebookAI/roberta-base', 'roberta-base'),
}

NAME_MAP = {
    'Fixed_SGD':'Fixed SGD','Cosine_SGD':'Cosine Decay SGD','Discriminative':'Discriminative LR',
    'STLR_Disc':'STLR+Discriminative','SAM_Disc':'SAM+Discriminative',
    'DALS_Fast':'DALS-Fast','DALS_Acc':'DALS-Acc',
}
def log(msg):
    """Print to stdout and append to progress.log."""
    print(msg, flush=True)
    with open('progress.log', 'a', encoding='utf-8') as f:
        f.write(msg + '\n')


def get_device(kind):
    if kind == 'cuda':
        if not torch.cuda.is_available():
            raise SystemExit('CUDA requested but torch.cuda.is_available() is False. '
                             'Check driver: nvidia-smi / kernel module mismatch.')
        return torch.device('cuda'), 'cuda'
    if kind.startswith('dml'):
        import torch_directml
        idx = int(kind.split(':')[1]) if ':' in kind else 0
        return torch_directml.device(idx), f'dml{idx}'
    return torch.device('cpu'), 'cpu'


def build_model(cfg, n_labels, device):
    m = cfg['model']
    if m == 'MLP':       return data.MLP().to(device)
    if m == 'ConvNet':   return data.ConvNet(n_classes=n_labels).to(device)
    if m == 'ConvNet_lg': return data.ConvNet100(n_classes=n_labels).to(device)
    from transformers import AutoModelForSequenceClassification
    hub_id, local_dir = MODEL_REGISTRY[m]
    local_path = os.path.join(MODEL_DIR, local_dir)
    # Prefer local copy; fall back to hub id (works on CUDA machine with cache/network).
    src = local_path if os.path.isdir(local_path) else hub_id
    return AutoModelForSequenceClassification.from_pretrained(src, num_labels=n_labels).to(device)


def build_loaders(cfg, seed):
    ds = cfg['ds']
    if ds == 'synthetic': return data.synthetic_loaders(seed=seed, batch=64), 10
    if ds in ('cifar10', 'cifar100'):
        return data.cifar_loaders(ds, seed=seed, batch=128), (100 if ds=='cifar100' else 10)
    hub_id, local_dir = MODEL_REGISTRY[cfg['model']]
    local_path = os.path.join(MODEL_DIR, local_dir)
    tok_name = local_path if os.path.isdir(local_path) else hub_id
    tr, va, te, nl = data.nlp_loaders(cfg['nlp'], seed=seed, batch=32, tokenizer_name=tok_name)
    return (tr, va, te), nl


def build_opt(name, model, lr_sgd, lr_adam, T, ep=None):
    s=None; p=None
    if name == 'Fixed_SGD':        return opt.FixedLRSGD(model.parameters(),lr=lr_sgd,momentum=0.9,weight_decay=1e-4),s,p
    if name == 'Cosine_SGD':       return opt.CosineAnnealingSGD(model.parameters(),lr=lr_sgd,T_max=T,momentum=0.9,weight_decay=5e-4),s,p
    if name == 'SGDR':             return opt.SGDRWithRestarts(model.parameters(),lr=lr_sgd,T_0=max(1,T//5),T_mult=2,momentum=0.9,weight_decay=5e-4),s,p
    if name == 'Adam':             return torch.optim.Adam(model.parameters(),lr=lr_adam),s,p
    if name == 'AdamW':            return torch.optim.AdamW(model.parameters(),lr=lr_adam,weight_decay=1e-2),s,p
    if name == 'AdaBound':
        return opt.AdaBound(model.parameters(),lr=lr_adam,final_lr=0.1,weight_decay=0),s,p
    if name == 'LARS':             return opt.LARS(model.parameters(),lr=lr_sgd,momentum=0.9,weight_decay=5e-4),s,p
    if name == 'Discriminative':
        g=opt.build_discriminative_param_groups(model,lr_sgd,decay_factor=2.6)
        return opt.FixedLRSGD(g,lr=lr_sgd,momentum=0.9,weight_decay=1e-4),s,p
    if name == 'RAdam':            return opt.RAdam(model.parameters(),lr=lr_adam),s,p
    if name == 'Lion':             return opt.Lion(model.parameters(),lr=lr_adam*0.1,weight_decay=1e-2),s,p
    if name == 'Lookahead':
        inner=torch.optim.AdamW(model.parameters(),lr=lr_adam,weight_decay=1e-2)
        return LookaheadW(inner,k=5,alpha=0.5),s,p
    if name == 'SAM':
        o=SAMW(model.parameters(), torch.optim.SGD, rho=0.05, adaptive=True,
               lr=lr_sgd, momentum=0.9, weight_decay=5e-4)
        return o,s,p
    if name == 'Grokfast':         return opt.Grokfast(model.parameters(),lr=lr_sgd,momentum=0.9,weight_decay=5e-4,alpha=0.6),s,p
    if name == 'STLR_Disc':
        g=opt.build_discriminative_stlr_param_groups(model,lr_sgd,decay_factor=2.6)
        o=opt.FixedLRSGD(g,lr=lr_sgd,momentum=0.9,weight_decay=1e-4)
        # param groups already carry per-layer eta_max from the builder;
        # do NOT apply decay_factor again here (would double-decay).
        return o,opt.STLRScheduler(o,T=T,cut_frac=0.1,ratio=32,decay_factor=None),p
    if name == 'SAM_Disc':
        g=opt.build_discriminative_param_groups(model,lr_sgd,decay_factor=2.6)
        o=SAMW(g, torch.optim.SGD, rho=0.05, adaptive=True,
               lr=lr_sgd, momentum=0.9, weight_decay=5e-4)
        return o,s,p
    dlr=0.1 if lr_sgd>5e-4 else lr_sgd
    if name == 'DALS':
        o=opt.DALS(model,lr=dlr,momentum=0.9,weight_decay=1e-4,T_max=T); return o,s,o.update_phase
    if name == 'DALS_Fast':
        o=opt.DALSFast(model,lr=dlr,momentum=0.85,weight_decay=1e-4,T_max=T); return o,s,o.update_phase
    if name == 'DALS_Acc':
        # T_0 fixed to 10 epochs (manuscript §3.4): T_0 = steps_per_epoch * 10.
         # steps_per_epoch = total_steps / ep. Old code used T_0=T//2 (wrong)
         # and later a bare T_0=10 (steps, not epochs) which caused collapse.
         steps_per_epoch = max(1, T // ep) if ep else max(1, T)
         # lr: from-scratch use 0.03 (original submission; dlr=0.1 collapsed cifar10
         # because SGDR cosine has no warmup); fine-tuning keep lr_sgd (5e-4)
         # since 0.03 is far too large for DistilBERT fine-tuning (collapses to random).
         acc_lr = 0.03 if lr_sgd > 5e-4 else lr_sgd
         o=opt.DALSAcc(model,lr=acc_lr,momentum=0.9,weight_decay=5e-4,T_max=T,T_0=steps_per_epoch*10,T_mult=2); return o,s,o.update_phase
    raise ValueError(name)


class LookaheadW(torch.optim.Optimizer):
    """Lookahead wrapper over an inner optimizer (k steps forward, 1 step back)."""
    def __init__(self, inner, k=5, alpha=0.5):
        self.inner=inner; self.k=k; self.alpha=alpha; self.step_n=0
        self.slow={}
        for g in inner.param_groups:
            for p in g['params']:
                self.slow[id(p)]=p.data.clone()
    @property
    def param_groups(self): return self.inner.param_groups
    @property
    def state(self): return self.inner.state
    def zero_grad(self,**kw): self.inner.zero_grad(**kw)
    @torch.no_grad()
    def step(self,closure=None):
        loss=self.inner.step(closure); self.step_n+=1
        if self.step_n % self.k == 0:
            for g in self.inner.param_groups:
                for p in g['params']:
                    sp=self.slow[id(p)]
                    sp.add_(p-sp, alpha=self.alpha)
                    p.copy_(sp)
        return loss


class SAMW(torch.optim.Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2020).

    Standard SAM (ported from the original submission code/codes/optimizers.py):
    - adaptive perturbation: e_w = (p^2 if adaptive else 1) * grad * scale,
      with scale = rho / global_grad_norm (whole-network L2 ball ||eps||<=rho).
    - per-parameter e_w stored in state[p], restored exactly in second_step.
    - base_optimizer.step() applies the gradient at the perturbed point.
    The training loop (train.py) calls first_step()/second_step().
    """
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        self.rho = rho
        self.adaptive = adaptive
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    def zero_grad(self, set_to_none=False):
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    def _grad_norm(self):
        grad_norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norms.append(((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(2))
        if len(grad_norms) == 0:
            return torch.tensor(0.0, device=p.device if len(self.param_groups)>0 and len(self.param_groups[0]['params'])>0 else 'cpu')
        return torch.norm(torch.stack(grad_norms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w.detach().clone()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p in self.state and 'e_w' in self.state[p]:
                    p.add_(self.state[p]['e_w'], alpha=-1.0)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            return self.base_optimizer.step(closure)
        return self.base_optimizer.step()


def load_results(path):
    try:
        with open(path) as f: return json.load(f)
    except: return {}


def save_results(data, path):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f: json.dump(data, f, indent=2)
    os.replace(tmp, path)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--device',default='cuda')
    ap.add_argument('--ds',nargs='+',default=['synthetic'])
    ap.add_argument('--seeds',nargs='+',type=int,default=SEEDS)
    ap.add_argument('--out',default='results_comp')
    ap.add_argument('--strategies',nargs='+',default=None,
                    help='subset of STRATEGIES to run (default: all 18)')
    ap.add_argument('--grad_stats',type=int,default=0,
                    help='>0: collect per-layer gradient stats (reviewer R4-15)')
    args=ap.parse_args()

    device,dev_tag=get_device(args.device)
    os.makedirs(args.out,exist_ok=True)
    seeds=args.seeds
    configs=[c for c in CONFIGS if c['ds'] in args.ds]
    strategies = [s for s in STRATEGIES if args.strategies is None or s in args.strategies]
    if not strategies:
        raise SystemExit('No strategies selected')

    total_pending = 0
    for cfg in configs:
        ds=cfg['ds']
        out_json=os.path.join(args.out,f'{ds}_{dev_tag}.json')
        existing=load_results(out_json)
        for name in strategies:
            display_name = NAME_MAP.get(name, name)
            ex_seeds = set()
            if name in existing:
                for r in existing[name].get('per_seed', []):
                    if isinstance(r, dict) and r.get('acc') is not None:
                        ex_seeds.add(r.get('seed'))
            for seed in seeds:
                if seed not in ex_seeds:
                    total_pending += 1

    if total_pending == 0:
        log('ALL DONE - nothing to run')
        return

    log(f'Device={dev_tag} Pending={total_pending}')
    log('='*60)

    done=0; t0=time.time()
    for cfg in configs:
        ds=cfg['ds']; m=cfg['model']
        out_json=os.path.join(args.out,f'{ds}_{dev_tag}.json')
        results=load_results(out_json)

        # ensure all strategy keys exist
        for name in strategies:
            if name not in results:
                results[name] = {'per_seed': []}

        for name in strategies:
            display_name = NAME_MAP.get(name, name)
            ex_seeds = set()
            if name in results:
                for r in results[name].get('per_seed', []):
                    if isinstance(r, dict) and r.get('acc') is not None:
                        ex_seeds.add(r.get('seed'))

            pending = [s for s in seeds if s not in ex_seeds]
            if not pending:
                continue

            log(f' [{ds}] {display_name}: {len(pending)} seeds pending')
            from tqdm import tqdm
            for seed in tqdm(pending, desc=f'{ds:8s} {display_name}', ncols=100,
                             position=0, leave=True, ascii=True):
                data.set_seed(seed)
                try:
                    (tr,va,te),nl=build_loaders(cfg,seed)
                    total_steps=len(tr)*cfg['ep']
                    model=build_model(cfg,nl,device)
                    o,sched,pcb=build_opt(name,model,cfg['lr_sgd'],cfg['lr_adam'],total_steps,ep=cfg['ep'])
                    if pcb is None: pcb=lambda x:None
                    r=T.train_one(model,o,tr,va,te,device,n_labels=nl,epochs=cfg['ep'],
                                   sched=sched,dals_phase_update=pcb,select_by='val',verbose=0,
                                   grad_stats_every=args.grad_stats)
                    acc=r['test_acc_valsel']; f1=r['test_f1_valsel']
                    results[name].setdefault('per_seed',[]).append({
                        'seed':seed,'acc':acc,'f1':f1,
                        'time':r['wall_clock_s'],'mem':r.get('peak_gpu_mem_mb')
                    })
                    save_results(results,out_json)
                    done+=1; elapsed=time.time()-t0
                    eta=elapsed/done*(total_pending-done) if done>0 else 0
                    log(f'  [{done}/{total_pending}] {ds:8s} {display_name:20s} s={seed} acc={acc:.1f} f1={f1:.1f} t={r["wall_clock_s"]:.0f}s | {elapsed/60:.0f}m ETA {eta/60:.0f}m')
                except Exception as e:
                    results[name].setdefault('per_seed',[]).append({'seed':seed,'acc':None,'f1':None,'error':str(e)})
                    save_results(results,out_json)
                    done+=1
                    log(f'  [{done}/{total_pending}] {ds:8s} {display_name:20s} s={seed} FAIL: {e}')
                finally:
                    if 'model' in dir(): del model
                    if 'o' in dir(): del o,sched,pcb
                    if dev_tag=='cuda': torch.cuda.empty_cache()

            # compute aggregates for this strategy
            runs=results[name]['per_seed']
            accs=[x['acc'] for x in runs if x.get('acc') is not None]
            f1s=[x['f1'] for x in runs if x.get('f1') is not None]
            n=len(accs); ma=float(np.mean(accs)) if accs else None
            sa=float(np.std(accs)) if n>1 else 0.0
            mf=float(np.mean(f1s)) if f1s else None
            ci=0.0 if n<=1 else {2:12.706,3:4.303,4:3.182,5:2.776}.get(n,1.96)*sa/math.sqrt(n)
            results[name].update({'n':n,'acc_mean':ma,'acc_std':sa,'acc_ci95':ci,'f1_mean':mf})
            save_results(results,out_json)
            if ma is not None:
                log(f'  [{ds}] {display_name}: {ma:.2f}±{ci:.2f} (n={n})')

        save_results(results,out_json)
        valid={k:v for k,v in results.items() if v.get('acc_mean') is not None}
        if valid:
            best=max(valid.items(),key=lambda x:x[1]['acc_mean'])
            log(f'  [{ds}] BEST: {NAME_MAP.get(best[0],best[0])} = {best[1]["acc_mean"]:.2f}%')

    elapsed=time.time()-t0
    log(f'\nDONE in {elapsed/3600:.1f}h. Results: {args.out}/')


if __name__=='__main__':
    main()