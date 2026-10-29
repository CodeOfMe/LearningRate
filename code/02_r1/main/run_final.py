"""FULL benchmark with resume capability. Saves after EVERY experiment.
Progress: prints to stdout AND writes to progress.log.
Resume: loads existing JSON, skips completed (seed,strategy) pairs.
"""
import argparse, json, os, sys, time, math
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import optimizers as opt
import data
import train as T

STRATEGIES = [
    'Fixed_SGD', 'Cosine_SGD', 'Adam', 'AdamW',
    'LARS', 'Discriminative',
    'RAdam', 'Lion', 'SAM', 'Grokfast',
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
]

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
    if kind == 'cuda': return torch.device('cuda'), 'cuda'
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
    if m == 'DistilBERT':
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=n_labels).to(device)
    if m == 'BERT_base':
        from transformers import AutoModelForSequenceClassification
        return AutoModelForSequenceClassification.from_pretrained('google-bert/bert-base-uncased', num_labels=n_labels).to(device)
    raise ValueError(m)


def build_loaders(cfg, seed):
    ds = cfg['ds']
    if ds == 'synthetic': return data.synthetic_loaders(seed=seed, batch=64), 10
    if ds in ('cifar10', 'cifar100'):
        return data.cifar_loaders(ds, seed=seed, batch=128), (100 if ds=='cifar100' else 10)
    tr, va, te, nl = data.nlp_loaders(cfg['nlp'], seed=seed, batch=32)
    return (tr, va, te), nl


def build_opt(name, model, lr_sgd, lr_adam, T):
    s=None; p=None
    if name == 'Fixed_SGD':        return opt.FixedLRSGD(model.parameters(),lr=lr_sgd,momentum=0.9,weight_decay=1e-4),s,p
    if name == 'Cosine_SGD':       return opt.CosineAnnealingSGD(model.parameters(),lr=lr_sgd,T_max=T,momentum=0.9,weight_decay=5e-4),s,p
    if name == 'Adam':             return torch.optim.Adam(model.parameters(),lr=lr_adam),s,p
    if name == 'AdamW':            return torch.optim.AdamW(model.parameters(),lr=lr_adam,weight_decay=1e-2),s,p
    if name == 'LARS':             return opt.LARS(model.parameters(),lr=lr_sgd,momentum=0.9,weight_decay=5e-4),s,p
    if name == 'Discriminative':
        g=opt.build_discriminative_param_groups(model,lr_sgd,decay_factor=2.6)
        return opt.FixedLRSGD(g,lr=lr_sgd,momentum=0.9,weight_decay=1e-4),s,p
    if name == 'RAdam':            return opt.RAdam(model.parameters(),lr=lr_adam),s,p
    if name == 'Lion':             return opt.Lion(model.parameters(),lr=lr_adam*0.1,weight_decay=1e-2),s,p
    if name == 'SAM':
        base=torch.optim.SGD(model.parameters(),lr=lr_sgd,momentum=0.9,weight_decay=5e-4)
        return SAMW(base,rho=0.05),s,p
    if name == 'Grokfast':         return opt.Grokfast(model.parameters(),lr=lr_sgd,momentum=0.9,weight_decay=5e-4,alpha=0.6),s,p
    if name == 'STLR_Disc':
        g=opt.build_discriminative_stlr_param_groups(model,lr_sgd,decay_factor=2.6)
        o=opt.FixedLRSGD(g,lr=lr_sgd,momentum=0.9,weight_decay=1e-4)
        return o,opt.STLRScheduler(o,T=T,cut_frac=0.1,ratio=32,decay_factor=2.6),p
    if name == 'SAM_Disc':
        g=opt.build_discriminative_param_groups(model,lr_sgd,decay_factor=2.6)
        base=torch.optim.SGD(g,lr=lr_sgd,momentum=0.9,weight_decay=5e-4)
        return SAMW(base,rho=0.05),s,p
    dlr=0.1 if lr_sgd>5e-4 else lr_sgd
    if name == 'DALS':
        o=opt.DALS(model,lr=dlr,momentum=0.9,weight_decay=1e-4,T_max=T); return o,s,o.update_phase
    if name == 'DALS_Fast':
        o=opt.DALSFast(model,lr=dlr,momentum=0.85,weight_decay=1e-4,T_max=T); return o,s,o.update_phase
    if name == 'DALS_Acc':
        o=opt.DALSAcc(model,lr=dlr,momentum=0.9,weight_decay=5e-4,T_max=T,T_0=max(1,T//2),T_mult=2); return o,s,o.update_phase
    raise ValueError(name)


class SAMW(torch.optim.Optimizer):
    def __init__(self,b,r=0.05): self.base=b; self.rho=r; self._e=[]
    @property
    def param_groups(self): return self.base.param_groups
    def zero_grad(self,**kw): self.base.zero_grad(**kw)
    @torch.no_grad()
    def step(self,closure=None):
        if closure is None: return self.base.step()
        loss=closure(); loss.backward(); self._e.clear()
        for g in self.base.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    gn=p.grad.norm(2)+1e-12; eps=(p.grad/gn)*self.rho
                    p.add_(eps); self._e.append(eps)
        self.base.zero_grad(); loss2=closure(); loss2.backward(); i=0
        for g in self.base.param_groups:
            for p in g['params']:
                if i<len(self._e): p.sub_(self._e[i]); i+=1
        self.base.step(); return loss2


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
    args=ap.parse_args()

    device,dev_tag=get_device(args.device)
    os.makedirs(args.out,exist_ok=True)
    seeds=args.seeds
    configs=[c for c in CONFIGS if c['ds'] in args.ds]

    total_pending = 0
    for cfg in configs:
        ds=cfg['ds']
        out_json=os.path.join(args.out,f'{ds}_{dev_tag}.json')
        existing=load_results(out_json)
        for name in STRATEGIES:
            display_name = NAME_MAP.get(name, name)
            ex_seeds = set()
            if name in existing:
                for r in existing[name].get('per_seed', []):
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
        for name in STRATEGIES:
            if name not in results:
                results[name] = {'per_seed': []}

        for name in STRATEGIES:
            display_name = NAME_MAP.get(name, name)
            ex_seeds = set()
            if name in results:
                for r in results[name].get('per_seed', []):
                    ex_seeds.add(r.get('seed'))

            pending = [s for s in seeds if s not in ex_seeds]
            if not pending:
                continue

            log(f' [{ds}] {display_name}: {len(pending)} seeds pending')
            for seed in pending:
                data.set_seed(seed)
                try:
                    (tr,va,te),nl=build_loaders(cfg,seed)
                    total_steps=len(tr)*cfg['ep']
                    model=build_model(cfg,nl,device)
                    o,sched,pcb=build_opt(name,model,cfg['lr_sgd'],cfg['lr_adam'],total_steps)
                    if pcb is None: pcb=lambda x:None
                    r=T.train_one(model,o,tr,va,te,device,n_labels=nl,epochs=cfg['ep'],
                                   sched=sched,dals_phase_update=pcb,select_by='val',verbose=0)
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