"""R3 消融实验——CIFAR-100 组件交互效应量化。

完全沿用 R2 stage3_ablation/run_interaction.py 的规格（5 组件 + LOO 16 组合 + 5 种子），
仅新增 cifar100 数据分支，并把路径指向 R3 目录。组件开关语义、位掩码、交互公式、日志框架
一律不改。

用法：
  python run_interaction_cifar100.py --device cuda --datasets cifar100
  python run_interaction_cifar100.py --device cuda --datasets cifar100 --seeds 42 1337 2024 5555 9999
"""
import os
import sys
import json
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
CORE = os.path.join(HERE, 'core')
LIB = os.path.join(HERE, 'lib')
OUT = os.path.join(HERE, 'outputs', 'ablation_results')

sys.path.insert(0, CORE)
sys.path.insert(0, LIB)

# 数据/模型/缓存路径指向 R3 目录（保证可复现，不依赖外部缓存）
os.environ['DATA_DIR'] = os.path.join(ROOT, 'datasets')
os.environ['HF_DATASETS_CACHE'] = os.path.join(os.environ['DATA_DIR'], 'huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import numpy as np
import data
import train as T
import optimizers as opt
from logger import ExperimentLogger, ResumableRunner  # noqa: E402

COMPONENTS = ['cosine', 'trust', 'filter', 'phase', 'depth']
SEEDS = [42, 1337, 2024, 5555, 9999]


def mask_to_flags(mask):
    return {COMPONENTS[i]: bool(mask & (1 << i)) for i in range(len(COMPONENTS))}


def build_dals(model, lr, total_steps, flags):
    """严格复用 opt.DALS，仅覆盖 use_* 开关。"""
    o = opt.DALS(model, lr=lr, momentum=0.9, weight_decay=1e-4,
                 T_max=total_steps, variant='full')
    o.use_cosine = flags['cosine']
    o.use_trust = flags['trust']
    o.use_filter = flags['filter']
    o.use_phase = flags['phase']
    o.use_depth = flags['depth']
    return o


def generate_combos(mode):
    """mode='loo': full + 5 单移除 + 10 成对移除 = 16 组合。"""
    full_mask = (1 << len(COMPONENTS)) - 1
    combos = []
    if mode == 'full':
        for mask in range(1 << len(COMPONENTS)):
            combos.append((mask, f"{mask:05b}"))
    else:  # loo
        combos.append((full_mask, 'full'))
        for i, c in enumerate(COMPONENTS):
            combos.append((full_mask & ~(1 << i), f'no_{c}'))
        for a, b in itertools.combinations(range(len(COMPONENTS)), 2):
            mask = full_mask & ~(1 << a) & ~(1 << b)
            combos.append((mask, f'no_{COMPONENTS[a]}_no_{COMPONENTS[b]}'))
    return combos


def run_one_combo(ds, mask, seed, device):
    data.set_seed(seed)
    if ds == 'synthetic':
        tr, va, te = data.synthetic_loaders(seed=seed, batch=64)
        ep, nl = 80, 10
        model = data.MLP()
    elif ds == 'cifar10':
        tr, va, te = data.cifar10_loaders(seed=seed, batch=128)
        ep, nl = 50, 10
        model = data.ConvNet()
    elif ds == 'cifar100':
        tr, va, te = data.cifar_loaders('cifar100', seed=seed, batch=128)
        ep, nl = 50, 100
        model = data.ConvNet100(n_classes=100)
    else:
        raise ValueError(ds)
    total_steps = len(tr) * ep
    lr = 0.1  # from-scratch 任务统一 0.1（对齐 run_final.py 的 dlr）
    flags = mask_to_flags(mask)
    o = build_dals(model, lr, total_steps, flags)
    pcb = o.update_phase
    r = T.train_one(model, o, tr, va, te, device, n_labels=nl, epochs=ep,
                    dals_phase_update=pcb, select_by='val', verbose=0)
    del model, o
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return {'acc': r['test_acc_valsel'], 'f1': r['test_f1_valsel']}


def interaction_analysis(ds, combo_means):
    full_mask = (1 << len(COMPONENTS)) - 1
    full = combo_means.get(full_mask)
    main = {}
    if full is not None:
        for i, c in enumerate(COMPONENTS):
            loo = combo_means.get(full_mask & ~(1 << i))
            if loo is not None:
                main[c] = full - loo
    inter = {}
    if full is not None:
        for a, b in itertools.combinations(range(len(COMPONENTS)), 2):
            ca, cb = COMPONENTS[a], COMPONENTS[b]
            full_a = combo_means.get(full_mask & ~(1 << a))
            full_b = combo_means.get(full_mask & ~(1 << b))
            full_ab = combo_means.get(full_mask & ~(1 << a) & ~(1 << b))
            if None in (full_a, full_b, full_ab):
                continue
            inter[f'{ca}×{cb}'] = full_a + full_b - full_ab - full
    return {'main_effects': {k: round(v, 4) for k, v in main.items()},
            'interactions': {k: round(v, 4) for k, v in inter.items()}}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--datasets', nargs='+', default=['cifar100'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--mode', choices=['loo', 'full'], default='loo')
    args = ap.parse_args()

    device = torch.device('cuda') if args.device == 'cuda' else torch.device('cpu')
    os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'results'), exist_ok=True)
    logger = ExperimentLogger(os.path.join(OUT, 'logs'), run_name='stage3',
                              results_dir=os.path.join(OUT, 'results'))
    runner = ResumableRunner(logger)

    combos = generate_combos(args.mode)
    print(f'组合数：{len(combos)}（mode={args.mode}），数据集 {args.datasets}，种子 {args.seeds}', flush=True)

    tasks = []
    for ds in args.datasets:
        for mask, label in combos:
            for seed in args.seeds:
                tasks.append({'task_id': f'ablation|{ds}|{label}|{seed}',
                              'description': f'消融 {ds}/{label} seed={seed}',
                              'dataset': ds, 'mask': mask, 'label': label, 'seed': seed})

    def run_task(task):
        res = run_one_combo(task['dataset'], task['mask'], task['seed'], device)
        return res

    summary = runner.run_tasks(tasks, run_task)
    print(f'消融运行完成：成功 {summary["success"]}，失败 {summary["failed"]}，跳过 {summary["skipped"]}', flush=True)

    # 从日志结果聚合 per-seed acc
    results_dir = logger.results_dir
    combo_seeds = {}
    for t in tasks:
        tid = t['task_id']
        f = os.path.join(results_dir, f'{tid}.json')
        if not os.path.exists(f):
            continue
        rec = json.load(open(f, encoding='utf-8'))
        if rec.get('status') != 'success' or not rec.get('output'):
            continue
        key = (t['dataset'], t['mask'])
        combo_seeds.setdefault(key, []).append(rec['output'].get('acc'))

    combo_means = {}
    analysis = {}
    for (ds, mask), accs in combo_seeds.items():
        accs = [a for a in accs if a is not None]
        combo_means[(ds, mask)] = float(np.mean(accs)) if accs else None
    for ds in args.datasets:
        cm = {mask: v for (d, mask), v in combo_means.items() if d == ds}
        analysis[ds] = interaction_analysis(ds, cm)
        analysis[ds]['combo_means'] = {f'{m:05b}': round(v, 4) for m, v in cm.items() if v is not None}

    out = {'mode': args.mode, 'components': COMPONENTS, 'summary': summary,
           'dataset': args.datasets, 'seeds': args.seeds, 'analysis': analysis}
    out_path = os.path.join(OUT, 'stage3_interaction.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    md = ['# R3 消融：CIFAR-100 组件交互量化', '', f'模式 {args.mode}，组件 {COMPONENTS}', '']
    for ds in args.datasets:
        a = analysis[ds]
        md += [f'## {ds}', '', '### 主效应（full 移除该组件的精度损失）', '',
               '| 组件 | 移除损失 |', '|---|---:|']
        for c, v in sorted(a['main_effects'].items(), key=lambda x: -x[1]):
            md.append(f'| {c} | {v:+.4f} |')
        md += ['', '### 二阶交互项（>0 正交互，<0 负交互/冗余）', '',
               '| 组件对 | 交互值 |', '|---|---:|']
        for c, v in sorted(a['interactions'].items(), key=lambda x: -x[1]):
            md.append(f'| {c} | {v:+.4f} |')
        md.append('')
    md_path = os.path.join(OUT, 'stage3_interaction_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    for ds in args.datasets:
        a = analysis[ds]
        print(f'  [{ds}] 主效应: {a["main_effects"]}', flush=True)
        print(f'  [{ds}] 交互项: {a["interactions"]}', flush=True)
    print(f'JSON: {out_path}', flush=True)
    print(f'报告: {md_path}', flush=True)


if __name__ == '__main__':
    main()
