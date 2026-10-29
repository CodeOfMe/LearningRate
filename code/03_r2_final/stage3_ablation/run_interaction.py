"""阶段三：消融实验——组件交互效应量化。

【benchmark 消融代码的问题（参照 run_ablation.py 发现）】
  1. 只有 9 个「开/关式」变体（full + 8 个单/双组件），无法量化组件间交互项；
  2. 第 54 行 lr 三元表达式 `lr = 0.1 if ds in (...) else 0.1` 两分支相同（笔误/死代码）；
  3. 默认 `--seeds` 只有 1 个种子，多种子结果需手动传参；
  4. 只覆盖 synthetic + CIFAR-10，无 NLP 数据集。

【本脚本的改进】
  - 用 DALS 的 5 个可开关组件 (cosine, trust, filter, phase, depth) 构造任意组合，
    严格复用 benchmark 的 opt.DALS 类（仅覆盖 use_* 开关）；
  - leave-one-out：full 逐一去掉每个组件（5 个）+ full 本身 = 6 个；
  - 成对交互：同时去掉 A、B（C(5,2)=10 个），用于「单独移除 A / 单独移除 B / 同时移除 A,B」
    的差异量化交互项；
  - 可选 --full-factorial 跑全部 2^5=32 组合（覆盖所有交互，无遗漏）；
  - 5 种子，断点恢复，JSON 记录。

交互项公式（2^k 因子分析，leave-one-out 视角）：
  交互(A,B) = acc(full−A) + acc(full−B) − acc(full−A−B) − acc(full)
  > 0 表示 A、B 同时存在产生正交互（1+1>2）；< 0 表示负交互（冗余/冲突）。

用法：
  python stage3_ablation/run_interaction.py --device cuda --datasets synthetic
  python stage3_ablation/run_interaction.py --device cuda --datasets synthetic cifar10 --full-factorial
"""
import os
import sys
import json
import itertools

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)
CORE = os.path.join(AUDIT, 'core')
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(AUDIT, 'lib'))

ROOT = os.path.abspath(os.path.join(AUDIT, '..', '..'))
BENCH = ROOT
os.environ['MODEL_DIR'] = os.path.join(BENCH, 'models')
os.environ['DATA_DIR'] = os.path.join(BENCH, 'datasets')
os.environ['HF_DATASETS_CACHE'] = os.path.join(BENCH, 'datasets', 'huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import numpy as np
import data
import train as T
import optimizers as opt
from logger import ExperimentLogger, ResumableRunner  # noqa: E402

# 5 个可开关组件，顺序即位掩码顺序
COMPONENTS = ['cosine', 'trust', 'filter', 'phase', 'depth']
SEEDS = [42, 1337, 2024, 5555, 9999]


def mask_to_flags(mask):
    """位掩码 -> {component: bool}。mask 的 bit i 对应 COMPONENTS[i]。"""
    return {COMPONENTS[i]: bool(mask & (1 << i)) for i in range(len(COMPONENTS))}


def flags_to_mask(flags):
    m = 0
    for i, c in enumerate(COMPONENTS):
        if flags[c]:
            m |= (1 << i)
    return m


def build_dals(model, lr, total_steps, flags):
    """严格复用 benchmark 的 opt.DALS，仅覆盖 use_* 开关。"""
    o = opt.DALS(model, lr=lr, momentum=0.9, weight_decay=1e-4,
                 T_max=total_steps, variant='full')
    o.use_cosine = flags['cosine']
    o.use_trust = flags['trust']
    o.use_filter = flags['filter']
    o.use_phase = flags['phase']
    o.use_depth = flags['depth']
    return o


def generate_combos(mode):
    """生成要跑的组合列表 [(mask, label)]。

    mode='loo': full + 5 个 leave-one-out + 10 个成对移除 = 16 组合。
    mode='full': 全部 2^5 = 32 组合。
    """
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
    else:
        raise ValueError(ds)
    total_steps = len(tr) * ep
    lr = 0.1  # from-scratch 任务统一 0.1（对齐 run_final.py）
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
    """给定 {mask: mean_acc}，用 2^k 因子分析计算主效应 + 二阶交互。"""
    full_mask = (1 << len(COMPONENTS)) - 1
    full = combo_means.get(full_mask)
    main = {}
    if full is not None:
        for i, c in enumerate(COMPONENTS):
            loo = combo_means.get(full_mask & ~(1 << i))
            if loo is not None:
                main[c] = full - loo  # 移除该组件导致的精度损失
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
    ap.add_argument('--datasets', nargs='+', default=['synthetic'])
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--mode', choices=['loo', 'full'], default='loo')
    args = ap.parse_args()

    device = torch.device('cuda') if args.device == 'cuda' else torch.device('cpu')
    logger = ExperimentLogger(os.path.join(AUDIT, 'logs'), run_name='stage3')
    runner = ResumableRunner(logger)

    combos = generate_combos(args.mode)
    print(f'组合数：{len(combos)}（mode={args.mode}），数据集 {args.datasets}，种子 {args.seeds}')

    # 汇总每个 (dataset, mask) 的 per-seed acc
    agg = {}
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
    print(f'消融运行完成：成功 {summary["success"]}，失败 {summary["failed"]}，跳过 {summary["skipped"]}')

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

    # 计算 mean 与交互
    combo_means = {}
    analysis = {}
    for (ds, mask), accs in combo_seeds.items():
        accs = [a for a in accs if a is not None]
        combo_means[(ds, mask)] = float(np.mean(accs)) if accs else None
    for ds in args.datasets:
        cm = {mask: v for (d, mask), v in combo_means.items() if d == ds}
        analysis[ds] = interaction_analysis(ds, cm)
        analysis[ds]['combo_means'] = {f'{m:05b}': round(v, 4) for m, v in cm.items() if v is not None}

    out = {'mode': args.mode, 'components': COMPONENTS, 'summary': summary, 'analysis': analysis}
    out_path = os.path.join(AUDIT, 'results', 'stage3_interaction.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # markdown
    md = ['# 阶段三：消融组件交互量化', '', f'模式 {args.mode}，组件 {COMPONENTS}', '']
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
    md_path = os.path.join(AUDIT, 'results', 'stage3_interaction_report.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md))

    print(f'\n主效应与交互项：')
    for ds in args.datasets:
        a = analysis[ds]
        print(f'  [{ds}] 主效应: {a["main_effects"]}')
        print(f'  [{ds}] 交互项: {a["interactions"]}')
    print(f'\nJSON: {out_path}')
    print(f'报告: {md_path}')


if __name__ == '__main__':
    main()
