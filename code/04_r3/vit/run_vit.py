"""意见4：DALS 三变体在 ViT-Small 上的验证。

在 ViT-Small（ImageNet 预训练，patch16/224）上微调 CIFAR-100，跑 DALS / DALS-Fast / DALS-Acc
三个变体，各 5 个种子，验证 DALS 能否扩展到带残差连接的架构（回应 Reviewer #4）。

关键点：
  - 本地加载预训练权重（offline，hf-mirror 下载的 WinKawaks/vit-small-patch16-224）
  - depth 组件映射到 ViT 的 12 个 encoder block（而非顶层 children 的 2 个模块）
  - 图片 resize 到 224 + ImageNet 归一化
  - 断点续跑（每个 (variant, seed) 一个 task）
"""
import os
import sys
import json
import math

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
os.environ['DATA_DIR'] = os.path.join(ROOT, 'datasets')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import numpy as np
from tqdm import tqdm
import data
import optimizers as opt
import train as T
from logger import ExperimentLogger, ResumableRunner

WEIGHTS_DIR = os.path.join(ROOT, 'models', 'vit-small-patch16-224')
OUT = os.path.join(HERE, 'outputs', 'vit_results')

VARIANTS = ['DALS', 'DALS_Fast', 'DALS_Acc']
SEEDS = [42, 1337, 2024, 5555, 9999]
EPOCHS = 30
BATCH = 64
LR_SGD = 5e-4          # 微调 regime（对齐论文 GLUE 微调）
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_vit():
    from transformers import ViTForImageClassification
    m = ViTForImageClassification.from_pretrained(
        WEIGHTS_DIR, num_labels=100, ignore_mismatched_sizes=True, local_files_only=True)
    return m


def build_vit_param_groups(model, lr):
    """把 ViT 的 12 个 encoder block + embeddings + 分类头 映射成 depth 层。"""
    blocks = list(model.vit.encoder.layer)          # 12 个 ViTLayer
    head_mods = [model.vit.layernorm]
    if hasattr(model.vit, 'pooler') and model.vit.pooler is not None:
        head_mods.append(model.vit.pooler)
    head_mods.append(model.classifier)
    all_layers = [model.vit.embeddings] + blocks + head_mods
    n = len(all_layers)
    groups = []
    for i, layer in enumerate(all_layers):
        ps = [p for p in layer.parameters() if p.requires_grad]
        if not ps:
            continue
        name_map = {id(p): nm for nm, p in layer.named_parameters()}
        groups.append({'params': ps, 'lr': lr,
                       'depth_ratio': i / (n - 1) if n > 1 else 0.5,
                       'names': [name_map.get(id(p), 'param') for p in ps],
                       'name_of': name_map})
    return groups


def vit_loaders(seed, batch=BATCH, val_frac=0.1):
    import torchvision
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(224), T.RandomHorizontalFlip(), T.ToTensor(), T.Normalize(MEAN, STD)])
    tf_test = T.Compose([T.Resize(224), T.ToTensor(), T.Normalize(MEAN, STD)])
    data_dir = os.environ.get('DATA_DIR', HERE)
    root = None
    for r in [data_dir, './data', '../data', './数据集', '../数据集']:
        if os.path.isdir(os.path.join(r, 'cifar-100-python')):
            root = r; break
    root = root or './data'
    train = torchvision.datasets.CIFAR100(root=root, train=True, download=False, transform=tf)
    test = torchvision.datasets.CIFAR100(root=root, train=False, download=False, transform=tf_test)
    n_val = int(len(train) * val_frac)
    n_tr = len(train) - n_val
    gen = torch.Generator().manual_seed(seed)
    tr, val = torch.utils.data.random_split(train, [n_tr, n_val], generator=gen)
    return (torch.utils.data.DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0),
            torch.utils.data.DataLoader(val, batch_size=batch, num_workers=0),
            torch.utils.data.DataLoader(test, batch_size=batch, num_workers=0))


def build_opt(variant, groups, steps_per_epoch):
    T_total = steps_per_epoch * EPOCHS
    if variant == 'DALS':
        return opt.DALS(groups, lr=LR_SGD, momentum=0.9, weight_decay=1e-4, T_max=T_total)
    if variant == 'DALS_Fast':
        return opt.DALSFast(groups, lr=LR_SGD * 4, momentum=0.9, weight_decay=1e-4,
                            warmup_frac=0.05, T_max=T_total, fast_phase0_bypass=False)
    if variant == 'DALS_Acc':
        return opt.DALSAcc(groups, lr=LR_SGD, momentum=0.9, weight_decay=5e-4,
                           T_max=T_total, T_0=steps_per_epoch * 10, T_mult=2)
    raise ValueError(variant)


def run_one(variant, seed, device):
    data.set_seed(seed)
    tr, va, te = vit_loaders(seed)
    model = build_vit().to(device)
    groups = build_vit_param_groups(model, LR_SGD)
    steps_per_epoch = len(tr)
    o = build_opt(variant, groups, steps_per_epoch)
    r = T.train_one(model, o, tr, va, te, device, n_labels=100, epochs=EPOCHS,
                    dals_phase_update=o.update_phase, select_by='val', verbose=0)
    return {'acc': r['test_acc_valsel'], 'f1': r['test_f1_valsel'],
            'wall_clock_s': r['wall_clock_s']}


def main():
    device = torch.device('cuda')
    os.makedirs(os.path.join(OUT, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(OUT, 'results'), exist_ok=True)
    logger = ExperimentLogger(os.path.join(OUT, 'logs'), run_name='vit',
                              results_dir=os.path.join(OUT, 'results'))
    runner = ResumableRunner(logger)
    tasks = [{'task_id': f'vit|{v}|{s}', 'description': f'ViT {v} seed={s}',
              'variant': v, 'seed': s} for v in VARIANTS for s in SEEDS]
    print(f'ViT 实验：{len(tasks)} 个 task（{len(VARIANTS)} 变体 × {len(SEEDS)} 种子）', flush=True)

    def run_task(t):
        return run_one(t['variant'], t['seed'], device)

    summary = runner.run_tasks(tasks, run_task)
    print(f'完成：成功 {summary["success"]}，失败 {summary["failed"]}，跳过 {summary["skipped"]}', flush=True)

    # 汇总
    rows = {}
    for t in tasks:
        f = os.path.join(OUT, 'results', f"{t['task_id']}.json")
        if os.path.exists(f):
            rec = json.load(open(f, encoding='utf-8'))
            if rec.get('status') == 'success' and rec.get('output'):
                rows.setdefault(t['variant'], []).append(rec['output'].get('acc'))
    print('\n=== ViT 结果（5-seed）===')
    for v in VARIANTS:
        accs = [a for a in rows.get(v, []) if a is not None]
        if accs:
            m = np.mean(accs); sd = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
            print(f'  {v}: acc={m:.2f} ± {sd:.2f}  ({accs})')


if __name__ == '__main__':
    main()
