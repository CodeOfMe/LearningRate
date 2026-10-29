"""意见2：深度-梯度噪声假设的实证验证。

验证 DALS 深度滤波所依赖的前提——「越底层梯度噪声越大、越需要强 EMA 滤波
(alpha = 0.7 - 0.4*d_l)」是否在实测中成立。

方法：在 CIFAR-100 上用**普通 SGD-momentum（无任何梯度滤波）**训练 ConvNet100，
每步采样原始梯度，按层（深度）聚合三个指标：
  1. 梯度范数方差（跨 mini-batch 的波动）
  2. 梯度信噪比 SNR = mean(||g||) / std(||g||)
  3. 相邻 mini-batch 梯度余弦相似度 cos(g_t, g_{t+1})
然后画「指标 vs 层深度」三条曲线，对照线性深度缩放假设。

注：用 SGD 而非 DALS 采样，是为了测「模型本身的原始梯度噪声」，避免循环论证。
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
import torch.nn as nn
import numpy as np

import data

DATASET = 'cifar100'
EPOCHS = 30
BATCH = 128
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
SEED = 42


def layer_key_of(name):
    """'features.0.weight' -> 'features.0'；'classifier.0.bias' -> 'classifier.0'"""
    parts = name.split('.')
    if parts[0] == 'classifier':
        return f'classifier.{parts[1]}'
    return f'{parts[0]}.{parts[1]}'


def build_layers(model):
    """按命名顺序把参数归到层，得到 layer_key -> depth_index 与 param 列表。"""
    ordered = []  # [(layer_key, [param_names])]
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        k = layer_key_of(name)
        if not ordered or ordered[-1][0] != k:
            ordered.append([k, []])
        ordered[-1][1].append(name)
    layers = []
    for i, (k, names) in enumerate(ordered):
        depth_ratio = i / (len(ordered) - 1) if len(ordered) > 1 else 0.5
        layers.append({'key': k, 'depth_ratio': depth_ratio, 'names': names,
                       'norm_sum': 0.0, 'norm_sq_sum': 0.0, 'count': 0,
                       'cos_sum': 0.0, 'cos_count': 0,
                       'prev_norm_sq': 0.0, 'prev_grads': {}})
    return layers


def collect_step(model, layers):
    """在 loss.backward() 之后、optimizer.step() 之前调用，逐层累计梯度统计。"""
    for L in layers:
        norm_sq = 0.0
        dot_prev = 0.0
        for name in L['names']:
            p = dict(model.named_parameters())[name]
            g = p.grad
            if g is None:
                continue
            g_flat = g.detach().flatten()
            norm_sq += float((g_flat * g_flat).sum())
            prev = L['prev_grads'].get(name)
            if prev is not None:
                dot_prev += float((g_flat * prev).sum())
            L['prev_grads'][name] = g_flat.clone()
        norm = math.sqrt(norm_sq) if norm_sq > 0 else 0.0
        L['norm_sum'] += norm
        L['norm_sq_sum'] += norm_sq
        L['count'] += 1
        if L['prev_norm_sq'] > 0 and norm_sq > 0:
            cos = dot_prev / (math.sqrt(L['prev_norm_sq']) * math.sqrt(norm_sq))
            cos = max(-1.0, min(1.0, cos))
            L['cos_sum'] += cos
            L['cos_count'] += 1
        L['prev_norm_sq'] = norm_sq


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    data.set_seed(SEED)

    device = torch.device('cuda')
    tr, va, te = data.cifar_loaders(DATASET, seed=SEED, batch=BATCH)
    model = data.ConvNet100(n_classes=100).to(device)
    layers = build_layers(model)
    print(f'层数 {len(layers)}，各层：')
    for L in layers:
        print(f"  depth={L['depth_ratio']:.2f}  {L['key']}  ({len(L['names'])} 个参数)")

    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                          weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    from tqdm import tqdm
    model.train()
    for ep in range(1, EPOCHS + 1):
        pbar = tqdm(tr, desc=f'epoch {ep}/{EPOCHS}', leave=False, ncols=80)
        running = 0.0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            collect_step(model, layers)   # 采原始梯度（step 前）
            opt.step()
            running = 0.9 * running + 0.1 * loss.item()
            pbar.set_postfix(loss=f'{running:.4f}')
        print(f'  epoch {ep}/{EPOCHS} 完成 (loss≈{running:.4f})', flush=True)

    # 汇总每层指标
    out = {'dataset': DATASET, 'optimizer': 'SGD-momentum', 'epochs': EPOCHS,
           'seed': SEED, 'layers': []}
    for L in layers:
        mean_norm = L['norm_sum'] / L['count'] if L['count'] else 0.0
        var_norm = (L['norm_sq_sum'] / L['count'] - mean_norm ** 2) if L['count'] else 0.0
        std_norm = math.sqrt(max(0.0, var_norm))
        snr = mean_norm / std_norm if std_norm > 0 else 0.0
        cos = L['cos_sum'] / L['cos_count'] if L['cos_count'] else 0.0
        out['layers'].append({'key': L['key'], 'depth_ratio': round(L['depth_ratio'], 4),
                              'mean_norm': round(mean_norm, 6), 'std_norm': round(std_norm, 6),
                              'variance': round(var_norm, 6), 'snr': round(snr, 4),
                              'cos_sim': round(cos, 4)})
        print(f"  depth={L['depth_ratio']:.2f} {L['key']:>14s}  "
              f"mean_norm={mean_norm:.4f} std={std_norm:.4f} SNR={snr:.2f} cos={cos:.4f}")

    OUT_DIR = os.path.join(HERE, 'outputs')
    out_path = os.path.join(OUT_DIR, 'grad_noise.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(out, open(out_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'\n结果已写入 {out_path}')

    # 画图
    plot(out, os.path.join(OUT_DIR, 'grad_noise.png'))


def plot(out, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ls = out['layers']
    depth = [l['depth_ratio'] for l in ls]
    keys = [l['key'] for l in ls]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    # 1) 梯度范数方差
    axes[0].bar(range(len(ls)), [l['variance'] for l in ls], color='tab:blue')
    axes[0].set_xticks(range(len(ls))); axes[0].set_xticklabels(keys, rotation=45, ha='right')
    axes[0].set_title('Gradient variance vs depth')
    axes[0].set_xlabel('layer (bottom -> top)')
    # 2) SNR
    axes[1].plot(depth, [l['snr'] for l in ls], 'o-', color='tab:green')
    axes[1].set_title('Gradient SNR vs depth')
    axes[1].set_xlabel('depth ratio'); axes[1].set_ylabel('SNR (higher = more stable)')
    # 3) cosine similarity
    axes[2].plot(depth, [l['cos_sim'] for l in ls], 's-', color='tab:red')
    axes[2].set_title('Consecutive-batch gradient cosine similarity')
    axes[2].set_xlabel('depth ratio'); axes[2].set_ylabel('cosine (higher = more stable)')
    fig.suptitle('Per-layer gradient noise on CIFAR-100 (SGD-momentum, 30 epochs)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f'图已保存 {path}')


if __name__ == '__main__':
    main()
