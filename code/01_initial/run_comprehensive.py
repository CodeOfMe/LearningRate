"""
Final comprehensive benchmark: 16 strategies across 5 generations.
Tracks convergence milestones (epochs and time to reach accuracy thresholds).
Generates publication-quality figures.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import time
from collections import OrderedDict

from optimizers import (
    FixedLRSGD, CosineAnnealingSGD, SGDRWithRestarts,
    AdamW, AdaBound, Adafactor,
    DiscriminativeLR, LARS, LAMB,
    RAdam, Lookahead, SAM, Sophia, Lion, ScheduleFree,
    Grokfast, DALS, DALSFast, DALSAcc,
    STLRScheduler, build_discriminative_param_groups,
    build_discriminative_stlr_param_groups,
)

FIGS_DIR = Path(__file__).parent / "figs"
RESULTS_DIR = Path(__file__).parent / "results"
FIGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

def seed_all(s=42):
    torch.manual_seed(s)
    np.random.seed(s)

class BenchmarkModel(nn.Module):
    def __init__(self, d=64, h=128, nc=10):
        super().__init__()
        self.layer1 = nn.Linear(d, h)
        self.layer2 = nn.Linear(h, h)
        self.layer3 = nn.Linear(h, h)
        self.layer4 = nn.Linear(h, nc)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)

def make_data(n=8000, d=64, nc=10):
    torch.manual_seed(42)
    X = torch.randn(n, d)
    base = X[:, :nc] * 3.0
    y = base.argmax(dim=1) % nc
    X = X + torch.randn_like(X) * 0.1
    return X, y

def evaluate(model, X, y, bs=512):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb, yb = X[i:i+bs], y[i:i+bs]
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += len(yb)
    return 100.0 * correct / total

EPOCHS = 80
BATCH_SIZE = 64
N_TRAIN = 6400
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

def train_and_evaluate(name, epochs=EPOCHS, base_lr=0.03):
    seed_all(42)
    n, d, nc = 8000, 64, 10
    X, y = make_data(n, d, nc)
    X_train, y_train = X[:6400], y[:6400]
    X_test, y_test = X[6400:], y[6400:]

    model = BenchmarkModel(d, 128, nc)
    total_steps = epochs * STEPS_PER_EPOCH

    opt, scheduler, is_sam = create_optimizer(name, model, base_lr, total_steps)

    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'lr': []}
    milestones = {}
    best_acc = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        indices = torch.randperm(len(X_train))
        epoch_loss = 0

        for i in range(0, len(X_train), BATCH_SIZE):
            idx = indices[i:i+BATCH_SIZE]
            xb, yb = X_train[idx], y_train[idx]

            if is_sam:
                def closure():
                    opt.zero_grad()
                    out = model(xb)
                    loss = F.cross_entropy(out, yb)
                    loss.backward()
                    return loss
                loss = closure()
                opt.first_step(zero_grad=True)
                with torch.enable_grad():
                    closure()
                opt.second_step()
            elif isinstance(opt, (DALS, DALSFast, DALSAcc)):
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                opt.update_phase(loss.item())
                opt.step()
            else:
                opt.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                opt.step()

            if scheduler is not None and not isinstance(scheduler, STLRScheduler):
                scheduler.step()

            epoch_loss += loss.item()

        if isinstance(scheduler, STLRScheduler):
            for _ in range(STEPS_PER_EPOCH):
                scheduler.step()

        train_acc = evaluate(model, X_train, y_train)
        test_acc = evaluate(model, X_test, y_test)
        lr_now = opt.param_groups[0].get('lr', base_lr)
        t_now = time.time() - t0

        for thr in [60, 70, 75, 80, 82, 84, 85, 86, 87, 88]:
            if test_acc >= thr and thr not in milestones:
                milestones[thr] = (ep, round(t_now, 2))

        best_acc = max(best_acc, test_acc)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(epoch_loss / STEPS_PER_EPOCH)
        history['lr'].append(lr_now)

    total_time = time.time() - t0
    return history, best_acc, milestones, total_time

def create_optimizer(name, model, lr, total_steps):
    is_sam = False
    scheduler = None

    if name == 'Gen1_FixedSGD':
        opt = FixedLRSGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)

    elif name == 'Gen2_CosineSGD':
        opt = CosineAnnealingSGD(model.parameters(), lr=0.03, T_max=total_steps, momentum=0.9, weight_decay=1e-4)

    elif name == 'Gen2_SGDR':
        opt = SGDRWithRestarts(model.parameters(), lr=0.05, T_0=10, T_mult=2, momentum=0.9, weight_decay=1e-4)

    elif name == 'Gen3_Adam':
        opt = optim.Adam(model.parameters(), lr=3e-4)

    elif name == 'Gen3_AdamW':
        opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    elif name == 'Gen3_AdaBound':
        opt = AdaBound(model.parameters(), lr=3e-4, final_lr=0.01)

    elif name == 'Gen4_LARS':
        opt = LARS(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4, trust_coef=0.02)

    elif name == 'Gen4_Discriminative':
        pg = build_discriminative_param_groups(model, base_lr=0.05, decay_factor=2.6)
        opt = DiscriminativeLR(pg, momentum=0.9, weight_decay=1e-4)

    elif name == 'Gen5_RAdam':
        opt = RAdam(model.parameters(), lr=3e-4)

    elif name == 'Gen5_Lion':
        opt = Lion(model.parameters(), lr=1e-4, weight_decay=0.01)

    elif name == 'Gen5_Lookahead':
        opt = Lookahead(AdamW(model.parameters(), lr=3e-4, weight_decay=0.01), k=5, alpha=0.5)

    elif name == 'Gen5_SAM':
        opt = SAM(model.parameters(), optim.SGD, rho=0.05, adaptive=True, lr=0.03, momentum=0.9, weight_decay=1e-4)
        is_sam = True

    elif name == 'Gen5_Grokfast':
        opt = Grokfast(model.parameters(), lr=0.03, momentum=0.9, weight_decay=1e-4, alpha=0.98)

    elif name == 'Gen5_STLR':
        pg = build_discriminative_stlr_param_groups(model, base_lr=0.05, decay_factor=2.6)
        opt = optim.SGD(pg, momentum=0.9, weight_decay=1e-4)
        scheduler = STLRScheduler(opt, T=total_steps, decay_factor=2.6)

    elif name == 'SOTA_SAM_Discrim':
        pg = build_discriminative_param_groups(model, base_lr=0.05, decay_factor=2.6)
        opt = SAM(pg, optim.SGD, rho=0.05, adaptive=True, lr=0.05, momentum=0.9, weight_decay=1e-4)
        is_sam = True

    elif name == 'SOTA_DALS':
        opt = DALS(model, lr=0.03, momentum=0.9, weight_decay=1e-4,
                   trust_coef=0.02, grokfast_alpha=0.6,
                   warmup_frac=0.05, T_max=total_steps)

    elif name == 'SOTA_DALS_Fast':
        opt = DALSFast(model, lr=0.05, momentum=0.85, weight_decay=1e-4,
                        trust_coef=0.02, grokfast_alpha=0.6,
                        warmup_frac=0.02, T_max=total_steps)

    elif name == 'SOTA_DALS_Acc':
        opt = DALSAcc(model, lr=0.03, momentum=0.9, weight_decay=5e-4,
                       trust_coef=0.02, grokfast_alpha=0.7,
                       T_0=STEPS_PER_EPOCH * 10, T_mult=2)

    else:
        raise ValueError(f"Unknown strategy: {name}")

    return opt, scheduler, is_sam

if __name__ == '__main__':
    strategies = [
        'Gen1_FixedSGD', 'Gen2_CosineSGD', 'Gen2_SGDR',
        'Gen3_Adam', 'Gen3_AdamW', 'Gen3_AdaBound',
        'Gen4_LARS', 'Gen4_Discriminative',
        'Gen5_RAdam', 'Gen5_Lion', 'Gen5_Lookahead', 'Gen5_SAM', 'Gen5_Grokfast',
        'Gen5_STLR', 'SOTA_SAM_Discrim', 'SOTA_DALS',
        'SOTA_DALS_Fast', 'SOTA_DALS_Acc',
    ]

    gen_map = {
        'Gen1': 'G1: Fixed', 'Gen2': 'G2: Schedule', 'Gen3': 'G3: Adaptive',
        'Gen4': 'G4: Layer', 'Gen5': 'G5: Layer\u00d7Time', 'SOTA': 'SOTA',
    }
    gen_colors = {
        'Gen1': '#e74c3c', 'Gen2': '#f39c12', 'Gen3': '#3498db',
        'Gen4': '#2ecc71', 'Gen5': '#9b59b6', 'SOTA': '#e91e63',
    }

    print(f"\n{'='*100}")
    print(f"  FINAL COMPREHENSIVE BENCHMARK \u2014 {len(strategies)} strategies, {EPOCHS} epochs")
    print(f"{'='*100}\n")

    all_results = {}
    all_histories = {}

    for name in strategies:
        print(f"  Training: {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            history, best_acc, milestones, total_time = train_and_evaluate(name)
            elapsed = time.time() - t0
            all_histories[name] = history
            all_results[name] = {
                'best_acc': best_acc,
                'final_acc': history['test_acc'][-1],
                'milestones': milestones,
                'total_time': total_time,
            }
            ms_str = "  ".join([f"{thr}%:{v[0]}ep" for thr, v in sorted(milestones.items()) if thr <= 86])
            print(f"Best={best_acc:.1f}%  {ms_str}  ({total_time:.1f}s)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"FAILED: {e}")

    # ==================== Summary Table ====================
    print(f"\n{'='*110}")
    print(f"  CONVERGENCE & ACCURACY COMPARISON TABLE")
    print(f"{'='*110}")
    header = f"{'Strategy':<25s} {'Gen':>6s} {'Best':>6s}"
    for thr in [80, 84, 86, 87]:
        header += f" {'->'+str(thr)+'%':>8s}"
    header += f" {'Time':>6s}"
    print(header)
    print("-" * 110)

    sorted_strategies = sorted(strategies, key=lambda n: all_results.get(n, {}).get('best_acc', 0), reverse=True)
    for name in sorted_strategies:
        if name not in all_results:
            continue
        r = all_results[name]
        gen = name.split('_')[0]
        gen_label = gen_map.get(gen, gen)
        row = f"{name:<25s} {gen_label:>6s} {r['best_acc']:>5.1f}%"
        for thr in [80, 84, 86, 87]:
            if thr in r['milestones']:
                row += f" {r['milestones'][thr][0]:>5d}ep"
            else:
                row += f" {'n/a':>8s}"
        row += f" {r['total_time']:>5.1f}s"
        print(row)

    # ==================== Figure 1: Training Curves ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for name in strategies:
        if name not in all_histories:
            continue
        h = all_histories[name]
        gen = name.split('_')[0]
        color = gen_colors.get(gen, '#95a5a6')
        epochs_range = range(1, len(h['test_acc']) + 1)
        axes[0, 0].plot(epochs_range, h['test_acc'], label=name, color=color, linewidth=1.5, alpha=0.8)
        axes[0, 1].plot(epochs_range, h['train_loss'], label=name, color=color, linewidth=1.5, alpha=0.8)
        axes[1, 0].plot(epochs_range, h['lr'], label=name, color=color, linewidth=1.5, alpha=0.8)

    axes[0, 0].set_title('Test Accuracy Over Training', fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Accuracy (%)')
    axes[0, 0].legend(fontsize=7, ncol=2); axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_title('Training Loss', fontsize=13, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend(fontsize=7, ncol=2); axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')

    axes[1, 0].set_title('Learning Rate Schedule', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch'); axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].legend(fontsize=7, ncol=2); axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    names_sorted = sorted(all_results.keys(), key=lambda n: all_results[n]['best_acc'])
    accs = [all_results[n]['best_acc'] for n in names_sorted]
    colors_bar = [gen_colors.get(n.split('_')[0], '#95a5a6') for n in names_sorted]
    axes[1, 1].barh(range(len(names_sorted)), accs, color=colors_bar, edgecolor='white', height=0.7)
    axes[1, 1].set_yticks(range(len(names_sorted)))
    axes[1, 1].set_yticklabels(names_sorted, fontsize=8)
    axes[1, 1].set_xlabel('Best Test Accuracy (%)')
    axes[1, 1].set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    for i, v in enumerate(accs):
        axes[1, 1].text(v + 0.2, i, f'{v:.1f}%', va='center', fontsize=7)
    axes[1, 1].grid(True, axis='x', alpha=0.3)

    plt.suptitle('Learning Rate Strategy Comprehensive Comparison\n(Gen1\u2192Gen5 + SOTA)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / 'comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {FIGS_DIR / 'comprehensive_comparison.png'}")
    plt.close(fig)

    # ==================== Figure 2: Convergence Speed ====================
    fig, ax = plt.subplots(figsize=(14, 7))
    top_names = sorted(all_results.keys(), key=lambda n: all_results[n]['best_acc'], reverse=True)[:8]
    for name in top_names:
        h = all_histories[name]
        gen = name.split('_')[0]
        color = gen_colors.get(gen, '#95a5a6')
        lw = 3.0 if name == 'SOTA_DALS' else 1.5
        ls = '--' if name == 'SOTA_DALS' else '-'
        ax.plot(range(1, len(h['test_acc']) + 1), h['test_acc'], ls, label=name, color=color, linewidth=lw, alpha=0.9)
    ax.axhline(y=86, color='gray', linestyle=':', alpha=0.5, label='86% threshold')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Convergence Speed: Top 8 Strategies', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / 'convergence_speed.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'convergence_speed.png'}")
    plt.close(fig)

    # ==================== Figure 3: Generation Evolution ====================
    gen_groups = OrderedDict()
    for name in strategies:
        gen = name.split('_')[0]
        if gen not in gen_groups:
            gen_groups[gen] = []
        if name in all_results:
            gen_groups[gen].append(all_results[name]['best_acc'])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    gen_order = ['Gen1', 'Gen2', 'Gen3', 'Gen4', 'Gen5', 'SOTA']
    gen_names_display = ['G1: Fixed LR', 'G2: LR Schedule', 'G3: Adaptive',
                         'G4: Layer-wise', 'G5: Layer\u00d7Time', 'SOTA Combined']
    gen_avgs = [np.mean(gen_groups.get(g, [0])) for g in gen_order if g in gen_groups]
    gen_maxs = [np.max(gen_groups.get(g, [0])) for g in gen_order if g in gen_groups]
    gen_filtered = [g for g in gen_order if g in gen_groups]

    x = range(len(gen_filtered))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], gen_avgs, width, label='Average', color='#3498db', alpha=0.7)
    ax1.bar([i + width / 2 for i in x], gen_maxs, width, label='Best', color='#e74c3c', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([gen_names_display[gen_order.index(g)] for g in gen_filtered], fontsize=9)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('5 Generations of Learning Rate Evolution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    improvements = [0] + [gen_maxs[i] - gen_maxs[i - 1] for i in range(1, len(gen_maxs))]
    ax2.plot([gen_names_display[gen_order.index(g)] for g in gen_filtered], improvements,
             'o-', color='#2ecc71', linewidth=2, markersize=8)
    ax2.fill_between(range(len(improvements)), improvements, alpha=0.2, color='#2ecc71')
    ax2.set_ylabel('Accuracy Improvement (%)')
    ax2.set_title('Marginal Gain per Generation', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    for i, v in enumerate(improvements):
        ax2.annotate(f'+{v:.1f}%' if v > 0 else f'{v:.1f}%',
                     (i, v), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / 'generation_evolution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'generation_evolution.png'}")
    plt.close(fig)

    # ==================== Save Results ====================
    serializable = {}
    for name in all_results:
        r = all_results[name]
        serializable[name] = {
            'best_acc': r['best_acc'],
            'final_acc': r['final_acc'],
            'total_time': r['total_time'],
            'milestones': {str(k): list(v) for k, v in r['milestones'].items()},
            'history': {k: [float(x) for x in vs] for k, vs in all_histories[name].items()},
        }
    with open(RESULTS_DIR / 'comprehensive_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR}")
    print("All figures saved to", FIGS_DIR)