"""
Synthetic benchmark: validates all optimizers on a controlled problem.
Produces the full comparison without needing GPU training time.
Generates publication-quality figures for the README.
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
import matplotlib.gridspec as gridspec
import json
import time
from collections import OrderedDict

from optimizers import (
    FixedLRSGD, CosineAnnealingSGD, SGDRWithRestarts,
    AdamW, AdaBound, Adafactor,
    DiscriminativeLR, LARS, LAMB,
    RAdam, Lookahead, SAM, Sophia, Lion, ScheduleFree,
    Grokfast, DALS,
    STLRScheduler, build_discriminative_param_groups,
    build_discriminative_stlr_param_groups,
)

DEVICE = torch.device("cpu")
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
    # Strong hierarchical label structure tied to input features
    # Lower dims → more universal features, higher dims → task-specific
    base = X[:, :nc] * 3.0  # amplify signal
    y = base.argmax(dim=1) % nc
    # Add noise to make it non-trivial
    X = X + torch.randn_like(X) * 0.1
    return X, y

h_dim = 128

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

def train_and_evaluate(name, epochs=30, lr=0.01):
    seed_all(42)
    n, d, nc = 8000, 64, 10
    X, y = make_data(n, d, nc)
    X_train, y_train = X[:6400], y[:6400]
    X_test, y_test = X[6400:], y[6400:]
    
    model = BenchmarkModel(d, h_dim, nc)
    total_steps = epochs * (6400 // 64)
    
    opt = create_optimizer(name, model, lr, total_steps)
    if isinstance(opt, tuple):
        optimizer, scheduler = opt
    else:
        optimizer, scheduler = opt, None
    
    is_sam = isinstance(optimizer, SAM)
    is_dals = isinstance(optimizer, DALS)
    
    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'lr': []}
    best_acc = 0
    
    for ep in range(epochs):
        model.train()
        indices = torch.randperm(len(X_train))
        epoch_loss = 0
        for i in range(0, len(X_train), 64):
            idx = indices[i:i+64]
            xb, yb = X_train[idx], y_train[idx]
            
            if is_sam:
                def closure():
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = F.cross_entropy(out, yb)
                    loss.backward()
                    return loss
                loss = closure()
                optimizer.first_step(zero_grad=True)
                with torch.enable_grad():
                    closure()
                optimizer.second_step()
            elif is_dals:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(xb), yb)
                loss.backward()
                optimizer.step()
            
            if scheduler and not isinstance(scheduler, STLRScheduler):
                scheduler.step()
            epoch_loss += loss.item()
        
        if isinstance(scheduler, STLRScheduler):
            for _ in range(6400 // 64):
                scheduler.step()
        
        train_acc = evaluate(model, X_train, y_train)
        test_acc = evaluate(model, X_test, y_test)
        lr_now = optimizer.param_groups[0].get('lr', lr) if hasattr(optimizer, 'param_groups') else lr
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(epoch_loss / (6400 // 64))
        history['lr'].append(lr_now)
        best_acc = max(best_acc, test_acc)
    
    return history, best_acc

def create_optimizer(name, model, lr, total_steps):
    if name == 'Gen1_FixedSGD':
        return FixedLRSGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif name == 'Gen2_CosineSGD':
        return CosineAnnealingSGD(model.parameters(), lr=lr, T_max=total_steps, momentum=0.9, weight_decay=1e-4)
    elif name == 'Gen2_SGDR':
        return SGDRWithRestarts(model.parameters(), lr=lr, T_0=5, momentum=0.9, weight_decay=1e-4)
    elif name == 'Gen3_Adam':
        return optim.Adam(model.parameters(), lr=3e-4)
    elif name == 'Gen3_AdamW':
        return AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    elif name == 'Gen3_AdaBound':
        return AdaBound(model.parameters(), lr=3e-4, final_lr=0.01)
    elif name == 'Gen4_LARS':
        return LARS(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    elif name == 'Gen4_Discriminative':
        pg = build_discriminative_param_groups(model, base_lr=0.05, decay_factor=2.6)
        return DiscriminativeLR(pg, momentum=0.9, weight_decay=1e-4)
    elif name == 'Gen5_RAdam':
        return RAdam(model.parameters(), lr=3e-4)
    elif name == 'Gen5_Lion':
        return Lion(model.parameters(), lr=1e-4, weight_decay=0.01)
    elif name == 'Gen5_Lookahead':
        return Lookahead(AdamW(model.parameters(), lr=3e-4, weight_decay=0.01), k=5, alpha=0.5)
    elif name == 'Gen5_SAM':
        return SAM(model.parameters(), optim.SGD, rho=0.05, adaptive=True, lr=lr, momentum=0.9, weight_decay=1e-4)
    elif name == 'Gen5_Grokfast':
        return Grokfast(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4, alpha=0.98)
    elif name == 'Gen5_STLR':
        pg = build_discriminative_stlr_param_groups(model, base_lr=0.05, decay_factor=2.6)
        opt = optim.SGD(pg, momentum=0.9, weight_decay=1e-4)
        sched = STLRScheduler(opt, T=total_steps, decay_factor=2.6)
        return opt, sched
    elif name == 'SOTA_SAM_Discrim':
        pg = build_discriminative_param_groups(model, base_lr=0.05, decay_factor=2.6)
        return SAM(pg, optim.SGD, rho=0.05, adaptive=True, lr=0.05, momentum=0.9, weight_decay=1e-4)
    elif name == 'SOTA_DALS':
        return DALS(model, lr=0.05, momentum=0.9, weight_decay=1e-4, decay_factor=2.6,
                   trust_coef=0.02, T_max=total_steps, sam_rho=0.05)


if __name__ == '__main__':
    optimizers = [
        'Gen1_FixedSGD', 'Gen2_CosineSGD', 'Gen2_SGDR',
        'Gen3_Adam', 'Gen3_AdamW', 'Gen3_AdaBound',
        'Gen4_LARS', 'Gen4_Discriminative',
        'Gen5_RAdam', 'Gen5_Lion', 'Gen5_Lookahead', 'Gen5_SAM',
        'Gen5_Grokfast', 'Gen5_STLR',
        'SOTA_SAM_Discrim', 'SOTA_DALS',
    ]
    
    print(f"\n{'='*60}")
    print(f"  Comprehensive LR Benchmark — {len(optimizers)} strategies, 30 epochs")
    print(f"{'='*60}\n")
    
    all_results = {}
    all_histories = {}
    
    for name in optimizers:
        print(f"  Training: {name}...", end=" ", flush=True)
        t0 = time.time()
        try:
            history, best_acc = train_and_evaluate(name, epochs=30)
            elapsed = time.time() - t0
            all_histories[name] = history
            all_results[name] = {
                'best_acc': best_acc,
                'final_acc': history['test_acc'][-1],
                'total_time': elapsed,
            }
            print(f"Best={best_acc:.1f}% Final={history['test_acc'][-1]:.1f}% ({elapsed:.1f}s)")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    # ==================== Summary Table ====================
    gen_colors = {
        'Gen1': '#e74c3c', 'Gen2': '#f39c12', 'Gen3': '#3498db',
        'Gen4': '#2ecc71', 'Gen5': '#9b59b6', 'SOTA': '#e91e63'
    }
    gen_labels = {
        'Gen1': 'G1: Fixed', 'Gen2': 'G2: Schedule', 'Gen3': 'G3: Adaptive',
        'Gen4': 'G4: Layer-wise', 'Gen5': 'G5: Layer×Time', 'SOTA': 'SOTA',
    }
    
    print(f"\n{'='*80}")
    print(f"  RESULTS SUMMARY — Hierarchical Learning Rate Evolution")
    print(f"{'='*80}")
    print(f"{'Optimizer':<28} {'Generation':>12} {'Best':>7} {'Final':>7} {'Time':>7}")
    print("-"*80)
    
    for name in optimizers:
        if name not in all_results:
            continue
        r = all_results[name]
        gen = name.split('_')[0]
        print(f"{name:<28} {gen_labels.get(gen, gen):>12} {r['best_acc']:>6.1f}% "
              f"{r['final_acc']:>6.1f}% {r['total_time']:>6.1f}s")
    
    best_name = max(all_results, key=lambda n: all_results[n]['best_acc'])
    print(f"\n  BEST: {best_name} = {all_results[best_name]['best_acc']:.1f}%")
    
    # ==================== Figure 1: Training Curves ====================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    for name in optimizers:
        if name not in all_histories:
            continue
        h = all_histories[name]
        gen = name.split('_')[0]
        color = gen_colors.get(gen, '#95a5a6')
        epochs = range(1, len(h['test_acc'])+1)
        axes[0, 0].plot(epochs, h['test_acc'], label=name, color=color, linewidth=1.5, alpha=0.8)
        axes[0, 1].plot(epochs, h['train_loss'], label=name, color=color, linewidth=1.5, alpha=0.8)
        axes[1, 0].plot(epochs, h['lr'], label=name, color=color, linewidth=1.5, alpha=0.8)
    
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
    
    # Bar chart
    names_sorted = sorted(all_results.keys(), key=lambda n: all_results[n]['best_acc'])
    accs = [all_results[n]['best_acc'] for n in names_sorted]
    colors_bar = [gen_colors.get(n.split('_')[0], '#95a5a6') for n in names_sorted]
    axes[1, 1].barh(range(len(names_sorted)), accs, color=colors_bar, edgecolor='white', height=0.7)
    axes[1, 1].set_yticks(range(len(names_sorted)))
    axes[1, 1].set_yticklabels(names_sorted, fontsize=8)
    axes[1, 1].set_xlabel('Best Test Accuracy (%)')
    axes[1, 1].set_title('Accuracy Comparison', fontsize=13, fontweight='bold')
    for i, v in enumerate(accs):
        axes[1, 1].text(v+0.2, i, f'{v:.1f}%', va='center', fontsize=7)
    axes[1, 1].grid(True, axis='x', alpha=0.3)
    
    plt.suptitle('Learning Rate Strategy Comprehensive Comparison\n(Gen1→Gen5 + SOTA)',
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / 'comprehensive_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {FIGS_DIR / 'comprehensive_comparison.png'}")
    plt.close(fig)
    
    # ==================== Figure 2: Generation Evolution ====================
    gen_groups = OrderedDict()
    for name in optimizers:
        gen = name.split('_')[0]
        if gen not in gen_groups:
            gen_groups[gen] = []
        if name in all_results:
            gen_groups[gen].append(all_results[name]['best_acc'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    gen_order = ['Gen1', 'Gen2', 'Gen3', 'Gen4', 'Gen5', 'SOTA']
    gen_names_display = ['G1: Fixed LR', 'G2: LR Schedule', 'G3: Adaptive',
                         'G4: Layer-wise', 'G5: Layer×Time', 'SOTA Combined']
    gen_avgs = [np.mean(gen_groups.get(g, [0])) for g in gen_order if g in gen_groups]
    gen_maxs = [np.max(gen_groups.get(g, [0])) for g in gen_order if g in gen_groups]
    gen_orders_filtered = [g for g in gen_order if g in gen_groups]
    
    x = range(len(gen_orders_filtered))
    width = 0.35
    ax1.bar([i - width/2 for i in x], gen_avgs, width, label='Average', color='#3498db', alpha=0.7)
    ax1.bar([i + width/2 for i in x], gen_maxs, width, label='Best', color='#e74c3c', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels([gen_names_display[gen_order.index(g)] for g in gen_orders_filtered], fontsize=9)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('5 Generations of Learning Rate Evolution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    improvements = [0] + [gen_maxs[i] - gen_maxs[i-1] for i in range(1, len(gen_maxs))]
    ax2.plot([gen_names_display[gen_order.index(g)] for g in gen_orders_filtered], improvements,
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
    
    # ==================== Figure 3: LR Schedule Comparison ====================
    fig, ax = plt.subplots(figsize=(12, 6))
    for name in optimizers:
        if name not in all_histories:
            continue
        h = all_histories[name]
        gen = name.split('_')[0]
        color = gen_colors.get(gen, '#95a5a6')
        style = '-' if 'SOTA' not in name else '--'
        lw = 1.5 if 'SOTA' not in name else 2.5
        ax.plot(range(1, len(h['lr'])+1), h['lr'], style, label=name, color=color, linewidth=lw, alpha=0.8)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedules Across 5 Generations', fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGS_DIR / 'lr_schedule_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {FIGS_DIR / 'lr_schedule_comparison.png'}")
    plt.close(fig)
    
    # Save results
    serializable = {}
    for name in all_results:
        serializable[name] = {
            **all_results[name],
            'history': {k: [float(v) for v in vs] for k, vs in all_histories[name].items()}
        }
    with open(RESULTS_DIR / 'comprehensive_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\nAll results saved to {RESULTS_DIR}")
    print("All figures saved to", FIGS_DIR)