"""
Fast CIFAR-10 benchmark: 10 key optimizers, 10 epochs, SimpleCNN.
Optimized for MPS with small model and reasonable training time.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import numpy as np
import time
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from optimizers import (
    FixedLRSGD, CosineAnnealingSGD, SGDRWithRestarts,
    AdamW, AdaBound, Adafactor,
    DiscriminativeLR, LARS, LAMB,
    RAdam, Lookahead, SAM, Lion, ScheduleFree,
    Grokfast, DALS,
    STLRScheduler, build_discriminative_param_groups,
    build_discriminative_stlr_param_groups,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def get_data(batch_size=256):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    testloader = DataLoader(testset, batch_size=batch_size*2, shuffle=False, num_workers=0)
    return trainloader, testloader

def evaluate(model, testloader):
    model.eval()
    correct = total = 0
    loss_sum = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss_sum += F.cross_entropy(outputs, labels).item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100.0 * correct / total, loss_sum / total

def train(model, optimizer, trainloader, testloader, epochs, scheduler=None):
    model.to(DEVICE)
    history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'lr_history': [], 'epoch_time': []}
    is_sam = isinstance(optimizer, SAM)
    is_dals = isinstance(optimizer, DALS)
    best_acc = 0
    best_epoch = 0
    
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total_loss = correct = total = 0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            if is_sam:
                def closure():
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = F.cross_entropy(outputs, targets)
                    loss.backward()
                    return loss
                loss = closure()
                optimizer.first_step(zero_grad=True)
                with torch.enable_grad():
                    closure()
                optimizer.second_step()
                outputs = model(inputs)
            elif is_dals:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = F.cross_entropy(outputs, targets)
                loss.backward()
                optimizer.step()
            
            if scheduler and not isinstance(scheduler, STLRScheduler):
                scheduler.step()
            
            total_loss += loss.item() * targets.size(0)
            with torch.no_grad():
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total += targets.size(0)
        
        if isinstance(scheduler, STLRScheduler):
            for _ in range(len(trainloader)):
                scheduler.step()
        
        epoch_time = time.time() - t0
        train_acc = 100.0 * correct / total
        train_loss = total_loss / total
        test_acc, test_loss = evaluate(model, testloader)
        
        lr_val = optimizer.param_groups[0]['lr'] if hasattr(optimizer, 'param_groups') else 0
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr_history'].append(lr_val)
        history['epoch_time'].append(epoch_time)
        
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
        
        print(f"  Epoch {epoch+1:2d}/{epochs} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% "
              f"(Best: {best_acc:.2f}%@{best_epoch+1}) | LR: {lr_val:.6f} | {epoch_time:.1f}s")
    
    return {'history': history, 'best_acc': best_acc, 'best_epoch': best_epoch,
            'final_acc': history['test_acc'][-1], 'total_time': sum(history['epoch_time'])}


def make_optimizer(name, model, total_steps):
    if name == 'Gen1_FixedSGD':
        return FixedLRSGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif name == 'Gen2_CosineSGD':
        return CosineAnnealingSGD(model.parameters(), lr=0.1, T_max=total_steps, momentum=0.9, weight_decay=5e-4)
    elif name == 'Gen3_AdamW':
        return AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    elif name == 'Gen4_Discriminative':
        pg = build_discriminative_param_groups(model, base_lr=0.1, decay_factor=2.6)
        return DiscriminativeLR(pg, momentum=0.9, weight_decay=5e-4)
    elif name == 'Gen4_LARS':
        return LARS(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif name == 'Gen5_Lion':
        return Lion(model.parameters(), lr=1e-4, weight_decay=0.01)
    elif name == 'Gen5_LookaheadAdamW':
        return Lookahead(AdamW(model.parameters(), lr=3e-4, weight_decay=0.05), k=5, alpha=0.5)
    elif name == 'Gen5_SAM':
        return SAM(model.parameters(), optim.SGD, rho=0.05, adaptive=True, lr=0.1, momentum=0.9, weight_decay=5e-4)
    elif name == 'Gen5_Grokfast':
        return Grokfast(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, alpha=0.98)
    elif name == 'Gen5_STLR':
        pg = build_discriminative_stlr_param_groups(model, base_lr=0.1, decay_factor=2.6)
        opt = optim.SGD(pg, momentum=0.9, weight_decay=5e-4)
        sched = STLRScheduler(opt, T=total_steps, decay_factor=2.6)
        return opt, sched
    elif name == 'SOTA_DALS':
        return DALS(model, lr=0.1, momentum=0.9, weight_decay=5e-4, decay_factor=2.6,
                   trust_coef=0.02, T_max=total_steps)
    elif name == 'SOTA_SAM_Discriminative':
        pg = build_discriminative_param_groups(model, base_lr=0.1, decay_factor=2.6)
        return SAM(pg, optim.SGD, rho=0.05, adaptive=True, lr=0.1, momentum=0.9, weight_decay=5e-4)
    return None


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='quick', choices=['quick', 'full'])
    parser.add_argument('--epochs', type=int, default=None)
    args = parser.parse_args()
    
    epochs = 10 if args.mode == 'quick' else 30
    if args.epochs:
        epochs = args.epochs
    
    print(f"\n{'='*60}")
    print(f"  CIFAR-10 Benchmark  |  Device: {DEVICE}  |  Epochs: {epochs}")
    print(f"{'='*60}\n")
    
    trainloader, testloader = get_data()
    
    all_results = {}
    gen_colors = {
        'Gen1': '#e74c3c', 'Gen2': '#f39c12', 'Gen3': '#3498db',
        'Gen4': '#2ecc71', 'Gen5': '#9b59b6', 'SOTA': '#e91e63'
    }
    
    configs = [
        'Gen1_FixedSGD', 'Gen2_CosineSGD', 'Gen3_AdamW', 'Gen4_Discriminative',
        'Gen4_LARS', 'Gen5_Lion', 'Gen5_LookaheadAdamW', 'Gen5_SAM',
        'Gen5_Grokfast', 'Gen5_STLR', 'SOTA_SAM_Discriminative', 'SOTA_DALS'
    ]
    
    total_steps = epochs * len(trainloader)
    
    for name in configs:
        print(f"\n>>> {name}")
        seed_everything(42)
        model = SmallCNN()
        
        result = make_optimizer(name, model, total_steps)
        if result is None:
            continue
        
        if isinstance(result, tuple):
            optimizer, scheduler = result
        elif name == 'SOTA_DALS':
            optimizer = result
            scheduler = None
        else:
            optimizer = result
            scheduler = None
        
        try:
            r = train(model, optimizer, trainloader, testloader, epochs, scheduler)
            all_results[name] = r
            print(f"  => Best: {r['best_acc']:.2f}% @ epoch {r['best_epoch']+1}")
        except Exception as e:
            print(f"  => FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*70}")
    print(f"  FINAL RESULTS — CIFAR-10 ({epochs} epochs, SmallCNN)")
    print(f"{'='*70}")
    header = f"{'Optimizer':<28} {'Best':>7} {'Final':>7} {'Best@':>6} {'Time':>7} {'Gen':>5}"
    print(header)
    print("-"*len(header))
    
    for name, r in sorted(all_results.items(), key=lambda x: -x[1]['best_acc']):
        gen = name.split('_')[0]
        print(f"{name:<28} {r['best_acc']:>6.2f}% {r['final_acc']:>6.2f}% "
              f"{r['best_epoch']+1:>5d} {r['total_time']:>6.1f}s {gen:>5}")
    
    best = max(all_results.items(), key=lambda x: x[1]['best_acc'])
    print(f"\n  BEST: {best[0]} = {best[1]['best_acc']:.2f}%")
    baseline = next((r for n, r in all_results.items() if 'Gen1' in n), None)
    if baseline:
        print(f"  vs Gen1 baseline: +{best[1]['best_acc'] - baseline['best_acc']:.2f}% improvement")
    
    RESULTS_DIR = Path(__file__).parent / "results"
    FIGS_DIR = Path(__file__).parent / "figs"
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGS_DIR.mkdir(exist_ok=True)
    
    serializable = {}
    for name, r in all_results.items():
        serializable[name] = {
            'best_acc': r['best_acc'], 'best_epoch': r['best_epoch'],
            'final_acc': r['final_acc'], 'total_time': r['total_time'],
            'history': r['history'],
        }
    with open(RESULTS_DIR / f"cifar10_{epochs}ep_results.json", 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for i, (name, r) in enumerate(all_results.items()):
            h = r['history']
            gen = name.split('_')[0]
            color = gen_colors.get(gen, '#95a5a6')
            axes[0].plot(range(1, len(h['test_acc'])+1), h['test_acc'], label=name, color=color, linewidth=1.5)
            axes[1].plot(range(1, len(h['test_loss'])+1), h['test_loss'], label=name, color=color, linewidth=1.5)
            axes[2].plot(range(1, len(h['lr_history'])+1), h['lr_history'], label=name, color=color, linewidth=1.5)
        axes[0].set_title('Test Accuracy'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend(fontsize=7, ncol=2); axes[0].grid(True, alpha=0.3)
        axes[1].set_title('Test Loss'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss')
        axes[1].legend(fontsize=7, ncol=2); axes[1].grid(True, alpha=0.3)
        axes[2].set_title('Learning Rate'); axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('LR')
        axes[2].set_yscale('log'); axes[2].legend(fontsize=7, ncol=2); axes[2].grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIGS_DIR / f"cifar10_{epochs}ep_curves.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGS_DIR / f'cifar10_{epochs}ep_curves.png'}")
        plt.close(fig)
        
        names = sorted(all_results.keys(), key=lambda n: all_results[n]['best_acc'])
        accs = [all_results[n]['best_acc'] for n in names]
        colors = [gen_colors.get(n.split('_')[0], '#95a5a6') for n in names]
        fig, ax = plt.subplots(figsize=(10, max(6, len(names)*0.4)))
        ax.barh(range(len(names)), accs, color=colors, edgecolor='white', height=0.7)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel('Best Test Accuracy (%)')
        ax.set_title(f'Best Accuracy Comparison — CIFAR-10 ({epochs} epochs)')
        for i, v in enumerate(accs):
            ax.text(v+0.1, i, f'{v:.2f}%', va='center', fontsize=8)
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        fig.savefig(FIGS_DIR / f"cifar10_{epochs}ep_bar.png", dpi=150, bbox_inches='tight')
        print(f"Saved: {FIGS_DIR / f'cifar10_{epochs}ep_bar.png'}")
        plt.close(fig)
    except Exception as e:
        print(f"Plotting error: {e}")