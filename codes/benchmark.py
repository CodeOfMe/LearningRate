"""
Comprehensive Learning Rate Benchmark Suite
=============================================
Benchmarks all 5 generations of LR strategies across:
  - CIFAR-10 / CIFAR-100 image classification (ResNet-18)
  - Transfer learning (pretrained ResNet-50 → CIFAR-10)
  - NLP text classification (with discriminative fine-tuning)

Produces unified comparison tables and visualizations.
"""

import os
import sys
import json
import time
import copy
import math
import warnings
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as tv_models

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from optimizers import (
    FixedLRSGD, CosineAnnealingSGD, SGDRWithRestarts,
    AdamW, AdaBound, Adafactor,
    DiscriminativeLR, LARS, LAMB,
    RAdam, Lookahead, SAM, Sophia, Lion, ScheduleFree,
    Grokfast, DALS,
    STLRScheduler, build_discriminative_param_groups,
    build_discriminative_stlr_param_groups,
)

BASE_DIR = Path(__file__).parent
RESULTS_DIR = BASE_DIR / "results"
FIGS_DIR = BASE_DIR / "figs"
RESULTS_DIR.mkdir(exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def seed_everything(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    elif torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cifar10_loaders(data_dir="./data", batch_size=128, num_workers=2):
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
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                            download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                           download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size * 2, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return trainloader, testloader


def get_cifar100_loaders(data_dir="./data", batch_size=128, num_workers=2):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True,
                                             download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False,
                                            download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                             num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size * 2, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return trainloader, testloader


def get_resnet18_cifar(num_classes=10):
    model = torchvision.models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def get_pretrained_resnet50_cifar(num_classes=10, freeze_layers=True):
    model = torchvision.models.resnet50(weights='IMAGENET1K_V1')
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    if freeze_layers:
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
    return model


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = F.cross_entropy(outputs, labels)
            loss_sum += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    acc = 100.0 * correct / total
    avg_loss = loss_sum / total
    return acc, avg_loss


def train_one_epoch(model, optimizer, trainloader, device, scheduler=None):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    is_sam = isinstance(optimizer, SAM)
    is_dals = isinstance(optimizer, DALS)
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
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
                loss2 = closure()
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
        if scheduler is not None and not isinstance(scheduler, STLRScheduler):
            scheduler.step()
        total_loss += loss.item() * targets.size(0)
        with torch.no_grad():
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    if isinstance(scheduler, STLRScheduler):
        for _ in range(len(trainloader)):
            scheduler.step()
    return total_loss / total, 100.0 * correct / total


def run_experiment(config, device=None):
    if device is None:
        device = get_device()
    seed_everything(config.get('seed', 42))
    dataset = config.get('dataset', 'cifar10')
    num_classes = 100 if dataset == 'cifar100' else 10
    if dataset == 'cifar100':
        trainloader, testloader = get_cifar100_loaders(batch_size=config.get('batch_size', 128))
    else:
        trainloader, testloader = get_cifar10_loaders(batch_size=config.get('batch_size', 128))

    model_type = config.get('model', 'resnet18')
    if model_type == 'resnet18':
        model = get_resnet18_cifar(num_classes).to(device)
    elif model_type == 'resnet50_pretrained':
        model = get_pretrained_resnet50_cifar(num_classes).to(device)
    elif model_type == 'simplecnn':
        model = SimpleCNN(num_classes).to(device)
    else:
        model = get_resnet18_cifar(num_classes).to(device)

    optimizer_name = config['optimizer']
    epochs = config.get('epochs', 30)
    lr = config.get('lr', 0.1)
    weight_decay = config.get('weight_decay', 5e-4)
    total_steps = epochs * len(trainloader)

    optimizer = create_optimizer(optimizer_name, model, lr, weight_decay, total_steps, config)
    scheduler = create_scheduler(optimizer_name, optimizer, total_steps, config)

    history = {
        'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [],
        'lr_history': [], 'epoch_time': []
    }
    best_acc = 0
    best_epoch = 0
    for epoch in range(epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, optimizer, trainloader, device, scheduler)
        epoch_time = time.time() - t0
        test_acc, test_loss = evaluate(model, testloader, device)
        current_lr = get_current_lr(optimizer)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['lr_history'].append(current_lr)
        history['epoch_time'].append(epoch_time)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  [{optimizer_name}] Epoch {epoch+1}/{epochs} | "
                  f"Train: {train_acc:.2f}% | Test: {test_acc:.2f}% (Best: {best_acc:.2f}% @ {best_epoch+1}) | "
                  f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")
    return {
        'config': config,
        'history': history,
        'best_acc': best_acc,
        'best_epoch': best_epoch,
        'final_acc': history['test_acc'][-1],
        'total_time': sum(history['epoch_time']),
    }


def create_optimizer(name, model, lr, weight_decay, total_steps, config):
    if name == 'fixed_sgd':
        return FixedLRSGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'cosine_sgd':
        return CosineAnnealingSGD(model.parameters(), lr=lr, T_max=total_steps,
                                   momentum=0.9, weight_decay=weight_decay)
    elif name == 'sgdr':
        return SGDRWithRestarts(model.parameters(), lr=lr, T_0=config.get('T_0', 10),
                                 momentum=0.9, weight_decay=weight_decay)
    elif name == 'adam':
        return optim.Adam(model.parameters(), lr=config.get('adam_lr', 1e-3),
                          weight_decay=weight_decay)
    elif name == 'adamw':
        return AdamW(model.parameters(), lr=config.get('adam_lr', 1e-3),
                     weight_decay=config.get('adam_wd', 0.01))
    elif name == 'adabound':
        return AdaBound(model.parameters(), lr=config.get('adam_lr', 1e-3),
                        weight_decay=weight_decay, final_lr=config.get('final_lr', 0.1))
    elif name == 'adafactor':
        return Adafactor(model.parameters(), lr=config.get('adam_lr', 1e-3),
                         weight_decay=weight_decay)
    elif name == 'lars':
        return LARS(model.parameters(), lr=lr, momentum=0.9,
                    weight_decay=weight_decay, trust_coef=config.get('trust_coef', 0.02))
    elif name == 'lamb':
        return LAMB(model.parameters(), lr=config.get('adam_lr', 1e-3),
                    weight_decay=weight_decay)
    elif name == 'radam':
        return RAdam(model.parameters(), lr=config.get('adam_lr', 1e-3),
                     weight_decay=weight_decay)
    elif name == 'lion':
        return Lion(model.parameters(), lr=config.get('lion_lr', 1e-4),
                    weight_decay=config.get('adam_wd', 0.01))
    elif name == 'schedule_free':
        return ScheduleFree(model.parameters(), lr=config.get('adam_lr', 1e-3),
                           momentum=0.9, weight_decay=weight_decay)
    elif name == 'grokfast':
        return Grokfast(model.parameters(), lr=config.get('adam_lr', 1e-3),
                       momentum=0.9, weight_decay=weight_decay,
                       alpha=config.get('grokfast_alpha', 0.98))
    elif name == 'discriminative_sgd':
        param_groups = build_discriminative_param_groups(
            model, base_lr=lr, decay_factor=config.get('decay_factor', 2.6))
        discriminative_config = {
            'params': param_groups,
            'momentum': 0.9,
            'weight_decay': weight_decay,
        }
        return DiscriminativeLR(param_groups, momentum=0.9, weight_decay=weight_decay)
    elif name == 'discriminative_adamw':
        param_groups = build_discriminative_param_groups(
            model, base_lr=config.get('adam_lr', 1e-3),
            decay_factor=config.get('decay_factor', 2.6))
        return AdamW(param_groups, weight_decay=config.get('adam_wd', 0.01))
    elif name == 'discriminative_stlr':
        param_groups = build_discriminative_stlr_param_groups(
            model, base_lr=lr, decay_factor=config.get('decay_factor', 2.6))
        base_opt = optim.SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
        return base_opt
    elif name == 'sam':
        base_opt_cls = config.get('sam_base', optim.SGD)
        sam_kwargs = {'lr': lr, 'momentum': 0.9, 'weight_decay': weight_decay}
        return SAM(model.parameters(), base_opt_cls,
                   rho=config.get('sam_rho', 0.05),
                   adaptive=config.get('adaptive_sam', True),
                   **sam_kwargs)
    elif name == 'sam_discriminative':
        param_groups = build_discriminative_param_groups(
            model, base_lr=lr, decay_factor=config.get('decay_factor', 2.6))
        return SAM(param_groups, optim.SGD,
                   rho=config.get('sam_rho', 0.05),
                   adaptive=config.get('adaptive_sam', True),
                   lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif name == 'lookahead_adamw':
        base = AdamW(model.parameters(), lr=config.get('adam_lr', 1e-3),
                     weight_decay=config.get('adam_wd', 0.01))
        return Lookahead(base, k=config.get('lookahead_k', 5),
                         alpha=config.get('lookahead_alpha', 0.5))
    elif name == 'dals':
        return DALS(model, lr=lr, momentum=0.9, weight_decay=weight_decay,
                   decay_factor=config.get('decay_factor', 2.6),
                   trust_coef=config.get('trust_coef', 0.02),
                   stlr_cut_frac=config.get('stlr_cut_frac', 0.1),
                   stlr_ratio=config.get('stlr_ratio', 32),
                   sam_rho=config.get('sam_rho', 0.05),
                   grokfast_alpha=config.get('grokfast_alpha', 0.98),
                   T_max=total_steps,
                   warmup_steps=config.get('warmup_steps', total_steps // 10))
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def create_scheduler(name, optimizer, total_steps, config):
    if name == 'discriminative_stlr' or name == 'discriminative_adamw_stlr':
        decay_factor = config.get('decay_factor', 2.6) if 'discriminative' in name else None
        return STLRScheduler(optimizer, T=total_steps,
                            cut_frac=config.get('stlr_cut_frac', 0.1),
                            ratio=config.get('stlr_ratio', 32),
                            decay_factor=decay_factor)
    if name in ('adam', 'adamw', 'radam', 'lion', 'adabound', 'adafactor',
                'lamb', 'schedule_free', 'grokfast', 'lookahead_adamw',
                'discriminative_adamw', 'dals'):
        return None
    return None


def get_current_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']
    return 0


# ============================================================================
# Experiment Configurations
# ============================================================================

def get_cifar10_configs(epochs=30):
    base = {'dataset': 'cifar10', 'model': 'resnet18', 'epochs': epochs, 'batch_size': 128, 'seed': 42}
    configs = {
        'Gen1_FixedSGD':        {**base, 'optimizer': 'fixed_sgd', 'lr': 0.1},
        'Gen2_CosineSGD':       {**base, 'optimizer': 'cosine_sgd', 'lr': 0.1},
        'Gen2_SGDR':            {**base, 'optimizer': 'sgdr', 'lr': 0.1, 'T_0': 10},
        'Gen3_Adam':            {**base, 'optimizer': 'adam', 'adam_lr': 1e-3},
        'Gen3_AdamW':           {**base, 'optimizer': 'adamw', 'adam_lr': 1e-3, 'adam_wd': 0.01},
        'Gen3_AdaBound':        {**base, 'optimizer': 'adabound', 'adam_lr': 1e-3, 'final_lr': 0.1},
        'Gen3_Adafactor':       {**base, 'optimizer': 'adafactor', 'adam_lr': 5e-3},
        'Gen4_LARS':            {**base, 'optimizer': 'lars', 'lr': 0.1, 'trust_coef': 0.02},
        'Gen4_LAMB':            {**base, 'optimizer': 'lamb', 'adam_lr': 1e-3},
        'Gen4_Discriminative':  {**base, 'optimizer': 'discriminative_sgd', 'lr': 0.1, 'decay_factor': 2.6},
        'Gen5_RAdam':           {**base, 'optimizer': 'radam', 'adam_lr': 1e-3},
        'Gen5_Lion':            {**base, 'optimizer': 'lion', 'lion_lr': 1e-4, 'adam_wd': 0.01},
        'Gen5_Lookahead':       {**base, 'optimizer': 'lookahead_adamw', 'adam_lr': 1e-3, 'adam_wd': 0.01},
        'Gen5_SAM':             {**base, 'optimizer': 'sam', 'lr': 0.1, 'sam_rho': 0.05},
        'Gen5_ScheduleFree':    {**base, 'optimizer': 'schedule_free', 'adam_lr': 1e-3},
        'Gen5_Grokfast':        {**base, 'optimizer': 'grokfast', 'adam_lr': 5e-3, 'grokfast_alpha': 0.98},
        'Gen5_STLR':            {**base, 'optimizer': 'discriminative_stlr', 'lr': 0.1,
                                 'decay_factor': 2.6, 'stlr_cut_frac': 0.1, 'stlr_ratio': 32},
        'SOTA_SAM_Discrim':     {**base, 'optimizer': 'sam_discriminative', 'lr': 0.1,
                                 'decay_factor': 2.6, 'sam_rho': 0.05},
        'SOTA_DALS_Ours':       {**base, 'optimizer': 'dals', 'lr': 0.1, 'decay_factor': 2.6,
                                 'trust_coef': 0.02, 'sam_rho': 0.05, 'grokfast_alpha': 0.98},
    }
    return configs


def get_transfer_configs(epochs=20):
    base = {'dataset': 'cifar10', 'model': 'resnet50_pretrained', 'epochs': epochs,
            'batch_size': 64, 'seed': 42}
    configs = {
        'TF_FixedSGD':         {**base, 'optimizer': 'fixed_sgd', 'lr': 0.01},
        'TF_AdamW':            {**base, 'optimizer': 'adamw', 'adam_lr': 1e-3, 'adam_wd': 0.01},
        'TF_Discriminative':   {**base, 'optimizer': 'discriminative_sgd', 'lr': 0.01, 'decay_factor': 2.6},
        'TF_SAM':              {**base, 'optimizer': 'sam', 'lr': 0.01, 'sam_rho': 0.05},
        'TF_SAM_Discrim':      {**base, 'optimizer': 'sam_discriminative', 'lr': 0.01,
                                'decay_factor': 2.6, 'sam_rho': 0.05},
        'TF_DALS_Ours':        {**base, 'optimizer': 'dals', 'lr': 0.01, 'decay_factor': 2.6,
                                'trust_coef': 0.02, 'sam_rho': 0.05, 'grokfast_alpha': 0.98},
        'TF_STLR':             {**base, 'optimizer': 'discriminative_stlr', 'lr': 0.01,
                                'decay_factor': 2.6, 'stlr_cut_frac': 0.1, 'stlr_ratio': 32},
        'TF_Grokfast':         {**base, 'optimizer': 'grokfast', 'adam_lr': 1e-3, 'grokfast_alpha': 0.98},
    }
    return configs


# ============================================================================
# Visualization
# ============================================================================

def plot_training_curves(all_results, title="Learning Rate Strategy Comparison",
                         save_path=None):
    """Plot comprehensive training curves for all optimizers."""
    n = len(all_results)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols

    fig = plt.figure(figsize=(6 * ncols, 5 * nrows * 2))
    gs = gridspec.GridSpec(nrows * 2, ncols, hspace=0.3, wspace=0.3)

    colors = plt.cm.tab20(np.linspace(0, 1, n))

    ax_acc = fig.add_subplot(gs[0, :])
    ax_loss = fig.add_subplot(gs[1, :])

    for i, (name, result) in enumerate(all_results.items()):
        h = result['history']
        epochs = range(1, len(h['test_acc']) + 1)
        ax_acc.plot(epochs, h['test_acc'], label=name, color=colors[i], linewidth=1.5)
        ax_loss.plot(epochs, h['test_loss'], label=name, color=colors[i], linewidth=1.5)

    ax_acc.set_xlabel('Epoch')
    ax_acc.set_ylabel('Test Accuracy (%)')
    ax_acc.set_title(f'{title} - Test Accuracy')
    ax_acc.legend(fontsize=7, loc='lower right', ncol=2)
    ax_acc.grid(True, alpha=0.3)

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Test Loss')
    ax_loss.set_title(f'{title} - Test Loss')
    ax_loss.legend(fontsize=7, loc='upper right', ncol=2)
    ax_loss.grid(True, alpha=0.3)

    ax_lr = fig.add_subplot(gs[2, :])
    for i, (name, result) in enumerate(all_results.items()):
        h = result['history']
        epochs = range(1, len(h['lr_history']) + 1)
        ax_lr.plot(epochs, h['lr_history'], label=name, color=colors[i], linewidth=1.5)
    ax_lr.set_xlabel('Epoch')
    ax_lr.set_ylabel('Learning Rate')
    ax_lr.set_title('Learning Rate Schedule')
    ax_lr.legend(fontsize=7, loc='upper right', ncol=2)
    ax_lr.grid(True, alpha=0.3)
    ax_lr.set_yscale('log')

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_summary_bar(all_results, metric='best_acc', title="Best Accuracy Comparison",
                     save_path=None):
    """Bar chart comparing final metric across all optimizers."""
    names = list(all_results.keys())
    values = [all_results[n][metric] for n in names]

    sorted_idx = np.argsort(values)
    names = [names[i] for i in sorted_idx]
    values = [values[i] for i in sorted_idx]

    gen_colors = {
        'Gen1': '#e74c3c', 'Gen2': '#f39c12', 'Gen3': '#3498db',
        'Gen4': '#2ecc71', 'Gen5': '#9b59b6', 'SOTA': '#e91e63',
        'TF': '#607d8b'
    }
    colors = []
    for n in names:
        for prefix, c in gen_colors.items():
            if n.startswith(prefix):
                colors.append(c)
                break
        else:
            colors.append('#95a5a6')

    fig, ax = plt.subplots(figsize=(12, max(6, len(names) * 0.4)))
    bars = ax.barh(range(len(names)), values, color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel('Best Test Accuracy (%)' if metric == 'best_acc' else metric)
    ax.set_title(title)
    ax.grid(True, axis='x', alpha=0.3)

    for bar, val in zip(bars, values):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', fontsize=8)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


def plot_generation_evolution(all_results, save_path=None):
    """Radar/funnel chart showing evolution across 5 generations."""
    gen_metrics = {}
    for name, result in all_results.items():
        gen = None
        for g in ['Gen1', 'Gen2', 'Gen3', 'Gen4', 'Gen5']:
            if name.startswith(g):
                gen = g
                break
        if gen:
            if gen not in gen_metrics:
                gen_metrics[gen] = []
            gen_metrics[gen].append(result['best_acc'])

    gen_order = ['Gen1', 'Gen2', 'Gen3', 'Gen4', 'Gen5']
    gen_avg = [np.mean(gen_metrics.get(g, [0])) for g in gen_order]
    gen_max = [np.max(gen_metrics.get(g, [0])) for g in gen_order]
    gen_labels = ['Fixed LR', 'LR Schedule', 'Adaptive', 'Layer-wise', 'Layer×Time']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    x = range(len(gen_order))
    width = 0.35
    ax1.bar([i - width/2 for i in x], gen_avg, width, label='Average', color='#3498db', alpha=0.7)
    ax1.bar([i + width/2 for i in x], gen_max, width, label='Best', color='#e74c3c', alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(gen_labels, fontsize=10)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Learning Rate Evolution: 5 Generations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    improvements = [0] + [gen_max[i] - gen_max[i-1] for i in range(1, len(gen_max))]
    ax2.plot(gen_labels, improvements, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    ax2.fill_between(gen_labels, improvements, alpha=0.2, color='#2ecc71')
    ax2.set_ylabel('Accuracy Improvement (%)')
    ax2.set_title('Marginal Improvement per Generation')
    ax2.grid(True, alpha=0.3)
    for i, v in enumerate(improvements):
        ax2.annotate(f'+{v:.1f}%' if v > 0 else f'{v:.1f}%',
                     (gen_labels[i], v), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close(fig)


def generate_report(all_results, dataset_name="CIFAR-10"):
    """Generate text report of all results."""
    report = []
    report.append(f"\n{'='*80}")
    report.append(f"  Learning Rate Strategy Comparison Report — {dataset_name}")
    report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"{'='*80}\n")

    header = f"{'Optimizer':<25} {'Best Acc':>10} {'Final Acc':>10} {'Best Ep':>8} {'Time(s)':>10} {'Gen':>6}"
    report.append(header)
    report.append("-" * len(header))

    gen_map = {}
    for name, result in all_results.items():
        gen = 'SOTA'
        for g in ['Gen1', 'Gen2', 'Gen3', 'Gen4', 'Gen5', 'TF']:
            if name.startswith(g):
                gen = g
                break
        gen_map[name] = gen
        r = result
        report.append(f"{name:<25} {r['best_acc']:>9.2f}% {r['final_acc']:>9.2f}% "
                     f"{r['best_epoch']+1:>8} {r['total_time']:>9.1f}s {gen:>6}")

    best_name = max(all_results, key=lambda n: all_results[n]['best_acc'])
    best_acc = all_results[best_name]['best_acc']
    report.append(f"\n  Best: {best_name} = {best_acc:.2f}%")

    baseline_name = None
    for name in all_results:
        if 'Gen1' in name or 'TF_Fixed' in name:
            baseline_name = name
            break
    if baseline_name:
        baseline_acc = all_results[baseline_name]['best_acc']
        improvement = best_acc - baseline_acc
        report.append(f"  vs Baseline ({baseline_name}): +{improvement:.2f}% improvement")

    return "\n".join(report)


def save_results(all_results, filepath):
    serializable = {}
    for name, result in all_results.items():
        serializable[name] = {
            'best_acc': result['best_acc'],
            'best_epoch': result['best_epoch'],
            'final_acc': result['final_acc'],
            'total_time': result['total_time'],
            'history': result['history'],
            'config': {k: v for k, v in result['config'].items()
                      if not callable(v) and k != 'sam_base'},
        }
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)


# ============================================================================
# Main Runner
# ============================================================================

def run_full_benchmark(configs, dataset_name="CIFAR-10", prefix=""):
    device = get_device()
    print(f"\n{'='*60}")
    print(f"  Benchmark: {dataset_name}  |  Device: {device}")
    print(f"  {len(configs)} optimizers to compare")
    print(f"{'='*60}\n")

    all_results = {}
    for name, config in configs.items():
        print(f"\n>>> Running: {name}")
        try:
            result = run_experiment(config, device=device)
            all_results[name] = result
            print(f"    => Best: {result['best_acc']:.2f}% @ epoch {result['best_epoch']+1}")
        except Exception as e:
            print(f"    => FAILED: {e}")
            import traceback
            traceback.print_exc()

    report = generate_report(all_results, dataset_name)
    print(report)

    save_results(all_results, RESULTS_DIR / f"{prefix}results.json")

    try:
        plot_training_curves(all_results, f"LR Strategy — {dataset_name}",
                            FIGS_DIR / f"{prefix}training_curves.png")
        plot_summary_bar(all_results, title=f"Best Accuracy — {dataset_name}",
                        save_path=FIGS_DIR / f"{prefix}summary_bar.png")
        if any('Gen' in k for k in all_results):
            plot_generation_evolution(all_results,
                                     save_path=FIGS_DIR / f"{prefix}evolution.png")
    except Exception as e:
        print(f"Plotting error: {e}")

    return all_results


def run_quick_smoke_test():
    """Quick test with reduced epochs to verify everything works."""
    print("=== SMOKE TEST (3 epochs) ===")
    configs = get_cifar10_configs(epochs=3)
    subset = {k: configs[k] for k in list(configs.keys())[:5]}
    return run_full_benchmark(subset, "CIFAR-10 (smoke)", prefix="smoke_")


def run_full_cifar10():
    """Full CIFAR-10 benchmark."""
    configs = get_cifar10_configs(epochs=30)
    return run_full_benchmark(configs, "CIFAR-10", prefix="cifar10_")


def run_full_transfer():
    """Full transfer learning benchmark."""
    configs = get_transfer_configs(epochs=20)
    return run_full_benchmark(configs, "CIFAR-10 Transfer", prefix="transfer_")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Learning Rate Strategy Benchmark")
    parser.add_argument('--mode', type=str, default='smoke',
                       choices=['smoke', 'full', 'transfer', 'all'],
                       help='Benchmark mode')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    args = parser.parse_args()

    if args.mode == 'smoke':
        run_quick_smoke_test()
    elif args.mode == 'full':
        if args.epochs:
            configs = get_cifar10_configs(args.epochs)
            run_full_benchmark(configs, "CIFAR-10", prefix="cifar10_")
        else:
            run_full_cifar10()
    elif args.mode == 'transfer':
        run_full_transfer()
    elif args.mode == 'all':
        r1 = run_full_cifar10()
        r2 = run_full_transfer()