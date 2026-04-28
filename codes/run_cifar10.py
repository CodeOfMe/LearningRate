"""
CIFAR-10 benchmark: 18 strategies across 5 generations.
Uses a small ConvNet trained from scratch on CIFAR-10.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import json
import time
import pickle
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

DATA_DIR = Path(__file__).parent / "data" / "cifar-10-batches-py"

def seed_all(s=42):
    torch.manual_seed(s)
    np.random.seed(s)

def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

def load_cifar10(data_dir=DATA_DIR):
    """Load CIFAR-10 from local pickle files."""
    train_data = []
    train_labels = []
    for i in range(1, 6):
        batch = unpickle(data_dir / f'data_batch_{i}')
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    train_labels = np.array(train_labels)

    test_batch = unpickle(data_dir / 'test_batch')
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_labels = np.array(test_batch[b'labels'])

    return train_data, train_labels, test_data, test_labels

class SmallConvNet(nn.Module):
    """Small ConvNet for CIFAR-10: 3 conv layers + 2 FC layers."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 32->16
        x = self.pool(F.relu(self.conv2(x)))  # 16->8
        x = self.pool(F.relu(self.conv3(x)))  # 8->4
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def evaluate(model, X, y, bs=512, device='cpu'):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i+bs].to(device)
            yb = y[i:i+bs].to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += len(yb)
    return 100.0 * correct / total

EPOCHS = 50
BATCH_SIZE = 128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_and_evaluate(name, epochs=EPOCHS, base_lr=0.03):
    seed_all(42)
    train_data, train_labels, test_data, test_labels = load_cifar10()

    # Normalize
    train_mean = train_data.mean() / 255.0
    train_std = train_data.std() / 255.0
    X_train = ((train_data / 255.0) - train_mean) / (train_std + 1e-8)
    X_test = ((test_data / 255.0) - train_mean) / (train_std + 1e-8)

    X_train = torch.tensor(X_train, dtype=torch.float32).permute(0, 3, 1, 2)
    y_train = torch.tensor(train_labels, dtype=torch.long)
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    model = SmallConvNet().to(DEVICE)
    N_TRAIN = len(X_train)
    STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE
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
            xb = X_train[idx].to(DEVICE)
            yb = y_train[idx].to(DEVICE)

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

        train_acc = evaluate(model, X_train, y_train, bs=512, device=DEVICE)
        test_acc = evaluate(model, X_test, y_test, bs=512, device=DEVICE)
        lr_now = opt.param_groups[0].get('lr', base_lr)
        t_now = time.time() - t0

        for thr in [50, 60, 70, 75, 80]:
            if test_acc >= thr and thr not in milestones:
                milestones[thr] = (ep, round(t_now, 2))

        best_acc = max(best_acc, test_acc)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(epoch_loss / STEPS_PER_EPOCH)
        history['lr'].append(lr_now)

        if ep % 10 == 0 or ep == 1:
            print(f"  Epoch {ep}: train={train_acc:.1f}%, test={test_acc:.1f}%, lr={lr_now:.6f}, time={t_now:.1f}s")

    total_time = time.time() - t0
    return history, best_acc, milestones, total_time

def create_optimizer(name, model, lr, total_steps):
    is_sam = False
    scheduler = None

    if name == 'Gen1_FixedSGD':
        opt = FixedLRSGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    elif name == 'Gen2_CosineSGD':
        opt = CosineAnnealingSGD(model.parameters(), lr=0.03, T_max=total_steps, momentum=0.9, weight_decay=5e-4)

    elif name == 'Gen2_SGDR':
        opt = SGDRWithRestarts(model.parameters(), lr=0.05, T_0=10, T_mult=2, momentum=0.9, weight_decay=5e-4)

    elif name == 'Gen3_Adam':
        opt = optim.Adam(model.parameters(), lr=3e-4)

    elif name == 'Gen3_AdamW':
        opt = AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    elif name == 'Gen3_AdaBound':
        opt = AdaBound(model.parameters(), lr=3e-4, final_lr=0.01)

    elif name == 'Gen4_LARS':
        opt = LARS(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4, trust_coef=0.02)

    elif name == 'Gen4_Discriminative':
        pg = build_discriminative_param_groups(model, base_lr=0.05, decay_factor=2.6)
        opt = DiscriminativeLR(pg, momentum=0.9, weight_decay=5e-4)

    elif name == 'Gen5_RAdam':
        opt = RAdam(model.parameters(), lr=3e-4)

    elif name == 'Gen5_Lion':
        opt = Lion(model.parameters(), lr=1e-4, weight_decay=0.01)

    elif name == 'Gen5_Lookahead':
        opt = Lookahead(AdamW(model.parameters(), lr=3e-4, weight_decay=0.01), k=5, alpha=0.5)

    elif name == 'Gen5_SAM':
        opt = SAM(model.parameters(), optim.SGD, rho=0.05, adaptive=True, lr=0.03, momentum=0.9, weight_decay=5e-4)
        is_sam = True

    elif name == 'Gen5_Grokfast':
        opt = Grokfast(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4, alpha=0.98)

    elif name == 'Gen5_STLR':
        pg = build_discriminative_stlr_param_groups(model, base_lr=0.05, decay_factor=2.6)
        opt = optim.SGD(pg, momentum=0.9, weight_decay=5e-4)
        scheduler = STLRScheduler(opt, T=total_steps, decay_factor=2.6)

    elif name == 'SOTA_SAM_Discrim':
        pg = build_discriminative_param_groups(model, base_lr=0.05, decay_factor=2.6)
        opt = SAM(pg, optim.SGD, rho=0.05, adaptive=True, lr=0.05, momentum=0.9, weight_decay=5e-4)
        is_sam = True

    elif name == 'SOTA_DALS':
        opt = DALS(model, lr=0.03, momentum=0.9, weight_decay=5e-4,
                   trust_coef=0.02, grokfast_alpha=0.6,
                   warmup_frac=0.05, T_max=total_steps)

    elif name == 'SOTA_DALS_Fast':
        opt = DALSFast(model, lr=0.05, momentum=0.85, weight_decay=5e-4,
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

    print(f"\n{'='*100}")
    print(f"  CIFAR-10 BENCHMARK — {len(strategies)} strategies, {EPOCHS} epochs, device={DEVICE}")
    print(f"{'='*100}\n")

    all_results = {}
    all_histories = {}

    for name in strategies:
        print(f"\n  Training: {name}...", flush=True)
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
            ms_str = "  ".join([f"{thr}%:{v[0]}ep" for thr, v in sorted(milestones.items())])
            print(f"  => Best={best_acc:.1f}%  Final={history['test_acc'][-1]:.1f}%  {ms_str}  ({total_time:.1f}s)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  => FAILED: {e}")

    # Summary
    print(f"\n{'='*110}")
    print(f"  CIFAR-10 RESULTS SUMMARY")
    print(f"{'='*110}")
    sorted_strategies = sorted(all_results.keys(), key=lambda n: all_results.get(n, {}).get('best_acc', 0), reverse=True)
    for name in sorted_strategies:
        if name not in all_results:
            continue
        r = all_results[name]
        gen = name.split('_')[0]
        ms_str = "  ".join([f"{thr}%:{v[0]}ep" for thr, v in sorted(r['milestones'].items())])
        print(f"  {name:<25s} Best={r['best_acc']:>5.1f}%  Final={r['final_acc']:>5.1f}%  {ms_str}  ({r['total_time']:.1f}s)")

    # Save results
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
    with open(RESULTS_DIR / 'cifar10_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR / 'cifar10_results.json'}")
