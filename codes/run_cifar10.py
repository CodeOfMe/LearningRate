"""
CIFAR-10 benchmark: 18 strategies across 5 generations.
MPS/CUDA acceleration, multi-threaded DataLoader, mixed precision.
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
import os
from torch.utils.data import DataLoader, TensorDataset

from optimizers import (
    FixedLRSGD, CosineAnnealingSGD, SGDRWithRestarts,
    AdamW, AdaBound,
    DiscriminativeLR, LARS,
    RAdam, Lookahead, SAM, Lion,
    Grokfast, DALS, DALSFast, DALSAcc,
    STLRScheduler, build_discriminative_param_groups,
    build_discriminative_stlr_param_groups,
)

# ========== Hardware Setup ==========
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

NUM_WORKERS = 4 if DEVICE == 'cpu' else 0
PIN_MEMORY = DEVICE != 'cpu'
USE_AMP = DEVICE != 'cpu'

torch.set_num_threads(4)
print(f"Device: {DEVICE} | Workers: {NUM_WORKERS} | AMP: {USE_AMP} | Threads: {torch.get_num_threads()}")

FIGS_DIR = Path(__file__).parent / "figs"
RESULTS_DIR = Path(__file__).parent / "results"
FIGS_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR = Path(__file__).parent / "data" / "cifar-10-batches-py"

EPOCHS = 50
BATCH_SIZE = 128

def seed_all(s=42):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def unpickle(file):
    with open(file, 'rb') as fo:
        d = pickle.load(fo, encoding='bytes')
    return d

def load_cifar10(data_dir=DATA_DIR):
    train_data, train_labels = [], []
    for i in range(1, 6):
        batch = unpickle(data_dir / f'data_batch_{i}')
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32)
    train_labels = np.array(train_labels, dtype=np.int64)
    test_batch = unpickle(data_dir / 'test_batch')
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32)
    test_labels = np.array(test_batch[b'labels'], dtype=np.int64)
    return train_data, train_labels, test_data, test_labels

class SmallConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def create_optimizer(name, model, lr, total_steps):
    is_sam = False
    scheduler = None
    steps_per_epoch = 50000 // BATCH_SIZE

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
                       T_0=steps_per_epoch * 10, T_mult=2)
    else:
        raise ValueError(f"Unknown strategy: {name}")
    return opt, scheduler, is_sam

def train_and_evaluate(name, epochs=EPOCHS):
    seed_all(42)
    train_data, train_labels, test_data, test_labels = load_cifar10()

    mean = train_data.mean(axis=(0, 2, 3)) / 255.0
    std = train_data.std(axis=(0, 2, 3)) / 255.0

    X_train = torch.tensor(train_data, dtype=torch.float32) / 255.0
    for c in range(3):
        X_train[:, c] = (X_train[:, c] - mean[c]) / (std[c] + 1e-8)
    y_train = torch.tensor(train_labels, dtype=torch.long)

    X_test = torch.tensor(test_data, dtype=torch.float32) / 255.0
    for c in range(3):
        X_test[:, c] = (X_test[:, c] - mean[c]) / (std[c] + 1e-8)
    y_test = torch.tensor(test_labels, dtype=torch.long)

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
                               persistent_workers=NUM_WORKERS > 0)

    model = SmallConvNet().to(DEVICE)
    total_steps = epochs * len(train_loader)
    opt, scheduler, is_sam = create_optimizer(name, model, 0.03, total_steps)

    scaler = torch.amp.GradScaler('cuda', enabled=(USE_AMP and DEVICE == 'cuda'))
    amp_device_type = 'mps' if DEVICE == 'mps' else 'cuda'

    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'lr': []}
    milestones = {}
    best_acc = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=PIN_MEMORY), yb.to(DEVICE, non_blocking=PIN_MEMORY)

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
                with torch.autocast(amp_device_type, enabled=USE_AMP):
                    out = model(xb)
                    loss = F.cross_entropy(out, yb)
                if USE_AMP and DEVICE == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.update_phase(loss.item())
                    opt.step()
                opt.zero_grad(set_to_none=True)
            else:
                opt.zero_grad(set_to_none=True)
                with torch.autocast(amp_device_type, enabled=USE_AMP):
                    out = model(xb)
                    loss = F.cross_entropy(out, yb)
                if USE_AMP and DEVICE == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    opt.step()

            if scheduler is not None and not isinstance(scheduler, STLRScheduler):
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

        if isinstance(scheduler, STLRScheduler):
            for _ in range(len(train_loader)):
                scheduler.step()

        # Eval
        model.eval()
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=512,
                                      num_workers=0, pin_memory=False)
            correct = total = 0
            for xb, yb in test_loader:
                xb = xb.to(DEVICE, non_blocking=False)
                pred = model(xb).argmax(1).cpu()
                correct += (pred == yb).sum().item()
                total += len(yb)
            test_acc = 100.0 * correct / total

            train_loader_eval = DataLoader(TensorDataset(X_train, y_train), batch_size=512,
                                             num_workers=0, pin_memory=False)
            correct = total = 0
            for xb, yb in train_loader_eval:
                xb = xb.to(DEVICE, non_blocking=False)
                pred = model(xb).argmax(1).cpu()
                correct += (pred == yb).sum().item()
                total += len(yb)
            train_acc = 100.0 * correct / total

        lr_now = opt.param_groups[0].get('lr', 0.03)
        t_now = time.time() - t0

        for thr in [50, 60, 70, 75, 80]:
            if test_acc >= thr and str(thr) not in milestones:
                milestones[str(thr)] = [ep, round(t_now, 2)]

        best_acc = max(best_acc, test_acc)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(epoch_loss / max(n_batches, 1))
        history['lr'].append(lr_now)

        if ep % 10 == 0 or ep == 1:
            print(f"  Ep {ep:3d}: train={train_acc:.1f}%  test={test_acc:.1f}%  lr={lr_now:.6f}  t={t_now:.1f}s")

    total_time = time.time() - t0
    return history, best_acc, milestones, total_time

if __name__ == '__main__':
    strategies = [
        'Gen1_FixedSGD', 'Gen2_CosineSGD', 'Gen2_SGDR',
        'Gen3_Adam', 'Gen3_AdamW', 'Gen3_AdaBound',
        'Gen4_LARS', 'Gen4_Discriminative',
        'Gen5_RAdam', 'Gen5_Lion', 'Gen5_Lookahead', 'Gen5_SAM', 'Gen5_Grokfast',
        'Gen5_STLR', 'SOTA_SAM_Discrim', 'SOTA_DALS',
        'SOTA_DALS_Fast', 'SOTA_DALS_Acc',
    ]

    print(f"\n{'='*90}")
    print(f"  CIFAR-10 BENCHMARK — {len(strategies)} strategies, {EPOCHS} epochs, device={DEVICE}")
    print(f"{'='*90}\n")

    all_results = {}
    all_histories = {}

    for name in strategies:
        print(f"\n  [{name}]", flush=True)
        t0 = time.time()
        try:
            history, best_acc, milestones, total_time = train_and_evaluate(name)
            all_histories[name] = history
            all_results[name] = {
                'best_acc': best_acc,
                'final_acc': history['test_acc'][-1],
                'milestones': milestones,
                'total_time': total_time,
            }
            ms = "  ".join([f"{k}%:{v[0]}ep" for k, v in sorted(milestones.items())])
            print(f"  => Best={best_acc:.1f}%  Final={history['test_acc'][-1]:.1f}%  {ms}  ({total_time:.1f}s)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  => FAILED: {e}")

    print(f"\n{'='*110}")
    print(f"  CIFAR-10 FINAL RESULTS")
    print(f"{'='*110}")
    for name in sorted(all_results, key=lambda n: all_results[n]['best_acc'], reverse=True):
        r = all_results[name]
        ms = "  ".join([f"{k}%:{v[0]}ep" for k, v in sorted(r['milestones'].items())])
        print(f"  {name:<25s} Best={r['best_acc']:>5.1f}%  Final={r['final_acc']:>5.1f}%  {ms}  ({r['total_time']:.1f}s)")

    serializable = {}
    for name in all_results:
        r = all_results[name]
        serializable[name] = {
            'best_acc': r['best_acc'],
            'final_acc': r['final_acc'],
            'total_time': r['total_time'],
            'milestones': r['milestones'],
            'history': {k: [float(x) for x in vs] for k, vs in all_histories[name].items()},
        }
    with open(RESULTS_DIR / 'cifar10_results.json', 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'cifar10_results.json'}")