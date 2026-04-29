"""
Quick validation test for all optimizers.
Uses small MLP on synthetic data for fast verification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from optimizers import (
    FixedLRSGD, CosineAnnealingSGD, SGDRWithRestarts,
    AdamW, AdaBound, Adafactor,
    DiscriminativeLR, LARS, LAMB,
    RAdam, Lookahead, SAM, Sophia, Lion, ScheduleFree,
    Grokfast, DALS,
    STLRScheduler, build_discriminative_param_groups,
    build_discriminative_stlr_param_groups,
)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {DEVICE}")

def make_data(n=2000, d=32, nc=10):
    torch.manual_seed(42)
    X = torch.randn(n, d)
    y = torch.randint(0, nc, (n,))
    return torch.utils.data.TensorDataset(X, y)

def make_model(d=32, nc=10):
    return nn.Sequential(
        nn.Linear(d, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, nc),
    )

def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X).argmax(1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return 100.0 * correct / total

def train_loop(model, optimizer, loader, epochs=5, scheduler=None, is_sam=False):
    model.to(DEVICE)
    for ep in range(epochs):
        model.train()
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            if is_sam:
                def closure():
                    optimizer.zero_grad()
                    loss = F.cross_entropy(model(X), y)
                    loss.backward()
                    return loss
                optimizer.step(closure)
                continue
            elif isinstance(optimizer, DALS):
                optimizer.zero_grad()
                loss = F.cross_entropy(model(X), y)
                loss.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                loss = F.cross_entropy(model(X), y)
                loss.backward()
                optimizer.step()
            if scheduler is not None and not isinstance(scheduler, STLRScheduler):
                scheduler.step()
    if isinstance(scheduler, STLRScheduler):
        for _ in range(epochs * len(loader)):
            scheduler.step()
    return evaluate(model, loader)

dataset = make_data()
loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

results = {}

optimizers_config = {
    "Gen1_FixedSGD": lambda m: (FixedLRSGD(m.parameters(), lr=0.01, momentum=0.9), False, None),
    "Gen2_CosineSGD": lambda m: (CosineAnnealingSGD(m.parameters(), lr=0.01, T_max=50, momentum=0.9), False, None),
    "Gen2_SGDR": lambda m: (SGDRWithRestarts(m.parameters(), lr=0.01, T_0=10, momentum=0.9), False, None),
    "Gen3_Adam": lambda m: (torch.optim.Adam(m.parameters(), lr=1e-3), False, None),
    "Gen3_AdamW": lambda m: (AdamW(m.parameters(), lr=1e-3, weight_decay=0.01), False, None),
    "Gen3_AdaBound": lambda m: (AdaBound(m.parameters(), lr=1e-3, final_lr=0.01), False, None),
    "Gen3_Adafactor": lambda m: (Adafactor(m.parameters(), lr=5e-3), False, None),
    "Gen4_LARS": lambda m: (LARS(m.parameters(), lr=0.01, momentum=0.9), False, None),
    "Gen4_LAMB": lambda m: (LAMB(m.parameters(), lr=1e-3), False, None),
    "Gen4_Discriminative": lambda m: (
        DiscriminativeLR(build_discriminative_param_groups(m, base_lr=0.01), momentum=0.9), False, None),
    "Gen5_RAdam": lambda m: (RAdam(m.parameters(), lr=1e-3), False, None),
    "Gen5_Lion": lambda m: (Lion(m.parameters(), lr=1e-4), False, None),
    "Gen5_Lookahead": lambda m: (Lookahead(torch.optim.Adam(m.parameters(), lr=1e-3), k=5, alpha=0.5), False, None),
    "Gen5_SAM": lambda m: (SAM(m.parameters(), torch.optim.SGD, rho=0.05, lr=0.01, momentum=0.9), True, None),
    "Gen5_ScheduleFree": lambda m: (ScheduleFree(m.parameters(), lr=1e-3, momentum=0.9), False, None),
    "Gen5_Grokfast": lambda m: (Grokfast(m.parameters(), lr=1e-3, momentum=0.9, alpha=0.98), False, None),
    "Gen5_STLR": lambda m: _make_stlr(m),
    "SOTA_SAM_Discriminative": lambda m: (
        SAM(build_discriminative_param_groups(m, base_lr=0.01), torch.optim.SGD,
            rho=0.05, lr=0.01, momentum=0.9), True, None),
    "SOTA_DALS_Ours": lambda m: (DALS(m, lr=0.01, T_max=50), False, None),
}

def _make_stlr(model):
    pg = build_discriminative_stlr_param_groups(model, base_lr=0.01)
    opt = torch.optim.SGD(pg, lr=0.01, momentum=0.9)
    sched = STLRScheduler(opt, T=50 * len(loader), decay_factor=2.6)
    return opt, False, sched

for name, make_opt in optimizers_config.items():
    torch.manual_seed(42)
    model = make_model()
    opt, is_sam, sched = make_opt(model)
    try:
        acc = train_loop(model, opt, loader, epochs=5, scheduler=sched, is_sam=is_sam)
        results[name] = acc
        print(f"  {name:30s} => Acc: {acc:.2f}%  OK")
    except Exception as e:
        results[name] = -1
        print(f"  {name:30s} => FAILED: {e}")

print("\n" + "="*60)
print("  RESULTS SUMMARY")
print("="*60)
for name, acc in sorted(results.items(), key=lambda x: -x[1]):
    status = "OK" if acc >= 0 else "FAILED"
    gen = name.split("_")[0]
    print(f"  [{gen:5s}] {name:30s} {acc:7.2f}%  {status}")

print(f"\n  Best: {max(results, key=results.get)} = {max(results.values()):.2f}%")
print("  All optimizers validated!" if all(v > 0 for v in results.values()) else "  SOME OPTIMIZERS FAILED!")