"""
Test_All.py — Unified benchmark for 18 learning rate strategies across 5 datasets.

Datasets:
  1. Synthetic (4-layer MLP, 80 epochs, from scratch)
  2. CIFAR-10 (SmallConvNet, 50 epochs, from scratch)
  3. RTE       (DistilBERT, 10 epochs, fine-tune) — GLUE NLP, textual entailment
  4. TREC-6    (DistilBERT, 10 epochs, fine-tune) — ULMFiT benchmark, 6-class QA
  5. IMDb      (DistilBERT, 5 epochs, fine-tune)  — ULMFiT benchmark, sentiment

Usage:
  python Test_All.py                          # Run all benchmarks
  python Test_All.py --dataset synthetic       # Run only synthetic
  python Test_All.py --dataset cifar10         # Run only CIFAR-10
  python Test_All.py --dataset rte             # Run only RTE
  python Test_All.py --dataset trec6          # Run only TREC-6
  python Test_All.py --dataset imdb            # Run only IMDb
  python Test_All.py --dataset nlp             # Run all 3 NLP benchmarks
  python Test_All.py --list                    # List strategies and datasets
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import time
import argparse
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

# ============================================================================
# Hardware Setup
# ============================================================================
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

NUM_WORKERS = 4 if DEVICE == 'cuda' else 0
PIN_MEMORY = DEVICE == 'cuda'

torch.set_num_threads(4 if DEVICE != 'cuda' else 8)
print(f"{'='*70}")
print(f"  Device: {DEVICE} | Workers: {NUM_WORKERS} | Threads: {torch.get_num_threads()}")
print(f"{'='*70}")

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR = Path(__file__).parent / "data"
RESULTS_DIR.mkdir(exist_ok=True)

STRATEGIES = [
    'Gen1_FixedSGD', 'Gen2_CosineSGD', 'Gen2_SGDR',
    'Gen3_Adam', 'Gen3_AdamW', 'Gen3_AdaBound',
    'Gen4_LARS', 'Gen4_Discriminative',
    'Gen5_RAdam', 'Gen5_Lion', 'Gen5_Lookahead', 'Gen5_SAM', 'Gen5_Grokfast',
    'Gen5_STLR', 'SOTA_SAM_Discrim', 'SOTA_DALS',
    'SOTA_DALS_Fast', 'SOTA_DALS_Acc',
]


def seed_all(s=42):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# ============================================================================
# NLP Shared Components
# ============================================================================
class DistilBERTWrapper(nn.Module):
    """6-group wrapper for DALS depth-aware processing.
    Group 0 (lowest LR): embeddings
    Group 1-3: transformer layer pairs (0-1, 2-3, 4-5)
    Group 4: pre_classifier
    Group 5 (highest LR): classifier
    """
    def __init__(self, model):
        super().__init__()
        self.embeddings = model.distilbert.embeddings
        self.transformer_01 = nn.Sequential(*[model.distilbert.transformer.layer[0], model.distilbert.transformer.layer[1]])
        self.transformer_23 = nn.Sequential(*[model.distilbert.transformer.layer[2], model.distilbert.transformer.layer[3]])
        self.transformer_45 = nn.Sequential(*[model.distilbert.transformer.layer[4], model.distilbert.transformer.layer[5]])
        self.pre_classifier = model.pre_classifier
        self.classifier = model.classifier


def _get_nlp_layer_groups(model):
    groups = [
        {'name': 'embeddings', 'modules': [model.distilbert.embeddings]},
        {'name': 'transformer_01', 'modules': [model.distilbert.transformer.layer[0], model.distilbert.transformer.layer[1]]},
        {'name': 'transformer_23', 'modules': [model.distilbert.transformer.layer[2], model.distilbert.transformer.layer[3]]},
        {'name': 'transformer_45', 'modules': [model.distilbert.transformer.layer[4], model.distilbert.transformer.layer[5]]},
        {'name': 'pre_classifier', 'modules': [model.pre_classifier]},
        {'name': 'classifier', 'modules': [model.classifier]},
    ]
    param_groups = []
    for g in groups:
        params = []
        for m in g['modules']:
            params.extend(list(m.parameters()))
        param_groups.append({'params': params, 'name': g['name']})
    return param_groups, len(groups)


def _build_nlp_discriminative(model, base_lr, decay_factor=2.6):
    pg, num = _get_nlp_layer_groups(model)
    for i in range(num):
        lr = base_lr / (decay_factor ** (num - 1 - i))
        pg[i]['lr'] = lr
    return pg


def _build_nlp_stlr(model, base_lr, decay_factor=2.6):
    pg, num = _get_nlp_layer_groups(model)
    for i in range(num):
        lr = base_lr / (decay_factor ** (num - 1 - i))
        pg[i]['lr'] = lr
        pg[i]['eta_max'] = lr
    return pg


def _create_nlp_optimizer(name, model, total_steps, steps_per_epoch,
                           lr_adam=5e-5, lr_sgd=5e-4):
    is_sam = False
    scheduler = None

    if name == 'Gen1_FixedSGD':
        opt = FixedLRSGD(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=0.01)
    elif name == 'Gen2_CosineSGD':
        opt = CosineAnnealingSGD(model.parameters(), lr=lr_sgd, T_max=total_steps, momentum=0.9, weight_decay=0.01)
    elif name == 'Gen2_SGDR':
        opt = SGDRWithRestarts(model.parameters(), lr=lr_sgd * 2, T_0=max(steps_per_epoch, 50), T_mult=2, momentum=0.9, weight_decay=0.01)
    elif name == 'Gen3_Adam':
        opt = torch.optim.Adam(model.parameters(), lr=lr_adam)
    elif name == 'Gen3_AdamW':
        opt = AdamW(model.parameters(), lr=lr_adam, weight_decay=0.01)
    elif name == 'Gen3_AdaBound':
        opt = AdaBound(model.parameters(), lr=lr_adam, final_lr=lr_sgd)
    elif name == 'Gen4_LARS':
        opt = LARS(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=0.01, trust_coef=0.02)
    elif name == 'Gen4_Discriminative':
        pg = _build_nlp_discriminative(model, base_lr=lr_sgd)
        opt = DiscriminativeLR(pg, momentum=0.9, weight_decay=0.01)
    elif name == 'Gen5_RAdam':
        opt = RAdam(model.parameters(), lr=lr_adam)
    elif name == 'Gen5_Lion':
        opt = Lion(model.parameters(), lr=lr_adam * 0.5, weight_decay=0.01)
    elif name == 'Gen5_Lookahead':
        opt = Lookahead(AdamW(model.parameters(), lr=lr_adam, weight_decay=0.01), k=5, alpha=0.5)
    elif name == 'Gen5_SAM':
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, adaptive=True, lr=lr_sgd, momentum=0.9, weight_decay=0.01)
        is_sam = True
    elif name == 'Gen5_Grokfast':
        opt = Grokfast(model.parameters(), lr=lr_sgd, momentum=0.9, weight_decay=0.01, alpha=0.98)
    elif name == 'Gen5_STLR':
        pg = _build_nlp_stlr(model, base_lr=lr_sgd)
        opt = torch.optim.SGD(pg, momentum=0.9, weight_decay=0.01)
        scheduler = STLRScheduler(opt, T=total_steps, decay_factor=2.6)
    elif name == 'SOTA_SAM_Discrim':
        pg = _build_nlp_discriminative(model, base_lr=lr_sgd)
        opt = SAM(pg, torch.optim.SGD, rho=0.05, adaptive=True, lr=lr_sgd, momentum=0.9, weight_decay=0.01)
        is_sam = True
    elif name == 'SOTA_DALS':
        wrapper = DistilBERTWrapper(model)
        opt = DALS(wrapper, lr=lr_sgd, momentum=0.9, weight_decay=0.01,
                   trust_coef=0.02, grokfast_alpha=0.6, warmup_frac=0.1, T_max=total_steps)
    elif name == 'SOTA_DALS_Fast':
        wrapper = DistilBERTWrapper(model)
        opt = DALSFast(wrapper, lr=lr_sgd * 2, momentum=0.85, weight_decay=0.01,
                       trust_coef=0.02, grokfast_alpha=0.6, warmup_frac=0.05, T_max=total_steps)
    elif name == 'SOTA_DALS_Acc':
        wrapper = DistilBERTWrapper(model)
        opt = DALSAcc(wrapper, lr=lr_sgd, momentum=0.9, weight_decay=0.01,
                      trust_coef=0.02, grokfast_alpha=0.7,
                      T_0=max(steps_per_epoch * 3, 50), T_mult=2)
    else:
        raise ValueError(f"Unknown strategy: {name}")
    return opt, scheduler, is_sam


def _run_nlp_benchmark(strategy_name, dataset_name, epochs, train_loader, val_loader,
                        num_labels=2, lr_adam=5e-5, lr_sgd=5e-4,
                        milestones_thresholds=None):
    from transformers import DistilBertForSequenceClassification

    seed_all(42)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=num_labels
    ).to(DEVICE)

    # Freeze embeddings for more stable fine-tuning (optional — off by default)
    # for param in model.distilbert.embeddings.parameters():
    #     param.requires_grad = False

    total_steps = epochs * len(train_loader)
    steps_per_epoch = len(train_loader)
    opt, scheduler, is_sam = _create_nlp_optimizer(
        strategy_name, model, total_steps, steps_per_epoch, lr_adam, lr_sgd
    )
    is_dals = isinstance(opt, (DALS, DALSFast, DALSAcc))

    if milestones_thresholds is None:
        milestones_thresholds = [60, 65, 70, 75, 80]

    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'lr': []}
    milestones = {}
    best_acc = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0, 0
        correct_train, total_train = 0, 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            if is_sam:
                opt.zero_grad()
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
                opt.first_step(zero_grad=True)
                with torch.enable_grad():
                    out2 = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss2 = out2.loss
                    loss2.backward()
                opt.second_step()
            elif is_dals:
                if isinstance(opt, Lookahead):
                    opt.zero_grad()
                else:
                    opt.zero_grad(set_to_none=True)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
                opt.update_phase(loss.item())
                opt.step()
            elif isinstance(opt, Lookahead):
                opt.zero_grad()
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
                opt.step()

            if scheduler is not None and not isinstance(scheduler, STLRScheduler):
                scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1

            with torch.no_grad():
                preds = out.logits.argmax(dim=-1)
                correct_train += (preds == labels).sum().item()
                total_train += len(labels)

        if isinstance(scheduler, STLRScheduler):
            for _ in range(len(train_loader)):
                scheduler.step()

        # Evaluate
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['label'].to(DEVICE)
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = out.logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(labels)
        test_acc = 100.0 * correct / total
        train_acc = 100.0 * correct_train / max(total_train, 1)

        lr_now = opt.param_groups[0].get('lr', lr_adam if 'Adam' in strategy_name or 'RAdam' in strategy_name or 'Lion' in strategy_name or 'Lookahead' in strategy_name else lr_sgd)
        t_now = time.time() - t0

        for thr in milestones_thresholds:
            if test_acc >= thr and str(thr) not in milestones:
                milestones[str(thr)] = [ep, round(t_now, 2)]

        best_acc = max(best_acc, test_acc)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(epoch_loss / max(n_batches, 1))
        history['lr'].append(lr_now)

        print(f"    Ep {ep:2d}/{epochs}: train={train_acc:.1f}%  val={test_acc:.1f}%  lr={lr_now:.2e}  t={t_now:.1f}s", flush=True)

    total_time = time.time() - t0
    return history, best_acc, milestones, total_time


# ============================================================================
# 1. Synthetic Benchmark
# ============================================================================
class SyntheticMLP(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=256, output_dim=10):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim)
        self.layer4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


def run_synthetic(strategy_name, epochs=80):
    seed_all(42)

    torch.manual_seed(42)
    n_train, n_test = 10000, 2000
    X_train = torch.randn(n_train, 20)
    y_train = (X_train[:, :5].sum(dim=1) > 0).long()
    X_test = torch.randn(n_test, 20)
    y_test = (X_test[:, :5].sum(dim=1) > 0).long()

    torch.manual_seed(43)
    X2 = torch.randn(n_train, 20) * 0.5
    y2 = ((X2[:, 0] * X2[:, 1] + X2[:, 2] * X2[:, 3]) > 0).long()
    X_train = torch.cat([X_train, X2])
    y_train = torch.cat([y_train, y2])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True)
    X_test, y_test = X_test.to(DEVICE), y_test.to(DEVICE)

    model = SyntheticMLP().to(DEVICE)
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    opt, scheduler, is_sam = _create_synthetic_optimizer(strategy_name, model, total_steps, steps_per_epoch)

    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'lr': []}
    milestones = {}
    best_acc = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            if is_sam:
                opt.zero_grad()
                out = model(xb)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                opt.first_step(zero_grad=True)
                with torch.enable_grad():
                    out2 = model(xb)
                    loss2 = F.cross_entropy(out2, yb)
                    loss2.backward()
                opt.second_step()
            elif isinstance(opt, (DALS, DALSFast, DALSAcc)):
                out = model(xb)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                opt.update_phase(loss.item())
                opt.step()
                opt.zero_grad(set_to_none=True)
            elif isinstance(opt, Lookahead):
                opt.zero_grad()
                out = model(xb)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                opt.step()
            if scheduler is not None and not isinstance(scheduler, STLRScheduler):
                scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        if isinstance(scheduler, STLRScheduler):
            for _ in range(len(train_loader)):
                scheduler.step()

        model.eval()
        with torch.no_grad():
            preds = model(X_test).argmax(1)
            test_acc = 100.0 * (preds == y_test).float().mean().item()
            train_preds = model(X_train.to(DEVICE)).argmax(1)
            train_acc = 100.0 * (train_preds == y_train.to(DEVICE)).float().mean().item()

        lr_now = opt.param_groups[0].get('lr', 0.03)
        t_now = time.time() - t0

        for thr in [60, 70, 80, 85, 90]:
            if test_acc >= thr and str(thr) not in milestones:
                milestones[str(thr)] = [ep, round(t_now, 2)]

        best_acc = max(best_acc, test_acc)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(epoch_loss / max(n_batches, 1))
        history['lr'].append(lr_now)

        if ep % 10 == 0 or ep == 1:
            print(f"    Ep {ep:3d}: train={train_acc:.1f}%  test={test_acc:.1f}%  t={t_now:.1f}s", flush=True)

    total_time = time.time() - t0
    return history, best_acc, milestones, total_time


def _create_synthetic_optimizer(name, model, total_steps, steps_per_epoch):
    is_sam = False
    scheduler = None

    if name == 'Gen1_FixedSGD':
        opt = FixedLRSGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif name == 'Gen2_CosineSGD':
        opt = CosineAnnealingSGD(model.parameters(), lr=0.03, T_max=total_steps, momentum=0.9, weight_decay=5e-4)
    elif name == 'Gen2_SGDR':
        opt = SGDRWithRestarts(model.parameters(), lr=0.05, T_0=10, T_mult=2, momentum=0.9, weight_decay=5e-4)
    elif name == 'Gen3_Adam':
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
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
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, adaptive=True, lr=0.03, momentum=0.9, weight_decay=5e-4)
        is_sam = True
    elif name == 'Gen5_Grokfast':
        opt = Grokfast(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4, alpha=0.98)
    elif name == 'Gen5_STLR':
        pg = build_discriminative_stlr_param_groups(model, base_lr=0.05, decay_factor=2.6)
        opt = torch.optim.SGD(pg, momentum=0.9, weight_decay=5e-4)
        scheduler = STLRScheduler(opt, T=total_steps, decay_factor=2.6)
    elif name == 'SOTA_SAM_Discrim':
        pg = build_discriminative_param_groups(model, base_lr=0.05, decay_factor=2.6)
        opt = SAM(pg, torch.optim.SGD, rho=0.05, adaptive=True, lr=0.05, momentum=0.9, weight_decay=5e-4)
        is_sam = True
    elif name == 'SOTA_DALS':
        opt = DALS(model, lr=0.03, momentum=0.9, weight_decay=5e-4,
                   trust_coef=0.02, grokfast_alpha=0.6, warmup_frac=0.05, T_max=total_steps)
    elif name == 'SOTA_DALS_Fast':
        opt = DALSFast(model, lr=0.05, momentum=0.85, weight_decay=5e-4,
                       trust_coef=0.02, grokfast_alpha=0.6, warmup_frac=0.02, T_max=total_steps)
    elif name == 'SOTA_DALS_Acc':
        opt = DALSAcc(model, lr=0.03, momentum=0.9, weight_decay=5e-4,
                      trust_coef=0.02, grokfast_alpha=0.7,
                      T_0=steps_per_epoch * 10, T_mult=2)
    else:
        raise ValueError(f"Unknown strategy: {name}")
    return opt, scheduler, is_sam


# ============================================================================
# 2. CIFAR-10 Benchmark
# ============================================================================
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


def run_cifar10(strategy_name, epochs=50):
    seed_all(42)

    data_path = DATA_DIR / 'cifar-10-batches-py'
    if not data_path.exists():
        import urllib.request
        import tarfile
        print("  Downloading CIFAR-10...")
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        tar_path = DATA_DIR / 'cifar-10-python.tar.gz'
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not tar_path.exists():
            proxy = urllib.request.ProxyHandler({
                'http': os.environ.get('http_proxy', ''),
                'https': os.environ.get('https_proxy', ''),
            })
            opener = urllib.request.build_opener(proxy)
            data = opener.open(url).read()
            with open(tar_path, 'wb') as f:
                f.write(data)
        print("  Extracting CIFAR-10...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(DATA_DIR)

    def unpickle(file):
        with open(file, 'rb') as fo:
            d = pickle.load(fo, encoding='bytes')
        return d

    train_data, train_labels = [], []
    for i in range(1, 6):
        batch = unpickle(data_path / f'data_batch_{i}')
        train_data.append(batch[b'data'])
        train_labels.extend(batch[b'labels'])
    train_data = np.vstack(train_data).reshape(-1, 3, 32, 32)
    train_labels = np.array(train_labels, dtype=np.int64)
    test_batch = unpickle(data_path / 'test_batch')
    test_data = test_batch[b'data'].reshape(-1, 3, 32, 32)
    test_labels = np.array(test_batch[b'labels'], dtype=np.int64)

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

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=128, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    model = SmallConvNet().to(DEVICE)
    total_steps = epochs * len(train_loader)
    steps_per_epoch = len(train_loader)
    opt, scheduler, is_sam = _create_cifar10_optimizer(strategy_name, model, total_steps, steps_per_epoch)

    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'lr': []}
    milestones = {}
    best_acc = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE, non_blocking=PIN_MEMORY), yb.to(DEVICE, non_blocking=PIN_MEMORY)
            if is_sam:
                opt.zero_grad()
                out = model(xb)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                opt.first_step(zero_grad=True)
                with torch.enable_grad():
                    out2 = model(xb)
                    loss2 = F.cross_entropy(out2, yb)
                    loss2.backward()
                opt.second_step()
            elif isinstance(opt, (DALS, DALSFast, DALSAcc)):
                out = model(xb)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                opt.update_phase(loss.item())
                opt.step()
                opt.zero_grad(set_to_none=True)
            elif isinstance(opt, Lookahead):
                opt.zero_grad()
                out = model(xb)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                opt.step()
            else:
                opt.zero_grad(set_to_none=True)
                out = model(xb)
                loss = F.cross_entropy(out, yb)
                loss.backward()
                opt.step()
            if scheduler is not None and not isinstance(scheduler, STLRScheduler):
                scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        if isinstance(scheduler, STLRScheduler):
            for _ in range(len(train_loader)):
                scheduler.step()

        model.eval()
        with torch.no_grad():
            test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=512, num_workers=0)
            correct = total = 0
            for xb, yb in test_loader:
                xb = xb.to(DEVICE)
                pred = model(xb).argmax(1).cpu()
                correct += (pred == yb).sum().item()
                total += len(yb)
            test_acc = 100.0 * correct / total

            train_eval_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=512, num_workers=0)
            correct = total = 0
            for xb, yb in train_eval_loader:
                xb = xb.to(DEVICE)
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
            print(f"    Ep {ep:3d}: train={train_acc:.1f}%  test={test_acc:.1f}%  t={t_now:.1f}s", flush=True)

    total_time = time.time() - t0
    return history, best_acc, milestones, total_time


def _create_cifar10_optimizer(name, model, total_steps, steps_per_epoch):
    is_sam = False
    scheduler = None

    if name == 'Gen1_FixedSGD':
        opt = FixedLRSGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    elif name == 'Gen2_CosineSGD':
        opt = CosineAnnealingSGD(model.parameters(), lr=0.03, T_max=total_steps, momentum=0.9, weight_decay=5e-4)
    elif name == 'Gen2_SGDR':
        opt = SGDRWithRestarts(model.parameters(), lr=0.05, T_0=10, T_mult=2, momentum=0.9, weight_decay=5e-4)
    elif name == 'Gen3_Adam':
        opt = torch.optim.Adam(model.parameters(), lr=3e-4)
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
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, adaptive=True, lr=0.03, momentum=0.9, weight_decay=5e-4)
        is_sam = True
    elif name == 'Gen5_Grokfast':
        opt = Grokfast(model.parameters(), lr=0.03, momentum=0.9, weight_decay=5e-4, alpha=0.98)
    elif name == 'Gen5_STLR':
        pg = build_discriminative_stlr_param_groups(model, base_lr=0.05, decay_factor=2.6)
        opt = torch.optim.SGD(pg, momentum=0.9, weight_decay=5e-4)
        scheduler = STLRScheduler(opt, T=total_steps, decay_factor=2.6)
    elif name == 'SOTA_SAM_Discrim':
        pg = build_discriminative_param_groups(model, base_lr=0.05, decay_factor=2.6)
        opt = SAM(pg, torch.optim.SGD, rho=0.05, adaptive=True, lr=0.05, momentum=0.9, weight_decay=5e-4)
        is_sam = True
    elif name == 'SOTA_DALS':
        opt = DALS(model, lr=0.03, momentum=0.9, weight_decay=5e-4,
                   trust_coef=0.02, grokfast_alpha=0.6, warmup_frac=0.05, T_max=total_steps)
    elif name == 'SOTA_DALS_Fast':
        opt = DALSFast(model, lr=0.05, momentum=0.85, weight_decay=5e-4,
                       trust_coef=0.02, grokfast_alpha=0.6, warmup_frac=0.02, T_max=total_steps)
    elif name == 'SOTA_DALS_Acc':
        opt = DALSAcc(model, lr=0.03, momentum=0.9, weight_decay=5e-4,
                      trust_coef=0.02, grokfast_alpha=0.7,
                      T_0=steps_per_epoch * 10, T_mult=2)
    else:
        raise ValueError(f"Unknown strategy: {name}")
    return opt, scheduler, is_sam


# ============================================================================
# 3. RTE Benchmark (DistilBERT, 10 epochs)
# ============================================================================
def run_rte(strategy_name, epochs=10):
    from datasets import load_dataset
    from transformers import DistilBertTokenizer

    dataset = load_dataset('nyu-mll/glue', 'rte')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize(batch):
        return tokenizer(batch['sentence1'], batch['sentence2'],
                         padding='max_length', truncation=True, max_length=128)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(['sentence1', 'sentence2', 'idx'])
    tokenized.set_format('torch')

    train_loader = DataLoader(tokenized['train'], batch_size=32, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(tokenized['validation'], batch_size=32, shuffle=False, num_workers=NUM_WORKERS)

    return _run_nlp_benchmark(
        strategy_name, 'RTE', epochs, train_loader, val_loader,
        num_labels=2, lr_adam=5e-5, lr_sgd=5e-4,
        milestones_thresholds=[60, 65, 70, 75, 80]
    )


# ============================================================================
# 4. TREC-6 Benchmark (DistilBERT, 10 epochs)
# ============================================================================
def run_trec6(strategy_name, epochs=10):
    from transformers import DistilBertTokenizer

    # Load TREC-6 from local files (downloaded to data/trec)
    trec_dir = DATA_DIR / 'trec'
    train_file = trec_dir / 'train_5500.label'
    test_file = trec_dir / 'TREC_10.label'

    if not train_file.exists() or not test_file.exists():
        import urllib.request
        trec_dir.mkdir(parents=True, exist_ok=True)
        urls = {
            'train_5500.label': 'https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label',
            'TREC_10.label': 'https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label',
        }
        for fname, url in urls.items():
            fpath = trec_dir / fname
            if not fpath.exists():
                print(f"  Downloading TREC: {fname}...")
                proxy_handler = urllib.request.ProxyHandler({
                    'http': os.environ.get('http_proxy', ''),
                    'https': os.environ.get('https_proxy', ''),
                })
                opener = urllib.request.build_opener(proxy_handler)
                data = opener.open(url).read()
                with open(fpath, 'wb') as f:
                    f.write(data)

    def parse_trec(fpath):
        data = []
        with open(fpath, 'r', encoding='latin-1') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                first_colon = line.find(':')
                if first_colon < 0:
                    continue
                coarse = line[:first_colon]
                rest = line[first_colon + 1:]
                space_pos = rest.find(' ')
                if space_pos < 0:
                    continue
                text = rest[space_pos + 1:].strip()
                data.append({'coarse': coarse, 'text': text})
        return data

    train_data = parse_trec(train_file)
    test_data = parse_trec(test_file)

    label2id = {'ABBR': 0, 'DESC': 1, 'ENTY': 2, 'HUM': 3, 'LOC': 4, 'NUM': 5}

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    class TRECDataset(torch.utils.data.Dataset):
        def __init__(self, data, tokenizer, label2id, max_len=128):
            self.data = data
            self.tokenizer = tokenizer
            self.label2id = label2id
            self.max_len = max_len

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            enc = self.tokenizer(item['text'], padding='max_length', truncation=True, max_length=self.max_len, return_tensors='pt')
            return {
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'label': torch.tensor(self.label2id[item['coarse']], dtype=torch.long),
            }

    train_dataset = TRECDataset(train_data, tokenizer, label2id)
    test_dataset = TRECDataset(test_data, tokenizer, label2id)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=NUM_WORKERS)

    return _run_nlp_benchmark(
        strategy_name, 'TREC-6', epochs, train_loader, val_loader,
        num_labels=6, lr_adam=3e-5, lr_sgd=3e-4,
        milestones_thresholds=[70, 80, 85, 88, 90]
    )


# ============================================================================
# 5. IMDb Benchmark (DistilBERT, 5 epochs)
# ============================================================================
def run_imdb(strategy_name, epochs=5):
    from datasets import load_dataset
    from transformers import DistilBertTokenizer

    dataset = load_dataset('imdb')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(['text'])
    tokenized.set_format('torch')

    train_loader = DataLoader(tokenized['train'], batch_size=32, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(tokenized['test'], batch_size=64, shuffle=False, num_workers=NUM_WORKERS)

    return _run_nlp_benchmark(
        strategy_name, 'IMDb', epochs, train_loader, val_loader,
        num_labels=2, lr_adam=5e-5, lr_sgd=5e-4,
        milestones_thresholds=[80, 85, 88, 90, 92]
    )


# ============================================================================
# Main Runner
# ============================================================================
BENCHMARK_CONFIGS = {
    'synthetic': {
        'run_fn': run_synthetic,
        'epochs': 80,
        'checkpoint': 'synthetic_results.json',
        'description': '4-layer MLP, 80 epochs, from scratch',
    },
    'cifar10': {
        'run_fn': run_cifar10,
        'epochs': 50,
        'checkpoint': 'cifar10_results.json',
        'description': 'SmallConvNet, 50 epochs, from scratch',
    },
    'rte': {
        'run_fn': run_rte,
        'epochs': 10,
        'checkpoint': 'rte_results.json',
        'description': 'DistilBERT fine-tune on RTE (GLUE), 10 epochs',
    },
    'trec6': {
        'run_fn': run_trec6,
        'epochs': 10,
        'checkpoint': 'trec6_results.json',
        'description': 'DistilBERT fine-tune on TREC-6, 10 epochs',
    },
    'imdb': {
        'run_fn': run_imdb,
        'epochs': 5,
        'checkpoint': 'imdb_results.json',
        'description': 'DistilBERT fine-tune on IMDb, 5 epochs',
    },
}


def run_benchmark(dataset_name, strategies=None):
    config = BENCHMARK_CONFIGS[dataset_name]
    epochs = config['epochs']
    run_fn = config['run_fn']
    checkpoint_file = RESULTS_DIR / config['checkpoint']

    if strategies is None:
        strategies = STRATEGIES

    print(f"\n{'='*90}")
    print(f"  {dataset_name.upper()} BENCHMARK — {len(strategies)} strategies, {epochs} epochs, device={DEVICE}")
    print(f"  {config['description']}")
    print(f"{'='*90}\n")

    all_results = {}

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            saved = json.load(f)
        for name in saved:
            all_results[name] = {k: v for k, v in saved[name].items()}
        completed = set(all_results.keys())
        print(f"  Resumed {len(completed)} completed strategies: {sorted(completed)}\n")
    else:
        completed = set()

    for name in strategies:
        if name in completed:
            r = all_results[name]
            print(f"  [{name}] — already done: Best={r['best_acc']:.1f}%", flush=True)
            continue
        print(f"\n  [{name}]", flush=True)
        try:
            history, best_acc, milestones, total_time = run_fn(name, epochs)
            all_results[name] = {
                'best_acc': best_acc,
                'final_acc': history['test_acc'][-1],
                'milestones': milestones,
                'total_time': total_time,
                'history': {k: [float(x) for x in vs] for k, vs in history.items()},
            }
            ms = "  ".join([f"{k}%:{v[0]}ep" for k, v in sorted(milestones.items())])
            print(f"  => Best={best_acc:.1f}%  Final={history['test_acc'][-1]:.1f}%  {ms}  ({total_time:.1f}s)")
            with open(checkpoint_file, 'w') as f:
                json.dump(all_results, f, indent=2)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  => FAILED: {e}")

    print(f"\n{'='*110}")
    print(f"  {dataset_name.upper()} FINAL RESULTS")
    print(f"{'='*110}")
    for name in sorted(all_results, key=lambda n: all_results[n]['best_acc'], reverse=True):
        r = all_results[name]
        ms = "  ".join([f"{k}%:{v[0]}ep" for k, v in sorted(r['milestones'].items())])
        print(f"  {name:<25s} Best={r['best_acc']:>5.1f}%  Final={r['final_acc']:>5.1f}%  {ms}  ({r['total_time']:.1f}s)")

    with open(RESULTS_DIR / config['checkpoint'], 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / config['checkpoint']}")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Unified benchmark: 18 strategies × 5 datasets')
    parser.add_argument('--dataset', type=str, default='all',
                        choices=['all', 'synthetic', 'cifar10', 'rte', 'trec6', 'imdb', 'nlp'],
                        help='Which benchmark dataset to run')
    parser.add_argument('--list', action='store_true', help='List strategies and datasets')
    args = parser.parse_args()

    if args.list:
        print(f"\n{'='*60}")
        print(f"  Available strategies ({len(STRATEGIES)}):")
        print(f"{'='*60}")
        for i, s in enumerate(STRATEGIES, 1):
            print(f"  {i:2d}. {s}")
        print(f"\n{'='*60}")
        print(f"  Available benchmarks:")
        print(f"{'='*60}")
        for name, cfg in BENCHMARK_CONFIGS.items():
            print(f"  {name:<12s} {cfg['description']}")
        print()
        sys.exit(0)

    if args.dataset == 'all':
        datasets = list(BENCHMARK_CONFIGS.keys())
    elif args.dataset == 'nlp':
        datasets = ['rte', 'trec6', 'imdb']
    else:
        datasets = [args.dataset]

    for ds in datasets:
        print(f"\n\n{'#'*90}")
        print(f"  Starting benchmark: {ds.upper()}")
        print(f"{'#'*90}")
        run_benchmark(ds)