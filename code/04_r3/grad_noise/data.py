"""Models and datasets for the revised benchmark.

Design choices addressing reviewer comments:
  - Synthetic: 3-hidden-layer MLP 64->128->128->10 (no batchnorm, no dropout).
  - CIFAR-10: small ConvNet, 50 epochs.
  - RTE / TREC-6 / IMDb: DistilBERT fine-tuning (pretrained weights).
  - Validation split: held-out from train for model selection; test set used once.
  - Metrics: accuracy and macro-F1.
"""
import random
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------
# Synthetic dataset
# ----------------------------------------------------------------------------
def make_synthetic(seed=42, n=8000, d=64, n_classes=10, signal_scale=2.0, noise_std=0.5):
    """Gaussian-mixture 10-class task in R^64.

    Each class k has its mean at e_k * signal_scale in the first n_classes dims,
    with all d dims carrying noise N(0, noise_std^2). Labels are argmax of the
    signal dims. The 54 noise dims force the optimiser to ignore irrelevant
    features; the moderate signal-to-noise ratio makes the task learnable but
    not trivial, so that optimiser differences (e.g., 97% vs 98%) are visible.
    """
    rng = np.random.RandomState(seed)
    y = rng.randint(0, n_classes, size=n)
    X = rng.randn(n, d).astype(np.float32) * noise_std
    for i, k in enumerate(y):
        X[i, k] += signal_scale
    return torch.from_numpy(X), torch.from_numpy(y).long()


class MLP(nn.Module):
    def __init__(self, in_dim=64, hidden=128, n_classes=10):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, n_classes)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.fc3(x)


def synthetic_loaders(seed=42, batch=64, val_frac=0.1):
    X, y = make_synthetic(seed=seed)
    ds = TensorDataset(X, y)
    n_val = int(len(ds) * val_frac)
    n_train = len(ds) - n_val
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=gen)
    # fixed test set
    Xte, yte = make_synthetic(seed=seed + 100, n=1600)
    test_ds = TensorDataset(Xte, yte)
    return (DataLoader(train_ds, batch_size=batch, shuffle=True),
            DataLoader(val_ds, batch_size=batch),
            DataLoader(test_ds, batch_size=batch))


# ----------------------------------------------------------------------------
# CIFAR-10 ConvNet
# ----------------------------------------------------------------------------
class ConvNet(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(128 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, n_classes))

    def forward(self, x):
        return self.classifier(self.features(x))


def cifar_loaders(dataset='cifar10', seed=42, batch=128, val_frac=0.1):
    import torchvision, os
    import torchvision.transforms as T
    tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()])
    tf_test = T.Compose([T.ToTensor()])
    data_dir = os.environ.get('DATA_DIR', os.path.dirname(os.path.abspath(__file__)))
    root_candidates = [os.path.join(data_dir, 'cifar10'), data_dir, './data', '../codes/data', '../data',
                       'C:/Users/frede/Documents/GitHub/CodeOfMe/LearningRate/data',
                       'C:/Users/frede/Documents/GitHub/CodeOfMe/LearningRate/codes/data']
    root = None
    marker = 'cifar-100-python' if dataset == 'cifar100' else 'cifar-10-batches-py'
    for r in root_candidates:
        if os.path.isdir(os.path.join(r, marker)):
            root = r; break
    if root is None:
        root = './data'
    DS = torchvision.datasets.CIFAR100 if dataset == 'cifar100' else torchvision.datasets.CIFAR10
    try:
        train = DS(root=root, train=True, download=True, transform=tf)
        test = DS(root=root, train=False, download=True, transform=tf_test)
    except Exception:
        train = DS(root=root, train=True, download=False, transform=tf)
        test = DS(root=root, train=False, download=False, transform=tf_test)
    n_val = int(len(train) * val_frac)
    n_tr = len(train) - n_val
    gen = torch.Generator().manual_seed(seed)
    tr, val = random_split(train, [n_tr, n_val], generator=gen)
    return (DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0),
            DataLoader(val, batch_size=batch, num_workers=0),
            DataLoader(test, batch_size=batch, num_workers=0))

cifar10_loaders = lambda seed=42,batch=128,val_frac=0.1: cifar_loaders('cifar10',seed,batch,val_frac)


class ConvNet100(nn.Module):
    def __init__(self, n_classes=100):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.3), nn.Linear(256*8*8, 512), nn.ReLU(), nn.Linear(512, n_classes))
    def forward(self, x):
        return self.classifier(self.features(x))


# ----------------------------------------------------------------------------
# DistilBERT fine-tuning datasets (RTE, TREC-6, IMDb)
# ----------------------------------------------------------------------------
def nlp_loaders(task='rte', seed=42, batch=32, max_len=128, val_frac=0.1, tokenizer_name='distilbert-base-uncased'):
    from datasets import load_dataset, Dataset
    from transformers import AutoTokenizer
    data_dir = os.environ.get('DATA_DIR', os.path.dirname(os.path.abspath(__file__)))
    cache_dir = os.path.join(data_dir, 'huggingface')
    os.makedirs(cache_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    # --- local jsonl source (offline, RTE/TREC-6/IMDb) ---
    local_root = os.path.join(data_dir, task)
    if os.path.isdir(local_root):
        def from_local(name):
            p = os.path.join(local_root, name)
            return Dataset.from_json(p) if os.path.exists(p) else None
        if task == 'rte':
            train = from_local('train.jsonl'); test = from_local('test.jsonl')
            n_labels = 2
            def tok(b): return tokenizer(b['sentence1'], b['sentence2'], truncation=True, max_length=max_len)
        elif task == 'trec6':
            train = from_local('train.jsonl'); test = from_local('test.jsonl')
            n_labels = 6
            def tok(b): return tokenizer(b['text'], truncation=True, max_length=max_len)
        elif task == 'imdb':
            train = from_local('train.jsonl'); test = from_local('test.jsonl')
            n_labels = 2
            def tok(b): return tokenizer(b['text'], truncation=True, max_length=max_len)
        else:
            train = test = None; n_labels = 2
        if train is not None and test is not None:
            return _build_nlp_loaders(train, test, tok, n_labels, task,
                                      seed=seed, batch=batch, val_frac=val_frac,
                                      tokenizer=tokenizer)

    if task == 'rte':
        ds = load_dataset('nyu-mll/glue', 'rte', cache_dir=cache_dir)
        def tok(b): return tokenizer(b['sentence1'], b['sentence2'], truncation=True, max_length=max_len)
        label_key = 'label'
        n_labels = 2
    elif task == 'trec6':
        ds = load_dataset('SetFit/TREC-QC', cache_dir=cache_dir)
        def tok(b): return tokenizer(b['text'], truncation=True, max_length=max_len)
        label_key = 'label_coarse'
        n_labels = 6
    elif task == 'imdb':
        ds = load_dataset('stanfordnlp/imdb', cache_dir=cache_dir)
        def tok(b): return tokenizer(b['text'], truncation=True, max_length=max_len)
        label_key = 'label'
        n_labels = 2
    elif task == 'sst2':
        ds = load_dataset('nyu-mll/glue', 'sst2', cache_dir=cache_dir)
        def tok(b): return tokenizer(b['sentence'], truncation=True, max_length=max_len)
        label_key = 'label'
        n_labels = 2
    elif task == 'mrpc':
        ds = load_dataset('nyu-mll/glue', 'mrpc', cache_dir=cache_dir)
        def tok(b): return tokenizer(b['sentence1'], b['sentence2'], truncation=True, max_length=max_len)
        label_key = 'label'
        n_labels = 2
    else:
        raise ValueError(task)
    ds = ds.map(tok, batched=True)
    # standardise to train/val/test
    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer)
    cols = ['input_ids', 'attention_mask', label_key]
    def fmt(split):
        split = split.remove_columns([c for c in split.column_names if c not in cols])
        split = split.rename_column(label_key, 'labels')
        return split
    if task in ('rte', 'sst2', 'mrpc'):
        # GLUE splits: official test set has no labels (all -1). Use the
        # validation split as the held-out evaluation set, matching RTE.
        train = fmt(ds['train']); test = fmt(ds['validation'])
    elif task == 'trec6':
        train = fmt(ds['train']); test = fmt(ds['test'])
    else:  # imdb
        train = fmt(ds['train']); test = fmt(ds['test'])
    # hold out val from train
    n_val = int(len(train) * val_frac)
    n_tr = len(train) - n_val
    gen = torch.Generator().manual_seed(seed)
    tr_split, val_split = random_split(train, [n_tr, n_val], generator=gen)
    return (DataLoader(tr_split, batch_size=batch, shuffle=True, collate_fn=collator),
            DataLoader(val_split, batch_size=batch, collate_fn=collator),
            DataLoader(test, batch_size=batch, collate_fn=collator),
            n_labels)


def _build_nlp_loaders(train, test, tok, n_labels, task, seed=42, batch=32,
                       val_frac=0.1, tokenizer=None):
    """Build train/val/test loaders from local HF Datasets (jsonl source)."""
    from transformers import DataCollatorWithPadding
    collator = DataCollatorWithPadding(tokenizer)
    train = train.map(tok, batched=True)
    test = test.map(tok, batched=True)
    cols = ['input_ids', 'attention_mask', 'labels']
    def fmt(split):
        # rename BEFORE removing columns: 'label'/'label_coarse' must become
        # 'labels' first, otherwise remove_columns drops them (KeyError later).
        for k in ('label', 'label_coarse'):
            if k in split.column_names:
                split = split.rename_column(k, 'labels')
        split = split.remove_columns([c for c in split.column_names if c not in cols])
        return split
    train = fmt(train); test = fmt(test)
    n_val = int(len(train) * val_frac)
    n_tr = len(train) - n_val
    gen = torch.Generator().manual_seed(seed)
    tr_split, val_split = random_split(train, [n_tr, n_val], generator=gen)
    return (DataLoader(tr_split, batch_size=batch, shuffle=True, collate_fn=collator),
            DataLoader(val_split, batch_size=batch, collate_fn=collator),
            DataLoader(test, batch_size=batch, collate_fn=collator),
            n_labels)