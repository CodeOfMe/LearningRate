"""
RTE benchmark: DistilBERT fine-tuning with 18 strategies.
RTE (Recognizing Textual Entailment) from GLUE — 2.5k train, 277 val.
Validates transfer learning scenario (pretrained model + NLP task).
MPS/CUDA acceleration.
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
import os

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from datasets import load_dataset

from optimizers import (
    FixedLRSGD, CosineAnnealingSGD, SGDRWithRestarts,
    AdamW, AdaBound,
    DiscriminativeLR, LARS,
    RAdam, Lookahead, SAM, Lion,
    Grokfast, DALS, DALSFast, DALSAcc,
    STLRScheduler,
)

# ========== Hardware Setup ==========
if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

torch.set_num_threads(4)
print(f"Device: {DEVICE} | Threads: {torch.get_num_threads()}")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

EPOCHS = 10
BATCH_SIZE = 32
MAX_LEN = 128
LR_ADAM = 5e-5
LR_SGD = 5e-4


def seed_all(s=42):
    torch.manual_seed(s)
    np.random.seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def get_layer_groups(model):
    """Flatten DistilBERT into 6 layer groups for discriminative/DALS."""
    groups = [
        {'name': 'embeddings', 'modules': [model.distilbert.embeddings]},
        {'name': 'transformer_01', 'modules': [model.distilbert.transformer.layer[0], model.distilbert.transformer.layer[1]]},
        {'name': 'transformer_23', 'modules': [model.distilbert.transformer.layer[2], model.distilbert.transformer.layer[3]]},
        {'name': 'transformer_45', 'modules': [model.distilbert.transformer.layer[4], model.distilbert.transformer.layer[5]]},
        {'name': 'pre_classifier', 'modules': [model.pre_classifier]},
        {'name': 'classifier', 'modules': [model.classifier]},
    ]
    param_groups = []
    for i, g in enumerate(groups):
        params = []
        for m in g['modules']:
            params.extend(list(m.parameters()))
        param_groups.append({
            'params': params,
            'name': g['name'],
        })
    return param_groups, len(groups)


def build_discriminative_groups(model, base_lr, decay_factor=2.6):
    """Create discriminative param groups for DistilBERT."""
    pg, num = get_layer_groups(model)
    for i in range(num):
        lr = base_lr / (decay_factor ** (num - 1 - i))
        pg[i]['lr'] = lr
    return pg


def build_stlr_groups(model, base_lr, decay_factor=2.6):
    """Create discriminative + STLR param groups for DistilBERT."""
    pg, num = get_layer_groups(model)
    for i in range(num):
        lr = base_lr / (decay_factor ** (num - 1 - i))
        pg[i]['lr'] = lr
        pg[i]['eta_max'] = lr
    return pg


class DistilBERTWrapper(nn.Module):
    """Wrapper so DALS can use model.children() for depth detection."""
    def __init__(self, model):
        super().__init__()
        self.embeddings = model.distilbert.embeddings
        self.transformer_01 = nn.Sequential(*[model.distilbert.transformer.layer[0], model.distilbert.transformer.layer[1]])
        self.transformer_23 = nn.Sequential(*[model.distilbert.transformer.layer[2], model.distilbert.transformer.layer[3]])
        self.transformer_45 = nn.Sequential(*[model.distilbert.transformer.layer[4], model.distilbert.transformer.layer[5]])
        self.pre_classifier = model.pre_classifier
        self.classifier = model.classifier


def create_optimizer(name, model, total_steps):
    is_sam = False
    scheduler = None
    steps_per_epoch = 2490 // BATCH_SIZE + 1

    if name == 'Gen1_FixedSGD':
        opt = FixedLRSGD(model.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=0.01)
    elif name == 'Gen2_CosineSGD':
        opt = CosineAnnealingSGD(model.parameters(), lr=LR_SGD, T_max=total_steps, momentum=0.9, weight_decay=0.01)
    elif name == 'Gen2_SGDR':
        opt = SGDRWithRestarts(model.parameters(), lr=LR_SGD * 2, T_0=steps_per_epoch * 3, T_mult=2, momentum=0.9, weight_decay=0.01)
    elif name == 'Gen3_Adam':
        opt = torch.optim.Adam(model.parameters(), lr=LR_ADAM)
    elif name == 'Gen3_AdamW':
        opt = AdamW(model.parameters(), lr=LR_ADAM, weight_decay=0.01)
    elif name == 'Gen3_AdaBound':
        opt = AdaBound(model.parameters(), lr=LR_ADAM, final_lr=LR_SGD)
    elif name == 'Gen4_LARS':
        opt = LARS(model.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=0.01, trust_coef=0.02)
    elif name == 'Gen4_Discriminative':
        pg = build_discriminative_groups(model, base_lr=LR_SGD, decay_factor=2.6)
        opt = DiscriminativeLR(pg, momentum=0.9, weight_decay=0.01)
    elif name == 'Gen5_RAdam':
        opt = RAdam(model.parameters(), lr=LR_ADAM)
    elif name == 'Gen5_Lion':
        opt = Lion(model.parameters(), lr=LR_ADAM * 0.5, weight_decay=0.01)
    elif name == 'Gen5_Lookahead':
        opt = Lookahead(AdamW(model.parameters(), lr=LR_ADAM, weight_decay=0.01), k=5, alpha=0.5)
    elif name == 'Gen5_SAM':
        opt = SAM(model.parameters(), torch.optim.SGD, rho=0.05, adaptive=True, lr=LR_SGD, momentum=0.9, weight_decay=0.01)
        is_sam = True
    elif name == 'Gen5_Grokfast':
        opt = Grokfast(model.parameters(), lr=LR_SGD, momentum=0.9, weight_decay=0.01, alpha=0.98)
    elif name == 'Gen5_STLR':
        pg = build_stlr_groups(model, base_lr=LR_SGD, decay_factor=2.6)
        opt = torch.optim.SGD(pg, momentum=0.9, weight_decay=0.01)
        scheduler = STLRScheduler(opt, T=total_steps, decay_factor=2.6)
    elif name == 'SOTA_SAM_Discrim':
        pg = build_discriminative_groups(model, base_lr=LR_SGD, decay_factor=2.6)
        opt = SAM(pg, torch.optim.SGD, rho=0.05, adaptive=True, lr=LR_SGD, momentum=0.9, weight_decay=0.01)
        is_sam = True
    elif name == 'SOTA_DALS':
        wrapper = DistilBERTWrapper(model)
        opt = DALS(wrapper, lr=LR_SGD, momentum=0.9, weight_decay=0.01,
                   trust_coef=0.02, grokfast_alpha=0.6,
                   warmup_frac=0.1, T_max=total_steps)
    elif name == 'SOTA_DALS_Fast':
        wrapper = DistilBERTWrapper(model)
        opt = DALSFast(wrapper, lr=LR_SGD * 2, momentum=0.85, weight_decay=0.01,
                       trust_coef=0.02, grokfast_alpha=0.6,
                       warmup_frac=0.05, T_max=total_steps)
    elif name == 'SOTA_DALS_Acc':
        wrapper = DistilBERTWrapper(model)
        opt = DALSAcc(wrapper, lr=LR_SGD, momentum=0.9, weight_decay=0.01,
                      trust_coef=0.02, grokfast_alpha=0.7,
                      T_0=steps_per_epoch * 3, T_mult=2)
    else:
        raise ValueError(f"Unknown strategy: {name}")
    return opt, scheduler, is_sam


def train_and_evaluate(name, epochs=EPOCHS):
    seed_all(42)

    dataset = load_dataset('nyu-mll/glue', 'rte')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize(batch):
        return tokenizer(batch['sentence1'], batch['sentence2'],
                         padding='max_length', truncation=True, max_length=MAX_LEN)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.remove_columns(['sentence1', 'sentence2', 'idx'])
    tokenized.set_format('torch')

    train_loader = torch.utils.data.DataLoader(
        tokenized['train'], batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = torch.utils.data.DataLoader(
        tokenized['validation'], batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
    model = model.to(DEVICE)
    total_steps = epochs * len(train_loader)
    opt, scheduler, is_sam = create_optimizer(name, model, total_steps)

    is_dals = isinstance(opt, (DALS, DALSFast, DALSAcc))

    history = {'train_acc': [], 'test_acc': [], 'train_loss': [], 'lr': []}
    milestones = {}
    best_acc = 0
    t0 = time.time()

    for ep in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        n_batches = 0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            if is_sam:
                def closure():
                    opt.zero_grad()
                    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    return out.loss
                loss = closure()
                opt.first_step(zero_grad=True)
                with torch.enable_grad():
                    closure()
                opt.second_step()
            elif is_dals:
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
                opt.update_phase(loss.item())
                opt.step()
                opt.zero_grad(set_to_none=True)
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
                if not is_sam:
                    preds = out.logits.argmax(dim=-1)
                else:
                    preds = model(input_ids=input_ids, attention_mask=attention_mask).logits.argmax(dim=-1)
                correct_train += (preds == labels).sum().item()
                total_train += len(labels)

        if isinstance(scheduler, STLRScheduler):
            for _ in range(len(train_loader)):
                scheduler.step()

        # Evaluate
        model.eval()
        correct = total = 0
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

        lr_now = opt.param_groups[0].get('lr', LR_ADAM)
        t_now = time.time() - t0

        for thr in [60, 65, 70, 75, 80]:
            if test_acc >= thr and str(thr) not in milestones:
                milestones[str(thr)] = [ep, round(t_now, 2)]

        best_acc = max(best_acc, test_acc)
        history['train_acc'].append(train_acc)
        history['test_acc'].append(test_acc)
        history['train_loss'].append(epoch_loss / max(n_batches, 1))
        history['lr'].append(lr_now)

        print(f"  Ep {ep:2d}/{epochs}: train={train_acc:.1f}%  val={test_acc:.1f}%  lr={lr_now:.2e}  t={t_now:.1f}s", flush=True)

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
    print(f"  RTE BENCHMARK (DistilBERT) — {len(strategies)} strategies, {EPOCHS} epochs, device={DEVICE}")
    print(f"{'='*90}\n")

    all_results = {}
    checkpoint_file = RESULTS_DIR / 'rte_results.json'

    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            saved = json.load(f)
        for name in saved:
            all_results[name] = saved[name]
        completed = set(all_results.keys())
        print(f"  Resumed {len(completed)} completed strategies: {sorted(completed)}")
    else:
        completed = set()

    for name in strategies:
        if name in completed:
            r = all_results[name]
            print(f"\n  [{name}] — already done: Best={r['best_acc']:.1f}%", flush=True)
            continue
        print(f"\n  [{name}]", flush=True)
        try:
            history, best_acc, milestones, total_time = train_and_evaluate(name)
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
    print(f"  RTE FINAL RESULTS")
    print(f"{'='*110}")
    for name in sorted(all_results, key=lambda n: all_results[n]['best_acc'], reverse=True):
        r = all_results[name]
        ms = "  ".join([f"{k}%:{v[0]}ep" for k, v in sorted(r['milestones'].items())])
        print(f"  {name:<25s} Best={r['best_acc']:>5.1f}%  Final={r['final_acc']:>5.1f}%  {ms}  ({r['total_time']:.1f}s)")

    with open(RESULTS_DIR / 'rte_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_DIR / 'rte_results.json'}")