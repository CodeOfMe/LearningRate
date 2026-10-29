"""Run remaining 3 IMDb strategies."""
import sys, os, json, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pathlib import Path
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from optimizers import DALS, DALSFast, DALSAcc
from Test_All import seed_all, DistilBERTWrapper, _get_nlp_layer_groups, \
    _build_nlp_discriminative, _build_nlp_stlr, _create_nlp_optimizer, DEVICE, \
    NUM_WORKERS, PIN_MEMORY, RESULTS_DIR

if torch.backends.mps.is_available():
    DEVICE = 'mps'
elif torch.cuda.is_available():
    DEVICE = 'cuda'
else:
    DEVICE = 'cpu'

seed_all(42)

dataset = load_dataset('imdb')
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=256)

tokenized = dataset.map(tokenize, batched=True)
tokenized = tokenized.remove_columns(['text'])
tokenized.set_format('torch')

train_loader = DataLoader(tokenized['train'], batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(tokenized['test'], batch_size=64, shuffle=False, num_workers=0)

strategies = ['SOTA_DALS', 'SOTA_DALS_Fast', 'SOTA_DALS_Acc']
epochs = 5
lr_adam = 5e-5
lr_sgd = 5e-4
milestones_thresholds = [80, 85, 88, 90, 92]

results_file = RESULTS_DIR / 'imdb_results.json'
if results_file.exists():
    with open(results_file) as f:
        all_results = json.load(f)
else:
    all_results = {}

for name in strategies:
    if name in all_results:
        print(f"  [{name}] already done: Best={all_results[name]['best_acc']:.1f}%")
        continue

    seed_all(42)
    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased', num_labels=2
    ).to(DEVICE)

    total_steps = epochs * len(train_loader)
    steps_per_epoch = len(train_loader)

    opt, scheduler, is_sam = _create_nlp_optimizer(
        name, model, total_steps, steps_per_epoch, lr_adam=lr_adam, lr_sgd=lr_sgd
    )
    is_dals = isinstance(opt, (DALS, DALSFast, DALSAcc))

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

            if is_dals:
                opt.zero_grad()
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
                opt.update_phase(loss.item())
                opt.step()
            else:
                opt.zero_grad()
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = out.loss
                loss.backward()
                opt.step()

            if scheduler is not None and not isinstance(scheduler, type(None)):
                try:
                    scheduler.step()
                except:
                    pass

            epoch_loss += loss.item()
            n_batches += 1

            with torch.no_grad():
                preds = out.logits.argmax(dim=-1)
                correct_train += (preds == labels).sum().item()
                total_train += len(labels)

        if isinstance(scheduler, type(None)) or scheduler is None:
            pass
        else:
            try:
                for _ in range(len(train_loader)):
                    scheduler.step()
            except:
                pass

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

        lr_now = opt.param_groups[0].get('lr', lr_sgd)
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
    all_results[name] = {
        'best_acc': best_acc,
        'final_acc': history['test_acc'][-1],
        'milestones': milestones,
        'total_time': total_time,
        'history': {k: [float(x) for x in vs] for k, vs in history.items()},
    }
    print(f"  => Best={best_acc:.1f}%  Final={history['test_acc'][-1]:.1f}%  ({total_time:.1f}s)")

    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"  Saved to {results_file}")

print("\nDone!")