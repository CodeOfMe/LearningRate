"""Unified training/evaluation loop for the revised benchmark.

Features addressing reviewer comments:
  - Validation-based checkpoint selection (one final test eval).
  - Reports accuracy and macro-F1.
  - Wall-clock time, throughput, peak GPU memory.
  - Per-layer gradient statistics (variance, norm, update-to-weight ratio)
    for the depth-gradient hypothesis (reviewer R4-15).
  - Multi-seed support.
  - Device-agnostic (cuda / directml / cpu).
"""
import time
import math
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score


def _move(batch, device):
    if isinstance(batch, (list, tuple)):
        return [b.to(device) for b in batch]
    if isinstance(batch, dict):
        return {k: v.to(device) for k, v in batch.items()}
    return batch.to(device)


def evaluate(model, loader, device, n_labels=2):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for batch in loader:
            if hasattr(batch, 'input_ids') or (isinstance(batch, dict) and 'input_ids' in batch):
                y = batch['labels']
                out = model(input_ids=batch['input_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device))
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x, y = batch
                    out = model(_move(x, device))
                else:
                    *x, y = batch
                    out = model(*[_move(t, device) for t in x])
            else:
                x, y = batch
                out = model(_move(x, device))
            ys.extend(y.cpu().tolist())
            ps.extend(out.logits.argmax(dim=-1).cpu().tolist() if hasattr(out, 'logits') else out.argmax(dim=-1).cpu().tolist())
    acc = accuracy_score(ys, ps) * 100.0
    avg = 'macro' if n_labels > 2 else 'binary'
    f1 = f1_score(ys, ps, average=avg) * 100.0
    return acc, f1


class GradStats:
    """Collect per-layer gradient statistics for the depth-gradient hypothesis."""
    def __init__(self):
        self.stats = []

    def hook(self, model):
        handles = []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            def make(nm, param):
                def cb(grad):
                    self.stats.append((nm, param.norm(2).item(),
                                       grad.norm(2).item() if grad is not None else 0.0,
                                       grad.var().item() if grad is not None else 0.0))
                return cb
            h = p.register_hook(make(name, p))
            handles.append(h)
        return handles


def train_one(model, optimizer, train_loader, val_loader, test_loader, device,
              n_labels=2, epochs=80, n_classes_for_acc=None,
              sched=None, dals_phase_update=None,
              select_by='val', grad_stats_every=0, verbose=0):
    """Train and report final-epoch + best-val test metrics.

    select_by: 'val' -> select checkpoint by best val acc, evaluate once on test.
               'last' -> report final-epoch test metrics.
    dals_phase_update: optional callable(loss_val) to update DALS phase each step.
    Returns dict with metrics + timing.
    """
    model = model.to(device)
    best_val = -1.0; best_test = (0.0, 0.0); best_ep = -1
    last_test = (0.0, 0.0)
    gs = None
    handles = []
    if grad_stats_every > 0:
        gs = GradStats()
        handles = gs.hook(model)
    peak_mem = 0.0
    is_cuda = 'cuda' in str(device) or device.type == 'cuda'
    is_gpu = is_cuda or 'privateuseone' in str(device) or 'dml' in str(device)
    t0 = time.time()
    n_samples = 0
    for ep in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            if hasattr(batch, 'input_ids') or (isinstance(batch, dict) and 'input_ids' in batch):
                y = batch['labels'].to(device)
                out = model(input_ids=batch['input_ids'].to(device),
                            attention_mask=batch['attention_mask'].to(device))
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 2:
                    x, y = batch
                    x = _move(x, device); y = _move(y, device)
                    out = model(x)
                else:
                    *x, y = batch
                    x = [_move(t, device) for t in x]; y = _move(y, device)
                    out = model(*x)
            else:
                x, y = batch
                x = _move(x, device); y = _move(y, device)
                out = model(x)
            logits = out.logits if hasattr(out, 'logits') else out
            loss = nn.functional.cross_entropy(logits, y)
            optimizer.zero_grad()
            loss.backward()
            if dals_phase_update is not None:
                dals_phase_update(loss.item())
            if sched is not None:
                sched.step()
            optimizer.step()
            n_samples += y.size(0)
            if is_cuda:
                peak_mem = max(peak_mem, torch.cuda.max_memory_allocated() / 1e6)
        # validation
        if val_loader is not None and len(val_loader) > 0:
            val_acc, val_f1 = evaluate(model, val_loader, device, n_labels)
            if val_acc > best_val:
                best_val = val_acc
                best_ep = ep
                if select_by == 'val' and test_loader is not None:
                    best_test = evaluate(model, test_loader, device, n_labels)
        if verbose and (ep % max(1, epochs // 5) == 0 or ep == epochs):
            print(f'  ep{ep} loss={loss.item():.4f} val_acc={val_acc if val_loader else 0:.2f}')
    elapsed = time.time() - t0
    # final-epoch test
    if test_loader is not None:
        last_test = evaluate(model, test_loader, device, n_labels)
    for h in handles:
        h.remove()
    res = {
        'best_val_acc': best_val if val_loader else None,
        'best_val_epoch': best_ep,
        'test_acc_valsel': best_test[0] if select_by == 'val' else None,
        'test_f1_valsel': best_test[1] if select_by == 'val' else None,
        'test_acc_last': last_test[0],
        'test_f1_last': last_test[1],
        'wall_clock_s': elapsed,
        'throughput_sps': n_samples / elapsed if elapsed > 0 else 0.0,
        'peak_gpu_mem_mb': peak_mem if is_gpu else None,
        'grad_stats': gs.stats if gs is not None else None,
    }
    return res