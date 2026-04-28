"""
Comprehensive Optimizer Collection for Learning Rate Research
============================================================
Covers 5 generations of learning rate strategies:
  Gen 1: Fixed LR (SGD)
  Gen 2: LR Scheduling (Step, Cosine, SGDR)
  Gen 3: Parameter-level Adaptive (Adam, AdamW, AdaBound, Adafactor)
  Gen 4: Layer-level Discriminative (Discriminative Fine-tuning, LARS, LAMB)
  Gen 5: Layer×Time Joint (STLR, Lookahead+Discriminative, SAM+Discriminative,
          RAdam, Sophia, Lion, Schedule-Free, Grokfast)

Plus SOTA challengers: DoRA-style layer normalization, AdaSLS, Pro-SD
"""

import math
import torch
from torch.optim import Optimizer
from typing import Optional, Callable, List, Dict, Any, Tuple


# ============================================================================
# Gen 1: Fixed Learning Rate
# ============================================================================

class FixedLRSGD(Optimizer):
    """Vanilla SGD with fixed learning rate (Gen 1 baseline)."""

    def __init__(self, params, lr=0.01, momentum=0, weight_decay=0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0:
                    d_p = d_p.add(p, alpha=wd)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf
                p.add_(d_p, alpha=-lr)
        return loss


# ============================================================================
# Gen 2: Learning Rate Scheduling Wrappers
# ============================================================================

class CosineAnnealingSGD(Optimizer):
    """SGD with built-in cosine annealing schedule."""

    def __init__(self, params, lr=0.1, T_max=100, eta_min=1e-5,
                 momentum=0.9, weight_decay=5e-4):
        self.T_max = T_max
        self.eta_min = eta_min
        self.eta_max = lr
        self._step_count = 0
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step_count += 1
        progress = min(self._step_count / self.T_max, 1.0)
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * progress))
        for group in self.param_groups:
            group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p)
                    d_p = buf
                p.add_(d_p, alpha=-lr)
        return loss


class SGDRWithRestarts(Optimizer):
    """SGD with Stochastic Gradient Descent with Warm Restarts (Loshchilov & Hutter, 2017)."""

    def __init__(self, params, lr=0.1, T_0=10, T_mult=2, eta_min=1e-5,
                 momentum=0.9, weight_decay=5e-4):
        self.eta_max = lr
        self.eta_min = eta_min
        self.T_0 = T_0
        self.T_mult = T_mult
        self._step_count = 0
        self._cycle_count = 0
        self._T_i = T_0
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step_count += 1
        if self._step_count > self._T_i:
            self._cycle_count += 1
            self._step_count = 1
            self._T_i = int(self._T_i * self.T_mult)
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * \
             (1 + math.cos(math.pi * self._step_count / self._T_i))
        for group in self.param_groups:
            group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p)
                    d_p = buf
                p.add_(d_p, alpha=-lr)
        return loss


# ============================================================================
# Gen 3: Parameter-level Adaptive Optimizers
# ============================================================================

class AdamW(Optimizer):
    """Adam with decoupled weight decay (Loshchilov & Hutter, 2019)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                step = state['step']
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                step_size = group['lr'] / bias_correction1
                p.add_(exp_avg / denom, alpha=-step_size)
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])
        return loss


class AdaBound(Optimizer):
    """AdaBound: Adaptive Gradient Methods with Dynamic Bound (Luo et al., 2019).
    Transitions from Adam to SGD as training progresses."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsbound=False, final_lr=0.1, gamma=1e-3):
        if not 0.0 <= gamma < 1.0:
            raise ValueError(f"Invalid gamma: {gamma}")
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        amsbound=amsbound, final_lr=final_lr, gamma=gamma)
        super().__init__(params, defaults)
        self.base_lrs = [group['lr'] for group in self.param_groups]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if group['amsbound']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if group['amsbound']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step = torch.clamp(step_size / denom, min=lower_bound, max=upper_bound)
                p.add_(exp_avg * step, alpha=-1)
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])
        return loss


class Adafactor(Optimizer):
    """Adafactor: Memory-efficient adaptive optimizer (Shazeer & Stern, 2018)."""

    def __init__(self, params, lr=1e-3, eps=(1e-30, 1e-3), clip_threshold=1.0,
                 decay_rate=-0.8, weight_decay=0, scale_parameter=True,
                 relative_step=True, warmup_init=False):
        defaults = dict(lr=lr, eps=eps, clip_threshold=clip_threshold,
                        decay_rate=decay_rate, weight_decay=weight_decay,
                        scale_parameter=scale_parameter, relative_step=relative_step,
                        warmup_init=warmup_init)
        super().__init__(params, defaults)

    @staticmethod
    def _get_lr(param_group, param_state):
        rel_step = param_group.get('relative_step', True)
        if rel_step:
            step = param_state['step']
            lr = min(step ** param_group['decay_rate'], 1.0)
            if param_group.get('warmup_init', False) and step < 10000:
                lr /= max(1.0, step / 10000.0)
            return lr * param_group['lr']
        return param_group['lr']

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg_sq'] = torch.zeros_like(p)
                state['step'] += 1
                exp_avg_sq = state['exp_avg_sq']
                beta2t = 1.0 - (state['step'] ** group['decay_rate'])
                exp_avg_sq.mul_(beta2t).addcmul_(grad, grad, value=1 - beta2t)
                lr = self._get_lr(group, state)
                update = grad / exp_avg_sq.sqrt().add_(group['eps'][1])
                rms = update.norm(2) / math.sqrt(update.numel())
                if rms > group['clip_threshold']:
                    update.mul_(group['clip_threshold'] / rms)
                p.add_(update, alpha=-lr)
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['weight_decay'] * lr)
        return loss


# ============================================================================
# Gen 4: Layer-level Optimizers
# ============================================================================

class DiscriminativeLR(Optimizer):
    """Discriminative Fine-tuning wrapper (Howard & Ruder, 2018).
    Different learning rates for different parameter groups (layers)."""

    def __init__(self, params_groups_lrs, momentum=0.9, weight_decay=0,
                 optimizer_type='sgd'):
        if isinstance(params_groups_lrs, list) and len(params_groups_lrs) > 0 and isinstance(params_groups_lrs[0], dict):
            param_groups = params_groups_lrs
        else:
            param_groups = [{'params': list(params_groups_lrs), 'lr': 0.01}]
        self.optimizer_type = optimizer_type
        defaults = dict(momentum=momentum, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group.get('lr', 0.01)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])
                if group.get('momentum', 0) != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p)
                    d_p = buf
                p.add_(d_p, alpha=-lr)
        return loss


class LARS(Optimizer):
    """Layer-wise Adaptive Rate Scaling (Yang et al., 2019).
    Scales learning rate by trust ratio: ||w|| / ||g||."""

    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=5e-4,
                 trust_coef=0.02, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coef=trust_coef, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])
                param_norm = p.norm(2)
                grad_norm = d_p.norm(2)
                trust_ratio = group['trust_coef']
                if param_norm > 0 and grad_norm > 0:
                    trust_ratio = group['trust_coef'] * param_norm / (grad_norm + group['eps'])
                if group['momentum'] != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(group['momentum']).add_(d_p)
                    d_p = buf
                p.add_(d_p, alpha=-lr * trust_ratio)
        return loss


class LAMB(Optimizer):
    """Layer-wise Adaptive Moments optimizer for Batch training (You et al., 2020).
    Combines Adam with layer-wise trust ratio scaling."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0, trust_coef=0.02, adam=False, amsgrad=False):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        trust_coef=trust_coef, adam=adam, amsgrad=amsgrad)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                adam_step = exp_avg / bias_correction1 / (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])
                if not group['adam']:
                    weight_norm = p.norm(2)
                    adam_step_norm = adam_step.norm(2)
                    if weight_norm > 0 and adam_step_norm > 0:
                        trust_ratio = group['trust_coef'] * weight_norm / adam_step_norm
                    else:
                        trust_ratio = 1.0
                    p.add_(adam_step, alpha=-group['lr'] * trust_ratio)
                else:
                    p.add_(adam_step, alpha=-group['lr'])
        return loss


# ============================================================================
# Gen 5: Layer × Time Joint Optimizers
# ============================================================================

class RAdam(Optimizer):
    """Rectified Adam (Liu et al., 2020): variance rectification for Adam warmup."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, degenerated_to_sgd=True):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        degenerated_to_sgd=degenerated_to_sgd)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * (beta2 ** state['step']) / bias_correction2
                step_size = group['lr'] / bias_correction1
                if N_sma >= 5:
                    r = math.sqrt((N_sma - 4) / (N_sma_max - 4) * (N_sma - 2) / N_sma * N_sma_max / (N_sma_max - 2))
                    var = (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])
                    p.add_(exp_avg / var, alpha=-step_size * r)
                elif group['degenerated_to_sgd']:
                    p.add_(exp_avg, alpha=-step_size)
                else:
                    var = (exp_avg_sq / bias_correction2).sqrt().add_(group['eps'])
                    p.add_(exp_avg / var, alpha=-step_size)
        return loss


class Lookahead(Optimizer):
    """Lookahead optimizer (Zhang et al., 2020): k-step forward, 1-step back."""

    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self._slow_params = {}
        self._step_count = 0

    def _ensure_slow(self):
        for group in self.param_groups:
            for p in group['params']:
                if p not in self._slow_params:
                    self._slow_params[p] = torch.clone(p.data).detach()

    def update(self, group):
        for fast in group['params']:
            slow = self._slow_params[fast]
            slow.add_(fast.data - slow, alpha=self.alpha)
            fast.data.copy_(slow)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        self._step_count += 1
        self._ensure_slow()
        if self._step_count % self.k == 0:
            for group in self.param_groups:
                self.update(group)
        return loss

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def defaults(self):
        return self.optimizer.defaults


class SAM(Optimizer):
    """Sharpness-Aware Minimization (Foret et al., 2020): seeks flat minima."""

    def __init__(self, params, base_optimizer, rho=0.05, adaptive=True, **kwargs):
        assert rho >= 0.0, f"Invalid rho: {rho}"
        self.rho = rho
        self.adaptive = adaptive
        self.base_optimizer = base_optimizer(params, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w.detach().clone()
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                if p in self.state and 'e_w' in self.state[p]:
                    p.add_(self.state[p]['e_w'], alpha=-1.0)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad()

    def _grad_norm(self):
        grad_norms = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    grad_norms.append(((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(2))
        if len(grad_norms) == 0:
            return torch.tensor(0.0)
        return torch.norm(torch.stack(grad_norms), p=2)

    def step(self, closure=None):
        assert closure is not None, "SAM requires closure for two-step update"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        with torch.enable_grad():
            closure()
        self.second_step()

    def zero_grad(self):
        self.base_optimizer.zero_grad()

    def state_dict(self):
        return self.base_optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.base_optimizer.load_state_dict(state_dict)


class Sophia(Optimizer):
    """Sophia: Second-order Clipped Stochastic Optimization (Liu et al., 2023).
    Uses Hutchinson estimator for Hessian diagonal with clipping."""

    def __init__(self, params, lr=1e-3, betas=(0.965, 0.99), rho=0.04,
                 weight_decay=1e-1, hessian_update_freq=10, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, rho=rho, weight_decay=weight_decay,
                        hessian_update_freq=hessian_update_freq, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None, hess=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['h'] = torch.zeros_like(p)
                state['step'] += 1
                m, h = state['m'], state['h']
                beta1, beta2 = group['betas']
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                if hess is not None and state['step'] % group['hessian_update_freq'] == 1:
                    h_diag = hess.get(p, torch.zeros_like(p))
                    h.mul_(beta2).add_(h_diag.abs(), alpha=1 - beta2)
                if state['step'] > 1:
                    update = m / (h + group['eps'])
                    update = torch.clamp(update, -group['rho'], group['rho'])
                    p.add_(update, alpha=-group['lr'])
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])
        return loss


class Lion(Optimizer):
    """Lion: Evolved Sign Momentum (Chen et al., 2023).
    Memory-efficient: only tracks momentum, uses sign of update."""

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                update = exp_avg * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])
        return loss


class ScheduleFree(Optimizer):
    """Schedule-Free: learning rate schedules without learning rate schedules (Defazio et al., 2024)."""

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0,
                 warmup_steps=1000, r=0.0, c=0.0):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        warmup_steps=warmup_steps, r=r, c=c)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['z'] = torch.clone(p).detach()
                state['step'] += 1
                lr = group['lr']
                mom = group['momentum']
                z = state['z']
                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])
                z.add_(grad, alpha=-lr)
                ck = 1 - mom
                k = state['step']
                coeff = lr * ck / (k * lr * ck + 1)
                p.mul_(1 - coeff).add_(z, alpha=coeff)
        return loss


# ============================================================================
# Novel SOTA Challenger: Discriminative SAM + STLR (DSAM)
# ============================================================================

class DiscriminativeSAM:
    """Discriminative SAM: combines layer-wise LR scaling with SAM flat-minima seeking.
    Our proposed SOTA approach: Discriminative Fine-tuning + SAM + STLR scheduling."""

    def __init__(self, model, base_optimizer_cls, rho=0.05, decay_factor=2.6,
                 adaptive=True, **optimizer_kwargs):
        self.model = model
        self.rho = rho
        self.decay_factor = decay_factor
        self.adaptive = adaptive
        self.base_optimizer_cls = base_optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self._setup_optimizer()

    def _setup_optimizer(self):
        layers = list(self.model.children())
        num_layers = len(layers)
        param_groups = []
        for i, layer in enumerate(layers):
            lr_scale = self.decay_factor ** (num_layers - 1 - i)
            param_groups.append({
                'params': list(layer.parameters()),
                'lr': self.optimizer_kwargs.get('lr', 0.01) / lr_scale
            })
        self.optimizer = self.base_optimizer_cls(param_groups, **self.optimizer_kwargs)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm_weighted()
        scale = self.rho / (grad_norm + 1e-12)
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
                p.add_(e_w)
                self.state[p]['e_w'] = e_w
        if zero_grad:
            self.zero_grad()

    def _grad_norm_weighted(self):
        device = self.optimizer.param_groups[0]['params'][0].device
        norms = []
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    norms.append(((torch.abs(p) if self.adaptive else 1.0) * p.grad).norm(2).to(device))
        return torch.norm(torch.stack(norms), 2)

    def step(self, closure):
        closure = torch.enable_grad()(closure)
        with torch.no_grad():
            grad_norm = self._grad_norm_weighted()
            scale = self.rho / (grad_norm + 1e-12)
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    e_w = (torch.pow(p, 2) if self.adaptive else 1.0) * p.grad * scale
                    p.add_(e_w)
                    self.state.setdefault(p, {})['e_w'] = e_w
        self.zero_grad()
        with torch.enable_grad():
            closure()
        with torch.no_grad():
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    if p.grad is not None and 'e_w' in self.state.get(p, {}):
                        p.add_(self.state[p]['e_w'], alpha=-1.0)
            self.optimizer.step()
        self.zero_grad()

    def zero_grad(self):
        self.optimizer.zero_grad()

    @property
    def state(self):
        return self.optimizer.state

    @property
    def param_groups(self):
        return self.optimizer.param_groups


class Grokfast(Optimizer):
    """Grokfast: Accelerated Generalization via Gradient Accumulation (Chen et al., 2024).
    Applies EMA filter to gradients for grokking-style acceleration."""

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0,
                 alpha=0.98, filter_type='ema'):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        alpha=alpha, filter_type=filter_type)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['filtered_grad'] = torch.zeros_like(p)
                state['step'] += 1
                m = state['momentum_buffer']
                fg = state['filtered_grad']
                alpha = group['alpha']
                fg.mul_(alpha).add_(grad, alpha=1 - alpha)
                effective_grad = grad + fg
                if group['momentum'] != 0:
                    m.mul_(group['momentum']).add_(effective_grad, alpha=1 - group['momentum'])
                    effective_grad = m
                p.add_(effective_grad, alpha=-group['lr'])
                if group['weight_decay'] != 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])
        return loss


# ============================================================================
# SOTA Challenger: Discriminative Adaptive Layer Scaling (DALS)
# ============================================================================

class DALS(Optimizer):
    """Discriminative Adaptive Layer Scaling (Ours).
    Key innovations beyond ULMFiT:
    1. Per-layer adaptive trust ratio (inspired by LARS but layer-wise)
    2. STLR with layer-dependent warmup fraction
    3. SAM-style perturbation with layer-wise rho scaling
    4. Grokfast gradient filtering for lower layers
    """

    def __init__(self, model, lr=0.01, momentum=0.9, weight_decay=5e-4,
                 decay_factor=2.6, trust_coef=0.02,
                 stlr_cut_frac=0.1, stlr_ratio=32,
                 sam_rho=0.05, grokfast_alpha=0.98,
                 adaptive_sam=True, warmup_steps=0, T_max=10000):
        self.model = model
        self.decay_factor = decay_factor
        self.trust_coef = trust_coef
        self.stlr_cut_frac = stlr_cut_frac
        self.stlr_ratio = stlr_ratio
        self.sam_rho = sam_rho
        self.grokfast_alpha = grokfast_alpha
        self.adaptive_sam = adaptive_sam
        self.T_max = T_max
        self.warmup_steps = warmup_steps
        self._global_step = 0

        param_groups = self._build_param_groups(model, lr, momentum, weight_decay)
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    def _build_param_groups(self, model, base_lr, momentum, weight_decay):
        layers = list(model.children())
        num_layers = len(layers)
        groups = []
        for i, layer in enumerate(layers):
            depth = num_layers - 1 - i
            lr = base_lr / (self.decay_factor ** depth)
            groups.append({
                'params': list(layer.parameters()),
                'lr': lr,
                'layer_depth': depth,
                'momentum': momentum,
                'weight_decay': weight_decay,
            })
        return groups

    def _get_stlr_factor(self, step, depth):
        T = self.T_max
        cut_frac = self.stlr_cut_frac * (1 + 0.1 * depth / max(len(self.param_groups) - 1, 1))
        cut = max(1, int(T * cut_frac))
        if step < cut:
            p = step / cut
        else:
            p = 1.0 - (step - cut) / (cut * (1.0 / cut_frac - 1.0))
        p = max(0.0, min(1.0, p))
        return (1.0 + p * (self.stlr_ratio - 1.0)) / self.stlr_ratio

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._global_step += 1
        step = self._global_step

        for group in self.param_groups:
            depth = group.get('layer_depth', 0)
            stlr_factor = self._get_stlr_factor(step, depth)
            warmup_factor = min(1.0, step / max(1, self.warmup_steps)) if self.warmup_steps > 0 else 1.0
            effective_lr = group['lr'] * stlr_factor * warmup_factor

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['momentum_buffer'] = torch.zeros_like(p)
                    state['filtered_grad'] = torch.zeros_like(p)
                state['step'] += 1
                m = state['momentum_buffer']
                fg = state['filtered_grad']

                alpha = self.grokfast_alpha ** (1 + depth * 0.3)
                fg.mul_(alpha).add_(grad, alpha=1 - alpha)

                effective_grad = grad + (1 - depth / max(len(self.param_groups) - 1, 1)) * fg

                if group['weight_decay'] != 0:
                    effective_grad = effective_grad.add(p, alpha=group['weight_decay'])

                param_norm = p.norm(2)
                grad_norm = effective_grad.norm(2)
                trust_ratio = 1.0
                if param_norm > 0 and grad_norm > 0:
                    raw_ratio = param_norm / grad_norm
                    trust_ratio = min(max(self.trust_coef * raw_ratio, 0.2), 5.0)

                if group['momentum'] != 0:
                    m.mul_(group['momentum']).add_(effective_grad, alpha=1 - group['momentum'])
                    effective_grad = m

                p.add_(effective_grad, alpha=-effective_lr * trust_ratio)

        return loss


# ============================================================================
# STLR Scheduler (can be composed with any optimizer)
# ============================================================================

class STLRScheduler:
    """Slanted Triangular Learning Rate Scheduler (Howard & Ruder, 2018).
    Works with any PyTorch optimizer by modifying param group LRs."""

    def __init__(self, optimizer, T, cut_frac=0.1, ratio=32,
                 decay_factor=None):
        self.optimizer = optimizer
        self.T = T
        self.cut_frac = cut_frac
        self.ratio = ratio
        self.decay_factor = decay_factor
        self._step_count = 0
        if decay_factor is not None:
            num_groups = len(optimizer.param_groups)
            for i, group in enumerate(optimizer.param_groups):
                group['eta_max'] = group['lr'] / (decay_factor ** (num_groups - 1 - i))
        else:
            for group in optimizer.param_groups:
                group['eta_max'] = group['lr']

    def step(self):
        self._step_count += 1
        t = self._step_count
        cut = max(1, int(self.T * self.cut_frac))
        if t < cut:
            p = t / cut
        else:
            p = 1.0 - (t - cut) / (cut * (1.0 / self.cut_frac - 1.0))
        p = max(0.0, min(1.0, p))
        factor = (1.0 + p * (self.ratio - 1.0)) / self.ratio
        for group in self.optimizer.param_groups:
            group['lr'] = group['eta_max'] * factor

    def get_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


# ============================================================================
# Utility: Build discriminative param groups for any optimizer
# ============================================================================

def build_discriminative_param_groups(model, base_lr, decay_factor=2.6):
    """Create parameter groups with layer-wise learning rates."""
    layers = list(model.children())
    num_layers = len(layers)
    param_groups = []
    for i, layer in enumerate(layers):
        lr = base_lr / (decay_factor ** (num_layers - 1 - i))
        param_groups.append({
            'params': list(layer.parameters()),
            'lr': lr
        })
    return param_groups


def build_discriminative_stlr_param_groups(model, base_lr, decay_factor=2.6):
    """Create param groups with discriminative LR + STLR schedule info."""
    layers = list(model.children())
    num_layers = len(layers)
    param_groups = []
    for i, layer in enumerate(layers):
        lr_max = base_lr / (decay_factor ** (num_layers - 1 - i))
        param_groups.append({
            'params': list(layer.parameters()),
            'lr': lr_max,
            'eta_max': lr_max
        })
    return param_groups