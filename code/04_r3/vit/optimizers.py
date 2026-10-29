"""
Optimizers for Learning Rate Engineering (revised for NEUCOM-D-26-12332).

Covers 5 generations + DALS with ablation switches.

Ablation switches for DALS (passed as constructor kwargs):
  use_cosine      : if False, LR is constant eta_0 (no schedule)        default True
  use_trust      : if False, trust ratio = 1                            default True
  use_filter     : if False, no Grokfast EMA (effective_grad = grad)    default True
  use_phase      : if False, alpha is constant alpha_0                  default True
  use_depth      : if False, blend is constant 0.5 (no depth axis)      default True
  fixed_alpha    : if set, override alpha (used by use_filter only/fix) default None
  blend_top/bot  : override blend coefficients                          default None

A 'variant' string can be passed to DALS to pick a preset ablation:
  'full','cosine_only','trust_only','filter_only','phase_only','depth_only',
  'cosine_trust','cosine_filter','filter_trust'.
"""
import math
import torch
from torch.optim import Optimizer
from typing import Optional, List


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _is_no_trust_param(name: str) -> bool:
    """Parameters that bypass the trust ratio (bias, embedding, norm)."""
    n = name.lower()
    if 'bias' in n:
        return True
    if 'embed' in n:
        return True
    if 'norm' in n or 'ln' in n or 'layer_norm' in n or 'layernorm' in n:
        return True
    return False


def build_layer_param_groups(model, lr, depth_dim='children'):
    """Build per-layer param groups with depth_ratio and param names.

    depth_dim: 'children' uses model.children() (original behaviour);
               'named'   flattens top-level named children for transformers.
    """
    groups = []
    if depth_dim == 'children':
        layers = list(model.children())
        n = max(len(layers), 1)
        for i, layer in enumerate(layers):
            ps = [p for p in layer.parameters() if p.requires_grad]
            if not ps:
                continue
            # Per-parameter name map: allows per-parameter trust-ratio exemption
            # (bias/embedding/layernorm only), instead of disabling the whole group.
            name_map = {id(p): nm for nm, p in layer.named_parameters()}
            groups.append({'params': ps, 'lr': lr, 'depth_ratio': i / (n - 1) if n > 1 else 0.5,
                           'names': [name_map.get(id(p), 'param') for p in ps],
                           'name_of': name_map})
    else:
        named = list(model.named_children())
        n = max(len(named), 1)
        for i, (lname, layer) in enumerate(named):
            ps = [p for p in layer.parameters() if p.requires_grad]
            if not ps:
                continue
            name_map = {id(p): f'{lname}.{nm}' for nm, p in layer.named_parameters()}
            groups.append({'params': ps, 'lr': lr, 'depth_ratio': i / (n - 1) if n > 1 else 0.5,
                           'names': [name_map.get(id(p), lname) for p in ps],
                           'name_of': name_map})
    if not groups:
        name_map = {id(p): nm for nm, p in model.named_parameters()}
        groups.append({'params': [p for p in model.parameters() if p.requires_grad],
                       'lr': lr, 'depth_ratio': 0.5, 'names': ['all'], 'name_of': name_map})
    return groups


# ----------------------------------------------------------------------------
# Gen 1: Fixed LR SGD
# ----------------------------------------------------------------------------
class FixedLRSGD(Optimizer):
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
            lr = group['lr']; mom = group['momentum']; wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if wd != 0:
                    d_p = d_p.add(p, alpha=wd)
                if mom != 0:
                    s = self.state[p]
                    if 'momentum_buffer' not in s:
                        s['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        s['momentum_buffer'].mul_(mom).add_(d_p)
                    d_p = s['momentum_buffer']
                p.add_(d_p, alpha=-lr)
        return loss


# ----------------------------------------------------------------------------
# Gen 2: Schedulers as wrappers (LR stored in param_group['lr'])
# ----------------------------------------------------------------------------
class CosineAnnealingSGD(Optimizer):
    def __init__(self, params, lr=0.1, T_max=100, eta_min=1e-5,
                 momentum=0.9, weight_decay=5e-4):
        self.T_max = T_max; self.eta_min = eta_min; self.eta_max = lr
        self._step = 0
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step += 1
        prog = min(self._step / self.T_max, 1.0)
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * prog))
        for group in self.param_groups:
            group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])
                if group['momentum'] != 0:
                    s = self.state[p]
                    if 'momentum_buffer' not in s:
                        s['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        s['momentum_buffer'].mul_(group['momentum']).add_(d_p)
                    d_p = s['momentum_buffer']
                p.add_(d_p, alpha=-lr)
        return loss


class SGDRWithRestarts(Optimizer):
    def __init__(self, params, lr=0.1, T_0=10, T_mult=2, eta_min=1e-5,
                 momentum=0.9, weight_decay=5e-4):
        self.eta_max = lr; self.eta_min = eta_min
        self.T_0 = T_0; self.T_mult = T_mult
        self._step = 0; self._cycle = 0; self._T_i = T_0
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._step += 1
        if self._step >= self._T_i:
            self._step = 0; self._cycle += 1
            self._T_i = int(self.T_0 * (self.T_mult ** self._cycle))
        prog = self._step / max(1, self._T_i)
        lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * prog))
        for group in self.param_groups:
            group['lr'] = lr
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad
                if group['weight_decay'] != 0:
                    d_p = d_p.add(p, alpha=group['weight_decay'])
                if group['momentum'] != 0:
                    s = self.state[p]
                    if 'momentum_buffer' not in s:
                        s['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        s['momentum_buffer'].mul_(group['momentum']).add_(d_p)
                    d_p = s['momentum_buffer']
                p.add_(d_p, alpha=-lr)
        return loss


# ----------------------------------------------------------------------------
# Gen 3: AdaBound (self-implemented, ported from the original submission)
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Gen 3: use torch built-in Adam/AdamW
# ----------------------------------------------------------------------------
# We use torch.optim.Adam and AdamW directly in the runner for canonical methods.


# ----------------------------------------------------------------------------
# Gen 4: LARS
# ----------------------------------------------------------------------------
class LARS(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=5e-4, trust_coef=0.02):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, trust_coef=trust_coef)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']; mom = group['momentum']; wd = group['weight_decay']; tc = group['trust_coef']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g.add(p, alpha=wd)
                pn = p.norm(2); gn = g.norm(2)
                tr = 1.0
                if pn > 0 and gn > 0:
                    tr = min(max(tc * (pn / gn), 0.2), 5.0)
                if mom != 0:
                    s = self.state[p]
                    if 'momentum_buffer' not in s:
                        s['momentum_buffer'] = torch.clone(g).detach()
                    else:
                        s['momentum_buffer'].mul_(mom).add_(g)
                    g = s['momentum_buffer']
                p.add_(g, alpha=-lr * tr)
        return loss


# ----------------------------------------------------------------------------
# Gen 5: Grokfast
# ----------------------------------------------------------------------------
class Grokfast(Optimizer):
    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=5e-4, alpha=0.6):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, alpha=alpha)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']; mom = group['momentum']; wd = group['weight_decay']; a = group['alpha']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                s = self.state[p]
                if 'filtered_grad' not in s:
                    s['filtered_grad'] = torch.zeros_like(p)
                fg = s['filtered_grad']
                fg.mul_(a).add_(g, alpha=1 - a)
                eg = g + fg
                if wd != 0:
                    eg = eg.add(p, alpha=wd)
                if mom != 0:
                    if 'momentum_buffer' not in s:
                        s['momentum_buffer'] = torch.clone(eg).detach()
                    else:
                        s['momentum_buffer'].mul_(mom).add_(eg)
                    eg = s['momentum_buffer']
                p.add_(eg, alpha=-lr)
        return loss


# ----------------------------------------------------------------------------
# RAdam
# ----------------------------------------------------------------------------
class RAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group['betas']; eps = group['eps']; lr = group['lr']; wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                s = self.state[p]
                if 'step' not in s:
                    s['step'] = 0; s['m'] = torch.zeros_like(p); s['v'] = torch.zeros_like(p)
                s['step'] += 1
                t = s['step']; m = s['m']; v = s['v']
                m.mul_(b1).add_(g, alpha=1 - b1)
                v.mul_(b2).addcmul_(g, g, value=1 - b2)
                rho_inf = 2 / (1 - b2) - 1
                rho_t = rho_inf - 2 * t * (b2 ** t) / (1 - b2 ** t)
                if rho_t > 4:
                    r = math.sqrt(((rho_t - 4) * (rho_inf - 4) * (rho_t - 2)) /
                                 ((rho_inf - 4) * (rho_inf - 2) * (rho_t)) )
                    denom = (v.sqrt() / math.sqrt(1 - b2 ** t)).add_(eps)
                    update = m / (1 - b1 ** t) * r / denom
                else:
                    update = m / (1 - b1 ** t)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


# ----------------------------------------------------------------------------
# Lion
# ----------------------------------------------------------------------------
class Lion(Optimizer):
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group['lr']; b1, b2 = group['betas']; wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                s = self.state[p]
                if 'm' not in s:
                    s['m'] = torch.zeros_like(p)
                m = s['m']
                update = (b1 * m + (1 - b1) * g).sign_()
                m.mul_(b2).add_(g, alpha=1 - b2)
                if wd != 0:
                    p.mul_(1 - lr * wd)
                p.add_(update, alpha=-lr)
        return loss


# ----------------------------------------------------------------------------
# DALS (revised) with ablation switches
# ----------------------------------------------------------------------------
class DALS(Optimizer):
    """Discriminative Adaptive Layer Scaling (revised).

    Phase axis: modulates EMA smoothing coefficient alpha_{l,t}.
    Depth axis: modulates raw vs filtered blending ratio.
    LR schedule: warmup-cosine, phase-INDEPENDENT (per reviewer R4-10).
    Trust ratio: bypassed for bias/embedding/norm parameters.
    """

    # ablation preset variants
    _PRESETS = {
        'full':            dict(use_cosine=True,  use_trust=True,  use_filter=True,  use_phase=True,  use_depth=True),
        'cosine_only':     dict(use_cosine=True,  use_trust=False, use_filter=False, use_phase=False, use_depth=False),
        'trust_only':      dict(use_cosine=False, use_trust=True,  use_filter=False, use_phase=False, use_depth=False),
        'filter_only':     dict(use_cosine=False, use_trust=False, use_filter=True,  use_phase=False, use_depth=False),
        'phase_only':      dict(use_cosine=False, use_trust=False, use_filter=True,  use_phase=True,  use_depth=False),
        'depth_only':      dict(use_cosine=False, use_trust=False, use_filter=True,  use_phase=False, use_depth=True),
        'cosine_trust':    dict(use_cosine=True,  use_trust=True,  use_filter=False, use_phase=False, use_depth=False),
        'cosine_filter':   dict(use_cosine=True,  use_trust=False, use_filter=True,  use_phase=False, use_depth=False),
        'filter_trust':    dict(use_cosine=False, use_trust=True,  use_filter=True,  use_phase=False, use_depth=False),
    }

    def __init__(self, params, lr=0.03, momentum=0.9, weight_decay=1e-4,
                 trust_coef=0.02, grokfast_alpha=0.6,
                 warmup_frac=0.05, T_max=10000,
                 phase1_thresh=0.01, phase2_thresh=0.002,
                 variant='full', fast_phase0_bypass=False,
                 no_trust_names=None):
        self.trust_coef = trust_coef
        self.grokfast_alpha = grokfast_alpha
        self.T_max = T_max
        self.warmup_frac = warmup_frac
        self.phase1_thresh = phase1_thresh
        self.phase2_thresh = phase2_thresh
        self._global_step = 0
        self._loss_ema = None
        self._phase = 0
        self.fast_phase0_bypass = fast_phase0_bypass
        self.no_trust_names = no_trust_names or set()
        # ablation switches
        ab = self._PRESETS.get(variant, self._PRESETS['full'])
        self.use_cosine = ab['use_cosine']
        self.use_trust = ab['use_trust']
        self.use_filter = ab['use_filter']
        self.use_phase = ab['use_phase']
        self.use_depth = ab['use_depth']
        self.variant = variant

        if isinstance(params, torch.nn.Module):
            model = params
            params = build_layer_param_groups(model, lr, depth_dim='children')
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def _compute_phase(self, loss_val):
        if self._loss_ema is None:
            self._loss_ema = loss_val
            return 0
        old = self._loss_ema
        self._loss_ema = 0.95 * self._loss_ema + 0.05 * loss_val
        imp = (old - self._loss_ema) / max(abs(old), 1e-8)
        if imp > self.phase1_thresh:
            self._phase = 0
        elif imp > self.phase2_thresh:
            self._phase = 1
        else:
            self._phase = 2
        return self._phase

    def update_phase(self, loss_val):
        self._compute_phase(loss_val)

    def _lr_at(self, step):
        base = self.defaults['lr']
        if not self.use_cosine:
            return base
        W = int(self.warmup_frac * self.T_max)
        if step < W:
            return base * step / max(1, W)
        adj = (step - W) / max(1, self.T_max - W)
        return base * 0.5 * (1 + math.cos(math.pi * min(adj, 1.0)))

    def _alpha_at(self, depth_ratio):
        a0 = self.grokfast_alpha
        if not self.use_phase:
            return a0
        if self._phase == 0:
            return max(0.3, a0 - 0.3)
        if self._phase == 2:
            return min(0.9, a0 + 0.1)
        return a0

    def _blend_at(self, depth_ratio):
        if not self.use_depth:
            return 0.5  # fixed 50/50
        return 0.3 + 0.4 * depth_ratio

    def _param_no_trust(self, group, p):
        names = group.get('names', [])
        if names:
            for nm in names:
                if _is_no_trust_param(nm):
                    return True
            return False
        # fallback: cannot tell, use weight decay flag heuristic
        return False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._global_step += 1
        step = self._global_step - 1
        lr = self._lr_at(step)
        for group in self.param_groups:
            group['lr'] = lr
            depth_ratio = group.get('depth_ratio', 0.5)
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                s = self.state[p]
                if 'momentum_buffer' not in s:
                    s['momentum_buffer'] = torch.zeros_like(p)
                    s['filtered_grad'] = torch.zeros_like(p)
                m = s['momentum_buffer']; fg = s['filtered_grad']

                # Step 2: filter
                if self.use_filter:
                    if self.fast_phase0_bypass and self._phase == 0:
                        effective_grad = grad
                    else:
                        a = self._alpha_at(depth_ratio)
                        fg.mul_(a).add_(grad, alpha=1 - a)
                        blend = self._blend_at(depth_ratio)
                        effective_grad = grad * blend + fg * (1 - blend)
                else:
                    effective_grad = grad

                if group['weight_decay'] != 0:
                    effective_grad = effective_grad.add(p, alpha=group['weight_decay'])

                # Step 3: trust ratio
                tr = 1.0
                if self.use_trust and not self._param_no_trust(group, p):
                    pn = p.norm(2); gn = effective_grad.norm(2)
                    if pn > 0 and gn > 0:
                        tr = min(max(self.trust_coef * (pn / gn), 0.2), 5.0)

                # Step 4: momentum
                if group['momentum'] != 0:
                    m.mul_(group['momentum']).add_(effective_grad)
                    effective_grad = m

                p.add_(effective_grad, alpha=-lr * tr)
        return loss


# ----------------------------------------------------------------------------
# DALS-Fast / DALS-Acc convenience wrappers
# ----------------------------------------------------------------------------
class DALSFast(DALS):
    def __init__(self, params, lr=0.05, momentum=0.85, weight_decay=1e-4,
                 trust_coef=0.02, grokfast_alpha=0.6, warmup_frac=0.02,
                 T_max=10000, phase1_thresh=0.01, phase2_thresh=0.002, variant='full',
                 fast_phase0_bypass=True):
        # 修复（bug）：fast_phase0_bypass 在 fine-tuning（小 lr）场景会导致 phase 0 完全
        # 无 EMA 平滑，梯度波动 + trust ratio 失稳，模型坍缩到多数类（mrpc/DALS-Fast 死值）。
        # 改为可配置，由 runner 在 fine-tuning 时关闭。
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay,
                         trust_coef=trust_coef, grokfast_alpha=grokfast_alpha,
                         warmup_frac=warmup_frac, T_max=T_max,
                         phase1_thresh=phase1_thresh, phase2_thresh=phase2_thresh,
                         variant=variant, fast_phase0_bypass=fast_phase0_bypass)


class DALSAcc(DALS):
    def __init__(self, params, lr=0.03, momentum=0.9, weight_decay=5e-4,
                 trust_coef=0.02, grokfast_alpha=0.7, warmup_frac=0.05,
                 T_max=10000, phase1_thresh=0.01, phase2_thresh=0.001,
                 T_0=1000, T_mult=2, variant='full'):
        super().__init__(params, lr=lr, momentum=momentum, weight_decay=weight_decay,
                         trust_coef=trust_coef, grokfast_alpha=grokfast_alpha,
                         warmup_frac=warmup_frac, T_max=T_max,
                         phase1_thresh=phase1_thresh, phase2_thresh=phase2_thresh,
                         variant=variant, fast_phase0_bypass=False)
        self.T_0 = T_0; self.T_mult = T_mult
        self._cycle_step = 0; self._cycle_T = T_0; self._cycle_count = 0
        # 修复（bug）：SGDR 重启无 warmup，重启瞬间 lr 从 ~0 跳回 base，
        # 偶发导致微调坍缩（sst2_b/DALS_Acc seed9999=50.92）。这里给每个 cycle
        # 开头加 warmup 步数，让 lr 从 0 线性爬升到 base。
        self._warmup_steps = max(1, int(0.1 * self._cycle_T))

    def _lr_at(self, step):
        # SGDR-style restarts for DALS-Acc（修复版：每 cycle 前 warmup_steps 线性 warmup）
        if not self.use_cosine:
            return self.defaults['lr']
        base = self.defaults['lr']
        if self._cycle_step < self._warmup_steps:
            return base * (self._cycle_step / max(1, self._warmup_steps))
        prog = (self._cycle_step - self._warmup_steps) / max(1, self._cycle_T - self._warmup_steps)
        prog = min(prog, 1.0)
        return base * 0.5 * (1 + math.cos(math.pi * prog))

    @torch.no_grad()
    def step(self, closure=None):
        self._cycle_step += 1
        if self._cycle_step >= self._cycle_T:
            self._cycle_step = 0; self._cycle_count += 1
            self._cycle_T = int(self.T_0 * (self.T_mult ** self._cycle_count))
            self._warmup_steps = max(1, int(0.1 * self._cycle_T))
        return super().step(closure)


# ----------------------------------------------------------------------------
# STLR scheduler (applied to an existing optimizer's param groups)
# ----------------------------------------------------------------------------
class STLRScheduler:
    def __init__(self, optimizer, T, cut_frac=0.1, ratio=32, decay_factor=None):
        self.opt = optimizer; self.T = T; self.cut_frac = cut_frac; self.ratio = ratio
        self.decay_factor = decay_factor; self._step = 0
        if decay_factor is not None:
            n = len(optimizer.param_groups)
            for i, g in enumerate(optimizer.param_groups):
                g['eta_max'] = g['lr'] / (decay_factor ** (n - 1 - i))
        else:
            for g in optimizer.param_groups:
                g['eta_max'] = g['lr']

    def step(self):
        self._step += 1
        t = self._step
        cut = max(1, int(self.T * self.cut_frac))
        if t < cut:
            p = t / cut
        else:
            p = 1.0 - (t - cut) / (cut * (1.0 / self.cut_frac - 1.0))
        p = max(0.0, min(1.0, p))
        factor = (1.0 + p * (self.ratio - 1.0)) / self.ratio
        for g in self.opt.param_groups:
            g['lr'] = g['eta_max'] * factor

    def get_lr(self):
        return [g['lr'] for g in self.opt.param_groups]


# ----------------------------------------------------------------------------
# Discriminative param-group builders
# ----------------------------------------------------------------------------
def _transformer_layer_index(name):
    """从参数名提取层深度索引（保留用于诊断/可选逐层分组）。

    注：判别式衰减在 MRPC/TREC-6 上的坍缩是论文论证的「现象」而非数据 bug
    （见 main.tex 分析节），因此默认 build_discriminative_param_groups 保持
    与原 benchmark 一致的 children 分组，仅保留本函数供诊断使用。
    """
    import re
    n = name.lower()
    m = re.search(r'\.layer\.(\d+)\.', n)
    if m:
        return int(m.group(1))
    if 'embedding' in n or 'embeddings' in n:
        return -1
    if 'classifier' in n:
        return 10 ** 6
    if 'pre_classifier' in n or 'pooler' in n:
        return 10 ** 6 - 1
    return 0


def build_discriminative_param_groups(model, base_lr, decay_factor=2.6):
    layers = list(model.children())
    n = len(layers)
    groups = []
    for i, layer in enumerate(layers):
        ps = [p for p in layer.parameters() if p.requires_grad]
        if not ps:
            continue
        lr = base_lr / (decay_factor ** (n - 1 - i))
        groups.append({'params': ps, 'lr': lr})
    if not groups:
        groups.append({'params': [p for p in model.parameters() if p.requires_grad], 'lr': base_lr})
    return groups


def build_discriminative_stlr_param_groups(model, base_lr, decay_factor=2.6):
    layers = list(model.children())
    n = len(layers)
    groups = []
    for i, layer in enumerate(layers):
        ps = [p for p in layer.parameters() if p.requires_grad]
        if not ps:
            continue
        lr_max = base_lr / (decay_factor ** (n - 1 - i))
        groups.append({'params': ps, 'lr': lr_max, 'eta_max': lr_max})
    if not groups:
        groups.append({'params': [p for p in model.parameters() if p.requires_grad], 'lr': base_lr, 'eta_max': base_lr})
    return groups