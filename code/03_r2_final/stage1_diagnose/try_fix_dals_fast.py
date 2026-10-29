"""DALS-Fast 死值深度定位 + 多轮参数化修复尝试。

流程：
  - 严格参照 run_final.py 的 build_model / build_loaders / train_one 逻辑；
  - 通过 build_opt 的 DALS_Fast 分支，加一个 monkey-patch 以便注入待测超参；
  - 每条候选跑 mrpc seed=42（epoch=5），记录 acc/f1/best_val_epoch；
  - 判定：「数值正常」= acc 显著高于多数类基线 68.38 且 best_val_epoch > 1（即不止 epoch 1 是最优）；
  - 全部记录到 logs/try_fix_dals_fast_log.jsonl 与 results/try_fix_*.json。

用法：
  python code/try_fix_dals_fast.py --round 1
"""
import os
import sys
import json
import argparse
import time

HERE = os.path.dirname(os.path.abspath(__file__))
AUDIT = os.path.dirname(HERE)            # 返修材料包/
CORE = os.path.join(AUDIT, 'core')
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(AUDIT, 'lib'))

ROOT = os.path.abspath(os.path.join(AUDIT, '..', '..'))
BENCH = ROOT
os.environ['MODEL_DIR'] = os.path.join(BENCH, 'models')
os.environ['DATA_DIR'] = os.path.join(BENCH, 'datasets')
os.environ['HF_DATASETS_CACHE'] = os.path.join(BENCH, 'datasets', 'huggingface')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['HF_DATASETS_OFFLINE'] = '1'

import torch
import torch.nn as nn
import numpy as np
import data
import train as T
import optimizers as opt
from run_final import build_model, build_loaders, CONFIGS
from logger import ExperimentLogger


# 多轮候选配置：每轮只改一个变量，便于隔离根因。
CANDIDATES = {
    1: {  # momentum 0.85 -> 0.9（与 base DALS 一致）
        'name': 'c1_momentum_0.9',
        'description': 'momentum 0.85→0.9',
        'momentum': 0.9, 'warmup_frac': 0.02, 'trust_coef': 0.02,
        'grokfast_alpha': 0.6, 'phase1_thresh': 0.01, 'phase2_thresh': 0.002,
        'fast_phase0_bypass': False, 'weight_decay': 1e-4,
    },
    2: {  # warmup 0.02 -> 0.05（与 base DALS 一致）
        'name': 'c2_warmup_0.05',
        'description': 'warmup 0.02→0.05',
        'momentum': 0.85, 'warmup_frac': 0.05, 'trust_coef': 0.02,
        'grokfast_alpha': 0.6, 'phase1_thresh': 0.01, 'phase2_thresh': 0.002,
        'fast_phase0_bypass': False, 'weight_decay': 1e-4,
    },
    3: {  # momentum + warmup 同时调整
        'name': 'c3_mom0.9_warm0.05',
        'description': 'momentum 0.85→0.9 + warmup 0.02→0.05',
        'momentum': 0.9, 'warmup_frac': 0.05, 'trust_coef': 0.02,
        'grokfast_alpha': 0.6, 'phase1_thresh': 0.01, 'phase2_thresh': 0.002,
        'fast_phase0_bypass': False, 'weight_decay': 1e-4,
    },
    4: {  # momentum + warmup + 放大 phase 阈值（让 phase 检测更保守）
        'name': 'c4_phase_thresh_loose',
        'description': 'momentum 0.9 + warmup 0.05 + phase 阈值×3（更保守）',
        'momentum': 0.9, 'warmup_frac': 0.05, 'trust_coef': 0.02,
        'grokfast_alpha': 0.6, 'phase1_thresh': 0.03, 'phase2_thresh': 0.006,
        'fast_phase0_bypass': False, 'weight_decay': 1e-4,
    },
    5: {  # 关闭 trust ratio（排除 trust_ratio 失稳因素）
        'name': 'c5_no_trust',
        'description': 'momentum 0.9 + warmup 0.05 + trust_ratio 关闭',
        'momentum': 0.9, 'warmup_frac': 0.05, 'trust_coef': 0.02,
        'grokfast_alpha': 0.6, 'phase1_thresh': 0.01, 'phase2_thresh': 0.002,
        'fast_phase0_bypass': False, 'weight_decay': 1e-4,
        'variant': 'no_trust',  # 走预设表，关闭 use_trust
    },
    6: {  # 进一步放大 momentum（往 SGD 经典 0.99 走）
        'name': 'c6_mom0.95',
        'description': 'momentum 0.85→0.95 + warmup 0.05',
        'momentum': 0.95, 'warmup_frac': 0.05, 'trust_coef': 0.02,
        'grokfast_alpha': 0.6, 'phase1_thresh': 0.01, 'phase2_thresh': 0.002,
        'fast_phase0_bypass': False, 'weight_decay': 1e-4,
    },
    7: {  # 关闭 Grokfast 滤波（filter）排除 EMA 失稳因素
        'name': 'c7_no_filter',
        'description': 'momentum 0.9 + warmup 0.05 + 关闭 filter (Grokfast)',
        'momentum': 0.9, 'warmup_frac': 0.05, 'trust_coef': 0.02,
        'grokfast_alpha': 0.6, 'phase1_thresh': 0.01, 'phase2_thresh': 0.002,
        'fast_phase0_bypass': False, 'weight_decay': 1e-4,
        'variant': 'no_filter',
    },
    8: {  # 关闭 phase 检测（保留 cosine/trust/filter/depth）
        'name': 'c8_no_phase',
        'description': 'momentum 0.9 + warmup 0.05 + 关闭 phase 检测',
        'momentum': 0.9, 'warmup_frac': 0.05, 'trust_coef': 0.02,
        'grokfast_alpha': 0.6, 'phase1_thresh': 0.01, 'phase2_thresh': 0.002,
        'fast_phase0_bypass': False, 'weight_decay': 1e-4,
        'variant': 'phase_only',  # 保留 filter/cosine/depth，关闭 phase
    },
    9: {  # 提升 lr（fair_tuning 提示 DALS 在 4~6× 才起飞）
        'name': 'c9_lr4x',
        'description': 'momentum 0.9 + warmup 0.05 + lr×4 (=2e-3)',
        'momentum': 0.9, 'warmup_frac': 0.05, 'trust_coef': 0.02,
        'grokfast_alpha': 0.6, 'phase1_thresh': 0.01, 'phase2_thresh': 0.002,
        'fast_phase0_bypass': False, 'weight_decay': 1e-4,
        'lr_scale': 4,
    },
    10: {  # 提升 lr + 放大 momentum
        'name': 'c10_lr4x_mom0.9',
        'description': 'momentum 0.9 + warmup 0.05 + lr×4 + 关闭 phase',
        'momentum': 0.9, 'warmup_frac': 0.05, 'trust_coef': 0.02,
        'grokfast_alpha': 0.6, 'phase1_thresh': 0.01, 'phase2_thresh': 0.002,
        'fast_phase0_bypass': False, 'weight_decay': 1e-4,
        'lr_scale': 4,
        'variant': 'phase_only',
    },
    11: {  # lr×4 + 关闭 fast_phase0_bypass + momentum 0.9
        'name': 'c11_lr4x_phase0_bypass_off',
        'description': 'lr×4 + momentum 0.9 + warmup 0.05 + phase0 bypass OFF',
        'momentum': 0.9, 'warmup_frac': 0.05, 'trust_coef': 0.02,
        'grokfast_alpha': 0.6, 'phase1_thresh': 0.01, 'phase2_thresh': 0.002,
        'fast_phase0_bypass': False, 'weight_decay': 1e-4,
        'lr_scale': 4,
    },
    12: {  # lr×2 + 各种稳健化
        'name': 'c12_lr2x',
        'description': 'lr×2 + momentum 0.9 + warmup 0.05',
        'momentum': 0.9, 'warmup_frac': 0.05, 'trust_coef': 0.02,
        'grokfast_alpha': 0.6, 'phase1_thresh': 0.01, 'phase2_thresh': 0.002,
        'fast_phase0_bypass': False, 'weight_decay': 1e-4,
        'lr_scale': 2,
    },
}


def run_one_with_cfg(ds, strategy, seed, device, cfg_overrides):
    """对单个 (dataset, strategy, seed) 用自定义 DALS-Fast 超参跑一次。"""
    cfg = next(c for c in CONFIGS if c['ds'] == ds)
    data.set_seed(seed)
    (tr, va, te), nl = build_loaders(cfg, seed)
    total_steps = len(tr) * cfg['ep']
    model = build_model(cfg, nl, device)
    lr_sgd, lr_adam = cfg['lr_sgd'], cfg['lr_adam']
    dlr = 0.1 if lr_sgd > 5e-4 else lr_sgd
    if 'lr_scale' in cfg_overrides:
        dlr = dlr * cfg_overrides['lr_scale']

    # 用 DALS_Fast 自定义超参（绕过 build_opt 的硬编码）
    if strategy == 'DALS_Fast':
        o = opt.DALSFast(model, lr=dlr,
                         momentum=cfg_overrides['momentum'],
                         weight_decay=cfg_overrides['weight_decay'],
                         trust_coef=cfg_overrides['trust_coef'],
                         grokfast_alpha=cfg_overrides['grokfast_alpha'],
                         warmup_frac=cfg_overrides['warmup_frac'],
                         T_max=total_steps,
                         phase1_thresh=cfg_overrides['phase1_thresh'],
                         phase2_thresh=cfg_overrides['phase2_thresh'],
                         variant=cfg_overrides.get('variant', 'full'),
                         fast_phase0_bypass=cfg_overrides['fast_phase0_bypass'])
        sched = None
        phase_cb = o.update_phase
    else:
        raise ValueError(strategy)

    r = T.train_one(model, o, tr, va, te, device, n_labels=nl,
                    epochs=cfg['ep'], sched=sched, dals_phase_update=phase_cb,
                    select_by='val', verbose=0)
    res = {'test_acc_valsel': float(r['test_acc_valsel']),
           'test_f1_valsel': float(r['test_f1_valsel']),
           'best_val_epoch': int(r['best_val_epoch']),
           'best_val_acc': float(r.get('best_val_acc', -1))}
    del model, o, sched, phase_cb
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return res


def judge_pass(res, majority_baseline=68.38):
    """判定这次跑是否成功（数值非死值）。"""
    acc = res['test_acc_valsel']
    bv_ep = res['best_val_epoch']
    f1 = res['test_f1_valsel']
    # 判定准则：acc 比多数类高至少 2 点 OR acc >= 70，且 best_val_epoch > 1
    above_majority = acc > majority_baseline + 2.0
    healthy_early = bv_ep > 1 and acc >= 70
    return {
        'is_alive': above_majority or healthy_early,
        'above_majority_2pt': above_majority,
        'healthy_train': healthy_early,
        'majority_baseline': majority_baseline,
        'note': ('非死值，可继续 5-seed 验证' if (above_majority or healthy_early)
                 else '仍坍缩到多数类，需继续调参'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, required=True, choices=list(CANDIDATES.keys()),
                    help='候选编号 1~12')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--ds', default='mrpc')
    args = ap.parse_args()

    device = torch.device('cuda') if args.device == 'cuda' else torch.device('cpu')
    cfg_ov = CANDIDATES[args.round]
    logger = ExperimentLogger(os.path.join(AUDIT, 'logs'),
                              run_name='try_fix_dals_fast')

    step_id = f"tryfix|{args.ds}|DALS_Fast|r{args.round}_{cfg_ov['name']}|seed{args.seed}"
    if logger.is_done(step_id):
        print(f'  [skip] {step_id}  (已完成)')
        # 读出已有结果
        rec = json.load(open(os.path.join(AUDIT, 'results', f'{step_id}.json'),
                             encoding='utf-8'))
        print(f"      acc={rec['output']['test_acc_valsel']:.4f}  "
              f"f1={rec['output']['test_f1_valsel']:.4f}  "
              f"best_ep={rec['output']['best_val_epoch']}  "
              f"verdict={rec['output'].get('verdict', '?')}")
        return

    print(f'=== 轮次 {args.round}: {cfg_ov["description"]} ===')
    print(f'    配置: {cfg_ov}')
    t0 = time.time()
    res = run_one_with_cfg(args.ds, 'DALS_Fast', args.seed, device, cfg_ov)
    elapsed = time.time() - t0
    verdict = judge_pass(res)
    out = {**res, 'elapsed_sec': round(elapsed, 1),
           'cfg_overrides': cfg_ov, 'verdict': verdict}

    inp = {'dataset': args.ds, 'strategy': 'DALS_Fast', 'seed': args.seed,
           'round': args.round, 'candidate_name': cfg_ov['name']}
    logger.log(step_id, cfg_ov['description'], inp, out, 'success')
    print(f'\n>>> 结果: acc={res["test_acc_valsel"]:.4f}  f1={res["test_f1_valsel"]:.4f}  '
          f'best_val_epoch={res["best_val_epoch"]}  '
          f'verdict={verdict["note"]}')
    print(f'    耗时 {elapsed:.1f}s')


if __name__ == '__main__':
    main()