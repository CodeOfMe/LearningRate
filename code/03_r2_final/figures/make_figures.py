"""生成返修材料可视化图表（figures/），基于 5-seed 最终数据。

图 1：MRPC 等预算公平调参最优对比（bar，DALS 家族高亮，含大 lr 网格）
图 2：可调优性 span（worst→best 范围），三类画像标注
图 3：敏捷性（from-scratch 达阈值 epoch，5-seed mean±std）
"""
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(BASE, 'results')
FIG = os.path.join(BASE, 'figures')

DALS_COLORS = {'DALS': '#d62728', 'DALS_Fast': '#ff7f0e', 'DALS_Acc': '#2ca02c'}
DISC = {'Discriminative', 'STLR_Disc', 'SAM_Disc'}


def fig1_best_tuned():
    rows = json.load(open(os.path.join(RES, 'fair_tuning_merged.json')))
    mrpc = sorted([r for r in rows if r['dataset'] == 'mrpc'], key=lambda r: -r['best_acc'])
    names = [r['strategy'] for r in mrpc]
    vals = [r['best_acc'] for r in mrpc]
    colors = [DALS_COLORS.get(n, '#1f77b4') for n in names]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y = np.arange(len(names))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Best-tuned accuracy on MRPC (%)')
    ax.set_title('Equal-budget tuning (lr grid {0.5,1,2,4,6}x, 5 seeds)')
    for yi, v in zip(y, vals):
        ax.text(v + 0.1, yi, f'{v:.1f}', va='center', fontsize=7)
    ax.axvline(68.38, ls='--', color='gray', lw=0.8)
    ax.text(68.5, -0.6, 'majority class 68.38', fontsize=7, color='gray')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig1_fair_tuning_best.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIG, 'fig1_fair_tuning_best.png'), dpi=150)
    plt.close()


def fig2_tunability():
    rows = json.load(open(os.path.join(RES, 'fair_tuning_merged.json')))
    mrpc = sorted([r for r in rows if r['dataset'] == 'mrpc'], key=lambda r: -r['span'])
    names = [r['strategy'] for r in mrpc]
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y = np.arange(len(names))
    for yi, r in zip(y, mrpc):
        c = DALS_COLORS.get(r['strategy'], 'gray' if r['strategy'] in DISC else '#1f77b4')
        ax.plot([r['worst_acc'], r['best_acc']], [yi, yi], color=c, lw=2, marker='|', ms=10)
        if r['typical_acc'] is not None:
            ax.plot([r['typical_acc']], [yi], 'o', color='black', ms=4)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('Accuracy range over lr grid (%)')
    ax.set_title('Tunability: worst-to-best over learning rate (MRPC, 5 seeds)')
    ax.text(0.02, 0.98, 'dot = default lr; bar = worst to best', transform=ax.transAxes,
            va='top', fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig2_tunability_span.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIG, 'fig2_tunability_span.png'), dpi=150)
    plt.close()


def fig3_agility():
    rows = json.load(open(os.path.join(RES, 'agility.json')))
    # 全部 18 个策略（与正文 Table 2 一致）
    METHODS = ['Fixed_SGD', 'Cosine_SGD', 'SGDR', 'Adam', 'AdamW', 'AdaBound', 'LARS',
               'Discriminative', 'RAdam', 'Lion', 'Lookahead', 'SAM', 'Grokfast',
               'STLR_Disc', 'SAM_Disc', 'DALS', 'DALS_Fast', 'DALS_Acc']
    DALS_FAMILY = {'DALS', 'DALS_Fast', 'DALS_Acc'}
    fig, axes = plt.subplots(2, 1, figsize=(8, 8))
    for ax, ds, thresh, maxep in [
        (axes[0], 'synthetic', 95, None),
        (axes[1], 'cifar10', 80, 50),
    ]:
        dr = {r['strategy']: r for r in rows if r['dataset'] == ds}
        vals, errs, colors, labels = [], [], [], []
        for m in METHODS:
            r = dr.get(m)
            if r is None:
                vals.append(0); errs.append(0); colors.append('#eeeeee')
                labels.append(m + ' (n/a)')
                continue
            if r['reach_epoch_mean'] is not None:
                vals.append(r['reach_epoch_mean'])
                errs.append(r['reach_epoch_std'])
                colors.append(DALS_COLORS.get(m, '#1f77b4'))
                labels.append(m)
            else:
                vals.append(maxep)
                errs.append(0)
                colors.append('#cccccc')
                labels.append(m + ' (not reached)')
        x = np.arange(len(labels))
        ax.bar(x, vals, yerr=errs, color=colors, capsize=3, alpha=0.9)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5, rotation=60, ha='right')
        ax.set_ylabel(f'Epochs to reach {thresh}%')
        ax.set_title(f'{ds}: convergence speed (5 seeds)')
        if maxep is not None:
            ax.set_ylim(0, maxep * 1.15)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, 'fig3_agility_trajectory.pdf'), bbox_inches='tight')
    plt.savefig(os.path.join(FIG, 'fig3_agility_trajectory.png'), dpi=150)
    plt.close()
    print('figures 生成完成（5-seed 数据）')


if __name__ == '__main__':
    fig1_best_tuned()
    fig2_tunability()
    fig3_agility()
