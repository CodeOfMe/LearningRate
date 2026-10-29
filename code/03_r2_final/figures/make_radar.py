"""综合对比雷达图：敏捷性 × 精度 × 可调优性（8 方法 × 4 维度）。

4 个维度（归一化到 0~1，1=最好）：
  1. Synthetic agility （达到 95% epoch，越小越好）
  2. CIFAR-10 accuracy   （最终精度，越大越好）
  3. MRPC accuracy       （公平调参最优精度，越大越好）
  4. Tunability gain     （调参增益，越大越好）

方法：DALS-Fast, DALS-Acc, DALS, RAdam, Fixed_SGD, Cosine_SGD, SAM, SGDR
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, 'figures')
REPO = os.path.dirname(BASE)

# 读精度数据
fin = {}
for l in open(os.path.join(REPO, 'final data.jsonl')):
    r = json.loads(l)
    fin[(r['strategy'], r['dataset'])] = r['acc_mean']
# 读 MRPC 公平调参数据
ft = {}
for r in json.load(open(os.path.join(BASE, 'results', 'fair_tuning_merged.json'))):
    if r['dataset'] == 'mrpc':
        ft[r['strategy']] = r

METHODS = ['DALS_Fast', 'DALS_Acc', 'DALS', 'RAdam', 'Fixed_SGD', 'Cosine_SGD', 'SAM', 'SGDR']
DIMS = ['Synthetic\nagility', 'CIFAR-10\naccuracy', 'MRPC\naccuracy', 'Tunability\ngain']

# synthetic 敏捷性（达 95% epoch）
syn_agility = {'DALS_Fast': 1.0, 'DALS_Acc': 1.8, 'DALS': 2.0, 'RAdam': 4.6,
               'Fixed_SGD': 1.0, 'Cosine_SGD': 1.0, 'SAM': 1.0, 'SGDR': 1.0}
# cifar10 精度
cif_acc = {m: fin.get((m, 'cifar10')) for m in METHODS}
# MRPC 调优后精度
mrpc_acc = {m: ft[m]['best_acc'] for m in METHODS}
# 调参增益 = best - typical
tune_gain = {m: (ft[m]['best_acc'] - (ft[m]['typical_acc'] or 0)) for m in METHODS}

def norm_smaller(v, d):
    mn, mx = min(d.values()), max(d.values())
    return (mx - v) / (mx - mn) if mx > mn else 1.0

def norm_larger(v, d):
    return v / max(d.values()) if max(d.values()) > 0 else 0.0

scores = {}
for m in METHODS:
    scores[m] = [
        norm_smaller(syn_agility[m], syn_agility),
        norm_larger(cif_acc[m], cif_acc),
        norm_larger(mrpc_acc[m], mrpc_acc),
        norm_larger(tune_gain[m], tune_gain),
    ]

colors = {'DALS_Fast': '#ff7f0e', 'DALS_Acc': '#2ca02c', 'DALS': '#d62728',
          'RAdam': '#1f77b4', 'Fixed_SGD': '#8c564b', 'Cosine_SGD': '#e377c2',
          'SAM': '#7f7f7f', 'SGDR': '#bcbd22'}
# DALS 家族用实线粗线，baseline 用细虚线
DALS_FAMILY = {'DALS_Fast', 'DALS_Acc', 'DALS'}

N = len(DIMS)
angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(6.5, 6), subplot_kw=dict(polar=True))
for m in METHODS:
    vals = scores[m] + scores[m][:1]
    lw = 2.4 if m in DALS_FAMILY else 1.2
    ls = '-' if m in DALS_FAMILY else '--'
    ax.plot(angles, vals, ls, linewidth=lw, label=m, color=colors[m])
    ax.fill(angles, vals, alpha=0.06, color=colors[m])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(DIMS, fontsize=9)
ax.set_ylim(0, 1)
ax.set_yticks([0.25, 0.5, 0.75, 1.0])
ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0'], fontsize=7, color='gray')
ax.set_title('Agility, accuracy, and tunability (normalized, outer = better)', fontsize=10, pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.12), fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, 'fig4_radar_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()
print('fig4_radar_comparison.png 已生成（8 方法 × 4 维度）')
