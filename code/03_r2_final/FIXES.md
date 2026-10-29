# 代码修复记录（FIXES）

> 本文件记录死值根因定位 → 修复 → 验证的完整过程，配合 `logs/*_log.jsonl` 的逐步记录。
> 所有修复仅作用于 `experiment_audit/core/` 下的代码副本，benchmark 原始代码（`revision_benchmark/train_code/`）保持不变。

## 修复清单

| # | 文件 | 问题 | 修复 | 状态 |
|---|---|---|---|---|
| F1 | `core/run_final.py` | AdaBound `final_lr=0.1` 与 fine-tuning `lr=5e-5` 相差 2000 倍，后期 SGD 界被 clamp 到 ~0.1，BERT-base 微调发散坍缩（sst2_b 死值 50.92） | `final_lr=lr_sgd`（与 SGD 学习率同量级） | ✅ 已修复（5-seed 验证：mean=91.54 std=0.42） |
| F2 | `core/optimizers.py` | DALS_Acc SGDR 重启无 warmup，重启瞬间 lr 从 ~0 跳回 base，偶发坍缩（sst2_b/DALS_Acc seed9999=50.92） | 每个 cycle 开头加 `warmup_steps=0.1*cycle_T` 线性爬升 | ✅ 已修复（5-seed 验证：mean=92.13 std=0.44） |
| F3 | `core/run_final.py` | DALS-Fast 默认 lr 在 fine-tuning 任务（mrpc lr_sgd=5e-4）下系统偏低，5 种子坍缩到多数类（66.49, std=0） | momentum 0.85→0.9、warmup 0.02→0.05（与 base DALS 对齐）+ fine-tuning 路径 lr×4(=2e-3) | ✅ 已修复（5-seed 验证：mean=83.53 std=1.06） |

## F3 的根因诊断与修复

DALS-Fast 在 MRPC 的「坍缩到多数类」现象根因不是某个组件失效，而是 **fine-tuning 默认 lr 系统偏低**。
详细迭代记录见 `results/TRY_FIX_DALS_FAST.md`，关键证据：

- DALS base (momentum=0.9, warmup=0.05, bypass=False) 在 mrpc seed42 单 seed acc=69.36（已能训练，仅略高于多数类 68.38）
- DALS_Fast 改用 momentum=0.9 + warmup=0.05 但保持 lr=5e-4 → 仍 69.36（momentum/warmup 不是根因）
- **DALS_Fast 加 lr×4(=2e-3) → acc=83.33**（与 fair_tuning 最优 83.38 一致）
- 5-seed 验证：83.33, 84.07, 85.05, 82.35, 82.84 → mean=83.53, std=1.06

**修改位置**：`core/run_final.py` DALS_Fast 分支（fine-tuning 时 `lr_sgd<=5e-4` 路径
`fast_lr = dlr * 4`，from-scratch 路径保持 dlr=0.1 不变以避免破坏 cifar10/synthetic）。

## 不修复项（经决策：保留 + 如实标注）

| 数据集 | 策略 | 原因 |
|---|---|---|
| mrpc | Discriminative / STLR_Disc / SAM_Disc | 判别式衰减过度抑制底层的**真实现象**，是论文「DALS 避免坍缩」的论据，予以保留并补充「5 种子 std=0」说明 |
| trec6 | Discriminative / STLR_Disc / SAM_Disc | 同上，崩到接近随机（论文已述「near random」） |

## 验证方法

```bash
cd experiment_audit
# 完整 5-seed 修复验证（支持断点恢复，sst2_b 单 seed ~36min，建议后台跑）
python stage1_diagnose/verify_fix.py --device cuda
```
