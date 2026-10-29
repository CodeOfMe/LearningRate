# revision_benchmark — 修订版 Benchmark 运行包

针对 NEUCOM-D-26-12332 返修（R1-4 扩展基准、R4-5 验证集选模型、R4-16 macro-F1、
R1-10/R4-4 多 seed 统计、编辑部 RoBERTa 建议）整理的**可独立运行**实验包。
目标：把整个 `revision_benchmark/` 复制到有 CUDA 的机器上直接运行。

---

## 目录结构

```
revision_benchmark/
├── train_code/            # 训练代码（已修复 + 支持 RoBERTa/离线路径）
│   ├── run_final.py       # 主 benchmark（resume 能力，5-seed）
│   ├── optimizers.py      # 15 个优化器 + DALS（含消融开关）
│   ├── data.py            # 数据集 + 模型（支持本地路径 / 离线缓存）
│   ├── train.py           # 统一训练/评估（macro-F1，修复 SST-2 bug）
│   ├── run_ablation.py    # DALS 消融（9 变体）
│   └── run_analysis.py    # 敏感性 / 多 seed / 效率
├── models/                # 预训练模型
│   ├── bert-base-uncased/   ✅ 完整
│   ├── roberta-base/        ✅ 完整
│   └── distilbert-base-uncased/  ⚠️ 需补（见其 README.txt）
└── datasets/              # 数据集
    ├── cifar-10-batches-py/ ✅
    ├── cifar-100-python/     ✅
    └── huggingface/          ✅ GLUE (sst2/mrpc/mnli/qqp) 离线缓存
```

---

## 环境要求（目标机器）

```bash
pip install torch torchvision transformers datasets scikit-learn numpy
# 如用 DirectML（7900XTX）：另装 torch-directml
```

## 第一步：补 DistilBERT 模型

`models/distilbert-base-uncased/` 目前是空的（本机无缓存且无法下载）。
在目标 CUDA 机器上二选一：
1. **联网下载**：
   ```bash
   python -c "from transformers import AutoModel, AutoTokenizer; AutoModel.from_pretrained('distilbert-base-uncased'); AutoTokenizer.from_pretrained('distilbert-base-uncased')"
   ```
   然后把 `~/.cache/huggingface/hub/models--distilbert-base-uncased/snapshots/<hash>/*`
   复制到 `models/distilbert-base-uncased/`。
2. 从旧实验环境直接拷整个模型目录过来。

代码会自动优先使用 `models/` 下的本地副本；若不存在则 fallback 到 hub id。

---

## 运行

所有命令在 `train_code/` 目录内执行（`MODEL_DIR`/`DATA_DIR` 会自动指向上级目录）。

### 1. 补跑/新增的数据集（5 seed，验证集选模型）

```bash
# DistilBERT SST-2（修复 F1 bug 后补跑）
python run_final.py --device cuda --ds sst2

# BERT-base SST-2（修复 F1 bug 后补跑）
python run_final.py --device cuda --ds sst2_b

# RoBERTa（新增）
python run_final.py --device cuda --ds sst2_r mrpc_r

# 一次跑全部（含已完成的，会自动跳过已存在结果）
python run_final.py --device cuda --ds synthetic cifar10 cifar100 rte trec6 imdb sst2 mrpc sst2_b mrpc_b sst2_r mrpc_r
```

输出写到 `train_code/results_comp/*_cuda.json`（resume 安全，中断后重跑会跳过已完成 seed）。

### 2. 消融 / 敏感性 / 统计检验 / 效率

```bash
# 消融（DALS 组件拆解，9 变体，5-seed）—— R1-8/R4-9
python run_ablation.py --device cuda --seeds 42 1337 2024 5555 9999 --datasets synthetic cifar10

# 敏感性（phase阈值/α/blend/trust/warmup 网格）—— R4-14
python run_analysis.py --mode sensitivity --device cuda --seeds 42 1337 2024

# 统计检验（paired t-test + bootstrap CI，纯后处理，不训练）—— R1-10/R4-4
python run_significance.py --results results_comp --datasets all

# 效率汇总表（耗时/显存/吞吐）—— R1-15
python run_efficiency.py --results results_comp --datasets all
```

---

## 已修复的问题（相对 newcodes/ 原版）

| 问题 | 修复 |
|---|---|
| SST-2 全失败：`Target is multiclass but average='binary'` | `train.py` F1 一律用 `average='macro'`（同时满足 R4-16 macro-F1） |
| SAM/SAM+Discriminative 全失败：`unexpected keyword argument 'rho'` | `run_final.py` 的 `SAMW` 构造参数改为 `rho` |
| 无法离线加载模型/数据集 | `run_final.py` + `data.py` 支持 `MODEL_DIR`/`DATA_DIR` 本地路径，HF_DATASETS_CACHE 指向 `datasets/huggingface` |
| 无 RoBERTa | 新增 `sst2_r`/`mrpc_r` config，`MODEL_REGISTRY` 注册 RoBERTa |
| RTE/TREC-6/IMDb 无法离线加载 | `data.py` 支持本地 `datasets/{rte,trec6,imdb}` jsonl 优先读取 |
| 缺 SGDR/AdaBound/Lookahead 三基线 | `run_final.py` 策略列表补到 18 个 |
| 缺统计检验 | 新增 `run_significance.py`（paired t-test + bootstrap CI + Hedges' g） |
| 缺效率汇总 | 新增 `run_efficiency.py`（wall-clock/mem/throughput 表） |
| 消融 CIFAR-10 学习率与主表不一致 | `run_ablation.py` 统一为 0.1 |
| 梯度统计未启用（R4-15） | `run_final.py` 新增 `--grad_stats` 参数接线 `train.py` |

## 代码-功能对照（哪个代码跑哪个）

| 脚本 | 用途 | 对应返修 |
|---|---|---|
| `run_final.py` | 主 benchmark：18 策略 × 8 数据集 × 5-seed | R1-4/R4-5/R4-16/R1-10/R4-4 |
| `run_ablation.py` | DALS 消融（9 变体） | R1-8/R4-9 |
| `run_analysis.py` | 敏感性 / 多 seed / 效率 | R4-14/R1-10 |
| `run_significance.py` | 统计检验（后处理） | R1-10/R4-4 |
| `run_efficiency.py` | 效率汇总表（后处理） | R1-15 |
| `run_benchmark.py` | 备用完整版基准（18 策略，SGDR 等） | R1-4 |
| `train.py` | 统一训练/评估（macro-F1、val 选模型、梯度统计） | R4-5/R4-16/R4-15 |
| `data.py` | 数据集/模型加载（本地优先） | R4-17 |

## 结果文件说明

- `results_comp/{ds}_cuda.json`：每策略含 `n / acc_mean / acc_std / acc_ci95 / f1_mean / per_seed[]`
- 指标：accuracy + macro-F1 + wall-clock + peak GPU memory
- 协议：5 seed × 验证集选 checkpoint（`select_by='val'`），test 只评一次
