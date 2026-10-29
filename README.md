# DALS — Depth-Aware Learning-rate Schedule

Code and result tables for **DALS**, a depth-aware learning-rate schedule for deep networks.
DALS combines warmup-cosine scheduling, phase-/depth-aware Grokfast gradient filtering,
and LARS-style trust ratios.

> **Paper:** Ming-Hong Yao, Jian Cui, Di Wang, Qiu-Ye Yu.
> *Learning Rate Engineering: From Coarse Single Parameter to Layered Evolution*.
> Neurocomputing, 2026. DOI: [10.1016/j.neucom.2026.135504](https://doi.org/10.1016/j.neucom.2026.135504)

This repository contains the **code** and **all result tables**. The manuscript and
supplementary PDFs/DOCX are intentionally not included here.

## Publication details

| Field | Value |
|---|---|
| Title | Learning Rate Engineering: From Coarse Single Parameter to Layered Evolution |
| Journal | Neurocomputing |
| Article reference | NEUCOM_135504 |
| DOI | [10.1016/j.neucom.2026.135504](https://doi.org/10.1016/j.neucom.2026.135504) |
| First author | Ming-Hong Yao |
| Corresponding author | Qiu-Ye Yu |
| Authors | Ming-Hong Yao, Jian Cui, Di Wang, Qiu-Ye Yu |
| Received at Editorial Office | 10 Jun 2026 |
| Article revised | 10 Oct 2026 |
| Article accepted for publication | 25 Oct 2026 |

A `CITATION.cff` is included for GitHub's "Cite this repository" button.

```bibtex
@article{yao2026learning,
  title   = {Learning Rate Engineering: From Coarse Single Parameter to Layered Evolution},
  author  = {Yao, Ming-Hong and Cui, Jian and Wang, Di and Yu, Qiu-Ye},
  journal = {Neurocomputing},
  year    = {2026},
  doi     = {10.1016/j.neucom.2026.135504},
  note    = {Article reference: NEUCOM_135504}
}
```

---

## Layout

```
.
├── code/
│   ├── 01_initial/                 # DALS first implementation + 18 baselines
│   ├── 02_r1/
│   │   ├── main/                   # R1 main code (10 datasets, 5 seeds, val-based selection)
│   │   └── benchmark/              # standalone R1 benchmark package (ablation/significance/efficiency)
│   ├── 03_r2_final/
│   │   ├── core/                   # final code with the F1/F2/F3 numerical fixes
│   │   ├── stage1_diagnose/        # dead-value detection, root cause, re-run, verification
│   │   ├── stage2_statistics/      # significance correction, fair tuning, agility
│   │   ├── stage3_ablation/        # LOO + pairwise interaction ablation
│   │   ├── figures/                # figure generation
│   │   └── lib/
│   └── 04_r3/
│       ├── grad_noise/             # per-layer gradient noise on CIFAR-100
│       ├── vit/                    # ViT-Small fine-tuning check
│       └── ablation_cifar100/      # CIFAR-100 component ablation (16 combinations x 5 seeds)
└── tables/
    ├── main/                       # paper tables: gradient noise, accuracy, components, ablation, macro-F1
    ├── supplementary/              # all 63 supplementary tables (CSV + one Excel workbook)
    └── full_tables/                # final 180-cell master, per-seed accuracy/F1, dead values
```

### Code versions

| Directory | What it is |
|---|---|
| `code/01_initial/` | First DALS implementation and baselines (`optimizers.py`, `benchmark.py`, `run_*.py`) |
| `code/02_r1/main/` | R1 main code: 10 datasets, 5 seeds, validation-based checkpoint selection, macro-F1 |
| `code/02_r1/benchmark/` | Standalone R1 benchmark package (ablation, significance, efficiency) |
| `code/03_r2_final/core/` | **Final corrected code** (AdaBound `final_lr`, DALS-Acc SGDR warmup, DALS-Fast fine-tuning bypass) |
| `code/03_r2_final/stage*` | R2 diagnostic / statistics / ablation scripts |
| `code/04_r3/*` | R3 supplementary experiments, no changes to the optimizer logic |

The three R2 fixes are documented in `code/03_r2_final/FIXES.md`:

- **F1** AdaBound `final_lr` set to the SGD learning rate (SST-2/BERT collapse fixed)
- **F2** DALS-Acc SGDR cycles get a 10% linear warmup
- **F3** DALS-Fast disables `fast_phase0_bypass` during fine-tuning (with `lr x4`, momentum 0.9, warmup 0.05)

---

## Tables

### Main tables (`tables/main/`)

| File | Content |
|---|---|
| `gradient_noise.csv/.md` | Per-layer gradient variance, SNR and cosine similarity (CIFAR-100) |
| `accuracy.csv/.md` | Cross-dataset test accuracy, 18 strategies x 10 datasets, 5-seed mean |
| `component_origins.csv/.md` | DALS component origins and adaptations |
| `ablation_cifar100.csv/.md` | CIFAR-100 leave-one-out ablation main effects |
| `macro_f1.csv/.md/.tex/.docx` | **Macro-averaged F1**, 18 x 10, final corrected values |

> **Macro-F1 note.** The combined F1 table is rebuilt from the R3 supplementary S8 per-strategy
> per-seed tables. For legacy RTX-4070 runs on binary tasks (RTE, IMDb, MRPC, MRPC-BERT),
> positive-class F1 was analytically converted to macro-F1 using test accuracy and class
> distribution. The archived `f1_mean_archived` column in `final_data_master.csv` is therefore
> **not** a macro-F1 for those binary cells; use `macro_f1.csv` or `per_seed_macro_f1.csv`.

### Supplementary tables (`tables/supplementary/`)

- `T01...T63_*.csv` — all 63 tables of the R3 supplementary material, one CSV per table
- `manifest.csv` — maps table id → section/subsection → CSV file
- `supplementary_tables.xlsx` — the same 63 tables in one workbook (`INDEX` + `T01...T63` sheets)

### Full result tables (`tables/full_tables/`)

| File | Content |
|---|---|
| `final_data_master.csv` | 180 cells: dataset, strategy, GPU, accuracy mean/std/CI95, archived F1, issue |
| `final_data_master.jsonl` | same 180 cells in the original JSONL format (used by the R2 diagnostic script) |
| `per_seed_accuracy.csv` | 900 rows: dataset x strategy x seed accuracy |
| `per_seed_macro_f1.csv` | 900 rows: dataset x strategy x seed macro-F1 |
| `dead_values.csv` | 10 detected dead/collapsed cells |
| `initial_single_seed_accuracy.csv` | Initial single-seed benchmark summary |

---

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Datasets and pretrained weights are **not** included. Expected layout:

```
datasets/
├── cifar-10-batches-py/
├── cifar-100-python/
├── huggingface/
└── sst2/ mrpc/ rte/ trec6/ imdb/
models/
├── bert-base-uncased/
├── distilbert-base-uncased/
├── roberta-base/
└── vit-small-patch16-224/
```

CIFAR: <https://www.cs.toronto.edu/~kriz/cifar.html>  
GLUE: `nyu-mll/glue` (SST-2 / MRPC / RTE)  
IMDb: `stanfordnlp/imdb` · TREC-6: `CogComp/trec`

Example (final R2 code):

```bash
cd code/03_r2_final/core
python run_final.py --device cuda \
  --ds synthetic cifar10 cifar100 rte trec6 imdb sst2 mrpc sst2_b mrpc_b \
  --seeds 42 1337 2024 5555 9999
```

R3 experiments:

```bash
cd code/04_r3/grad_noise && python grad_noise.py
cd code/04_r3/vit && python run_vit.py
cd code/04_r3/ablation_cifar100 && python run_interaction_cifar100.py --device cuda --datasets cifar100 --mode loo
```

---

## License

GPL-3.0. See `LICENSE`.
