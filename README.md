# Defer to the Engineer: Calibrated Rejection of Rare and Unknown Wafer Defects under Fab Cost Asymmetry

**Team:** Shreyas Joshi, Xinhao Chen

A machine learning project studying open-set defect classification on semiconductor wafer maps: training a model to correctly classify known defect patterns while flagging genuinely novel/unseen defect signatures, calibrated against the asymmetric real-world costs of a fab acting on a wrong decision.

## Problem Statement

Automated wafer map defect classification works well on a fixed set of known defect patterns — but fabs continually produce new failure signatures that no labeled training set has seen. A closed-set classifier forces every wafer into one of its known classes, confidently, even when it's wrong — leading either to a healthy tool being halted or a failing tool continuing to scrap wafers. This project trains models that recognize known patterns while also flagging novel ones, with a calibrated confidence estimate suitable for a "send this to a human" decision.

## Dataset

**WM811K** ([Kaggle: qingyi/wm811k-wafer-map](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map)) — 811,457 real production wafer maps: 172,950 labeled across 8 defect types + "none", and 638,507 unlabeled.

## What's in here

- `Data_Preprocessing.ipynb` — all the EDA and preprocessing: padding/masking wafer maps, building the stratified splits, holding out the Near-full class
- `data_utils.py`, `model.py` — the Dataset class and the ResNet-18-style backbone shared by both training regimes
- `train_supervised.py` — Regime A, the from-scratch supervised baseline
- `train_mae.py` + `finetune.py` — Regime B, the MAE self-supervised pretraining stage and the fine-tuning stage that follows it
- `evaluate.py` — open-set scoring (energy + Mahalanobis), calibration, and the cost-sensitive threshold sweep
- `channel_ablation.py`, `check_feature_separation.py`, `random_search.py` — the feature-importance ablation, the energy-vs-Mahalanobis diagnostic, and the hyperparameter search
- `create_visualizations.py`, `feature_importance.py`, `fix_learning_curves.py` — scripts that generate the report figures
- `*.sbatch` — SLURM job scripts for running the longer stages on a cluster
- `mask_test.png`, `padding_test.png`, `wafer_examples.png` — sanity-check plots from preprocessing
- `Iteration2_Report.docx` — the full written report

## Pipeline

**1. Preprocessing** (`Data_Preprocessing.ipynb`)
Pads/masks wafer maps to 96×96 (2-channel: wafer map + validity mask), builds a custom stratified 70/15/15 train/val/test split, holds out the rarest class (**Near-full**, 149 examples) entirely as the "unknown" evaluation set, and saves everything to `.npz` files.

**2. Train Regime A — Supervised baseline**
```bash
python train_supervised.py --epochs 30 --batch_size 128 --lr 0.001
```

**3. Train Regime B — MAE pretraining, then fine-tuning**
```bash
python train_mae.py --epochs 20 --batch_size 256 --mask_ratio 0.6
python finetune.py --epochs 30 --batch_size 128 --lr 0.0005
```

**4. Evaluate both regimes**
```bash
python evaluate.py --regime supervised
python evaluate.py --regime finetuned
python check_feature_separation.py
```

**5. Ablation and tuning**
```bash
python channel_ablation.py --epochs 15
python random_search.py --n_configs 8 --epochs_per_config 5
```

**6. Generate figures**
```bash
python create_visualizations.py
python feature_importance.py
```

Corresponding `.sbatch` files are provided for each long-running step (submit via `sbatch <name>.sbatch` on a SLURM cluster).

## Key Results

| Metric | Regime A (Supervised) | Regime B (MAE-pretrained) |
|---|---|---|
| Test Accuracy | 0.9723 | 0.9749 |
| Test Macro-F1 | 0.8603 | 0.8663 |
| Test Balanced Accuracy | 0.8754 | 0.8704 |
| Open-Set AUROC (Energy) | 0.7885 | 0.5019 |
| Open-Set AUROC (Mahalanobis) | 0.9962 | 0.9974 |

**Headline finding:** the choice of open-set scoring method matters more than the training regime. Energy-based (logit) scoring made Regime B look like it had lost all ability to detect novel defects; Mahalanobis distance in feature space revealed both regimes separate known from unknown wafers almost perfectly, with Regime B marginally ahead. Under a cost-sensitive decision rule with Mahalanobis scoring, Regime B achieves 32–42% lower operational cost than Regime A.

## Environment

- Python 3.10, PyTorch 2.5.1+cu121
- `pandas<2.0`, `numpy<2.0` (required for loading the original `LSWMD.pkl`, serialized with an old pandas version)
- Trained on a single NVIDIA V100-SXM2-32GB (Northeastern Explorer HPC cluster)

## Known Issues

- `train_mae.py`'s checkpoint-resume logic does not reload `loss_history` on resume, truncating the plotted reconstruction-loss curve if the job is interrupted and resumed. `fix_learning_curves.py` regenerates the corrected plot from the raw training logs.

## Full Report

See `Iteration2_Report.docx` for the complete write-up: model architecture, training process, hyperparameter tuning, per-class performance, calibration, open-set detection analysis, feature importance, discussion, and answers to all research questions.
