"""
create_visualizations.py

Generates all report figures from the artifacts already saved during training and evaluation:
  1. learning_curves.png       - train/val metrics over epochs, both regimes + MAE pretraining
  2. confusion_matrices.png    - side-by-side confusion matrices, both regimes
  3. score_distributions.png   - known vs. unknown score histograms (energy + Mahalanobis), both regimes
  4. roc_curves.png            - actual ROC curves for open-set detection, both regimes, both scoring methods
  5. performance_comparison.png - bar chart comparing key metrics across regimes

Usage:
    python create_visualizations.py
"""

import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # no display on cluster - just save files
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.covariance import EmpiricalCovariance

from data_utils import KNOWN_CLASSES

RESULTS_BASE = Path('/scratch/joshi.shreyas/wafer_results')
OUTPUT_DIR = RESULTS_BASE / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def mahalanobis_scores(test_features, test_labels, unknown_features, num_classes=8):
    centroids = np.stack([test_features[test_labels == c].mean(axis=0) for c in range(num_classes)])
    precision = EmpiricalCovariance().fit(test_features).precision_

    def min_maha(feats):
        dists = [np.sqrt(np.sum((feats - centroids[c]) @ precision * (feats - centroids[c]), axis=1))
                 for c in range(num_classes)]
        return np.min(np.stack(dists), axis=0)

    return min_maha(test_features), min_maha(unknown_features)


# ----------------------------------------------------------------------
# 1. Learning curves
# ----------------------------------------------------------------------
print("Generating learning curves...")
sup_history = load_json(RESULTS_BASE / 'supervised' / 'history.json')
fin_history = load_json(RESULTS_BASE / 'finetuned' / 'history.json')
mae_checkpoint = torch.load(RESULTS_BASE / 'mae' / 'mae_checkpoint.pt', map_location='cpu', weights_only=False)
mae_loss_history = mae_checkpoint['loss_history']

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(range(1, len(mae_loss_history) + 1), mae_loss_history, marker='o', color='steelblue')
axes[0].set_title('MAE Pretraining: Reconstruction Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Avg Reconstruction Loss (MSE)')
axes[0].grid(alpha=0.3)

epochs_a = range(1, len(sup_history['val_macro_f1']) + 1)
epochs_b = range(1, len(fin_history['val_macro_f1']) + 1)
axes[1].plot(epochs_a, sup_history['val_macro_f1'], marker='o', label='Regime A (Supervised)', color='darkorange')
axes[1].plot(epochs_b, fin_history['val_macro_f1'], marker='s', label='Regime B (MAE-pretrained)', color='steelblue')
axes[1].set_title('Validation Macro-F1 over Training')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Macro-F1')
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].plot(epochs_a, sup_history['train_loss'], '--', label='Regime A - Train Loss', color='darkorange', alpha=0.6)
axes[2].plot(epochs_a, sup_history['val_loss'], label='Regime A - Val Loss', color='darkorange')
axes[2].plot(epochs_b, fin_history['train_loss'], '--', label='Regime B - Train Loss', color='steelblue', alpha=0.6)
axes[2].plot(epochs_b, fin_history['val_loss'], label='Regime B - Val Loss', color='steelblue')
axes[2].set_title('Train vs. Validation Loss')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Cross-Entropy Loss')
axes[2].legend(fontsize=8)
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'learning_curves.png', dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 2. Confusion matrices
# ----------------------------------------------------------------------
print("Generating confusion matrices...")
sup_eval = load_json(RESULTS_BASE / 'eval_supervised' / 'evaluation_results.json')
fin_eval = load_json(RESULTS_BASE / 'eval_finetuned' / 'evaluation_results.json')

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, eval_data, title in [(axes[0], sup_eval, 'Regime A (Supervised)'),
                               (axes[1], fin_eval, 'Regime B (MAE-pretrained)')]:
    cm = np.array(eval_data['confusion_matrix'])
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(len(KNOWN_CLASSES)))
    ax.set_yticks(range(len(KNOWN_CLASSES)))
    ax.set_xticklabels(KNOWN_CLASSES, rotation=45, ha='right')
    ax.set_yticklabels(KNOWN_CLASSES)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(f'{title}\nConfusion Matrix (row-normalized)')
    for i in range(len(KNOWN_CLASSES)):
        for j in range(len(KNOWN_CLASSES)):
            val = cm_norm[i, j]
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    color='white' if val > 0.5 else 'black', fontsize=7)
    plt.colorbar(im, ax=ax, fraction=0.046)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrices.png', dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 3. Score distributions (energy + Mahalanobis, both regimes)
# ----------------------------------------------------------------------
print("Generating score distributions...")
sup_raw = np.load(RESULTS_BASE / 'eval_supervised' / 'raw_scores.npz')
fin_raw = np.load(RESULTS_BASE / 'eval_finetuned' / 'raw_scores.npz')

sup_maha_known, sup_maha_unknown = mahalanobis_scores(
    sup_raw['test_features'], sup_raw['test_labels'], sup_raw['unknown_features'])
fin_maha_known, fin_maha_unknown = mahalanobis_scores(
    fin_raw['test_features'], fin_raw['test_labels'], fin_raw['unknown_features'])

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

for ax, known, unknown, title in [
    (axes[0, 0], sup_raw['known_energy'], sup_raw['unknown_energy'], 'Regime A - Energy Score'),
    (axes[0, 1], fin_raw['known_energy'], fin_raw['unknown_energy'], 'Regime B - Energy Score'),
    (axes[1, 0], sup_maha_known, sup_maha_unknown, 'Regime A - Mahalanobis Distance'),
    (axes[1, 1], fin_maha_known, fin_maha_unknown, 'Regime B - Mahalanobis Distance'),
]:
    ax.hist(known, bins=50, alpha=0.6, label='Known (test set)', color='steelblue', density=True)
    ax.hist(unknown, bins=30, alpha=0.6, label='Unknown (Near-full)', color='crimson', density=True)
    ax.set_title(title)
    ax.set_xlabel('Score')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'score_distributions.png', dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 4. ROC curves
# ----------------------------------------------------------------------
print("Generating ROC curves...")
fig, ax = plt.subplots(figsize=(8, 8))

for known, unknown, label, style in [
    (sup_raw['known_energy'], sup_raw['unknown_energy'], 'Regime A - Energy', '--'),
    (fin_raw['known_energy'], fin_raw['unknown_energy'], 'Regime B - Energy', '--'),
    (sup_maha_known, sup_maha_unknown, 'Regime A - Mahalanobis', '-'),
    (fin_maha_known, fin_maha_unknown, 'Regime B - Mahalanobis', '-'),
]:
    labels = np.concatenate([np.zeros(len(known)), np.ones(len(unknown))])
    scores = np.concatenate([known, unknown])
    fpr, tpr, _ = roc_curve(labels, scores)
    auroc = roc_auc_score(labels, scores)
    ax.plot(fpr, tpr, style, label=f'{label} (AUROC={auroc:.3f})', linewidth=2)

ax.plot([0, 1], [0, 1], 'k:', alpha=0.4, label='Random (AUROC=0.5)')
ax.set_xlabel('False Positive Rate (known wafers flagged as unknown)')
ax.set_ylabel('True Positive Rate (Near-full wafers correctly caught)')
ax.set_title('Open-Set Detection: ROC Curves')
ax.legend(loc='lower right')
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roc_curves.png', dpi=150)
plt.close()

# ----------------------------------------------------------------------
# 5. Performance comparison bar chart
# ----------------------------------------------------------------------
print("Generating performance comparison...")
sup_test = load_json(RESULTS_BASE / 'supervised' / 'test_results.json')
fin_test = load_json(RESULTS_BASE / 'finetuned' / 'test_results.json')

sup_maha_auroc = roc_auc_score(
    np.concatenate([np.zeros(len(sup_maha_known)), np.ones(len(sup_maha_unknown))]),
    np.concatenate([sup_maha_known, sup_maha_unknown]))
fin_maha_auroc = roc_auc_score(
    np.concatenate([np.zeros(len(fin_maha_known)), np.ones(len(fin_maha_unknown))]),
    np.concatenate([fin_maha_known, fin_maha_unknown]))

metrics = ['Test Accuracy', 'Test Macro-F1', 'Test Balanced-Acc', 'Open-Set AUROC\n(Mahalanobis)']
regime_a_vals = [sup_test['accuracy'], sup_test['macro_f1'], sup_test['balanced_accuracy'], sup_maha_auroc]
regime_b_vals = [fin_test['accuracy'], fin_test['macro_f1'], fin_test['balanced_accuracy'], fin_maha_auroc]

x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, regime_a_vals, width, label='Regime A (Supervised)', color='darkorange')
bars2 = ax.bar(x + width/2, regime_b_vals, width, label='Regime B (MAE-pretrained)', color='steelblue')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

ax.set_ylabel('Score')
ax.set_title('Regime A vs. Regime B: Performance Comparison')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'performance_comparison.png', dpi=150)
plt.close()

print(f"\nAll figures saved to: {OUTPUT_DIR}")
print("Files created:")
for f in sorted(OUTPUT_DIR.glob('*.png')):
    print(f"  - {f.name}")
