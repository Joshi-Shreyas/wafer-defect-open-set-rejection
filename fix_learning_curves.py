import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_BASE = Path('/scratch/joshi.shreyas/wafer_results')
OUTPUT_DIR = RESULTS_BASE / 'figures'

mae_loss_history = [
    0.01490, 0.00900, 0.00863, 0.00961, 0.00866, 0.00842, 0.00833, 0.00826,
    0.00822, 0.00819, 0.00817, 0.00815, 0.00813, 0.00812, 0.00811, 0.00810,
    0.00810, 0.00809, 0.00809, 0.00808
]

with open(RESULTS_BASE / 'supervised' / 'history.json') as f:
    sup_history = json.load(f)
with open(RESULTS_BASE / 'finetuned' / 'history.json') as f:
    fin_history = json.load(f)

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
print("Fixed learning_curves.png saved.")
