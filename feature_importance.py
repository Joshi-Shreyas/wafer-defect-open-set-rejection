"""
feature_importance.py
Generates the feature importance visualization: since there are no hand-engineered
tabular features, the equivalent is "which input channels matter" - the validity
mask channel ablation.
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

RESULTS_BASE = Path('/scratch/joshi.shreyas/wafer_results')
OUTPUT_DIR = RESULTS_BASE / 'figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(RESULTS_BASE / 'supervised' / 'test_results.json') as f:
    two_channel = json.load(f)
with open(RESULTS_BASE / 'ablation_no_mask' / 'test_results.json') as f:
    one_channel = json.load(f)

metrics = ['accuracy', 'macro_f1', 'balanced_accuracy']
labels = ['Test Accuracy', 'Test Macro-F1', 'Test Balanced-Acc']
two_channel_vals = [two_channel[m] for m in metrics]
one_channel_vals = [one_channel[m] for m in metrics]

x = np.arange(len(labels))
width = 0.35
fig, ax = plt.subplots(figsize=(9, 6))
bars1 = ax.bar(x - width/2, two_channel_vals, width, label='Wafer Map + Validity Mask (2 channels)', color='steelblue')
bars2 = ax.bar(x + width/2, one_channel_vals, width, label='Wafer Map Only (1 channel)', color='lightcoral')

for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.4f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontsize=9)

gap = two_channel_vals[1] - one_channel_vals[1]
ax.annotate(f'Δ = {gap:.4f}', xy=(1, max(two_channel_vals[1], one_channel_vals[1]) + 0.04),
            ha='center', fontsize=10, fontweight='bold', color='darkred')

ax.set_ylabel('Score')
ax.set_title('Input Channel (Feature) Importance:\nValidity Mask Channel Ablation Study')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 1.05)
ax.legend(loc='lower right')
ax.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'feature_importance.png', dpi=150)
plt.close()

print(f"Feature importance plot saved to: {OUTPUT_DIR / 'feature_importance.png'}")
print(f"2-channel: Macro-F1={two_channel['macro_f1']:.4f} | 1-channel: Macro-F1={one_channel['macro_f1']:.4f} | Gap: {gap:.4f}")
