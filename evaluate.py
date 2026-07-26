"""
evaluate.py

The core comparison: open-set scoring + temperature-scaling calibration, applied
IDENTICALLY to both Regime A (supervised) and Regime B (MAE-pretrained + fine-tuned),
then evaluated on:
  1. Standard closed-set test performance (per-class, not just aggregate)
  2. Open-set rejection quality on the held-out Near-full class (energy-based scoring)
  3. Calibration quality (before/after temperature scaling)
  4. A cost-sensitive decision rule sweep (the "Defer to the Engineer" mechanism)

Usage:
    python evaluate.py --regime supervised
    python evaluate.py --regime finetuned
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    f1_score, balanced_accuracy_score
)

from data_utils import get_dataloaders, NUM_CLASSES, KNOWN_CLASSES
from model import build_model


def get_logits_and_features(model, loader, device):
    """Runs the model over a loader, collecting logits, features, and true labels."""
    model.eval()
    all_logits, all_features, all_labels = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits, features = model(x, return_features=True)
            all_logits.append(logits.cpu())
            all_features.append(features.cpu())
            all_labels.append(y)

    return torch.cat(all_logits), torch.cat(all_features), torch.cat(all_labels)


def energy_score(logits, temperature=1.0):
    """
    Energy-based open-set score: E(x) = -T * logsumexp(logits / T).
    LOWER energy = more confident/in-distribution. HIGHER energy = more likely unknown.
    (Liu et al., 2020 - standard, simple, doesn't require retraining anything.)
    """
    return -temperature * torch.logsumexp(logits / temperature, dim=1)


def fit_temperature(logits, labels, device):
    """
    Fits a single scalar temperature via gradient descent to minimize NLL on validation data.
    Standard temperature scaling (Guo et al., 2017) - rescales confidence without
    changing the model's ranking/accuracy at all.
    """
    logits = logits.to(device)
    labels = labels.to(device)
    temperature = torch.nn.Parameter(torch.ones(1, device=device))
    optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return temperature.item()


def expected_calibration_error(probs, labels, n_bins=15):
    """
    Standard ECE: bins predictions by confidence, compares average confidence
    to actual accuracy within each bin.
    """
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(labels)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = torch.zeros(1)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            acc_in_bin = accuracies[in_bin].float().mean()
            conf_in_bin = confidences[in_bin].mean()
            ece += torch.abs(conf_in_bin - acc_in_bin) * prop_in_bin

    return ece.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/scratch/joshi.shreyas/wafer_processed')
    parser.add_argument('--regime', type=str, required=True, choices=['supervised', 'finetuned'],
                         help='Which trained model to evaluate')
    parser.add_argument('--fn_cost', type=float, default=20.0,
                         help='Relative cost of a missed defect (false negative on rejection) vs. an unnecessary review')
    parser.add_argument('--fp_cost', type=float, default=1.0,
                         help='Relative cost of an unnecessary human review (false positive on rejection)')
    args = parser.parse_args()

    model_paths = {
        'supervised': '/scratch/joshi.shreyas/wafer_results/supervised/best_model.pt',
        'finetuned': '/scratch/joshi.shreyas/wafer_results/finetuned/best_model.pt',
    }
    output_dir = Path(f'/scratch/joshi.shreyas/wafer_results/eval_{args.regime}')
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating regime: {args.regime}")
    print(f"Using device: {device}")

    print("\nLoading data...")
    train_loader, val_loader, test_loader, unknown_loader = get_dataloaders(
        args.data_dir, batch_size=128, num_workers=4
    )

    model = build_model(num_classes=NUM_CLASSES, in_channels=2).to(device)
    checkpoint = torch.load(model_paths[args.regime], map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # ------------------------------------------------------------------
    # 1. Standard closed-set test performance, per-class
    # ------------------------------------------------------------------
    print("\n--- Closed-set test performance (per-class) ---")
    test_logits, test_features, test_labels = get_logits_and_features(model, test_loader, device)
    test_preds = test_logits.argmax(dim=1)

    report = classification_report(test_labels, test_preds, target_names=KNOWN_CLASSES,
                                    zero_division=0, output_dict=True)
    print(classification_report(test_labels, test_preds, target_names=KNOWN_CLASSES, zero_division=0))

    cm = confusion_matrix(test_labels, test_preds)

    # ------------------------------------------------------------------
    # 2. Temperature scaling calibration (fit on validation set)
    # ------------------------------------------------------------------
    print("\n--- Calibration ---")
    val_logits, val_features, val_labels = get_logits_and_features(model, val_loader, device)

    val_probs_uncal = F.softmax(val_logits, dim=1)
    ece_before = expected_calibration_error(val_probs_uncal, val_labels)

    temperature = fit_temperature(val_logits, val_labels, device)
    val_probs_cal = F.softmax(val_logits.to(device) / temperature, dim=1).cpu()
    ece_after = expected_calibration_error(val_probs_cal, val_labels)

    print(f"Fitted temperature: {temperature:.4f}")
    print(f"ECE before calibration: {ece_before:.4f}")
    print(f"ECE after calibration:  {ece_after:.4f}")

    # ------------------------------------------------------------------
    # 3. Open-set rejection: energy scores on known (test) vs. unknown (Near-full)
    # ------------------------------------------------------------------
    print("\n--- Open-set rejection (Near-full as unknown) ---")
    unknown_logits, unknown_features, _ = get_logits_and_features(model, unknown_loader, device)

    known_energy = energy_score(test_logits, temperature=temperature)
    unknown_energy = energy_score(unknown_logits, temperature=temperature)

    print(f"Known (test) energy   - mean: {known_energy.mean():.3f}, std: {known_energy.std():.3f}")
    print(f"Unknown (Near-full) energy - mean: {unknown_energy.mean():.3f}, std: {unknown_energy.std():.3f}")
    print("(Higher energy = more 'unknown-like'. We want unknown_energy notably higher than known_energy.)")

    # AUROC for the binary "is this unknown?" task, using energy score as the detector
    binary_labels = np.concatenate([np.zeros(len(known_energy)), np.ones(len(unknown_energy))])
    binary_scores = np.concatenate([known_energy.numpy(), unknown_energy.numpy()])
    open_set_auroc = roc_auc_score(binary_labels, binary_scores)
    print(f"\nOpen-set detection AUROC (Near-full vs. known): {open_set_auroc:.4f}")
    print("(1.0 = perfect separation, 0.5 = no better than random)")

    # ------------------------------------------------------------------
    # 4. Cost-sensitive decision rule sweep
    # ------------------------------------------------------------------
    print(f"\n--- Cost-sensitive threshold sweep (FN cost={args.fn_cost}, FP cost={args.fp_cost}) ---")
    # For each candidate energy threshold, wafers above threshold get "deferred to engineer" (rejected).
    # False negative here = an unknown (Near-full) wafer that was NOT rejected (missed novel defect).
    # False positive here = a known wafer that WAS rejected unnecessarily (wasted human review).
    all_thresholds = np.linspace(binary_scores.min(), binary_scores.max(), 200)
    best_cost = float('inf')
    best_threshold = None

    for thresh in all_thresholds:
        rejected = binary_scores > thresh
        # Among knowns (label 0): rejected = false positive (unnecessary review)
        fp = ((binary_labels == 0) & rejected).sum()
        # Among unknowns (label 1): NOT rejected = false negative (missed novel defect)
        fn = ((binary_labels == 1) & ~rejected).sum()

        total_cost = args.fn_cost * fn + args.fp_cost * fp
        if total_cost < best_cost:
            best_cost = total_cost
            best_threshold = thresh

    rejected_at_best = binary_scores > best_threshold
    fp_best = int(((binary_labels == 0) & rejected_at_best).sum())
    fn_best = int(((binary_labels == 1) & ~rejected_at_best).sum())

    print(f"Best threshold: {best_threshold:.4f}")
    print(f"At this threshold: {fp_best} false positives (unnecessary reviews), {fn_best} false negatives (missed novel defects)")
    print(f"Total expected cost: {best_cost:.1f}")

    # ------------------------------------------------------------------
    # Save everything for later comparison between regimes
    # ------------------------------------------------------------------
    results = {
        'regime': args.regime,
        'per_class_report': report,
        'confusion_matrix': cm.tolist(),
        'temperature': temperature,
        'ece_before_calibration': ece_before,
        'ece_after_calibration': ece_after,
        'known_energy_mean': known_energy.mean().item(),
        'known_energy_std': known_energy.std().item(),
        'unknown_energy_mean': unknown_energy.mean().item(),
        'unknown_energy_std': unknown_energy.std().item(),
        'open_set_auroc': open_set_auroc,
        'best_threshold': float(best_threshold),
        'best_cost': float(best_cost),
        'fp_at_best_threshold': fp_best,
        'fn_at_best_threshold': fn_best,
        'fn_cost_used': args.fn_cost,
        'fp_cost_used': args.fp_cost,
    }

    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    # Also save raw energy scores and features for plotting later (t-SNE, histograms, etc.)
    np.savez(output_dir / 'raw_scores.npz',
             known_energy=known_energy.numpy(),
             unknown_energy=unknown_energy.numpy(),
             test_features=test_features.numpy(),
             test_labels=test_labels.numpy(),
             unknown_features=unknown_features.numpy())

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
