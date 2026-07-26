"""
random_search.py

Hyperparameter tuning via random search, using reduced-epoch training runs as a fast
proxy for full convergence (standard practice when full grid search at full epoch
count is computationally prohibitive - each full run here takes ~2 hours).

Searches over learning rate and weight decay for the supervised (Regime A) baseline.
Compares against the default hyperparameters already used for the full 30-epoch run
(lr=0.001, weight_decay=1e-4), to test whether that default was actually a good choice.

Usage:
    python random_search.py --n_configs 8 --epochs_per_config 5
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, balanced_accuracy_score

from data_utils import get_dataloaders, NUM_CLASSES
from model import build_model


def compute_class_weights(train_loader, num_classes, device):
    counts = torch.zeros(num_classes)
    for _, labels in train_loader:
        for c in range(num_classes):
            counts[c] += (labels == c).sum()
    weights = 1.0 / (counts + 1e-6)
    weights = weights / weights.sum() * num_classes
    return weights.to(device)


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return {
        'macro_f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
    }


def train_one_config(lr, weight_decay, epochs, train_loader, val_loader, class_weights, device):
    model = build_model(num_classes=NUM_CLASSES, in_channels=2).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

    return evaluate(model, val_loader, device, criterion)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/scratch/joshi.shreyas/wafer_processed')
    parser.add_argument('--n_configs', type=int, default=8)
    parser.add_argument('--epochs_per_config', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--output_dir', type=str, default='/scratch/joshi.shreyas/wafer_results/hp_search')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading data...")
    train_loader, val_loader, _, _ = get_dataloaders(args.data_dir, batch_size=args.batch_size, num_workers=4)
    class_weights = compute_class_weights(train_loader, NUM_CLASSES, device)

    lr_choices = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
    wd_choices = [0.0, 1e-5, 1e-4, 1e-3]

    configs = [{'lr': random.choice(lr_choices), 'weight_decay': random.choice(wd_choices)}
               for _ in range(args.n_configs)]
    configs.append({'lr': 1e-3, 'weight_decay': 1e-4, 'is_default': True})

    results = []
    print(f"\nSearching {len(configs)} configs, {args.epochs_per_config} epochs each...\n")

    for i, cfg in enumerate(configs):
        start = time.time()
        metrics = train_one_config(
            cfg['lr'], cfg['weight_decay'], args.epochs_per_config,
            train_loader, val_loader, class_weights, device
        )
        elapsed = time.time() - start

        tag = " (DEFAULT - used in full run)" if cfg.get('is_default') else ""
        print(f"Config {i+1}/{len(configs)}{tag}: lr={cfg['lr']}, wd={cfg['weight_decay']} "
              f"-> Val Macro-F1: {metrics['macro_f1']:.4f}, Val Balanced-Acc: {metrics['balanced_accuracy']:.4f} "
              f"({elapsed:.1f}s)")

        results.append({**cfg, **metrics})

    results_sorted = sorted(results, key=lambda r: r['macro_f1'], reverse=True)

    print("\n=== RANKED RESULTS (best to worst, by Val Macro-F1) ===")
    for r in results_sorted:
        tag = " <- DEFAULT" if r.get('is_default') else ""
        print(f"lr={r['lr']}, wd={r['weight_decay']}: Macro-F1={r['macro_f1']:.4f}, "
              f"Balanced-Acc={r['balanced_accuracy']:.4f}{tag}")

    default_rank = next(i for i, r in enumerate(results_sorted) if r.get('is_default'))
    print(f"\nDefault hyperparameters ranked #{default_rank + 1} out of {len(results_sorted)} configs tested.")

    with open(output_dir / 'search_results.json', 'w') as f:
        json.dump(results_sorted, f, indent=2)

    print(f"\nResults saved to: {output_dir / 'search_results.json'}")


if __name__ == "__main__":
    main()
