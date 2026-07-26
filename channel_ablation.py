"""
channel_ablation.py

Tests whether the validity mask channel (added during preprocessing to distinguish
real scanned area from artificial padding) actually contributes to classification
performance, or whether the model does just as well using only the raw wafer map channel.

Trains a 1-channel (wafer only) version of the SAME architecture, supervised-only,
for a shorter run than the full Regime A training (since we just need a directional
comparison, not a fully-converged model) - then compares to Regime A's already-recorded
2-channel test results.

Usage:
    python channel_ablation.py --epochs 15
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, balanced_accuracy_score

from data_utils import WaferDataset, NUM_CLASSES
from model import build_model


class WaferOnlyWrapper(torch.utils.data.Dataset):
    """Wraps a WaferDataset but strips the mask channel, returning only the wafer channel."""
    def __init__(self, base_dataset):
        self.base = base_dataset

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x, y = self.base[idx]
        return x[0:1], y  # keep only channel 0 (wafer map), drop channel 1 (mask)


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
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)
            preds = logits.argmax(dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return {
        'loss': total_loss / len(loader.dataset),
        'accuracy': (all_preds == all_labels).mean(),
        'macro_f1': f1_score(all_labels, all_preds, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(all_labels, all_preds),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/scratch/joshi.shreyas/wafer_processed')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output_dir', type=str, default='/scratch/joshi.shreyas/wafer_results/ablation_no_mask')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("Running channel ablation: WAFER MAP ONLY (mask channel removed)\n")

    print("Loading data (stripping mask channel)...")
    train_ds = WaferOnlyWrapper(WaferDataset(f"{args.data_dir}/train_data.npz"))
    val_ds = WaferOnlyWrapper(WaferDataset(f"{args.data_dir}/val_data.npz"))
    test_ds = WaferOnlyWrapper(WaferDataset(f"{args.data_dir}/test_data.npz"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")

    class_weights = compute_class_weights(train_loader, NUM_CLASSES, device)

    # Build model with in_channels=1 this time (no mask channel)
    model = build_model(num_classes=NUM_CLASSES, in_channels=1).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_f1 = -1.0
    epochs_without_improvement = 0
    patience = 5

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * x.size(0)

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_metrics['macro_f1'])
        epoch_time = time.time() - epoch_start

        print(f"Epoch {epoch}/{args.epochs} ({epoch_time:.1f}s) | Train Loss: {train_loss:.4f} | "
              f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | Val Balanced-Acc: {val_metrics['balanced_accuracy']:.4f}")

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            epochs_without_improvement = 0
            torch.save({'model_state_dict': model.state_dict()}, output_dir / 'best_model.pt')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    test_metrics = evaluate(model, test_loader, device, criterion)

    print(f"\n=== ABLATION RESULT: Wafer-map-only (no validity mask channel) ===")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")

    print(f"\n=== COMPARISON: Regime A with BOTH channels (from earlier run) ===")
    print(f"Test Accuracy: 0.9723")
    print(f"Test Macro-F1: 0.8603")
    print(f"Test Balanced Accuracy: 0.8754")

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
