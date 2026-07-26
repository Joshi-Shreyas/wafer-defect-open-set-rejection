"""
finetune.py

Regime B, stage 2: Fine-tuning.
Loads the MAE-pretrained encoder (from train_mae.py) and fine-tunes it on the SAME
120,960 labeled training wafers that Regime A (train_supervised.py) used - this keeps
the comparison fair, since the only difference between Regime A and Regime B is
whether the backbone started from random weights or MAE-pretrained weights.

Usage:
    python finetune.py --epochs 30 --batch_size 128 --lr 0.0005
"""

import argparse
import json
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
    parser.add_argument('--pretrained_encoder', type=str,
                         default='/scratch/joshi.shreyas/wafer_results/mae/pretrained_encoder.pt')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4,
                         help='Lower than Regime A default - fine-tuning a pretrained model typically needs a smaller LR')
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='/scratch/joshi.shreyas/wafer_results/finetuned')
    parser.add_argument('--patience', type=int, default=7)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading data...")
    train_loader, val_loader, test_loader, unknown_loader = get_dataloaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Train: {len(train_loader.dataset)}  Val: {len(val_loader.dataset)}  Test: {len(test_loader.dataset)}")

    print("\nComputing class weights...")
    class_weights = compute_class_weights(train_loader, NUM_CLASSES, device)

    # Build the same architecture, then load MAE-pretrained weights instead of random init
    model = build_model(num_classes=NUM_CLASSES, in_channels=2).to(device)

    print(f"\nLoading pretrained encoder weights from: {args.pretrained_encoder}")
    pretrained_state = torch.load(args.pretrained_encoder, map_location=device, weights_only=True)

    # The pretrained encoder was saved as a full WaferResNet state_dict (including a randomly-initialized
    # fc head that was never actually trained during MAE pretraining, since MAE only supervises the decoder).
    # We load everything except fc, since that classification head needs to be learned fresh during fine-tuning.
    model_dict = model.state_dict()
    filtered_state = {k: v for k, v in pretrained_state.items() if not k.startswith('fc.')}
    model_dict.update(filtered_state)
    model.load_state_dict(model_dict)
    print(f"Loaded {len(filtered_state)} pretrained layers (excluding classification head, which starts fresh).")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': [], 'val_macro_f1': [], 'val_balanced_acc': [], 'val_accuracy': []}
    best_val_f1 = -1.0
    epochs_without_improvement = 0

    print(f"\nStarting fine-tuning for up to {args.epochs} epochs...\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x.size(0)

            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        train_loss = running_loss / len(train_loader.dataset)
        val_metrics = evaluate(model, val_loader, device, criterion)
        scheduler.step(val_metrics['macro_f1'])
        epoch_time = time.time() - epoch_start

        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val Macro-F1: {val_metrics['macro_f1']:.4f} | Val Balanced-Acc: {val_metrics['balanced_accuracy']:.4f}")

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_macro_f1'].append(val_metrics['macro_f1'])
        history['val_balanced_acc'].append(val_metrics['balanced_accuracy'])
        history['val_accuracy'].append(val_metrics['accuracy'])

        if val_metrics['macro_f1'] > best_val_f1:
            best_val_f1 = val_metrics['macro_f1']
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_macro_f1': best_val_f1,
            }, output_dir / 'best_model.pt')
            print(f"  -> New best model saved (Val Macro-F1: {best_val_f1:.4f})")
        else:
            epochs_without_improvement += 1
            print(f"  -> No improvement ({epochs_without_improvement}/{args.patience})")

        print("-" * 70)

        if epochs_without_improvement >= args.patience:
            print(f"\nEarly stopping triggered after {epoch} epochs.")
            break

    with open(output_dir / 'history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nFine-tuning complete. Best Val Macro-F1: {best_val_f1:.4f}")

    print("\nLoading best checkpoint for final test evaluation...")
    checkpoint = torch.load(output_dir / 'best_model.pt', map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])

    test_metrics = evaluate(model, test_loader, device, criterion)
    print(f"\nFinal Test Results (Regime B - MAE-pretrained + fine-tuned):")
    print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"  Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    print(f"  Test Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)


if __name__ == "__main__":
    main()
