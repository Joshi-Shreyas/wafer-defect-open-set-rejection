"""
train_mae.py

Regime B, stage 1: Self-supervised (masked reconstruction) pretraining on the
638,507-wafer unlabeled pool, using the same ResNet backbone architecture as the
supervised baseline.

Masking is restricted to patches that are substantially real (scanned) wafer area,
using the validity mask generated during preprocessing - this avoids the trivial
"reconstruct pure padding" shortcut discussed in the project proposal.

Usage:
    python train_mae.py --epochs 20 --batch_size 256 --mask_ratio 0.6
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from data_utils import get_unlabeled_dataloader
from model import build_model


def generate_patch_mask(validity_mask, patch_size=8, mask_ratio=0.6, min_valid_frac=0.5):
    """
    Given a batch of validity masks (B, H, W), select a random subset of patches to mask,
    restricted to patches that are at least `min_valid_frac` real (non-padding) area.

    Returns a pixel-level binary mask (B, H, W): 1 = masked (reconstruction target), 0 = visible.
    """
    B, H, W = validity_mask.shape
    n = H // patch_size  # patches per side

    # Reshape into patch grid and compute each patch's fraction of real (valid) pixels
    v = validity_mask.view(B, n, patch_size, n, patch_size)
    patch_valid_frac = v.float().mean(dim=(2, 4))  # (B, n, n)
    eligible = patch_valid_frac >= min_valid_frac   # (B, n, n) bool

    patch_mask = torch.zeros(B, n, n, device=validity_mask.device)

    for b in range(B):
        elig_idx = eligible[b].nonzero(as_tuple=False)  # (num_eligible, 2)
        num_elig = elig_idx.shape[0]
        num_to_mask = int(num_elig * mask_ratio)
        if num_to_mask > 0:
            perm = torch.randperm(num_elig, device=validity_mask.device)[:num_to_mask]
            chosen = elig_idx[perm]
            patch_mask[b, chosen[:, 0], chosen[:, 1]] = 1

    # Upsample patch-level mask back to pixel level
    pixel_mask = patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    return pixel_mask  # (B, H, W)


class MAEDecoder(nn.Module):
    """
    Lightweight decoder: upsamples the encoder's spatial feature map (12x12x512,
    given our 96x96 input and three stride-2 stages) back to a 96x96x1 reconstruction
    of the wafer channel only (the validity mask channel is given as input, not predicted).
    """

    def __init__(self, feature_dim=512):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feature_dim, 256, kernel_size=4, stride=2, padding=1),  # 12 -> 24
            nn.BatchNorm2d(256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 24 -> 48
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 48 -> 96
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=3, padding=1),  # reconstruct wafer channel
        )

    def forward(self, x):
        return self.decoder(x)


class MAEModel(nn.Module):
    """Wraps the shared encoder (from model.py) with the lightweight MAE decoder."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = MAEDecoder(feature_dim=encoder.feature_dim)

    def encode_spatial(self, x):
        """Returns the pre-pooling spatial feature map (B, 512, 12, 12)."""
        x = self.encoder.stem(x)
        x = self.encoder.layer1(x)
        x = self.encoder.layer2(x)
        x = self.encoder.layer3(x)
        x = self.encoder.layer4(x)
        return x

    def forward(self, x):
        feat = self.encode_spatial(x)
        recon = self.decoder(feat)
        return recon


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/scratch/joshi.shreyas/wafer_processed')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--mask_ratio', type=float, default=0.6)
    parser.add_argument('--patch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--output_dir', type=str, default='/scratch/joshi.shreyas/wafer_results/mae')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint if present')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("\nLoading unlabeled pool...")
    unlabeled_loader = get_unlabeled_dataloader(
        args.data_dir, batch_size=args.batch_size, num_workers=args.num_workers
    )
    print(f"Unlabeled pool: {len(unlabeled_loader.dataset)} samples, {len(unlabeled_loader)} batches/epoch")

    encoder = build_model(num_classes=8, in_channels=2).to(device)  # num_classes unused here, just need the encoder
    mae_model = MAEModel(encoder).to(device)
    print(f"\nEncoder parameters: {sum(p.numel() for p in encoder.parameters()):,}")
    print(f"Decoder parameters: {sum(p.numel() for p in mae_model.decoder.parameters()):,}")

    optimizer = torch.optim.AdamW(mae_model.parameters(), lr=args.lr, weight_decay=1e-4)

    start_epoch = 1
    checkpoint_path = output_dir / 'mae_checkpoint.pt'
    if args.resume and checkpoint_path.exists():
        print(f"\nResuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        mae_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    print(f"\nStarting MAE pretraining: epochs {start_epoch} to {args.epochs}, mask_ratio={args.mask_ratio}\n")

    loss_history = []

    for epoch in range(start_epoch, args.epochs + 1):
        mae_model.train()
        epoch_start = time.time()
        running_loss = 0.0
        num_batches = 0

        for batch_idx, (x, _) in enumerate(unlabeled_loader):
            x = x.to(device)  # (B, 2, 96, 96): channel 0 = wafer, channel 1 = validity mask
            wafer = x[:, 0]
            validity = x[:, 1]

            patch_mask = generate_patch_mask(
                validity, patch_size=args.patch_size, mask_ratio=args.mask_ratio
            )

            # Zero out the wafer channel wherever masked; keep validity mask channel visible
            # (it only reveals padding vs. real-area layout, not the defect pattern itself)
            masked_input = x.clone()
            masked_input[:, 0] = wafer * (1 - patch_mask)

            optimizer.zero_grad()
            recon = mae_model(masked_input).squeeze(1)  # (B, 96, 96)

            # Loss only over masked AND genuinely real pixels
            loss_mask = patch_mask * validity
            mse = ((recon - wafer) ** 2) * loss_mask
            loss = mse.sum() / (loss_mask.sum() + 1e-6)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num_batches += 1

            if batch_idx % 200 == 0:
                print(f"  Epoch {epoch} | Batch {batch_idx}/{len(unlabeled_loader)} | Recon Loss: {loss.item():.5f}")

        avg_loss = running_loss / num_batches
        epoch_time = time.time() - epoch_start
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s) | Avg Reconstruction Loss: {avg_loss:.5f}")
        print("-" * 70)

        loss_history.append(avg_loss)

        # Checkpoint every epoch (important for long jobs that might hit time limits)
        torch.save({
            'epoch': epoch,
            'model_state_dict': mae_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_history': loss_history,
        }, checkpoint_path)

    # Save just the encoder weights separately, ready for finetune.py to load
    torch.save(encoder.state_dict(), output_dir / 'pretrained_encoder.pt')
    print(f"\nMAE pretraining complete. Pretrained encoder saved to: {output_dir / 'pretrained_encoder.pt'}")


if __name__ == "__main__":
    main()
