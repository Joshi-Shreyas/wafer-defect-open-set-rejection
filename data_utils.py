"""
data_utils.py

PyTorch Dataset classes for the wafer defect open-set rejection project.
Loads the preprocessed .npz files (wafers + masks + labels) saved during preprocessing.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# Fixed class list for known classes (excludes Near-full, which is held out)
KNOWN_CLASSES = ['none', 'Center', 'Donut', 'Edge-Loc', 'Edge-Ring', 'Loc', 'Random', 'Scratch']
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(KNOWN_CLASSES)}
IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}
NUM_CLASSES = len(KNOWN_CLASSES)


class WaferDataset(Dataset):
    """
    Loads a preprocessed split (.npz file with 'wafers', 'masks', and optionally 'labels').
    Returns a 2-channel tensor per sample: [wafer_map, validity_mask].

    For the unlabeled pool (no 'labels' key), label defaults to -1.
    """

    def __init__(self, npz_path, has_labels=True):
        data = np.load(npz_path, allow_pickle=True)
        self.wafers = data['wafers']  # (N, 96, 96), uint8, values 0/1/2
        self.masks = data['masks']    # (N, 96, 96), uint8, values 0/1
        self.has_labels = has_labels

        if has_labels and 'labels' in data:
            raw_labels = data['labels']
            self.labels = np.array([
                CLASS_TO_IDX.get(str(lbl), -1) for lbl in raw_labels
            ], dtype=np.int64)
        else:
            self.labels = np.full(len(self.wafers), -1, dtype=np.int64)

    def __len__(self):
        return len(self.wafers)

    def __getitem__(self, idx):
        wafer = self.wafers[idx].astype(np.float32)
        mask = self.masks[idx].astype(np.float32)

        # Normalize wafer values (0=background, 1=normal die, 2=defective die) to [0, 1]
        wafer = wafer / 2.0

        # Stack into 2-channel tensor: [wafer_map, validity_mask]
        combined = np.stack([wafer, mask], axis=0)  # (2, 96, 96)

        x = torch.from_numpy(combined).float()
        y = torch.tensor(self.labels[idx], dtype=torch.long)

        return x, y


def get_dataloaders(data_dir, batch_size=128, num_workers=4):
    """
    Convenience function to build train/val/test/unknown DataLoaders
    from the saved .npz files.
    """
    from torch.utils.data import DataLoader

    train_ds = WaferDataset(f"{data_dir}/train_data.npz")
    val_ds = WaferDataset(f"{data_dir}/val_data.npz")
    test_ds = WaferDataset(f"{data_dir}/test_data.npz")
    unknown_ds = WaferDataset(f"{data_dir}/unknown_data.npz")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                               num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    unknown_loader = DataLoader(unknown_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, unknown_loader


def get_unlabeled_dataloader(data_dir, batch_size=256, num_workers=4):
    """
    Separate loader for the large unlabeled pool, used for MAE pretraining.
    """
    from torch.utils.data import DataLoader

    unlabeled_ds = WaferDataset(f"{data_dir}/unlabeled_data.npz", has_labels=False)
    unlabeled_loader = DataLoader(unlabeled_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=num_workers, pin_memory=True)
    return unlabeled_loader


if __name__ == "__main__":
    # Quick sanity check when run directly
    data_dir = "/scratch/joshi.shreyas/wafer_processed"
    train_loader, val_loader, test_loader, unknown_loader = get_dataloaders(data_dir, batch_size=64)

    print(f"Train batches: {len(train_loader)}  ({len(train_loader.dataset)} samples)")
    print(f"Val batches: {len(val_loader)}  ({len(val_loader.dataset)} samples)")
    print(f"Test batches: {len(test_loader)}  ({len(test_loader.dataset)} samples)")
    print(f"Unknown batches: {len(unknown_loader)}  ({len(unknown_loader.dataset)} samples)")

    x, y = next(iter(train_loader))
    print(f"\nSample batch shape: {x.shape}  (expected: [batch, 2, 96, 96])")
    print(f"Label batch shape: {y.shape}")
    print(f"Sample labels: {y[:10].tolist()}")
    print(f"Value range - wafer channel: [{x[:,0].min():.3f}, {x[:,0].max():.3f}]")
    print(f"Value range - mask channel: [{x[:,1].min():.3f}, {x[:,1].max():.3f}]")
