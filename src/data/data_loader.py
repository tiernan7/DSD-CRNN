"""
Data loader utilities for creating train/validation/test loaders.
"""

from torch.utils.data import DataLoader
import torch

def get_data_loaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: bool | None = None,
    persistent_workers: bool | None = None,
    device: str | torch.device | None = None,
):
    """
    Create data loaders for training, validation, and testing.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = str(device)

    if pin_memory is None:
        pin_memory = device.startswith("cuda")

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=False,
    )

    return train_loader, val_loader, test_loader