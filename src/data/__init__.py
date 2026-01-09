"""
Data module for loading and processing datasets.

Public API:
- CustomDataset
- get_data_loaders
"""

from .dataset import CustomDataset
from .data_loader import get_data_loaders
from .synthetic_data import SyntheticDataGenerator

__all__ = ["CustomDataset", "get_data_loaders", "SyntheticDataGenerator"]
