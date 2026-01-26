"""
Custom dataset class for the project.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Any, Union

import torch
from torch.utils.data import Dataset


Sample = Dict[str, Any]
Transform = Callable[[Sample], Sample]



class Normalize(object):
    """Rescale all concentration by the largest concentration in the dataset.
    """
    def __init__(self, max_concentration):
        self.max_c0 = max_concentration

    def __call__(self, sample):
        # find max concentration across all c0 values
        new_sample = sample.copy()
        new_sample["c0"] = sample["c0"] / self.max_c0
        if "y_full" in sample:
            new_sample["y_full"] = sample["y_full"] / self.max_c0
        if "y" in sample:
            new_sample["y"] = sample["y"] / self.max_c0
        return new_sample


class KineticDataset(Dataset):
    """
    Dataset for samples produced by SyntheticDataGenerator.

    Supports:
      - passing an in-memory batch dict directly
      - passing a .pt file path created by torch.save(batch, path)

    Each item is a dict with keys like:
      t: (T,)
      y: (T,)
      full: (T, n_species)
      c0: (n_species,)
      c0_unit: (n_species,)
      log_c0_scale: ()
      k: (n_rxns,)
      label: python dict (metadata)
    """

    def __init__(
        self,
        data: Union[str, Dict[str, Any]],
        transform: Optional[Transform] = None,
        return_full: bool = True,
        return_label: bool = False,
    ):
        """
        Args:
            data: Either
                - path to a .pt file (saved batch dict), or
                - an in-memory batch dict from generator.generate_dataset(...)
            transform: Optional callable applied to each sample dict
            return_full: include "full" trajectories in returned sample
            return_label: include python "label" dict in returned sample
        """
        if isinstance(data, str):
            batch = torch.load(data, map_location="cpu")
        elif isinstance(data, dict):
            batch = data
        else:
            raise TypeError("data must be a file path (str) or a batch dict")

        self.batch = batch
        self.transform = transform
        self.return_full = return_full
        self.return_label = return_label

        self._n = int(self.batch["y"].shape[0])  # (B, T)


    def __len__(self) -> int:
        return self._n

    def get_t(self) -> torch.Tensor:
        """Get shared time grid tensor (T,)"""
        return self.batch["t"]

    def get_B_null_true(self) -> torch.Tensor:
        return self.batch["B_null_true"]

    def get_B_null_full(self) -> torch.Tensor:
        return self.batch["B_null_full"]
    
    def get_output_index(self) -> torch.Tensor:
        """Get output index tensor (1,)"""
        return self.batch["out_index"]

    def __getitem__(self, idx: int) -> Sample:
        # Shared time grid is stored once (T,)
        t = self.batch["t"]

        # Per-sample fields
        sample: Sample = {
            "y": self.batch["y"][idx],
        }

        # Add optional numeric fields if they exist
        for key in ["c0",  "k"]:
            if key in self.batch:
                val = self.batch[key]
                # batched tensors: index first dim, scalars: handle carefully
                if torch.is_tensor(val):
                    sample[key] = val[idx] if val.ndim >= 1 and val.shape[0] == self._n else val
                else:
                    sample[key] = val

        if self.return_full and "y_full" in self.batch:
            sample["y_full"] = self.batch["y_full"][idx]

        if self.return_label and "label" in self.batch:
            # labels is a list[dict]
            sample["label"] = self.batch["label"][idx]

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
