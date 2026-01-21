import torch
import torch.nn as nn

class RangeNormalizedMAE(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        # y_*: (B, T) or (B, T, S)
        y_range = y_true.amax(dim=1, keepdim=True).clamp_min(self.eps).detach()
        return torch.mean(torch.abs(y_pred - y_true) / y_range)
    
    
class RangeNormalizedMSE(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        y_pred, y_true: (B, T) or (B, T, S)
        Normalizes by range computed from y_true along time dimension.
        """
        # compute per-sample (and per-species if present) range from y_true only
        y_min = y_true.amin(dim=1, keepdim=True)
        y_max = y_true.amax(dim=1, keepdim=True)
        y_range = (y_max - y_min).clamp_min(self.eps)

        y_pred_n = (y_pred) / y_range
        y_true_n = (y_true) / y_range
        return torch.mean((y_pred_n - y_true_n)**2)