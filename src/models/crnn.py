import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
from scipy.linalg import null_space

class CRNN(nn.Module):
    def __init__(self, N: np.ndarray, num_reactions: int, eps: float = 1e-12):
        super().__init__()
        B = torch.tensor(null_space(N).astype(np.float32))  # (S, K)
        self.register_buffer("B", B)
        K = B.shape[1]

        # IMPORTANT: nonzero init
        self.W = nn.Parameter(1e-2 * torch.randn(K, num_reactions))
        self.log_k = nn.Parameter(torch.zeros(num_reactions))

        self.eps = eps

    def nu(self):
        # (S,R)
        return self.B @ self.W

    def rhs_C(self, t, C):
        # C: (B, S)
        C = C.clamp_min(self.eps)

        nu = self.nu()                  # (S, R)
        alpha = torch.relu(-nu)         # (S, R) reactant stoich/orders

        # log r = log k + log(C) @ alpha
        logC = torch.log(C)
        log_r = self.log_k + logC @ alpha      # (B, R)
        r = torch.exp(log_r)                   # (B, R)

        dCdt = r @ nu.T                        # (B, S)
        return dCdt


    def forward(self, t, c0):
        t = t.flatten()                 # ensure 1D
        C0 = c0.clamp_min(self.eps)
        C_traj = odeint(self.rhs_C, C0, t, method="dopri5")  # (T,B,S)
        return C_traj.permute(1,0,2)    # (B,T,S)

