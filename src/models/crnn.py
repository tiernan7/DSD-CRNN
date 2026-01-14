import torch
import torch.nn as nn
from torchdiffeq import odeint
import numpy as np
from scipy.linalg import null_space

class CRNN(nn.Module):
    def __init__(self, N: np.ndarray, num_reactions: int, eps: float = 1e-12):
        super().__init__()
        # A: (M, N) M = number of atom types, N = number of species
        # B: (N, K) K = dimension of null space
        # # B * v = 0 for any valid reaction flux v
        # W: (K, R) R = number of reactions
        # nu: (N, R) stoichiometric matrix
        # log_k: (R,) log reaction rates
        # C: (X, N) concentrations for X batch samples
        
        
        B = torch.tensor(null_space(N).astype(np.float32))  # (N, K)
        self.register_buffer("B", B)
        N = B.shape[0]
        K = B.shape[1]
        R = num_reactions

        # nonzero init
        self.W = nn.Parameter(1e-2 * torch.randn(K, R))
        self.log_k = nn.Parameter(torch.full((R,), np.log(2e-4), dtype=torch.float32))

        self.eps = eps

    def nu(self):
        # (N,R)
        return self.B @ self.W

    def rhs_C(self, t, C):
        # C: (X, N)
        C = C.clamp_min(self.eps)

        nu = self.nu()                  # (N, R) 
        alpha = torch.relu(nu)         # (N, R) which species are reactants
        beta = -nu                     # (N, R) reaction stoichiometry
        # log r = log k + log(C) @ alpha
        logC = torch.log(C)
        log_r = self.log_k + logC @ alpha      # (X, R)
        r = torch.exp(log_r)                   # (X, R)

        dCdt = r @ beta.T                     # (X, N)
        return dCdt


    def forward(self, t, c0):
        t = t.flatten()                 # ensure 1D
        C0 = c0.clamp_min(self.eps)
        C_traj = odeint(self.rhs_C, C0, t, method="dopri5", rtol=1e-7, atol=1e-9)# (T,B,S)
        return C_traj.permute(1,0,2)    # (B,T,S)

