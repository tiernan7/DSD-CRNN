"""
Synthetic data generation for model training and testing.

This module provides functions to generate synthetic data that can be used
for training and testing the models when real data is not available.
"""

from dataclasses import dataclass
import itertools
import itertools
from operator import le
from unicodedata import name
from networkx import sigma
from numpy.linalg import lstsq
import numpy as np
from numpy.linalg import matrix_rank, svd
import torch
from  itertools import combinations_with_replacement
from scipy.integrate import solve_ivp
import multiprocessing
from multiprocessing import Pool
@dataclass
class Partition:
    inputs: list[str]
    hidden: list[str]
    output: str = "S_out"

    @property
    def species(self) -> list[str]:
        return self.inputs + self.hidden + [self.output]
        
    @property
    def input_indices(self) -> list[int]:
        return [self.species.index(name) for name in self.inputs]

    @property
    def output_indices(self) -> list[int]:
        return [self.species.index(self.output)]
    
    
@dataclass(frozen=True)
class Reaction:
    reactants: tuple[int, ...]
    products: tuple[int, ...]
    


class SyntheticDataGenerator:
    def __init__(
        self,
        mechanism: str,
        reaction_rates,
        initial_concentration_ranges: dict,
        E_bounds: tuple[float, float],
        noise_level=0.05,
        t_end=40.0,
        n_timepoints=50,
        seed=None,
    ):
        
        self.rng = np.random.default_rng(seed)
        self.mechanism = mechanism
        self.k = reaction_rates
        self.initial_concentration_ranges = initial_concentration_ranges
        self.noise_level = noise_level
        self.E_bounds = E_bounds
        self.t_end = t_end
        self.n_timepoints = n_timepoints

        self.seed = seed

        (   self.composition_matrix,
            self.partition,
            self.reactions,
        ) = self.mech_to_f(mechanism)()
    
        self.c0 = [-1.0] * len(self.partition.species)  # placeholder; will be sampled later

        assert len(self.k) == len(self.reactions)
        assert len(self.c0) == len(self.partition.species)
        assert set(self.initial_concentration_ranges.keys()) == set(self.partition.species)

       
    
    def mech_to_f(self, mechanism: str):
        """
        Convert mechanism string to function.
        Currently only supports hardcoded mechanisms.
        """
        if mechanism == "one_step":
            return self.setup_hardcoded_single_step
        elif mechanism == "one_step_1latent":
            return self.setup_hardcoded_1latent_example
        else:
            raise ValueError(f"Unknown mechanism: {mechanism}")


    def _sample_c0(self, c_range):
        """
        Sample initial concentrations.
        Inputs get nonzero concentrations; hidden/output start at 0.
        """
        low, high = c_range
        if low == high:
            return low
        else:
            return self.rng.uniform(low, high)
        
    def _build_stoichiometric_matrix(self):
        """
        Build the stoichiometric matrix S from the reactions.
        Returns:
            np.ndarray: Stoichiometric matrix S of shape (n_species, n_reactions)
        """
        n_species = len(self.partition.species)
        n_rxns = len(self.reactions)
        S = np.zeros((n_species, n_rxns), dtype=float)

        for j, rxn in enumerate(self.reactions):
            for i in rxn.reactants:
                S[i, j] -= 1
            for i in rxn.products:
                S[i, j] += 1

        return S

    def _rate_vector(self, c):
        r = np.zeros(len(self.reactions), dtype=float)
        c = np.clip(c, 0.0, np.inf)

        for j, rxn in enumerate(self.reactions):
            val = self.k[j]
            for i in rxn.reactants:
                val *= c[i]
            r[j] = val

        return r



    def sample_noise(self, C):
        sigma = self._compute_sigma(C)
        C_noisy = C + self.noise_level * sigma * self.rng.standard_normal(C.shape)
        return C_noisy

    
    
    

    @staticmethod
    def _compute_sigma(C):
        sigma = np.std(C, axis=0) + 1e-12
        return sigma

    def _compute_sensitivites(self, c0):
        """
        Compute sensitivities of concentrations to parameters.
        Usinging the formulas: 
            t: timepoints
            c0: initial concentrations (num_species,)
            phi = log(k) (num_params,)
            f(t) = dC/dt (num_species,)
            U = dC/dphi (num_species, num_params)
            A = df/dC (num_species, num_species)
            B = df/dphi (num_species, num_params)
            df/dphi = df/dk * dk/dphi = df/dk * k
            dU/dt = A U + B (num_species, num_params)
             = (df/dC) dC/dphi + df/dphi

            
            Assumes multiplicities are always 1
        """
        k = self.k
        n_times = self.n_timepoints
        n_species = len(self.partition.species)
        n_params = len(self.k)
        gamma = self._build_stoichiometric_matrix() #(n_species,  n_reactions) 
        num_reactions = len(self.reactions) 
        U0 = 0.0 * np.ones((n_species, n_params))
        alpha = np.zeros((n_species, num_reactions))
        for r in range(gamma.shape[1]):
            for s in self.reactions[r].reactants:
                alpha[s, r] += 1 # 
                
                
        def Bc(c, gamma):
            # df(t)/dphi (num_species, num_params)
            v = self._rate_vector(c)  # (num_params,)
            B = []
            for ri in range(gamma.shape[1]):
                bi = gamma[:, ri] * v[ri]
                B.append(bi)
            return np.array(B).T
                
        def Ac(c, gamm, alpha):
            # df(t)/dC (num_species, num_species)
            V = [] # (num_reaction, num_species)
            for r in range(gamma.shape[1]):
                row = []
                for l in range(len(c)):
                    if not(alpha[l][r] > 0):
                        row.append(0.0)
                    else:
                        prod = np.prod([c[j] for j in self.reactions[r].reactants if j != l])
                        row.append(self.k[r] * prod)
                V.append(row)
            V = np.array(V)
            return gamma @ V
        
        def rhs(t, z, gamma, alpha):
            c = z[:n_species]
            
            U_flat = z[n_species:]
            
            
            U = U_flat.reshape((n_species, n_params))
            A = Ac(c, gamma, alpha)
            B = Bc(c, gamma)
            
            dc_dt = gamma @ self._rate_vector(c)
            dU_dt = (A @ U + B).flatten()
            
            dz_dt = np.concatenate([dc_dt, dU_dt])
            return dz_dt
        
        z0 = np.concatenate([c0, U0.flatten()])
        S = solve_ivp(rhs, (0, self.t_end), z0, t_eval=np.linspace(0, self.t_end, self.n_timepoints), args=(gamma, alpha), method="BDF", rtol=1e-10, atol=1e-12)
        C = S.y[:n_species, :].T
        U = S.y[n_species:, :].T.reshape((n_times, n_species, n_params))
        return C, U
                      
    def _compute_fim_output(self, c0, summed = True):
        """
        Compute Fisher Information Matrix.
        FIM_ij = sum_t (1/sigma_t^2) * (dC_out/dphi_i)T(dC_out/dphi_j) (outer prod)
        where sigma_t is the noise std at time t
        """
        C, U = self._compute_sensitivites(c0)
        out_idx = self.partition.output_indices[0]
        sigma_out = self._compute_sigma(C)[out_idx]
        fim = np.zeros((self.n_timepoints, len(self.k), len(self.k)))
        for ti in range(self.n_timepoints):
            Ut = U[ti, out_idx]
            fim[ti] += (1 / (sigma_out ** 2)) * np.outer(Ut, Ut)
        if summed:
            fim = np.sum(fim, axis=0)
        return fim
     
     
     
     
    @staticmethod
    def _SVD(fim):
        U, s, Vh = svd(fim)
        return U, s, Vh
     
    @staticmethod
    def _restric_fim(svd_fim):
        U, s, Vh = svd_fim
        if s.ndim == 1:
            rank = s
            U_r = U
            s_r = s
            Vh_r = Vh
        else:
            rank = len([i for i in np.diagonal(s) if i > 1e-8])
            U_r = U[:, :rank]
            s_r = s[:rank]
            Vh_r = Vh[:rank, :]
        fim_restricted = U_r @ np.diag(s_r) @ Vh_r
        return fim_restricted
     
     
    
    @staticmethod
    def _E_from_fim(svd_fim):
        """
        Compute the E-optimality criterion (trace of inverse FIM).
        """
        U, s, Vh = svd_fim
        if s.ndim == 1:
            eigenvalues = s**2
        else:
            eigenvalues = [s**2 for s in np.diagonal(s) if s > 1e-8]
        return np.min(eigenvalues)
     
    def simulate_mean(self):
        """
        Simulate the ODE system without noise.
        Returns:
            t_eval (np.ndarray): Time points of shape (n_timepoints,)
            C (np.ndarray): Concentrations of shape (n_timepoints, n_species)"""
        S = self._build_stoichiometric_matrix()

        def rhs(t, c):
            return S @ self._rate_vector(c)

        t_eval = np.linspace(0, self.t_end, self.n_timepoints)

        sol = solve_ivp(
            rhs,
            (0, self.t_end),
            self.c0,
            t_eval=t_eval,
            method="LSODA",
            rtol = 1e-6,
            atol = 1e-9,
        )

        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")


        C = sol.y.T


        return t_eval, C


    
    def setup_hardcoded_single_step(self):
        """
        Hardcode a 1-step cascade with intermediate:
            S_1 + S_2 → S_out
        With a valid atom-conserving composition matrix.
        """
        # Partition matches naming convention
        partition = Partition(inputs=['S_1', 'S_2'], hidden=[], output='S_out')
        # Index order: [S_1, S_2, S_out]
        # Atoms: A, B, and C
        composition_matrix = np.array([
            [1, 0, 1],  # Atom A
            [0, 1, 1],  # Atom B
            [0, 1, 1],  # Atom C
        ], dtype = int)


        # Use indices [0,1,2] for species in partition.species
        reactions = [
            Reaction(reactants=(0, 1), products=(2,)),  # S_1 + S_2 → S_out
        ]

        return composition_matrix, partition, reactions


    
    def setup_hardcoded_1latent_example(self):
        """
        Hardcode a 1-step cascade with intermediate:
            S_1 + S_2 → S_3
            S_3 → S_4 + S_out
        With a valid atom-conserving composition matrix.
        """
        # Partition matches naming convention
        partition = Partition(inputs=['S_1', 'S_2'], hidden=['S_3', 'S_4'], output='S_out')
        # Index order: [S_1, S_2, S_3, S_4, S_out]
        # Atoms: A, B, and C
        composition_matrix = np.array([
            [1, 0, 1, 1, 0],  # Atom A
            [0, 1, 1, 0, 1],  # Atom B
            [0, 1, 1, 1, 0],  # Atom C
        ], dtype = int)


        # Use indices [0,1,2,3,4] for species in partition.species
        reactions = [
            Reaction(reactants=(0, 1), products=(2,)),  # S_1 + S_2 → S_3
            Reaction(reactants=(2,), products=(3, 4)),   # S_3 → S_4 + S_out
        ]

        return composition_matrix, partition, reactions


    def generate_admisable_c0(self, E_bound: tuple[float, float]):
        stop = False
        attempt = 0
        successful_c0 = []
        while attempt < 1000:
            c0 = [self._sample_c0(self.initial_concentration_ranges[s]) for s in self.partition.species]
            fim = self._compute_fim_output(c0, summed=True)
            svd_fim = self._SVD(fim)
            fim_restricted = self._restric_fim(svd_fim)
            E = self._E_from_fim(self._SVD(fim_restricted))
            if E_bound[0] < E < E_bound[1]:
                return c0
            attempt += 1
        
        raise RuntimeError("Could not find admissable c0 in 1000 attempts")
    
    
    def grid_search(self, 
                    sampling_dim,
                    c_ranges,
                    k_ranges,
                    parallel = False):
        samples = []
        c0 = [np.linspace(c_ranges[s][0], c_ranges[s][1], sampling_dim) if not c_ranges[s][0] == c_ranges[s][1] else [c_ranges[s][0]] for s in range(len(self.partition.species))]
        k = [np.linspace(k_ranges[r][0], k_ranges[r][1], sampling_dim) if not k_ranges[r][0] == k_ranges[r][1] else [k_ranges[r][0]] for r in range(len(self.reactions))]
        
        cartesian_prod = itertools.product(*c0, *k)
        
        for sample in cartesian_prod:
            c0 = sample[:len(self.partition.species)]
            k = sample[len(self.partition.species):]

            fim = self._compute_fim_output(c0, summed=True)
            svd_fim = self._SVD(fim)
            fim_restricted = self._restric_fim(svd_fim)
            E = self._E_from_fim(self._SVD(fim_restricted))
            samples.append({
                "c0": c0,
                "k": k,
                "E": E
            })

    
        return samples

    
    def generate_sample(
        self,
    ):

        self.c0 = self.generate_admisable_c0(E_bound=self.E_bounds)
        
        t, C = self.simulate_mean()
        C_noisy = self.sample_noise(C)

        out_idx = self.partition.output_indices[0]

        return {
            "t": torch.tensor(t, dtype=torch.float32),
            "c0": torch.tensor(self.c0.copy(), dtype=torch.float32),
            "y": torch.tensor(C_noisy[:, out_idx], dtype=torch.float32),
            "y_full": torch.tensor(C_noisy, dtype=torch.float32),
            "E": self._E_from_fim(self._SVD(self._compute_fim_output(self.c0, summed=True))),
            "label": {
                "mechanism": self.mechanism,
                "k": self.k,
                "c0": self.c0.copy(),
                "reactions": self.reactions,
            },
        }





    
    def generate_batch(self, batch_size: int):
        samples = [self.generate_sample() for _ in range(batch_size)]

        # shared time grid (T,)
        t = samples[0]["t"]                                      # (T,)

        # outputs
        y = torch.stack([s["y"] for s in samples], dim=0)              # (B, T)
        y_full = torch.stack([s["y_full"] for s in samples], dim=0)        # (B, T, n_species)
        c0 = torch.stack([s["c0"] for s in samples], dim=0)            # (B, n_species)

        # metadata / labels (keep python objects)
        labels = [s["label"] for s in samples]
        k = torch.tensor([lab["k"] for lab in labels], dtype=torch.float32)  # (B, n_rxns)

        return {
            "t": t,                        # (T,)
            "y": y,                        # (B, T)
            "y_full": y_full,                  # (B, T, n_species)
            "c0": c0,                  # (B, n_species)
            'A': torch.tensor(self.composition_matrix, dtype=torch.float32),  # (M, N)
            "label": labels,
            "k": k,                        # (B, n_rxns)
            "out_index": self.partition.output_indices[0],  # (1,)
            "E": torch.tensor([s["E"] for s in samples], dtype=torch.float32),  # (B,)
        }



    
    def generate_dataset(self, num_samples: int, save_path: str | None = None):
        batch = self.generate_batch(num_samples)

        if save_path is not None:
            # labels are python objects; torch.save can handle them
            torch.save(batch, save_path)

        return batch

    