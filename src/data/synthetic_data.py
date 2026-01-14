"""
Synthetic data generation for model training and testing.

This module provides functions to generate synthetic data that can be used
for training and testing the models when real data is not available.
"""

from dataclasses import dataclass
from unicodedata import name
from numpy.linalg import lstsq
import numpy as np
from numpy.linalg import matrix_rank
import torch
from  itertools import combinations_with_replacement
from scipy.integrate import solve_ivp

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


    def simulate(self):

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
        sigma = np.std(C, axis=0) + 1e-12
        C_noisy = C + self.noise_level * sigma * self.rng.standard_normal(C.shape)
        C_noisy = np.clip(C_noisy, 0.0, np.inf)


        return t_eval, C_noisy


    
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
            Reaction(reactants=(0, 1), products=(2,)),  # S_1 + S_2 → S_3 + S_4
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
            [0, 1, 1, 1, 0],  # Atom B
            [0, 1, 1, 0, 1],  # Atom C
        ], dtype = int)


        # Use indices [0,1,2,3,4] for species in partition.species
        reactions = [
            Reaction(reactants=(0, 1), products=(2,)),  # S_1 + S_2 → S_3
            Reaction(reactants=(2,), products=(3, 4)),   # S_3 → S_4 + S_out
        ]

        return composition_matrix, partition, reactions


    
    def generate_sample(
        self,
    ):

        self.c0 = [0.0] * len(self.partition.species)
        for name in self.partition.inputs:
            i = self.partition.species.index(name)
            self.c0[i] = self._sample_c0(self.initial_concentration_ranges[name])

        
        t, C = self.simulate()

        out_idx = self.partition.output_indices[0]

        return {
            "t": torch.tensor(t, dtype=torch.float32),
            "c0": torch.tensor(self.c0.copy(), dtype=torch.float32),
            "y": torch.tensor(C[:, out_idx], dtype=torch.float32),
            "y_full": torch.tensor(C, dtype=torch.float32),
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
        full = torch.stack([s["full"] for s in samples], dim=0)        # (B, T, n_species)
        c0 = torch.stack([s["c0"] for s in samples], dim=0)            # (B, n_species)

        # metadata / labels (keep python objects)
        labels = [s["label"] for s in samples]
        k = torch.tensor([lab["k"] for lab in labels], dtype=torch.float32)  # (B, n_rxns)

        return {
            "t": t,                        # (T,)
            "y": y,                        # (B, T)
            "full": full,                  # (B, T, n_species)
            "c0": c0,                  # (B, n_species)
            'A': torch.tensor(self.composition_matrix, dtype=torch.float32),  # (M, N)
            "label": labels,
            "k": k,                        # (B, n_rxns)
            "out_index": self.partition.output_indices[0],  # (1,)
        }



    
    def generate_dataset(self, num_samples: int, save_path: str | None = None):
        batch = self.generate_batch(num_samples)

        if save_path is not None:
            # labels are python objects; torch.save can handle them
            torch.save(batch, save_path)

        return batch

    