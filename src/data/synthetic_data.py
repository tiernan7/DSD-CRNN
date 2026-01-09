"""
Synthetic data generation for model training and testing.

This module provides functions to generate synthetic data that can be used
for training and testing the models when real data is not available.
"""

from dataclasses import dataclass
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
    def n_inputs(self) -> int: return len(self.inputs)

    @property
    def n_hidden(self) -> int: return len(self.hidden)

    @property
    def n_species(self) -> int: return len(self.species)
        
    @property
    def input_indices(self) -> list[int]:
        return [self.species.index(name) for name in self.inputs]

    @property
    def hidden_indices(self) -> list[int]:
        return [self.species.index(name) for name in self.hidden]

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
        number_of_nonoutput_species=None,
        reaction_rate_ranges=(1e-3, 1e-1),
        initial_concentration_range=(1.0, 10.0),
        noise_level=0.05,
        t_end=40.0,
        n_timepoints=50,
        seed=None,
    ):
        self.rng = np.random.default_rng(seed)

        self.number_of_nonoutput_species = number_of_nonoutput_species
        self.reaction_rate_ranges = reaction_rate_ranges
        self.initial_concentration_range = initial_concentration_range
        self.noise_level = noise_level

        self.t_end = t_end
        self.n_timepoints = n_timepoints

        self.seed = seed


        self.partitions = (
            self.generate_partitions()
            if number_of_nonoutput_species is not None
            else None
        )

        
    @classmethod
    def from_hardcoded_1latent(
        cls,
        reaction_rate_ranges=(1e-3, 1e-1),
        initial_concentration_range=(1.0, 10.0),
        # optional fixed values (override ranges if provided)
        reaction_rates=None,
        initial_concentrations=None,
        noise_level=0.05,
        t_end=40.0,
        n_timepoints=50,
        seed=None,
    ):
        obj = cls(
            number_of_nonoutput_species=None,
            initial_concentration_range=initial_concentration_range,
            noise_level=noise_level,
            t_end=t_end,
            n_timepoints=n_timepoints,
            seed=seed,
        )

        (
            obj.composition_matrix,
            obj.partition,
            obj.reactions,
        ) = obj.setup_hardcoded_1latent_example()

        # Initialize k and c0 either from fixed values or by sampling from ranges
        obj.k = reaction_rates 

        obj.c0 = (
            np.asarray(initial_concentrations, dtype=float)
            if initial_concentrations is not None
            else obj._sample_c0()
        )

        assert len(obj.k) == len(obj.reactions)
        assert len(obj.c0) == obj.partition.n_species

        obj._mode = "hardcoded_1latent"
        return obj


   
    def _sample_k(self):
        """
        Sample reaction rate constants (log-uniform).
        """
        lo, hi = self.reaction_rate_ranges
        log_k = self.rng.uniform(np.log(lo), np.log(hi), size=len(self.reactions))
        return np.exp(log_k)


    def _sample_c0(self):
        """
        Sample initial concentrations.
        Inputs get nonzero concentrations; hidden/output start at 0.
        """
        lo, hi = self.initial_concentration_range
        c0 = np.zeros(self.partition.n_species,  dtype=float)

        for idx in self.partition.input_indices:
            c0[idx] = self.rng.uniform(lo, hi)

        return c0

    def generate_partitions(self):
        """
        Generate all possible partitions between input, hidden, and output species.
        Preserves at least 1 input and 1 output species.
        May be no hidden species.
        
        Returns:
            list: List of possible reaction configurations
        """
        output_species = 'S_out'
        non_output_species = [f'S_{i}' for i in range(1, self.number_of_nonoutput_species + 1 )]
        partitions = []
        for number_of_inputs in range(1, self.number_of_nonoutput_species + 1):
            input_species = list(non_output_species[:number_of_inputs])
            hidden_species = list(non_output_species[number_of_inputs:])
            partitions.append(Partition(inputs=input_species, hidden=hidden_species))
        return partitions
        
    def is_column_space_reachable(self, C, partition, tol=1e-8):
        # Output must be reachable from input + hidden
        C_in_hid = C[:, partition.input_indices + partition.hidden_indices]
        C_out = C[:, partition.output_indices]

        for j in range(C_out.shape[1]):
            c_out_j = C_out[:, j]
            x, res, _, _ = lstsq(C_in_hid, c_out_j)
            residual = np.linalg.norm(C_in_hid @ x - c_out_j)
            if residual > tol:
                return False

        # Additional check: if there are hidden species,
        # output must also be reachable from hidden alone
        if partition.n_hidden > 0:
            C_hid = C[:, partition.hidden_indices]
            for j in range(C_out.shape[1]):
                c_out_j = C_out[:, j]
                x, res, _, _ = lstsq(C_hid, c_out_j)
                residual = np.linalg.norm(C_hid @ x - c_out_j)
                if residual > tol:
                    return False

        return True

        
    def _build_stoichiometric_matrix(self):
        """
        Build the stoichiometric matrix S from the reactions.
        Returns:
            np.ndarray: Stoichiometric matrix S of shape (n_species, n_reactions)
        """
        n_species = self.partition.n_species
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

        # Optional check
        assert self.is_column_space_reachable(composition_matrix, partition), "Composition not reachable"

        # Use indices [0,1,2,3,4] for species in partition.species
        reactions = [
            Reaction(reactants=(0, 1), products=(2,)),  # S_1 + S_2 → S_3
            Reaction(reactants=(2,), products=(3, 4)),   # S_3 → S_4 + S_out
        ]

        return composition_matrix, partition, reactions


    
    def generate_sample(
        self,
        resample_initial: bool = True,
    ):

        if resample_initial:
            self.c0 = self._sample_c0()

        t, C = self.simulate()

        out_idx = self.partition.output_indices[0]

        return {
            "t": torch.tensor(t, dtype=torch.float32),
            "c0": torch.tensor(self.c0.copy(), dtype=torch.float32),
            "y": torch.tensor(C[:, out_idx], dtype=torch.float32),
            "full": torch.tensor(C, dtype=torch.float32),
            "label": {
                "has_hidden": self.partition.n_hidden > 0,
                "n_hidden": self.partition.n_hidden,
                "k": self.k,
                "c0": self.c0.copy(),
                "reactions": self.reactions,
            },
        }





    
    def generate_batch(self, batch_size: int):
        samples = [self.generate_sample() for _ in range(batch_size)]

        # shared time grid (T,)
        t = samples[0]["t"]

        # outputs
        y = torch.stack([s["y"] for s in samples], dim=0)              # (B, T)
        full = torch.stack([s["full"] for s in samples], dim=0)        # (B, T, n_species)

        # initial conditions
        c0 = torch.stack([s["c0"] for s in samples], dim=0)        # (B, n_species)                                                    # (B,)

        # metadata / labels (keep python objects)
        labels = [s["label"] for s in samples]
        k = torch.tensor([lab["k"] for lab in labels], dtype=torch.float32)  # (B, n_rxns)

        return {
            "t": t,                        # (T,)
            "y": y,                        # (B, T)
            "full": full,                  # (B, T, n_species)
            "c0": c0,                  # (B, n_species)
            "label": labels,
            "k": k,                        # (B, n_rxns)
        }



    
    def generate_dataset(self, num_samples: int, save_path: str | None = None):
        batch = self.generate_batch(num_samples)

        if save_path is not None:
            # labels are python objects; torch.save can handle them
            torch.save(batch, save_path)

        return batch

    