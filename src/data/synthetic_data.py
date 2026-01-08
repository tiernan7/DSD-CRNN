"""
Synthetic data generation for model training and testing.

This module provides functions to generate synthetic data that can be used
for training and testing the models when real data is not available.
"""

from dataclasses import dataclass
from matplotlib.pylab import lstsq
import numpy as np
from numpy.linalg import matrix_rank
import torch
from  itertools import combinations_with_replacement

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
    """
    Generator for synthetic training data.
    
    The training data consists of time-course trajectories of a single output species from a combinatorial set of idealized chemical reactions
    """
    #TODO: incorporate stoichiometry in reactions
    def __init__(self, 
                 number_of_nonoutput_species, 
                 reaction_rate_ranges, 
                 initial_concentration_range, 
                 noise_level=0.1):
        """
        Initialize the synthetic data generator.
        Can pull from all possible reaction configurations given the number of non-output species, assuming 1 output species and at least 1 input species.      
        """
        # TODO: Implement initialization
        # This should include:
        # - Storing data generation parameters
        # - Setting random seeds for reproducibility
        
        self.number_of_nonoutput_species = number_of_nonoutput_species
        self.reaction_rate_ranges = reaction_rate_ranges
        self.initial_concentration_range = initial_concentration_range
        self.noise_level = noise_level

        
        # Create all possible divisions of species into input, hidden, and output
        self.partitions = self.generate_partitions() 

   

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

        
        
    
    def sample_composition_matrix(self, partition, atomic_components, max_components_per_species, rank_target, seed):
        """
        Sample a composition matrix for a given partition.
        
        Args:
            partition (Partition): The partition of species across input, hidden, and output
            atomic_components (int): Number of atomic components to use in the composition matrix
            max_components_per_species (int): Maximum number of components per species
            rank_target (int): Target rank for the composition matrix
            seed (int): Random seed for reproducibility
            
        """
        def sample_valid_species_column(M, max_atoms, rng, seen):
            while True:
                col = np.zeros(M, dtype=int)
                values = list(range(0, max_atoms + 1))
                remaining = max(values)
                for i in range(M):
                    if not values:
                        break
                    value = rng.choice(values)
                    col[i] = value
                    remaining -= value
                    values = [v for v in values if v <= remaining]
                    if remaining <= 0:
                        break
                if col.sum() == 0:
                    continue
                if tuple(col) in seen:
                    continue
                seen.add(tuple(col))
                return col

        
        
        
        N = partition.n_species
        M = atomic_components
        
        
        
        
        rng = np.random.default_rng(seed)

        is_reachable = False
        while not(is_reachable):
            composition_matrix = np.zeros((M,N), dtype=int)
            seen = set()
            j = 0
            while j < N:
                col = sample_valid_species_column(M, max_components_per_species, rng, seen)
                composition_matrix[:, j] = col
                j += 1
            is_reachable = self.is_column_space_reachable(composition_matrix, partition)
        return composition_matrix
        
                
            
    def structure_metadata(self, A, partition):
        """
        Structure metadata from a composition matrix.
        
        Returns:
            dict: Mapping from species to atom count vectors and type
        """
        return {
            name: {
                "composition": A[:, i],
                "type": ("input" if name in partition.inputs else
                        "output" if name == partition.output else
                        "hidden")
            }
            for i, name in enumerate(partition.species)
        }

    
    def generate_candidate_reactions(self, partition, composition_matrix):
        """
        Generate all candidate stoichiometrically valid reactions
        for a given partition and composition matrix.

        Args:
            partition (Partition): Defines the species roles
            composition_matrix (np.ndarray): Shape (num_atoms, num_species)

        Returns:
            list[Reaction]: Set of valid, mass-conserving reactions
        """
        num_species = partition.n_species
        reactions = set()

        species_indices = list(range(num_species))
        # reactant/product pairs of size 1 or 2
        pairings = list(combinations_with_replacement(species_indices, 2)) + [(i,) for i in species_indices]

        for reactants in pairings:
            lhs = sum(composition_matrix[:, r] for r in reactants)

            for products in pairings:
                rhs = sum(composition_matrix[:, p] for p in products)

                if np.array_equal(lhs, rhs) and set(reactants) != set(products):
                    # sorted tuples prevent duplicates due to ordering
                    reactions.add(Reaction(
                        reactants=tuple(sorted(reactants)),
                        products=tuple(sorted(products))
                    ))

        return list(reactions)
    
    
    def prune_unused_species(self, composition_matrix, partition, reactions):
        """
        Remove species not participating in any reaction.

        Returns:
            - pruned composition_matrix
            - pruned partition
            - mapping from old to new species indices (optional)
        """
        used_species = set()
        for r in reactions:
            used_species.update(r.reactants)
            used_species.update(r.products)
        used_species = sorted(used_species)

        # Prune matrix
        pruned_matrix = composition_matrix[:, used_species]

        # Rebuild partition species list
        species = partition.species
        used_names = [species[i] for i in used_species]
        new_inputs = [s for s in used_names if s in partition.inputs]
        new_hidden = [s for s in used_names if s in partition.hidden]
        new_output = partition.output if partition.output in \
        used_names else None

        pruned_partition = Partition(
            inputs=new_inputs,
            hidden=new_hidden,
            output=new_output
        )

        return pruned_matrix, pruned_partition
    
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
        ])

        # Optional check
        assert self.is_column_space_reachable(composition_matrix, partition), "Composition not reachable"

        # Use indices [0,1,2,3,4] for species in partition.species
        reactions = [
            Reaction(reactants=(0, 1), products=(2,)),  # S_1 + S_2 → S_3
            Reaction(reactants=(2,), products=(3, 4)),   # S_3 → S_4 + S_out
        ]

        return composition_matrix, partition, reactions


    
    
    def enumerate_label_only_reactions(species: list[str], max_order=2):
        """
        Enumerate all possible reactions up to max_order (default: bimolecular)
        without atom constraints, using just species names.
        
        Returns:
            list[Reaction]: All nontrivial reactions (reactants ≠ products)
        """
        # All uni/bi-molecular reactant and product combinations
        reactant_combos = list(combinations_with_replacement(species, r=1)) + \
                        list(combinations_with_replacement(species, r=2))
        product_combos = list(combinations_with_replacement(species, r=1)) + \
                        list(combinations_with_replacement(species, r=2))
        
        reactions = set()
        for r in reactant_combos:
            for p in product_combos:
                if sorted(r) != sorted(p):  # exclude trivial identity reactions
                    reactions.add(Reaction(
                        reactants=tuple(sorted(r)),
                        products=tuple(sorted(p))
                    ))
        
        return sorted(reactions, key=lambda rxn: (rxn.reactants, rxn.products))
    
    def generate_all_mass_conserving_reactions(partition: Partition, composition_matrix: np.ndarray):
        """
        Given a species partition and composition matrix (atoms × species),
        generate all stoichiometrically valid reactions (1 or 2 reactants → 1 or 2 products).
        """

        n_species = partition.n_species
        reactions = set()

        # Species indices
        indices = list(range(n_species))

        # Reaction forms: uni/bi → uni/bi
        reactant_combos = list(combinations_with_replacement(indices, 2)) + [(i,) for i in indices]
        product_combos  = list(combinations_with_replacement(indices, 2)) + [(i,) for i in indices]

        for reactants in reactant_combos:
            lhs = sum(composition_matrix[:, i] for i in reactants)

            for products in product_combos:
                rhs = sum(composition_matrix[:, i] for i in products)

                # Valid if atom-balanced and not a trivial identity
                if np.array_equal(lhs, rhs) and set(reactants) != set(products):
                    reactions.add(Reaction(
                        reactants=tuple(sorted(reactants)),
                        products=tuple(sorted(products))
                    ))

        return list(reactions)
    
    def generate_sample(self, params):
        """
        Generate a single synthetic data sample.
        Args:
            params (dict): Parameters for sample generation. Includes:
                - 'class_label': Whether or not there are hidden species
                - 'sample_label': A unique label for the sample
                - list of reactions
                - Inital concentration in nM for each species across the reactions
                - rate constants (for nM input concentrations) for each reaction
                
        
        Returns:
            tuple: (data, label) pair where data is a tensor of species concentrations and label is
        """
        # TODO: Implement single sample generation
        # This should include:
        # - Generating synthetic features (e.g., random patterns, shapes)
        # - Assigning appropriate labels
        # - Adding noise for robustness
        pass
    
    def generate_batch(self, batch_size):
        """
        Generate a batch of synthetic data samples.
        
        Args:
            batch_size (int): Number of samples to generate
            
        Returns:
            tuple: (data_batch, labels_batch) where data_batch is a tensor of shape
                   (batch_size, *data_shape) and labels_batch is a tensor of shape (batch_size,)
        """
        # TODO: Implement batch generation
        # This should include:
        # - Calling generate_sample multiple times
        # - Stacking samples into a batch tensor
        pass
    
    def generate_dataset(self, num_samples, save_path=None):
        """
        Generate a complete synthetic dataset.
        
        Args:
            num_samples (int): Total number of samples to generate
            save_path (str, optional): Path to save the generated dataset
            
        Returns:
            tuple: (data, labels) tensors for the entire dataset
        """
        # TODO: Implement full dataset generation
        # This should include:
        # - Generating all samples
        # - Optionally saving to disk
        # - Returning the complete dataset
        pass
    
    def add_augmentation(self, data):
        """
        Apply data augmentation to synthetic data.
        
        Args:
            data: Input data tensor
            
        Returns:
            Augmented data tensor
        """
        # TODO: Implement augmentation strategies
        # This should include:
        # - Rotation
        # - Scaling
        # - Translation
        # - Color jittering (if applicable)
        pass


def create_synthetic_dataset(config):
    """
    Create a synthetic dataset based on configuration.
    
    Args:
        config (dict): Configuration dictionary containing:
            - data_shape: Shape of data samples
            - num_classes: Number of classes
            - num_train_samples: Number of training samples
            - num_val_samples: Number of validation samples
            - num_test_samples: Number of test samples
            - noise_level (optional): Level of noise to add (default: 0.1)
            
    Returns:
        dict: Dictionary containing train, validation, and test datasets
        
    Note:
        Missing required keys will raise KeyError during implementation.
        Optional keys will use default values if not provided.
    """
    # TODO: Implement dataset creation from config
    # This should include:
    # - Validating required configuration keys
    # - Creating generator instances
    # - Generating train/val/test splits
    # - Returning organized datasets
    pass
