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
        
                
            
    
    def canonicalize_composition_matrix(self, composition_matrix):
        """
        Canonicalize a composition matrix to ensure uniqueness.
        """
        pass
    
    def structure_metadata(self, A):
        """
        Structure metadata from a composition matrix.
        """
        pass
    
    def generate_candidate_reactions(self, partition, composition_matrix):
        """
        Generate all candidate reactions for a given partition and composition matrix
        """
        pass

    
    
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
