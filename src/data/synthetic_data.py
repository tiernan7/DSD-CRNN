"""
Synthetic data generation for model training and testing.

This module provides functions to generate synthetic data that can be used
for training and testing the models when real data is not available.
"""

import numpy as np
import torch
from  itertools import product

class Reaction:
    """
    Hold a representation of a chemical reaction, used for constructing synthetic data.
    """
    def __init__(self, reactants, products):
        """
        Initialize a Reaction instance.
        Reaction rates assume nM concentrations.
        Args:
            reactants (list): List of reactant species names
            products (list): List of product species names
            k_f (float): Forward reaction rate constant
            k_r (float): Reverse reaction rate constant
        """
        self.reactants = reactants
        self.products = products
        self.k_f = None
        self.k_r = None
        
    def set_rate_constants(self, k_f, k_r):
        self.k_f = k_f
        self.k_r = k_r
    
    def __repr__(self):
        return f"Reaction({self.reactants} <-> {self.products}, k_f={self.k_f}, k_r={self.k_r})"
    
    

class SyntheticDataGenerator:
    """
    Generator for synthetic training data.
    
    The training data consists of time-course trajectories of a single output species from a combinatorial set of idealized chemical reactions
    """
    
    def __init__(self, 
                 number_of_nonoutput_species, 
                 reaction_rate_ranges, 
                 initial_concentration_range, 
                 noise_level=0.1):
        """
        Initialize the synthetic data generator.
        Can pull from all possible reaction configurations given the number of non-output species, assuming 1 output species and at least 1 input species.

        
        Args:
            number_of_nonoutput_species (int): Number of hidden species in the reactions (total number of species will be this + 1 output species)
            reaction_rate_ranges (dict): Ranges for reaction rate constants
            initial_concentration_range (tuple): Range for initial concentrations of species
            noise_level (float): Level of noise to add to the data
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
        self.partitions = self.generate_all_species_partitions() 
        # For each parition, generate the set of all allowed reactions
        self.reaction_sets = []
        for partition in self.partitions:
            self.reaction_sets.append(self.generate_reaction_set(partition))
        
    def print_possible_sets(self):
        """
        List all possible reactions based on the current configuration.
        
        Returns:
            list: List of Reaction instances
        """
        for i, partition in enumerate(self.partitions):
            print(f"Partition {i+1}: Inputs: {partition['inputs']}, Hidden: {partition['hidden']}, Output: {partition['output']}")
            reactions = self.reaction_sets[i]
            for reaction in reactions:
                print(reaction)
            print("\n")
    
    def generate_reaction_set(self, partition):
        """
        Generate the set of all allowed reactions for a given partition of species.
        Includes all reactions of the form 
        A + B <-> C + D
        A + B <-> C
        A + B <-> D
        A <-> C + D
        B <-> C + D
        A <-> C
        B <-> C
        A <-> D
        B <-> D
        where A and B are input or hidden species, and C and D are hidden or output species.
        
        Args:
            partition (dict): A dictionary with keys 'inputs', 'hidden', 'output' listing species names
            
        Returns:
            list: List of Reaction instances
        """
        
        # Get the set of all tuples of input or hidden species of length 2 or 1
        reactant_candidates = list(product(partition['inputs'] + partition['hidden'], repeat=2)) + list(product(partition['inputs'] + partition['hidden'], repeat=1))
        # Get the set of all tuples of hidden or output species of length 2 or 1
        product_candidates = list(product(partition['hidden'] + [partition['output']], repeat=2)) + list(product(partition['hidden'] + [partition['output']], repeat=1))
        
        #Create all possible reactions from these candidates
        reactions = []
        for reactants in reactant_candidates:
            for products in product_candidates:
                # Avoid reactions where reactants and products are the same
                if set(reactants) != set(products):
                    # Generate random reaction rates within specified ranges
                    reactions.append(Reaction(reactants, products))
        return reactions
        
    def generate_all_species_partitions(self):
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
        for number_of_inputs in range(1, self.number_of_nonoutput_species):
            input_species = list(non_output_species[:number_of_inputs])
            hidden_species = list(non_output_species[number_of_inputs:])
            partitions.append({'inputs': input_species, 'hidden': hidden_species, 'output': output_species})
        return partitions
        
    
    def generate_reaction_set_from_species(self, reactants, products):
        """
        Generate a set of reaxtions and producs each length 2
        Return the reactions of the form

        
        Args:
            reactants (list): List of reactant species names
            products (list): List of product species names
        Returns:
            list: List of reaction dictionaries
        """
    
    def generate_parameter_set(self):
        """
        Generate a set of parameters for synthetic data generation.
        
        Retuirns:
            dict: A dictionary of parameters for data generation
        """
        ut  
    def write_paramaters_to_file(self, params, file_path):
        """
        Write generated parameters to a file for record-keeping.
        
        Args:
            params (dict): Parameters to write
            file_path (str): Path to the file where parameters will be saved
        """
        # TODO: Implement parameter writing
        # This should include:
        # - Formatting parameters
        # - Writing to a file
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
