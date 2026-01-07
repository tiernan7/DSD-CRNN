"""
Base model class for all models in the project.
"""

import torch
import torch.nn as nn


class BaseModel(nn.Module):
    """
    Base model class that all models should inherit from.
    
    This provides a common interface for all models.
    """
    
    def __init__(self):
        """Initialize the base model."""
        super(BaseModel, self).__init__()
        # TODO: Implement base model initialization
        pass
    
    def forward(self, x):
        """
        Forward pass of the model.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # TODO: Implement forward pass
        raise NotImplementedError("Subclasses must implement the forward method")
    
    def save_checkpoint(self, path):
        """
        Save model checkpoint.
        
        Args:
            path (str): Path to save the checkpoint
        """
        # TODO: Implement checkpoint saving
        pass
    
    def load_checkpoint(self, path):
        """
        Load model checkpoint.
        
        Args:
            path (str): Path to load the checkpoint from
        """
        # TODO: Implement checkpoint loading
        pass
