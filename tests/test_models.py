"""
Test cases for model classes.
"""

import pytest
import torch


def test_base_model_import():
    """Test that BaseModel can be imported."""
    from src.models.base_model import BaseModel
    assert BaseModel is not None


def test_crnn_import():
    """Test that CRNN model can be imported."""
    from src.models.crnn import CRNN
    assert CRNN is not None


# TODO: Add more tests once models are implemented
# Example tests to add:
# - test_crnn_initialization()
# - test_crnn_forward_pass()
# - test_model_checkpoint_save_load()
