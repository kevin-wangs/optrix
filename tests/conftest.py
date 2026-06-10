"""Shared test fixtures."""
import pytest
import numpy as np

@pytest.fixture
def random_matrix():
    return np.random.randn(16, 16).astype(np.float32)

@pytest.fixture
def random_vector():
    return np.random.randn(1024).astype(np.float32)
