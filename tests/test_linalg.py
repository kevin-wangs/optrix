"""Tests for optrix.linalg."""
import numpy as np
from optrix.linalg import matmul, dot, norm
from optrix import zeros, float32

class TestLinAlg:
    def test_matmul(self):
        a = zeros((4, 4), dtype=float32)
        b = zeros((4, 4), dtype=float32)
        a.to_device(np.eye(4, dtype=np.float32))
        b.to_device(np.ones((4, 4), dtype=np.float32))
        result = matmul(a, b)
        np.testing.assert_array_equal(result.to_host(), np.ones((4, 4)))

    def test_dot(self):
        a = zeros((3,), dtype=float32)
        b = zeros((3,), dtype=float32)
        a.to_device(np.array([1, 2, 3], dtype=np.float32))
        b.to_device(np.array([4, 5, 6], dtype=np.float32))
        assert dot(a, b) == 32.0

    def test_norm(self):
        a = zeros((3,), dtype=float32)
        a.to_device(np.array([3, 4, 0], dtype=np.float32))
        assert abs(norm(a) - 5.0) < 1e-5
