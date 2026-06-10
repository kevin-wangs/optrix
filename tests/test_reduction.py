"""Tests for optrix.reduction."""
import numpy as np
from optrix import dispatch, zeros, float32

class TestReductions:
    def test_reduce_sum(self):
        buf = zeros((5,), dtype=float32)
        buf.to_device(np.array([1, 2, 3, 4, 5], dtype=np.float32))
        result = dispatch("reduce_sum", buf)
        assert abs(result.to_host()[0] - 15.0) < 1e-5

    def test_reduce_max(self):
        buf = zeros((5,), dtype=float32)
        buf.to_device(np.array([1, 5, 3, 2, 4], dtype=np.float32))
        result = dispatch("reduce_max", buf)
        assert abs(result.to_host()[0] - 5.0) < 1e-5

    def test_reduce_mean(self):
        buf = zeros((4,), dtype=float32)
        buf.to_device(np.array([2, 4, 6, 8], dtype=np.float32))
        result = dispatch("reduce_mean", buf)
        assert abs(result.to_host()[0] - 5.0) < 1e-5
