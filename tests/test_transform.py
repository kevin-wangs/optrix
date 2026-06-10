"""Tests for optrix.transform."""
import numpy as np
from optrix import dispatch, zeros, float32

class TestTransforms:
    def test_relu(self):
        buf = zeros((4,), dtype=float32)
        buf.to_device(np.array([-1, 2, -3, 4], dtype=np.float32))
        result = dispatch("relu", buf)
        np.testing.assert_array_equal(result.to_host(), [0, 2, 0, 4])

    def test_softmax(self):
        buf = zeros((3,), dtype=float32)
        buf.to_device(np.array([1, 2, 3], dtype=np.float32))
        result = dispatch("softmax", buf)
        host = result.to_host()
        assert abs(sum(host) - 1.0) < 1e-5

    def test_exp(self):
        buf = zeros((3,), dtype=float32)
        buf.to_device(np.array([0, 1, 2], dtype=np.float32))
        result = dispatch("transform_exp", buf)
        np.testing.assert_allclose(result.to_host(), [1, np.e, np.e**2], rtol=1e-5)
