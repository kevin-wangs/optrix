"""Tests for optrix.dispatch."""
import numpy as np
import pytest
from optrix import dispatch, register_kernel, list_kernels, zeros, float32

class TestKernelRegistry:
    def test_register_and_list(self):
        @register_kernel("_test_kernel")
        def _test(x, output=None): pass
        assert "_test_kernel" in list_kernels()

    def test_dispatch_unknown_kernel(self):
        with pytest.raises(ValueError, match="not registered"):
            dispatch("nonexistent_kernel_xyz")

    def test_dispatch_vec_add(self):
        a = zeros((8,), dtype=float32)
        b = zeros((8,), dtype=float32)
        a.to_device(np.ones(8, dtype=np.float32))
        b.to_device(np.ones(8, dtype=np.float32) * 2)
        result = dispatch("vec_add", a, b)
        np.testing.assert_array_equal(result.to_host(), 3.0)
