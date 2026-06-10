"""Tests for optrix.memory."""
import numpy as np
import pytest
from optrix.memory import DeviceBuffer, zeros, empty, empty_like
from optrix.types import float32, int32

class TestDeviceBuffer:
    def test_zeros(self):
        buf = zeros((64, 64), dtype=float32)
        assert buf.shape == (64, 64)
        np.testing.assert_array_equal(buf.to_host(), 0.0)

    def test_roundtrip(self):
        buf = zeros((8,), dtype=float32)
        data = np.array([1,2,3,4,5,6,7,8], dtype=np.float32)
        buf.to_device(data)
        np.testing.assert_array_equal(buf.to_host(), data)

    def test_nbytes(self):
        buf = zeros((100,), dtype=float32)
        assert buf.nbytes == 400

    def test_empty_like(self):
        buf = zeros((10, 20), dtype=float32)
        buf2 = empty_like(buf)
        assert buf2.shape == buf.shape

    def test_repr(self):
        buf = zeros((16,))
        assert "DeviceBuffer" in repr(buf)
