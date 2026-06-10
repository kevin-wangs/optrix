"""Tests for optrix.types."""
import pytest
import numpy as np
from optrix.types import Dtype, float32, int32

class TestDtype:
    def test_np_dtype(self):
        assert float32.np_dtype == np.float32
        assert int32.np_dtype == np.int32

    def test_itemsize(self):
        assert float32.itemsize == 4
        assert Dtype.float64.itemsize == 8

    def test_enum_values(self):
        assert Dtype.float16.value == "float16"
        assert Dtype.bfloat16.value == "bfloat16"
