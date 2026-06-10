"""Tests for optrix.exceptions."""
import pytest
from optrix.exceptions import OptrixError, DeviceNotFoundError, ShapeMismatchError

class TestExceptions:
    def test_device_not_found(self):
        with pytest.raises(DeviceNotFoundError):
            raise DeviceNotFoundError(99)

    def test_shape_mismatch(self):
        with pytest.raises(ShapeMismatchError):
            raise ShapeMismatchError("shape error")

    def test_base_error(self):
        with pytest.raises(OptrixError):
            raise OptrixError("base error")
