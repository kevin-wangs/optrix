"""Tests for optrix.core."""
from optrix.core import detect_devices, synchronize

class TestDeviceDetection:
    def test_returns_list(self):
        assert isinstance(detect_devices(), list)

    def test_at_least_one_device(self):
        assert len(detect_devices()) >= 1

    def test_device_fields(self):
        dev = detect_devices()[0]
        assert dev.device_id >= 0
        assert dev.total_memory > 0
        assert dev.compute_units > 0

    def test_synchronize_no_error(self):
        synchronize(0)
