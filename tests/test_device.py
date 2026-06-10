"""Tests for optrix.device."""
from optrix.device import Device, get_device

class TestDevice:
    def test_create_device(self):
        d = Device(0)
        assert d.device_id == 0

    def test_get_device_singleton(self):
        d1 = get_device(0)
        d2 = get_device(0)
        assert d1 is d2

    def test_repr(self):
        d = Device(0)
        assert "Device" in repr(d)
