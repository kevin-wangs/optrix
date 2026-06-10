"""Optrix - High-performance parallel compute primitives."""
__version__ = "0.1.0"
__author__ = "Kevin Wangs"

from optrix.core import detect_devices, synchronize
from optrix.memory import zeros, empty, empty_like, DeviceBuffer
from optrix.types import float16, float32, float64, int32, int64
from optrix.dispatch import dispatch, register_kernel, list_kernels, auto_grid
from optrix.device import Device, get_device

__all__ = [
    "detect_devices", "synchronize",
    "zeros", "empty", "empty_like", "DeviceBuffer",
    "float16", "float32", "float64", "int32", "int64",
    "dispatch", "register_kernel", "list_kernels", "auto_grid",
    "Device", "get_device",
]
