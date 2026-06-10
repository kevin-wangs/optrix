"""Optrix - High-performance parallel compute primitives."""
__version__ = "0.1.1"
__author__ = "Kevin Wangs"

from optrix.core import detect_devices, synchronize
from optrix.memory import zeros, empty, ones, full, empty_like, zeros_like, DeviceBuffer
from optrix.types import float16, float32, float64, int32, int64
from optrix.dispatch import dispatch, register_kernel, list_kernels, auto_grid
from optrix.device import Device, get_device
from optrix.stream import Stream, StreamPool
from optrix.context import device_context, stream_context, current_context
from optrix.pool import MemoryPool, get_pool
from optrix.profiler import Profiler, profile
from optrix.linalg import matmul, dot, norm, solve, svd
from optrix.config import get_config

__version_info__ = tuple(int(x) for x in __version__.split("."))

__all__ = [
    "detect_devices", "synchronize",
    "zeros", "empty", "ones", "full", "empty_like", "zeros_like", "DeviceBuffer",
    "float16", "float32", "float64", "int32", "int64",
    "dispatch", "register_kernel", "list_kernels", "auto_grid",
    "Device", "get_device", "Stream", "StreamPool",
    "device_context", "stream_context", "current_context",
    "MemoryPool", "get_pool", "Profiler", "profile",
    "matmul", "dot", "norm", "solve", "svd",
    "get_config",
]
