"""RAII device buffers and memory management."""
from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import logging
from optrix.types import Dtype, float32

logger = logging.getLogger("optrix")

Shape = Union[int, Tuple[int, ...]]

def _parse_shape(shape: Shape) -> Tuple[int, ...]:
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)

class DeviceBuffer:
    """RAII wrapper around device memory. Auto-frees on scope exit."""
    def __init__(self, shape, dtype=float32, device_id=0, _ptr=None):
        self.shape = _parse_shape(shape)
        self.dtype = dtype
        self.device_id = device_id
        self._ptr = _ptr
        self._host_data = None
        self._allocate()

    @property
    def nbytes(self):
        n = 1
        for s in self.shape: n *= s
        return n * self.dtype.itemsize

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        n = 1
        for s in self.shape: n *= s
        return n

    def _allocate(self):
        try:
            from optrix._backend import hip
            self._ptr = hip.malloc(self.nbytes)
        except ImportError:
            self._host_data = np.zeros(self.shape, dtype=self.dtype.np_dtype)

    def to_host(self):
        if self._host_data is not None:
            return self._host_data.copy()
        from optrix._backend import hip
        out = np.empty(self.shape, dtype=self.dtype.np_dtype)
        hip.memcpy(self._ptr, out, self.nbytes, direction="device_to_host")
        return out

    def to_device(self, host_array):
        host_array = np.asarray(host_array, dtype=self.dtype.np_dtype)
        if host_array.shape != self.shape:
            from optrix.exceptions import ShapeMismatchError
            raise ShapeMismatchError(f"Expected {self.shape}, got {host_array.shape}")
        if self._host_data is not None:
            self._host_data[:] = host_array
            return
        from optrix._backend import hip
        hip.memcpy(host_array, self._ptr, self.nbytes, direction="host_to_device")

    def fill(self, value):
        self._host_data = np.full(self.shape, value, dtype=self.dtype.np_dtype)

    def copy(self):
        new_buf = DeviceBuffer(self.shape, self.dtype, self.device_id)
        new_buf.to_device(self.to_host())
        return new_buf

    def __del__(self):
        if self._ptr is not None:
            try:
                from optrix._backend import hip
                hip.free(self._ptr)
            except (ImportError, AttributeError): pass

    def __repr__(self):
        return f"DeviceBuffer(shape={self.shape}, dtype={self.dtype}, device={self.device_id})"

def zeros(shape, dtype=float32, device_id=0):
    return DeviceBuffer(shape, dtype, device_id)

def empty(shape, dtype=float32, device_id=0):
    return DeviceBuffer(shape, dtype, device_id)

def ones(shape, dtype=float32, device_id=0):
    buf = DeviceBuffer(shape, dtype, device_id)
    buf.fill(1.0)
    return buf

def full(shape, value, dtype=float32, device_id=0):
    buf = DeviceBuffer(shape, dtype, device_id)
    buf.fill(value)
    return buf

def empty_like(buf):
    return DeviceBuffer(buf.shape, buf.dtype, buf.device_id)

def zeros_like(buf):
    return zeros(buf.shape, buf.dtype, buf.device_id)
