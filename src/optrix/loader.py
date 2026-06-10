"""Data loading utilities for host-to-device transfer."""
from __future__ import annotations
import numpy as np
import logging
from pathlib import Path
from optrix.memory import zeros
from optrix.types import float32

logger = logging.getLogger("optrix")

def from_numpy(array, device_id=0):
    """Upload a numpy array to device."""
    buf = zeros(array.shape, dtype=float32, device_id=device_id)
    buf.to_device(array.astype(np.float32))
    return buf

def from_file(path, dtype=float32, device_id=0):
    """Load data from .npy file to device."""
    data = np.load(path)
    buf = zeros(data.shape, dtype=dtype, device_id=device_id)
    buf.to_device(data.astype(dtype.np_dtype))
    return buf

def from_raw(path, shape, dtype=float32, device_id=0):
    """Load raw binary file to device."""
    data = np.fromfile(path, dtype=dtype.np_dtype).reshape(shape)
    buf = zeros(shape, dtype=dtype, device_id=device_id)
    buf.to_device(data)
    return buf

def to_numpy(buf):
    """Download device buffer to numpy array."""
    return buf.to_host()

def to_file(buf, path):
    """Save device buffer to .npy file."""
    np.save(path, buf.to_host())
