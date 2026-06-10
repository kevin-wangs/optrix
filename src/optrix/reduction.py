"""Parallel reduction operations."""
from __future__ import annotations
import numpy as np
import logging
from optrix.memory import DeviceBuffer, zeros
from optrix.dispatch import register_kernel

logger = logging.getLogger("optrix")

@register_kernel("reduce_sum")
def reduce_sum(a, output=None):
    """Sum reduction."""
    result = np.sum(a.to_host())
    out = output or zeros((1,), dtype=a.dtype, device_id=a.device_id)
    out.to_device(np.array([result], dtype=a.dtype.np_dtype))
    return out

@register_kernel("reduce_max")
def reduce_max(a, output=None):
    """Max reduction."""
    result = np.max(a.to_host())
    out = output or zeros((1,), dtype=a.dtype, device_id=a.device_id)
    out.to_device(np.array([result], dtype=a.dtype.np_dtype))
    return out

@register_kernel("reduce_min")
def reduce_min(a, output=None):
    """Min reduction."""
    result = np.min(a.to_host())
    out = output or zeros((1,), dtype=a.dtype, device_id=a.device_id)
    out.to_device(np.array([result], dtype=a.dtype.np_dtype))
    return out

@register_kernel("reduce_mean")
def reduce_mean(a, output=None):
    """Mean reduction."""
    result = np.mean(a.to_host())
    out = output or zeros((1,), dtype=a.dtype, device_id=a.device_id)
    out.to_device(np.array([result], dtype=a.dtype.np_dtype))
    return out

@register_kernel("reduce_argmax")
def reduce_argmax(a, output=None):
    """Argmax reduction."""
    result = np.argmax(a.to_host())
    out = output or zeros((1,), dtype=a.dtype, device_id=a.device_id)
    out.to_device(np.array([result], dtype=np.int64))
    return out
