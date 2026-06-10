"""GPU-accelerated sorting operations."""
from __future__ import annotations
import numpy as np
from optrix.memory import DeviceBuffer, empty_like
from optrix.dispatch import register_kernel

@register_kernel("sort_radix")
def sort_radix(a, output=None):
    """Radix sort (simulated via numpy sort)."""
    output = output or empty_like(a)
    host = a.to_host()
    output.to_device(np.sort(host))
    return output

@register_kernel("argsort")
def argsort_kernel(a, output=None):
    """Arg sort — returns indices that would sort the array."""
    from optrix.memory import zeros
    from optrix.types import int64
    output = output or zeros(a.shape, dtype=int64, device_id=a.device_id)
    output.to_device(np.argsort(a.to_host()))
    return output

@register_kernel("topk")
def topk(a, k=10, output=None):
    """Top-K selection."""
    from optrix.memory import zeros
    host = a.to_host()
    result = np.sort(host)[-k:][::-1]
    output = output or zeros((k,), dtype=a.dtype, device_id=a.device_id)
    output.to_device(result)
    return output
