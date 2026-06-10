"""Prefix scan (parallel scan) operations."""
from __future__ import annotations
import numpy as np
from optrix.memory import DeviceBuffer, zeros, empty_like
from optrix.dispatch import register_kernel

@register_kernel("prefix_sum")
def prefix_sum(a, output=None):
    """Inclusive prefix sum (cumsum)."""
    output = output or empty_like(a)
    output.to_device(np.cumsum(a.to_host()))
    return output

@register_kernel("prefix_prod")
def prefix_prod(a, output=None):
    """Inclusive prefix product (cumprod)."""
    output = output or empty_like(a)
    output.to_device(np.cumprod(a.to_host()))
    return output
