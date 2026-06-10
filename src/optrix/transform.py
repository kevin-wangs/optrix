"""Element-wise transform operations."""
from __future__ import annotations
import numpy as np
from optrix.memory import DeviceBuffer, zeros, empty_like
from optrix.dispatch import register_kernel

@register_kernel("transform_abs")
def transform_abs(a, output=None):
    output = output or empty_like(a)
    output.to_device(np.abs(a.to_host()))
    return output

@register_kernel("transform_exp")
def transform_exp(a, output=None):
    output = output or empty_like(a)
    output.to_device(np.exp(a.to_host()))
    return output

@register_kernel("transform_log")
def transform_log(a, output=None):
    output = output or empty_like(a)
    output.to_device(np.log(a.to_host()))
    return output

@register_kernel("transform_sqrt")
def transform_sqrt(a, output=None):
    output = output or empty_like(a)
    output.to_device(np.sqrt(a.to_host()))
    return output

@register_kernel("transform_tanh")
def transform_tanh(a, output=None):
    output = output or empty_like(a)
    output.to_device(np.tanh(a.to_host()))
    return output

@register_kernel("transform_clip")
def transform_clip(a, lo=0.0, hi=1.0, output=None):
    output = output or empty_like(a)
    output.to_device(np.clip(a.to_host(), lo, hi))
    return output
