"""Data compression utilities for memory-efficient transfers."""
from __future__ import annotations
import numpy as np
import logging

logger = logging.getLogger("optrix")

def quantize_fp32_to_fp16(buf):
    """Quantize float32 buffer to float16 for memory savings."""
    from optrix.memory import zeros
    from optrix.types import float16
    host = buf.to_host().astype(np.float16)
    out = zeros(host.shape, dtype=float16, device_id=buf.device_id)
    out.to_device(host)
    return out

def quantize_fp32_to_int8(buf, scale=None):
    """Quantize float32 to int8 with dynamic scaling."""
    from optrix.memory import zeros
    from optrix.types import Dtype
    host = buf.to_host()
    if scale is None:
        scale = np.max(np.abs(host)) / 127.0
    quantized = np.clip(host / scale, -128, 127).astype(np.int8)
    out = zeros(host.shape, dtype=Dtype.int8, device_id=buf.device_id)
    out.to_device(quantized)
    return out, scale

def dequantize_int8_to_fp32(buf, scale):
    """Dequantize int8 buffer back to float32."""
    from optrix.memory import zeros
    from optrix.types import float32
    host = buf.to_host().astype(np.float32) * scale
    out = zeros(host.shape, dtype=float32, device_id=buf.device_id)
    out.to_device(host)
    return out
