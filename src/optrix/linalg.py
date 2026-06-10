"""Linear algebra operations with rocBLAS integration."""
from __future__ import annotations
import numpy as np
import logging
from optrix.memory import DeviceBuffer, zeros
from optrix.types import float32, float64

logger = logging.getLogger("optrix")

def matmul(a: DeviceBuffer, b: DeviceBuffer) -> DeviceBuffer:
    """Matrix multiplication with automatic backend selection."""
    if a.shape[1] != b.shape[0]:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    output = zeros((a.shape[0], b.shape[1]), dtype=a.dtype, device_id=a.device_id)
    try:
        from optrix._backend import hip
        hip.matmul(a._ptr, b._ptr, output._ptr, a.shape[0], b.shape[1], a.shape[1])
    except (ImportError, AttributeError):
        result = a.to_host() @ b.to_host()
        output.to_device(result)
    return output

def dot(a: DeviceBuffer, b: DeviceBuffer) -> float:
    """Dot product of two vectors."""
    return float(a.to_host() @ b.to_host())

def norm(a: DeviceBuffer, ord=2) -> float:
    """Compute vector norm."""
    return float(np.linalg.norm(a.to_host(), ord=ord))

def solve(A: DeviceBuffer, b: DeviceBuffer) -> DeviceBuffer:
    """Solve linear system Ax = b."""
    result = np.linalg.solve(A.to_host(), b.to_host())
    output = zeros(result.shape, dtype=A.dtype, device_id=A.device_id)
    output.to_device(result)
    return output

def svd(a: DeviceBuffer):
    """Singular Value Decomposition."""
    U, S, Vt = np.linalg.svd(a.to_host(), full_matrices=False)
    dev_id = a.device_id
    u_buf = zeros(U.shape, dtype=a.dtype, device_id=dev_id)
    s_buf = zeros(S.shape, dtype=a.dtype, device_id=dev_id)
    vt_buf = zeros(Vt.shape, dtype=a.dtype, device_id=dev_id)
    u_buf.to_device(U)
    s_buf.to_device(S)
    vt_buf.to_device(Vt)
    return u_buf, s_buf, vt_buf
