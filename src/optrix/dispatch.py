"""Kernel dispatch engine with registry and auto-grid."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple, Any
import logging
from optrix.memory import DeviceBuffer
from optrix.device import get_device

logger = logging.getLogger("optrix")

@dataclass
class KernelConfig:
    grid_dim: Tuple[int,int,int] = (1,1,1)
    block_dim: Tuple[int,int,int] = (256,1,1)
    shared_mem: int = 0
    stream: Optional[int] = None

@dataclass
class KernelEntry:
    name: str
    func: Callable
    source: Optional[str] = None
    compiled: bool = False
    _handle: Optional[int] = None
    invocation_count: int = 0

_REGISTRY: Dict[str, KernelEntry] = {}

def register_kernel(name, func=None, source=None):
    def _dec(fn):
        _REGISTRY[name] = KernelEntry(name=name, func=fn, source=source)
        return fn
    if func is not None:
        _dec(func); return func
    return _dec

def list_kernels(): return list(_REGISTRY.keys())

def auto_grid(n, block=256):
    blocks = (n + block - 1) // block
    return (blocks,1,1), (block,1,1)

def dispatch(kernel_name, *args, output=None, config=None, device_id=0):
    if kernel_name not in _REGISTRY:
        raise ValueError(f"Kernel '{kernel_name}' not registered. Available: {list_kernels()}")
    entry = _REGISTRY[kernel_name]
    device = get_device(device_id)
    if output is None and args and isinstance(args[0], DeviceBuffer):
        from optrix.memory import empty_like
        output = empty_like(args[0])
    try:
        from optrix._backend import hip
        cfg = config or KernelConfig()
        hip.launchKernel(entry._handle, cfg.grid_dim, cfg.block_dim, cfg.shared_mem, cfg.stream, list(args)+[output])
    except ImportError:
        entry.func(*args, output=output)
    entry.invocation_count += 1
    return output
