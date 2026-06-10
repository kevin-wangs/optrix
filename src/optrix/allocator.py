"""Memory allocation strategies."""
from __future__ import annotations
from abc import ABC, abstractmethod
import logging
from optrix.memory import DeviceBuffer
from optrix.types import Dtype, float32

logger = logging.getLogger("optrix")

class Allocator(ABC):
    @abstractmethod
    def allocate(self, shape, dtype=float32, device_id=0): ...

    @abstractmethod
    def free(self, buf): ...

class DefaultAllocator(Allocator):
    """Direct allocation via hipMalloc."""
    def allocate(self, shape, dtype=float32, device_id=0):
        return DeviceBuffer(shape, dtype, device_id)

    def free(self, buf):
        del buf

class PoolAllocator(Allocator):
    """Pool-based allocation for buffer reuse."""
    def __init__(self, max_size_mb=512):
        from optrix.pool import MemoryPool
        self.pool = MemoryPool(max_size_mb)

    def allocate(self, shape, dtype=float32, device_id=0):
        return self.pool.acquire(shape, dtype, device_id)

    def free(self, buf):
        self.pool.release(buf)

class PinnedAllocator(Allocator):
    """Pinned (page-locked) memory for faster DMA."""
    def allocate(self, shape, dtype=float32, device_id=0):
        try:
            from optrix._backend import hip
            size = 1
            for s in shape: size *= s
            size *= dtype.itemsize
            ptr = hip.hostMalloc(size)
            buf = DeviceBuffer(shape, dtype, device_id, _ptr=ptr)
            return buf
        except ImportError:
            return DeviceBuffer(shape, dtype, device_id)

    def free(self, buf):
        try:
            from optrix._backend import hip
            hip.hostFree(buf._ptr)
        except (ImportError, AttributeError):
            del buf
