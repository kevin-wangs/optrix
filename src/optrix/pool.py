"""Memory pool for efficient buffer reuse."""
from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple
import threading
import logging
from optrix.memory import DeviceBuffer
from optrix.types import Dtype, float32

logger = logging.getLogger("optrix")

class MemoryPool:
    """Buddy-allocator style memory pool for DeviceBuffers.

    Reduces allocation overhead by reusing freed buffers.
    Thread-safe for multi-stream environments.
    """
    def __init__(self, max_size_mb=512):
        self.max_size = max_size_mb * 1024 * 1024
        self._free: Dict[Tuple, List[DeviceBuffer]] = defaultdict(list)
        self._used: int = 0
        self._lock = threading.Lock()
        self._hits: int = 0
        self._misses: int = 0

    def acquire(self, shape, dtype=float32, device_id=0):
        key = (shape, dtype, device_id)
        with self._lock:
            if self._free[key]:
                buf = self._free[key].pop()
                self._hits += 1
                logger.debug("Pool hit for %s (hit rate: %.1f%%)", key, self.hit_rate)
                return buf
            self._misses += 1
        return DeviceBuffer(shape, dtype, device_id)

    def release(self, buf):
        key = (buf.shape, buf.dtype, buf.device_id)
        with self._lock:
            if self._used < self.max_size:
                self._free[key].append(buf)
                self._used += buf.nbytes

    @property
    def hit_rate(self):
        total = self._hits + self._misses
        return (self._hits / total * 100) if total > 0 else 0

    def clear(self):
        with self._lock:
            self._free.clear()
            self._used = 0

    def stats(self):
        with self._lock:
            return {
                "free_buffers": sum(len(v) for v in self._free.values()),
                "used_bytes": self._used,
                "hit_rate": self.hit_rate,
                "hits": self._hits,
                "misses": self._misses,
            }

_default_pool = None

def get_pool(max_size_mb=512):
    global _default_pool
    if _default_pool is None:
        _default_pool = MemoryPool(max_size_mb)
    return _default_pool
