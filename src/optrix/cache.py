"""L2 cache-aware data layout utilities."""
from __future__ import annotations
import numpy as np
import logging
from optrix.arch_detect import detect_arch

logger = logging.getLogger("optrix")

class CacheOptimizer:
    """Optimize data layout for GPU cache hierarchy."""
    def __init__(self, device_id=0):
        arch = detect_arch(device_id)
        self.l2_kb = arch.l2_cache_kb if arch else 4096
        self.cache_line_bytes = 64  # standard

    def optimal_tile_size(self, dtype_bytes=4):
        """Compute optimal tile size for tiled matrix operations."""
        # Fit tile in L2 cache (use ~75% of L2)
        usable = self.l2_kb * 1024 * 0.75
        elements = usable / dtype_bytes
        side = int(np.sqrt(elements))
        # Round to power of 2
        side = 1 << (side.bit_length() - 1)
        return side

    def suggest_block_tiling(self, matrix_shape, dtype_bytes=4):
        """Suggest block tiling for matrix operations."""
        tile = self.optimal_tile_size(dtype_bytes)
        rows, cols = matrix_shape
        return {
            "tile_m": min(tile, rows),
            "tile_n": min(tile, cols),
            "tile_k": min(tile, min(rows, cols)),
            "cache_usage_kb": tile * tile * dtype_bytes / 1024,
        }
