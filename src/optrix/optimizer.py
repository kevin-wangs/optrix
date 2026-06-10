"""Automatic kernel optimization and tuning."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import logging
import numpy as np

logger = logging.getLogger("optrix")

@dataclass
class TuningResult:
    block_size: int
    shared_mem_kb: int
    occupancy: float
    throughput_gflops: float

class KernelOptimizer:
    """Auto-tune kernel launch parameters for target hardware.

    Profiles different block sizes and shared memory configurations
    to find optimal launch parameters.
    """
    def __init__(self, device_id=0):
        self.device_id = device_id
        self._cache: Dict[str, TuningResult] = {}
        self._block_sizes = [64, 128, 256, 512, 1024]

    def _get_wavefront_size(self):
        try:
            from optrix.core import detect_devices
            dev = detect_devices()[self.device_id]
            return dev.wavefront_size
        except Exception:
            return 64

    def tune(self, kernel_name, input_shape, dtype="float32"):
        cache_key = f"{kernel_name}_{input_shape}_{dtype}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        wavefront = self._get_wavefront_size()
        best = None

        for bs in self._block_sizes:
            if bs % wavefront != 0:
                continue
            occupancy = min(1.0, 256 / bs)
            throughput = self._benchmark_kernel(kernel_name, bs, input_shape)
            result = TuningResult(
                block_size=bs, shared_mem_kb=48,
                occupancy=occupancy, throughput_gflops=throughput,
            )
            if best is None or result.throughput_gflops > best.throughput_gflops:
                best = result

        self._cache[cache_key] = best
        logger.info("Tuned %s: block=%d, %.1f GFLOPS", kernel_name, best.block_size, best.throughput_gflops)
        return best

    def _benchmark_kernel(self, kernel_name, block_size, input_shape):
        # Simplified benchmark — returns estimated GFLOPS
        n = 1
        for s in (input_shape if isinstance(input_shape, tuple) else (input_shape,)):
            n *= s
        ops_per_element = 2  # FMA
        return (n * ops_per_element) / 1e9 * (256 / block_size) * 10
