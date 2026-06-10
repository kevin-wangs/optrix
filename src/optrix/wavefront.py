"""Wavefront-aware compute utilities."""
from __future__ import annotations
from typing import Tuple
import logging
from optrix.arch_detect import detect_arch, is_rdna, is_cdna

logger = logging.getLogger("optrix")

def optimal_wavefront_size(device_id=0):
    arch = detect_arch(device_id)
    if arch:
        return arch.wavefront_size
    return 64

def wavefront_aligned_grid(n_elements, device_id=0):
    """Compute grid dimensions aligned to wavefront boundaries."""
    wf = optimal_wavefront_size(device_id)
    block = min(256, max(wf, ((n_elements + wf - 1) // wf) * wf))
    block = min(block, 1024)
    grid = (n_elements + block - 1) // block
    return (grid, 1, 1), (block, 1, 1)

def occupancy_estimate(block_size, device_id=0):
    """Estimate occupancy for given block size."""
    arch = detect_arch(device_id)
    if not arch:
        return 0.5
    wf = arch.wavefront_size
    waves_per_block = (block_size + wf - 1) // wf
    max_blocks = arch.compute_units * 10  # simplified
    return min(1.0, waves_per_block * max_blocks / (arch.compute_units * 64))

def lds_size_per_workgroup(device_id=0):
    """Get available LDS (Local Data Share) per workgroup."""
    arch = detect_arch(device_id)
    if arch and is_cdna(arch):
        return 65536  # 64KB for CDNA
    return 32768  # 32KB for RDNA
