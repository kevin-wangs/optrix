"""GPU architecture detection and feature flags."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import logging
import subprocess

logger = logging.getLogger("optrix")

@dataclass
class GPUArch:
    """GPU architecture descriptor."""
    name: str
    gfx: str
    generation: str
    wavefront_size: int
    compute_units: int
    max_workgroup_size: int
    has_matrix_cores: bool
    l2_cache_kb: int

ARCH_DB = {
    "gfx1100": GPUArch("RDNA 3", "gfx1100", "RDNA3", 64, 96, 1024, False, 4096),
    "gfx1101": GPUArch("RDNA 3", "gfx1101", "RDNA3", 64, 64, 1024, False, 2048),
    "gfx1030": GPUArch("RDNA 2", "gfx1030", "RDNA2", 64, 80, 1024, False, 4096),
    "gfx1031": GPUArch("RDNA 2", "gfx1031", "RDNA2", 64, 40, 1024, False, 2048),
    "gfx90a": GPUArch("CDNA 2", "gfx90a", "CDNA2", 64, 110, 1024, True, 8192),
    "gfx908": GPUArch("CDNA 1", "gfx908", "CDNA1", 64, 120, 1024, True, 8192),
    "gfx942": GPUArch("CDNA 3", "gfx942", "CDNA3", 64, 304, 1024, True, 16384),
}

def detect_arch(device_id=0) -> Optional[GPUArch]:
    """Detect GPU architecture via ROCm SMI."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showproductname", "--showmeminfo", "vram"],
            capture_output=True, text=True, timeout=5,
        )
        for gfx, arch in ARCH_DB.items():
            if gfx in result.stdout.lower():
                return arch
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    try:
        from optrix.core import detect_devices
        dev = detect_devices()[device_id]
        if dev.arch in ARCH_DB:
            return ARCH_DB[dev.arch]
    except Exception:
        pass

    return None

def is_rdna(arch):
    return arch and arch.generation.startswith("RDNA")

def is_cdna(arch):
    return arch and arch.generation.startswith("CDNA")

def has_matrix_cores(arch):
    return arch and arch.has_matrix_cores
