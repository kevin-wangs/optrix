"""ROCm utility functions."""
from __future__ import annotations
import os
import subprocess
import logging
from typing import Optional

logger = logging.getLogger("optrix")

def get_rocm_path() -> Optional[str]:
    """Find ROCm installation path."""
    paths = ["/opt/rocm", "/usr/lib/rocm", os.environ.get("ROCM_PATH", "")]
    for p in paths:
        if p and os.path.isdir(p):
            return p
    return None

def get_rocm_version() -> Optional[str]:
    path = get_rocm_path()
    if not path:
        return None
    ver_file = os.path.join(path, ".info", "version")
    if os.path.exists(ver_file):
        return open(ver_file).read().strip()
    return None

def get_smi_info() -> dict:
    """Query rocm-smi for all device info."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--json", "--showallinfo"],
            capture_output=True, text=True, timeout=10,
        )
        import json
        return json.loads(result.stdout)
    except Exception as e:
        logger.warning("rocm-smi query failed: %s", e)
        return {}

def check_rocm_available() -> bool:
    """Check if ROCm runtime is available."""
    path = get_rocm_path()
    if not path:
        return False
    try:
        result = subprocess.run(
            ["rocm-smi", "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False
