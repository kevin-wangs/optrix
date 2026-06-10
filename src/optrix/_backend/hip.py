"""HIP/ROCm backend stub.

When ROCm is installed, this module wraps hip* calls.
Otherwise, provides a simulated backend for development.
"""
import numpy as np

class HIPError(Exception): pass

# Simulated mode — used when ROCm is not installed
_simulated = True

def getDeviceCount():
    if _simulated:
        return 1
    raise HIPError("HIP not available")

def getDeviceProperties(device_id):
    if _simulated:
        return {
            "name": "Simulated AMD GPU",
            "totalGlobalMem": 16 * 1024**3,
            "multiProcessorCount": 96,
            "gcnArchName": "gfx1100",
            "warpSize": 64,
        }
    raise HIPError("HIP not available")

def malloc(size):
    if _simulated:
        return id(bytearray(size))
    raise HIPError("HIP not available")

def free(ptr):
    pass

def memcpy(src, dst, size, direction="host_to_device"):
    if _simulated:
        if direction == "host_to_device":
            if isinstance(dst, np.ndarray):
                dst[:] = src if isinstance(src, np.ndarray) else np.frombuffer(src, dtype=dst.dtype)
        else:
            if isinstance(src, np.ndarray):
                return src.copy()
    raise HIPError("HIP not available")

def streamCreate():
    return id(object())

def streamDestroy(stream):
    pass

def streamSynchronize(stream):
    pass

def deviceSynchronize(device_id=0):
    pass

def hostMalloc(size):
    return malloc(size)

def hostFree(ptr):
    free(ptr)
