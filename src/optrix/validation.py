"""Input validation utilities."""
from optrix.exceptions import ShapeMismatchError, OptrixError

def validate_shape(buf, expected):
    if isinstance(expected, int):
        expected = (expected,)
    if buf.shape != expected:
        raise ShapeMismatchError(f"Expected shape {expected}, got {buf.shape}")

def validate_dtype(buf, expected):
    if buf.dtype != expected:
        raise TypeError(f"Expected dtype {expected}, got {buf.dtype}")

def validate_device_count(device_id, available):
    if device_id >= available:
        from optrix.exceptions import DeviceNotFoundError
        raise DeviceNotFoundError(device_id)

def validate_kernel_args(func, args, kwargs):
    """Validate kernel function signature matches provided args."""
    import inspect
    sig = inspect.signature(func)
    try:
        sig.bind(*args, **kwargs)
    except TypeError as e:
        raise OptrixError(f"Kernel argument mismatch: {e}")
