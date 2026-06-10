"""Utility decorators for Optrix."""
import functools
import time
import logging

logger = logging.getLogger("optrix")

def timer(func):
    """Log execution time of a function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.debug("%s took %.4fs", func.__name__, elapsed)
        return result
    return wrapper

def experimental(func):
    """Mark a function as experimental."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.warning("Calling experimental function: %s", func.__name__)
        return func(*args, **kwargs)
    wrapper._experimental = True
    return wrapper

def requires_rocm(func):
    """Ensure ROCm is available before calling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import optrix._backend.hip
        except ImportError:
            raise RuntimeError(f"{func.__name__} requires ROCm/HIP")
        return func(*args, **kwargs)
    return wrapper
