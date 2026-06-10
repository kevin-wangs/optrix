"""Custom exceptions for Optrix."""

class OptrixError(Exception):
    """Base exception for all Optrix errors."""
    pass

class DeviceNotFoundError(OptrixError):
    """Raised when requested device is not available."""
    def __init__(self, device_id):
        super().__init__(f"Device {device_id} not found")
        self.device_id = device_id

class MemoryError(OptrixError):
    """Raised on device memory allocation failure."""
    pass

class KernelError(OptrixError):
    """Raised on kernel compilation or dispatch failure."""
    pass

class ShapeMismatchError(OptrixError):
    """Raised when buffer shapes don't match expected dimensions."""
    pass
