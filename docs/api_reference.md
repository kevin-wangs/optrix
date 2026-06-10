# Optrix API Reference

## Core

### `detect_devices() -> List[DeviceInfo]`
Detect all available ROCm/HIP compute devices.

### `synchronize(device_id=0) -> None`
Block until all pending work on device finishes.

### `DeviceInfo`
- `.device_id: int`
- `.name: str`
- `.total_memory: int` (bytes)
- `.compute_units: int`
- `.arch: str` (e.g., "gfx1100")
- `.wavefront_size: int`
- `.memory_gb: float`

## Memory

### `zeros(shape, dtype=float32, device_id=0) -> DeviceBuffer`
Allocate zero-initialized device buffer.

### `empty(shape, dtype=float32, device_id=0) -> DeviceBuffer`
Allocate uninitialized device buffer.

### `ones(shape, dtype=float32, device_id=0) -> DeviceBuffer`
Allocate one-initialized device buffer.

### `DeviceBuffer`
- `.shape: Tuple[int, ...]`
- `.dtype: Dtype`
- `.nbytes: int`
- `.to_host() -> np.ndarray`
- `.to_device(array: np.ndarray) -> None`
- `.fill(value) -> None`
- `.copy() -> DeviceBuffer`

## Dispatch

### `register_kernel(name, func=None) -> Callable`
Register a compute kernel.

### `dispatch(kernel_name, *args, output=None) -> DeviceBuffer`
Execute a registered kernel.

### `list_kernels() -> List[str]`
List all registered kernel names.

### `auto_grid(n_elements, block_size=256) -> Tuple`
Compute optimal grid/block dimensions.

## Linear Algebra

### `matmul(a, b) -> DeviceBuffer`
Matrix multiplication.

### `dot(a, b) -> float`
Dot product.

### `norm(a, ord=2) -> float`
Vector norm.

### `solve(A, b) -> DeviceBuffer`
Solve linear system Ax = b.

### `svd(a) -> Tuple[DeviceBuffer, DeviceBuffer, DeviceBuffer]`
Singular Value Decomposition.
