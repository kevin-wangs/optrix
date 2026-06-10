# API Reference

## Core
- `detect_devices() -> List[DeviceInfo]`
- `synchronize(device_id=0)`
- `DeviceInfo`: .name, .arch, .memory_gb, .compute_units

## Memory
- `zeros(shape, dtype, device_id)` / `empty()` / `ones()` / `full()`
- `DeviceBuffer`: .to_host(), .to_device(), .fill(), .copy()

## Dispatch
- `register_kernel(name, func)` — register a compute kernel
- `dispatch(kernel_name, *args, output=None)` — execute kernel
- `list_kernels()` — list registered kernels
- `auto_grid(n, block=256)` — compute grid/block dims

## Linear Algebra
- `matmul(a, b)` / `dot(a, b)` / `norm(a)` / `solve(A, b)` / `svd(a)`

## Profiling
- `Profiler` — context manager for timing
- `profile(func)` — decorator for timing
