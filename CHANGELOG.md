# Changelog

## [0.1.0] - 2025-11-01

### Added
- Core device detection with HIP/CPU fallback
- DeviceBuffer with RAII memory management
- Kernel dispatch engine with registry
- Built-in kernels: vec_add, matmul, relu, softmax, reductions
- Memory pool with buddy-allocator
- Stream and StreamPool management
- Execution context (thread-local)
- Profiler with context manager support
- Linear algebra module (matmul, dot, norm, solve, svd)
- Architecture detection (RDNA/CDNA)
- Wavefront-aware grid optimization
- L2 cache tiling optimizer
- ROCm utilities
- Data loader and serializer
- Quantization utilities (fp32→fp16, fp32→int8)
- Metrics collector
- CLI framework
- Comprehensive test suite
- CI/CD pipeline

### Changed
- N/A (initial release)

### Fixed
- N/A (initial release)
