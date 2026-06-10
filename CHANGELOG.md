# Changelog

## [0.2.0] - 2025-11-15

### Added
- ones(), full(), zeros_like() buffer allocators
- DeviceBuffer.fill() and .copy() methods
- CLI with info/bench/kernels commands
- Monte Carlo Pi estimation example
- ML inference pipeline example
- Benchmark example with timing
- Comprehensive API reference docs
- Performance optimization guide
- ROCm integration guide
- Security policy
- Code of conduct
- GitHub Actions CI (multi-Python matrix)
- Issue and PR templates

### Fixed
- Shape parsing for integer inputs
- Kernel argument validation
- Memory allocation edge cases

### Changed
- Version bump to 0.2.0
- Ruff formatting applied

## [0.1.0] - 2025-11-01

### Added
- Initial release
- Core device detection with HIP/CPU fallback
- DeviceBuffer with RAII memory management
- Kernel dispatch engine with registry
- Built-in kernels
- Memory pool
- Stream management
- Profiler
- Linear algebra module
- Architecture detection
- Test suite
