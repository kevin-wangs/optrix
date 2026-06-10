# Optrix Architecture

## Layer Model

Optrix is structured in four layers:

### Layer 1: Hardware Abstraction (HAL)
Wraps HIP/ROCm runtime calls. Simulated CPU backend for development.

### Layer 2: Memory Manager
RAII `DeviceBuffer` objects with pool allocation and pinned memory.

### Layer 3: Kernel Dispatch Engine
Named kernel registry with auto-grid tuning and async launch.

### Layer 4: Python API
Clean numpy-like interface hiding all complexity.

## Data Flow

```
User Code → optrix.dispatch() → Kernel Registry → Memory Manager
    → Device Layer → HAL → HIP/ROCm → GPU
```

## Thread Safety

All public APIs are thread-safe via internal locking:
- MemoryPool uses `threading.Lock`
- MetricsCollector uses `threading.Lock`
- ExecutionContext is `threading.local`

## AMD Optimizations

1. Wavefront-aware grid sizing (64 threads)
2. LDS bank conflict avoidance
3. Arch-specific tuning (gfx1100, gfx90a, gfx942)
4. rocBLAS fallback for linear algebra
