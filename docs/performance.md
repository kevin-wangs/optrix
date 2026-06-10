# Performance Guide

## Memory Management

### Pool Allocation
```python
from optrix import get_pool
pool = get_pool(max_size_mb=1024)
buf = pool.acquire((4096, 4096), dtype=optrix.float32)
# ... use buffer ...
pool.release(buf)
```

### Pinned Memory
```python
from optrix.allocator import PinnedAllocator
alloc = PinnedAllocator()
buf = alloc.allocate((1024, 1024))
```

## Kernel Optimization

### Wavefront Alignment
```python
grid, block = optrix.auto_grid(n_elements, block_size=256)
```

### Cache Tiling
```python
from optrix.cache import CacheOptimizer
opt = CacheOptimizer()
tiling = opt.suggest_block_tiling((4096, 4096))
```

## Profiling
```python
with optrix.Profiler() as p:
    result = optrix.dispatch("matmul", a, b)
print(p.result().summary())
```
