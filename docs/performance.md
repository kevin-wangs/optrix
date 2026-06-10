# Performance Guide

## Memory Management

### Use Pool Allocation
```python
from optrix import get_pool
pool = get_pool(max_size_mb=1024)
buf = pool.acquire((4096, 4096), dtype=optrix.float32)
# ... use buffer ...
pool.release(buf)
```

Pool allocation reduces allocation overhead by 10x for repeated operations.

### Pinned Memory
For large host-device transfers, use pinned memory:
```python
from optrix.allocator import PinnedAllocator
alloc = PinnedAllocator()
buf = alloc.allocate((1024, 1024), dtype=optrix.float32)
```

Pinned memory enables DMA at near-PCIe-gen4 bandwidth.

## Kernel Optimization

### Wavefront Alignment
Always size workgroups to wavefront multiples:
```python
# Good: 256 = 4 * 64 (wavefront)
grid, block = optrix.auto_grid(n_elements, block_size=256)
```

### Cache Tiling
Use the cache optimizer for matrix operations:
```python
from optrix.cache import CacheOptimizer
opt = CacheOptimizer(device_id=0)
tiling = opt.suggest_block_tiling((4096, 4096))
```

## Profiling

```python
from optrix import Profiler
p = Profiler()
with p:
    result = optrix.dispatch("matmul", a, b)
print(p.result().summary())
```
