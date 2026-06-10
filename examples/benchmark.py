#!/usr/bin/env python3
"""Benchmark Optrix operations."""
import time
import numpy as np
import optrix

def bench(name, fn, iterations=100):
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    avg = np.mean(times) * 1000
    p50 = np.percentile(times, 50) * 1000
    p99 = np.percentile(times, 99) * 1000
    print(f"{name:<30} avg={avg:.3f}ms  p50={p50:.3f}ms  p99={p99:.3f}ms")

print("Optrix Benchmarks")
print("=" * 60)

for size in [256, 512, 1024, 2048]:
    a = optrix.zeros((size, size), dtype=optrix.float32)
    b = optrix.zeros((size, size), dtype=optrix.float32)
    a.to_device(np.ones((size, size), dtype=np.float32))
    b.to_device(np.ones((size, size), dtype=np.float32))

    @optrix.register_kernel(f"matmul_{size}")
    def mm(x, y, output=None):
        output.to_device(x.to_host() @ y.to_host())

    bench(f"matmul {size}x{size}", lambda: optrix.dispatch(f"matmul_{size}", a, b))
