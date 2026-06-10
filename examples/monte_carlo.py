#!/usr/bin/env python3
"""Monte Carlo Pi estimation on GPU."""
import numpy as np
import time
import optrix

n = 10_000_000
print(f"Monte Carlo Pi ({n:,} samples)")

x = np.random.uniform(-1, 1, n).astype(np.float32)
y = np.random.uniform(-1, 1, n).astype(np.float32)

start = time.perf_counter()
cpu_pi = 4.0 * np.sum(x**2 + y**2 <= 1) / n
cpu_t = time.perf_counter() - start

bx, by = optrix.from_numpy(x), optrix.from_numpy(y)
optrix.register_kernel("count_inside", lambda x, y, output=None: output.to_device(np.array([np.sum(x.to_host()**2 + y.to_host()**2 <= 1)], dtype=np.float32)))

start = time.perf_counter()
r = optrix.dispatch("count_inside", bx, by)
gpu_pi = 4.0 * r.to_host()[0] / n
gpu_t = time.perf_counter() - start

print(f"CPU: pi={cpu_pi:.6f} ({cpu_t:.3f}s)")
print(f"GPU: pi={gpu_pi:.6f} ({gpu_t:.3f}s)")
print(f"Speedup: {cpu_t/gpu_t:.1f}x")
