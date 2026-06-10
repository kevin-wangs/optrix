#!/usr/bin/env python3
"""Monte Carlo Pi estimation on GPU."""
import numpy as np
import time
import optrix

n_samples = 10_000_000
print(f"Monte Carlo Pi Estimation ({n_samples:,} samples)")

# CPU
start = time.perf_counter()
x = np.random.uniform(-1, 1, n_samples)
y = np.random.uniform(-1, 1, n_samples)
cpu_pi = 4.0 * np.sum(x**2 + y**2 <= 1) / n_samples
cpu_time = time.perf_counter() - start
print(f"CPU: pi={cpu_pi:.6f}, time={cpu_time:.3f}s")

# GPU via Optrix
buf_x = optrix.from_numpy(x)
buf_y = optrix.from_numpy(y)

@optrix.register_kernel("count_inside")
def count_inside(x, y, output=None):
    hx, hy = x.to_host(), y.to_host()
    inside = np.sum(hx**2 + hy**2 <= 1.0)
    output.to_device(np.array([inside], dtype=np.float32))

start = time.perf_counter()
result = optrix.dispatch("count_inside", buf_x, buf_y)
gpu_pi = 4.0 * result.to_host()[0] / n_samples
gpu_time = time.perf_counter() - start
print(f"GPU: pi={gpu_pi:.6f}, time={gpu_time:.3f}s")
print(f"Speedup: {cpu_time/gpu_time:.1f}x")
