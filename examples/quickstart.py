#!/usr/bin/env python3
"""Optrix Quick Start Example."""
import numpy as np
import optrix

# Detect devices
devices = optrix.detect_devices()
print(f"Found {len(devices)} device(s): {devices[0]}")

# Allocate and initialize
n = 1024
a = optrix.zeros((n, n), dtype=optrix.float32)
b = optrix.zeros((n, n), dtype=optrix.float32)
a.to_device(np.random.randn(n, n).astype(np.float32))
b.to_device(np.random.randn(n, n).astype(np.float32))

# Register kernel
@optrix.register_kernel("my_matmul")
def my_matmul(x, y, output=None):
    result = x.to_host() @ y.to_host()
    output.to_device(result)

# Dispatch
result = optrix.dispatch("my_matmul", a, b)
print(f"Result shape: {result.shape}")
print(f"Result[0][0]: {result.to_host()[0][0]:.4f}")
