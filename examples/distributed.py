#!/usr/bin/env python3
"""Multi-device computation example."""
import numpy as np
import optrix

devices = optrix.detect_devices()
print(f"Available devices: {len(devices)}")

# Split work across devices
n = 10_000_000
chunk = n // len(devices)

for i, dev in enumerate(devices):
    buf = optrix.zeros(chunk, dtype=optrix.float32, device_id=i)
    buf.to_device(np.random.randn(chunk).astype(np.float32))
    result = optrix.dispatch("reduce_sum", buf)
    print(f"Device {i}: sum = {result.to_host()[0]:.2f}")
