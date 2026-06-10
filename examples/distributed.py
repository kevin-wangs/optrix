#!/usr/bin/env python3
"""Multi-device computation."""
import numpy as np
import optrix

devices = optrix.detect_devices()
print(f"Devices: {len(devices)}")

for i, dev in enumerate(devices):
    buf = optrix.zeros(1_000_000, dtype=optrix.float32, device_id=i)
    buf.to_device(np.random.randn(1_000_000).astype(np.float32))
    r = optrix.dispatch("reduce_sum", buf)
    print(f"  [{i}] {dev.name}: sum={r.to_host()[0]:.2f}")
