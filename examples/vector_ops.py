#!/usr/bin/env python3
"""Vector operations showcase."""
import numpy as np
import optrix

n = 1_000_000
a = optrix.from_numpy(np.random.randn(n).astype(np.float32))
b = optrix.from_numpy(np.random.randn(n).astype(np.float32))

# Vector add
c = optrix.dispatch("vec_add", a, b)
print(f"vec_add: {c.to_host()[:3]}")

# Dot product
d = optrix.dot(a, b)
print(f"dot: {d:.2f}")

# Norm
print(f"norm: {optrix.norm(a):.2f}")

# Softmax
s = optrix.from_numpy(np.array([1, 2, 3, 4, 5], dtype=np.float32))
r = optrix.dispatch("softmax", s)
print(f"softmax: {r.to_host()}")
