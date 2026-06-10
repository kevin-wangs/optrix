#!/usr/bin/env python3
"""Simple ML inference pipeline using Optrix."""
import numpy as np
import optrix

# Simulate a 2-layer MLP
def relu(buf):
    return optrix.dispatch("relu", buf)

def forward(x, w1, b1, w2, b2):
    h = optrix.dispatch("matmul", x, w1)
    h = relu(h)
    out = optrix.dispatch("matmul", h, w2)
    return out

# Setup
batch, input_dim, hidden, output_dim = 32, 784, 256, 10
x = optrix.zeros((batch, input_dim), dtype=optrix.float32)
w1 = optrix.zeros((input_dim, hidden), dtype=optrix.float32)
w2 = optrix.zeros((hidden, output_dim), dtype=optrix.float32)

x.to_device(np.random.randn(batch, input_dim).astype(np.float32))
w1.to_device(np.random.randn(input_dim, hidden).astype(np.float32) * 0.01)
w2.to_device(np.random.randn(hidden, output_dim).astype(np.float32) * 0.01)

result = forward(x, w1, None, w2, None)
print(f"Inference output shape: {result.shape}")
print(f"Predictions: {np.argmax(result.to_host(), axis=1)}")
