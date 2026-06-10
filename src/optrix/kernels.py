"""Built-in compute kernels."""
import numpy as np
from optrix.dispatch import register_kernel
from optrix.memory import DeviceBuffer

@register_kernel("vec_add")
def vec_add(a, b, output=None):
    """Element-wise vector addition."""
    ha, hb = a.to_host(), b.to_host()
    output.to_device(ha + hb)

@register_kernel("vec_mul")
def vec_mul(a, b, output=None):
    """Element-wise vector multiplication."""
    ha, hb = a.to_host(), b.to_host()
    output.to_device(ha * hb)

@register_kernel("matmul")
def matmul(a, b, output=None):
    """Matrix multiplication."""
    ha, hb = a.to_host(), b.to_host()
    output.to_device(ha @ hb)

@register_kernel("scale")
def scale(a, factor, output=None):
    """Scalar multiplication."""
    ha = a.to_host()
    output.to_device(ha * factor)

@register_kernel("relu")
def relu(a, output=None):
    """ReLU activation."""
    ha = a.to_host()
    output.to_device(np.maximum(0, ha))

@register_kernel("softmax")
def softmax(a, output=None):
    """Softmax along last axis."""
    ha = a.to_host()
    exp = np.exp(ha - np.max(ha, axis=-1, keepdims=True))
    output.to_device(exp / exp.sum(axis=-1, keepdims=True))
