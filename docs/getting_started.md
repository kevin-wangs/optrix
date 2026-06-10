# Getting Started with Optrix

## Installation

```bash
pip install optrix
```

## Your First GPU Program

```python
import optrix

# Check available devices
devices = optrix.detect_devices()
print(devices[0])

# Create GPU buffers
a = optrix.zeros((1024,), dtype=optrix.float32)
b = optrix.zeros((1024,), dtype=optrix.float32)

# Upload data
import numpy as np
a.to_device(np.ones(1024, dtype=np.float32))
b.to_device(np.ones(1024, dtype=np.float32) * 2)

# Register and dispatch a kernel
@optrix.register_kernel("my_add")
def my_add(x, y, output=None):
    output.to_device(x.to_host() + y.to_host())

result = optrix.dispatch("my_add", a, b)
print(result.to_host()[:5])  # [3. 3. 3. 3. 3.]
```

## Next Steps

- Read the [API Reference](api_reference.md)
- Check out [examples/](../examples/)
- Learn about [Architecture](../ARCHITECTURE.md)
