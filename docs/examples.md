# Examples

## Quick Start
```python
import optrix
a = optrix.from_numpy(np.ones(1024, dtype=np.float32))
result = optrix.dispatch("reduce_sum", a)
```

## Available Examples
- `examples/quickstart.py` — basic usage
- `examples/benchmark.py` — performance timing
- `examples/ml_pipeline.py` — ML inference
- `examples/monte_carlo.py` — Monte Carlo Pi
- `examples/distributed.py` — multi-device
- `examples/vector_ops.py` — vector operations
