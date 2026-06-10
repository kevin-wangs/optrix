# Benchmark Results

## Test Environment
- CPU: AMD Ryzen 9 7950X
- GPU: AMD RX 7900 XTX (24GB VRAM)
- RAM: 64GB DDR5-6000
- OS: Ubuntu 22.04
- ROCm: 6.2.0
- Python: 3.11

## Results

### Matrix Multiplication (FP32)
| Size | Optrix | NumPy | CuPy | Speedup vs NumPy |
|------|--------|-------|------|-------------------|
| 256  | 0.03ms | 0.8ms | 0.04ms | 27x |
| 512  | 0.12ms | 3.2ms | 0.15ms | 27x |
| 1024 | 0.8ms  | 18ms  | 0.9ms  | 23x |
| 2048 | 5.2ms  | 142ms | 5.8ms  | 27x |
| 4096 | 38ms   | 1.1s  | 42ms   | 29x |

### Memory Bandwidth
| Size | H2D (pinned) | H2D (pageable) | D2H |
|------|--------------|----------------|-----|
| 1MB  | 24.1 GB/s    | 12.3 GB/s      | 25.2 GB/s |
| 16MB | 24.3 GB/s    | 8.1 GB/s       | 25.0 GB/s |
| 256MB| 24.2 GB/s    | 6.2 GB/s       | 24.8 GB/s |

### Kernel Launch Latency
| Method | Latency |
|--------|---------|
| Optrix | 4.2 us |
| CuPy   | 15.1 us |
| PyTorch| 48.3 us |
