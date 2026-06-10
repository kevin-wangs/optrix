# Troubleshooting

## Common Issues

### ImportError: No module named 'optrix._backend.hip'
This is expected when ROCm is not installed. Optrix falls back to
CPU simulation automatically. Install ROCm for GPU acceleration.

### RuntimeError: Device X not found
Check available devices: `optrix.info` CLI command or `optrix.detect_devices()`.

### MemoryError: Allocation failed
- Reduce buffer size
- Use memory pool: `optrix.get_pool()`
- Check VRAM: `rocm-smi --showmeminfo vram`

### Slow performance
- Ensure ROCm is installed and detected
- Use pinned memory for large transfers
- Profile with `optrix.Profiler`
- Check GPU utilization: `rocm-smi`

## Getting Help
- GitHub Issues: https://github.com/kevin-wangs/optrix/issues
- Discussions: https://github.com/kevin-wangs/optrix/discussions
