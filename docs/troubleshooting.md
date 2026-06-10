# Troubleshooting

## No module 'optrix._backend.hip'
Expected without ROCm. Optrix falls back to CPU simulation.

## Device X not found
Run `optrix info` or `optrix.detect_devices()`.

## Allocation failed
- Use memory pool: `optrix.get_pool()`
- Check VRAM: `rocm-smi --showmeminfo vram`

## Slow performance
- Install ROCm for GPU acceleration
- Use pinned memory for large transfers
- Profile with `optrix.Profiler`
