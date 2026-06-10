# AMD ROCm Integration Guide

## Installation
```bash
sudo apt install rocm-dkms
sudo usermod -aG video,render $USER
ROCM_PATH=/opt/rocm pip install -e ".[rocm]"
```

## Architecture Features

### RDNA 3 (gfx1100) — RX 7900 XTX
- 64-thread wavefronts, 32KB LDS
- Best for: gaming, rendering, general compute

### CDNA 2 (gfx90a) — MI210/MI250
- 64-thread wavefronts, 64KB LDS, Matrix Cores
- Best for: HPC, AI training

### CDNA 3 (gfx942) — MI300X
- Enhanced Matrix Cores, 16MB L2
- Best for: large-scale AI, exascale HPC

## Troubleshooting
- "No GPU detected": check `rocm-smi`, verify groups
- Memory failures: check `rocm-smi --showmeminfo vram`
