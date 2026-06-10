# AMD ROCm Integration Guide

## Installation

### System Requirements
- Ubuntu 22.04+ or RHEL 9+
- AMD GPU (RDNA 2+ or CDNA 1+)
- ROCm 6.0+

### Install ROCm
```bash
sudo apt update && sudo apt install rocm-dkms
sudo usermod -aG video $USER
sudo usermod -aG render $USER
# Reboot required
```

### Install Optrix with ROCm
```bash
ROCM_PATH=/opt/rocm pip install -e ".[rocm]"
```

## Architecture-Specific Features

### RDNA 3 (gfx1100)
- 64-thread wavefronts
- 32KB LDS per workgroup
- No matrix cores
- Best for: gaming, rendering, general compute

### CDNA 2 (gfx90a)
- 64-thread wavefronts
- 64KB LDS per workgroup
- Matrix cores (MFMA)
- Best for: HPC, AI training, scientific computing

### CDNA 3 (gfx942)
- 64-thread wavefronts
- 64KB LDS per workgroup
- Enhanced matrix cores
- Best for: large-scale AI, exascale HPC

## ROCm SMI Monitoring

```python
from optrix.rocm_utils import get_smi_info
info = get_smi_info()
print(info)
```

## Troubleshooting

### "No GPU detected"
- Check: `rocm-smi`
- Ensure user is in `video` and `render` groups
- Verify: `ls /dev/kfd`

### Memory allocation failures
- Check VRAM: `rocm-smi --showmeminfo vram`
- Use memory pool to reduce fragmentation
