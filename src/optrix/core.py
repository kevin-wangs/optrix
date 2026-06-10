"""Device detection and runtime initialization."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
import os, logging

logger = logging.getLogger("optrix")

@dataclass
class DeviceInfo:
    device_id: int
    name: str
    total_memory: int
    compute_units: int
    arch: str
    wavefront_size: int
    is_integrated: bool = False

    @property
    def memory_gb(self):
        return self.total_memory / (1024 ** 3)

    def __repr__(self):
        return f"Device(id={self.device_id}, name='{self.name}', arch={self.arch}, mem={self.memory_gb:.1f}GB)"

def detect_devices():
    try:
        from optrix._backend import hip
        count = hip.getDeviceCount()
        devices = []
        for i in range(count):
            props = hip.getDeviceProperties(i)
            devices.append(DeviceInfo(
                device_id=i, name=props["name"],
                total_memory=props["totalGlobalMem"],
                compute_units=props["multiProcessorCount"],
                arch=props["gcnArchName"],
                wavefront_size=props["warpSize"],
            ))
        return devices
    except ImportError:
        return [DeviceInfo(
            device_id=0, name="Simulated CPU",
            total_memory=os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            compute_units=os.cpu_count() or 1, arch="cpu-sim",
            wavefront_size=1, is_integrated=True,
        )]

def synchronize(device_id=0):
    try:
        from optrix._backend import hip
        hip.deviceSynchronize(device_id)
    except ImportError:
        pass
