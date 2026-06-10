"""High-level device handle with stream management."""
from __future__ import annotations
from contextlib import contextmanager
from typing import Optional, List
import logging
from optrix.core import DeviceInfo, detect_devices

logger = logging.getLogger("optrix")

class Device:
    _registry = {}

    def __init__(self, device_id=0):
        self.device_id = device_id
        self._devices = detect_devices()
        if device_id >= len(self._devices):
            raise ValueError(f"Device {device_id} not found")
        self.info = self._devices[device_id]
        self._streams = []
        Device._registry[device_id] = self

    @property
    def name(self): return self.info.name
    @property
    def arch(self): return self.info.arch

    def create_stream(self):
        try:
            from optrix._backend import hip
            stream = hip.streamCreate()
        except ImportError:
            stream = len(self._streams)
        self._streams.append(stream)
        return stream

    def destroy_stream(self, stream):
        try:
            from optrix._backend import hip
            hip.streamDestroy(stream)
        except ImportError: pass
        if stream in self._streams: self._streams.remove(stream)

    @contextmanager
    def use_stream(self):
        stream = self.create_stream()
        try: yield stream
        finally: self.destroy_stream(stream)

    def query(self):
        try:
            from optrix._backend import hip
            return hip.deviceGetUtilization(self.device_id)
        except ImportError:
            return {"gpu_util": 0, "mem_util": 0, "temp_c": 0}

    @classmethod
    def default(cls):
        if 0 not in cls._registry: cls(0)
        return cls._registry[0]

    def __repr__(self):
        return f"Device(id={self.device_id}, name='{self.name}', arch={self.arch})"

def get_device(device_id=0):
    if device_id not in Device._registry: Device(device_id)
    return Device._registry[device_id]
