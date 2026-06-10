"""GPU profiling and performance measurement."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
import time
import logging

logger = logging.getLogger("optrix")

@dataclass
class ProfileEvent:
    name: str
    start: float
    end: float = 0.0
    device_id: int = 0
    metadata: Dict = field(default_factory=dict)

    @property
    def duration_ms(self):
        return (self.end - self.start) * 1000

@dataclass
class ProfileResult:
    events: List[ProfileEvent] = field(default_factory=list)

    def summary(self):
        lines = ["Profile Summary:", f"  {'Event':<30} {'Duration (ms)':>12} {'Device':>8}"]
        lines.append("  " + "-" * 52)
        for e in self.events:
            lines.append(f"  {e.name:<30} {e.duration_ms:>12.3f} {e.device_id:>8}")
        return "\n".join(lines)

class Profiler:
    """Context-based GPU profiler."""
    _instance = None

    def __init__(self):
        self.events: List[ProfileEvent] = []
        self._active = False

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def start(self, name, device_id=0):
        self._active = True
        event = ProfileEvent(name=name, start=time.perf_counter(), device_id=device_id)
        self.events.append(event)
        return len(self.events) - 1

    def stop(self, idx):
        if idx < len(self.events):
            self.events[idx].end = time.perf_counter()

    def clear(self):
        self.events.clear()

    def result(self):
        return ProfileResult(events=list(self.events))

    def __enter__(self):
        self._ctx_idx = self.start("profile_block")
        return self

    def __exit__(self, *exc):
        self.stop(self._ctx_idx)

def profile(func):
    """Decorator to profile a function."""
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        p = Profiler.instance()
        idx = p.start(func.__name__)
        result = func(*args, **kwargs)
        p.stop(idx)
        return result
    return wrapper
