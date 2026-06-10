"""Stream management for async execution."""
from __future__ import annotations
from contextlib import contextmanager
from typing import List
import logging

logger = logging.getLogger("optrix")

class Stream:
    """Async execution stream wrapper."""
    def __init__(self, device_id=0, priority=0):
        self.device_id = device_id
        self.priority = priority
        self._handle = None
        self._create()

    def _create(self):
        try:
            from optrix._backend import hip
            self._handle = hip.streamCreate()
        except ImportError:
            self._handle = id(self)

    def synchronize(self):
        try:
            from optrix._backend import hip
            hip.streamSynchronize(self._handle)
        except ImportError:
            pass

    def __del__(self):
        try:
            from optrix._backend import hip
            if self._handle: hip.streamDestroy(self._handle)
        except (ImportError, AttributeError): pass

    def __repr__(self):
        return f"Stream(device={self.device_id}, handle={self._handle})"

class StreamPool:
    """Pool of streams for round-robin scheduling."""
    def __init__(self, n_streams=4, device_id=0):
        self.streams = [Stream(device_id) for _ in range(n_streams)]
        self._idx = 0

    def acquire(self):
        s = self.streams[self._idx % len(self.streams)]
        self._idx += 1
        return s

    def synchronize_all(self):
        for s in self.streams:
            s.synchronize()

    @contextmanager
    def use(self):
        s = self.acquire()
        try: yield s
        finally: s.synchronize()
