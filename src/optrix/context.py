"""Execution context management."""
from __future__ import annotations
from contextlib import contextmanager
from typing import Optional
import threading
import logging

logger = logging.getLogger("optrix")

_local = threading.local()

class ExecutionContext:
    """Thread-local execution context."""
    def __init__(self, device_id=0, stream=None):
        self.device_id = device_id
        self.stream = stream
        self._prev = None

    def __enter__(self):
        self._prev = getattr(_local, "context", None)
        _local.context = self
        return self

    def __exit__(self, *exc):
        _local.context = self._prev

def current_context():
    return getattr(_local, "context", ExecutionContext())

@contextmanager
def device_context(device_id):
    with ExecutionContext(device_id=device_id) as ctx:
        yield ctx

@contextmanager
def stream_context(stream):
    ctx = current_context()
    with ExecutionContext(device_id=ctx.device_id, stream=stream) as new_ctx:
        yield new_ctx
