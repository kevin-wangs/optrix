"""Task scheduler for batch kernel dispatch."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Callable, Optional
from concurrent.futures import ThreadPoolExecutor
import logging

logger = logging.getLogger("optrix")

@dataclass
class ScheduledTask:
    kernel: str
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)
    priority: int = 0
    callback: Optional[Callable] = None

class BatchScheduler:
    """Schedule and batch multiple kernel dispatches."""
    def __init__(self, max_workers=4):
        self._queue: List[ScheduledTask] = []
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def add(self, kernel, *args, priority=0, **kwargs):
        self._queue.append(ScheduledTask(kernel, args, kwargs, priority))

    def execute(self):
        from optrix.dispatch import dispatch
        self._queue.sort(key=lambda t: t.priority, reverse=True)
        results = []
        for task in self._queue:
            result = dispatch(task.kernel, *task.args, **task.kwargs)
            results.append(result)
            if task.callback:
                task.callback(result)
        self._queue.clear()
        return results

    def __len__(self):
        return len(self._queue)
