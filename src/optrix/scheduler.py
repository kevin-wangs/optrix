"""Task scheduler for batch kernel dispatch."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Callable
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
    def __init__(self):
        self._queue: List[ScheduledTask] = []

    def add(self, kernel, *args, priority=0, **kwargs):
        self._queue.append(ScheduledTask(kernel, args, kwargs, priority))

    def execute(self):
        from optrix.dispatch import dispatch
        self._queue.sort(key=lambda t: t.priority, reverse=True)
        results = []
        for task in self._queue:
            result = dispatch(task.kernel, *task.args, **task.kwargs)
            results.append(result)
            if task.callback: task.callback(result)
        self._queue.clear()
        return results

    def __len__(self):
        return len(self._queue)
