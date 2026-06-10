"""Runtime metrics collection."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict
import time
import threading

@dataclass
class MetricPoint:
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)

class MetricsCollector:
    """Collect and aggregate runtime metrics."""
    _instance = None

    def __init__(self):
        self._metrics: Dict[str, List[float]] = defaultdict(list)
        self._lock = threading.Lock()

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def record(self, name: str, value: float, **tags):
        with self._lock:
            self._metrics[name].append(value)

    def mean(self, name: str) -> float:
        with self._lock:
            vals = self._metrics.get(name, [])
            return sum(vals) / len(vals) if vals else 0.0

    def p50(self, name: str) -> float:
        return self._percentile(name, 50)

    def p99(self, name: str) -> float:
        return self._percentile(name, 99)

    def _percentile(self, name, pct):
        import numpy as np
        with self._lock:
            vals = self._metrics.get(name, [])
            return float(np.percentile(vals, pct)) if vals else 0.0

    def summary(self) -> str:
        lines = ["Metrics Summary:"]
        for name, vals in sorted(self._metrics.items()):
            lines.append(f"  {name}: n={len(vals)}, mean={self.mean(name):.4f}")
        return "\n".join(lines)

    def clear(self):
        with self._lock:
            self._metrics.clear()
