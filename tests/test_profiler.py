"""Tests for optrix.profiler."""
import time
from optrix.profiler import Profiler

class TestProfiler:
    def test_profile_block(self):
        p = Profiler()
        idx = p.start("test")
        time.sleep(0.01)
        p.stop(idx)
        result = p.result()
        assert len(result.events) == 1
        assert result.events[0].duration_ms > 5

    def test_context_manager(self):
        p = Profiler()
        with p:
            time.sleep(0.01)
        assert len(p.events) >= 1
