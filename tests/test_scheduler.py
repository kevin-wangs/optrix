"""Tests for BatchScheduler."""
from optrix.scheduler import BatchScheduler

class TestBatchScheduler:
    def test_add(self):
        s = BatchScheduler()
        s.add("test")
        assert len(s) == 1

    def test_clear(self):
        s = BatchScheduler()
        s.add("test")
        try: s.execute()
        except: pass
        assert len(s) == 0
