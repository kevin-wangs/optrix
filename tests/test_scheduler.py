"""Tests for optrix.scheduler."""
from optrix.scheduler import BatchScheduler

class TestBatchScheduler:
    def test_add_and_len(self):
        s = BatchScheduler()
        s.add("vec_add")
        s.add("matmul")
        assert len(s) == 2

    def test_clear_on_execute(self):
        s = BatchScheduler()
        s.add("vec_add")
        # execute will fail without proper args, but queue should clear
        try: s.execute()
        except: pass
        assert len(s) == 0
