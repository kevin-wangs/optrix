"""Tests for optrix.stream."""
from optrix.stream import Stream, StreamPool

class TestStream:
    def test_create(self):
        s = Stream(device_id=0)
        assert s.device_id == 0

    def test_repr(self):
        s = Stream()
        assert "Stream" in repr(s)

class TestStreamPool:
    def test_pool_size(self):
        pool = StreamPool(n_streams=4)
        assert len(pool.streams) == 4

    def test_round_robin(self):
        pool = StreamPool(n_streams=3)
        s1 = pool.acquire()
        s2 = pool.acquire()
        s3 = pool.acquire()
        s4 = pool.acquire()
        assert s4 is s1  # wraps around
