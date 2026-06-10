"""Tests for optrix.pool."""
from optrix.pool import MemoryPool

class TestMemoryPool:
    def test_acquire_release(self):
        pool = MemoryPool(max_size_mb=64)
        buf = pool.acquire((10,), dtype=__import__("optrix").float32)
        assert buf.shape == (10,)
        pool.release(buf)
        assert pool.stats()["free_buffers"] == 1

    def test_hit_rate(self):
        pool = MemoryPool(max_size_mb=64)
        import optrix
        buf = pool.acquire((5,), dtype=optrix.float32)
        pool.release(buf)
        buf2 = pool.acquire((5,), dtype=optrix.float32)
        assert pool.stats()["hits"] == 1
