"""Tests for optrix.config."""
from optrix.config import Config, get_config

class TestConfig:
    def test_singleton(self):
        c1 = Config()
        c2 = Config()
        assert c1 is c2

    def test_defaults(self):
        c = get_config()
        assert c.get("default_device") == 0
        assert c.get("fallback_to_cpu") is True

    def test_set(self):
        c = get_config()
        c.set("test_key", 42)
        assert c.get("test_key") == 42
