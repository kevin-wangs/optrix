"""Global configuration management."""
import os
import json
import logging

logger = logging.getLogger("optrix")

_DEFAULTS = {
    "default_device": 0,
    "log_level": "WARNING",
    "memory_pool_size_mb": 512,
    "enable_profiling": False,
    "rocm_path": "/opt/rocm",
    "fallback_to_cpu": True,
}

class Config:
    _instance = None
    _data = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._data = dict(_DEFAULTS)
            cls._instance._load()
        return cls._instance

    def _load(self):
        path = os.environ.get("OPTRIX_CONFIG")
        if path and os.path.exists(path):
            with open(path) as f:
                self._data.update(json.load(f))
            logger.info("Loaded config from %s", path)

    def get(self, key, default=None):
        return self._data.get(key, default)

    def set(self, key, value):
        self._data[key] = value

    def __repr__(self):
        return f"Config({self._data})"

def get_config():
    return Config()
