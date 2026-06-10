"""Structured logging setup for Optrix."""
import logging
import sys

_initialized = False

def setup_logging(level=None):
    global _initialized
    if _initialized: return
    if level is None:
        from optrix.config import get_config
        level = get_config().get("log_level", "WARNING")
    fmt = "[%(asctime)s] %(name)s %(levelname)s: %(message)s"
    logging.basicConfig(level=getattr(logging, level), format=fmt, stream=sys.stderr)
    _initialized = True

def get_logger(name):
    setup_logging()
    return logging.getLogger(f"optrix.{name}")
