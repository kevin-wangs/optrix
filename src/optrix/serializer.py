"""Buffer serialization for checkpointing."""
from __future__ import annotations
import pickle
import json
import logging
from pathlib import Path
from optrix.memory import DeviceBuffer

logger = logging.getLogger("optrix")

def serialize_buffer(buf: DeviceBuffer) -> dict:
    """Serialize a DeviceBuffer to a dict."""
    return {
        "shape": buf.shape,
        "dtype": buf.dtype.value,
        "device_id": buf.device_id,
        "data": buf.to_host().tolist(),
    }

def deserialize_buffer(data: dict) -> DeviceBuffer:
    """Deserialize a dict back to a DeviceBuffer."""
    import numpy as np
    from optrix.types import Dtype
    dtype = Dtype(data["dtype"])
    arr = np.array(data["data"], dtype=dtype.np_dtype)
    from optrix.memory import zeros
    buf = zeros(tuple(data["shape"]), dtype=dtype, device_id=data["device_id"])
    buf.to_device(arr)
    return buf

def save_checkpoint(buffers: dict, path: str):
    """Save multiple buffers to a checkpoint file."""
    data = {k: serialize_buffer(v) for k, v in buffers.items()}
    with open(path, "wb") as f:
        pickle.dump(data, f)
    logger.info("Checkpoint saved: %s (%d buffers)", path, len(buffers))

def load_checkpoint(path: str) -> dict:
    """Load buffers from a checkpoint file."""
    with open(path, "rb") as f:
        data = pickle.load(f)
    buffers = {k: deserialize_buffer(v) for k, v in data.items()}
    logger.info("Checkpoint loaded: %s (%d buffers)", path, len(buffers))
    return buffers
