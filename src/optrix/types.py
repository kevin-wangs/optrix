"""Core type definitions for Optrix."""
from enum import Enum
import numpy as np

class Dtype(Enum):
    float16 = "float16"
    float32 = "float32"
    float64 = "float64"
    int8 = "int8"
    int16 = "int16"
    int32 = "int32"
    int64 = "int64"
    uint8 = "uint8"
    bfloat16 = "bfloat16"

    @property
    def np_dtype(self):
        _m = {"float16": np.float16, "float32": np.float32, "float64": np.float64,
              "int8": np.int8, "int16": np.int16, "int32": np.int32, "int64": np.int64,
              "uint8": np.uint8, "bfloat16": np.float16}
        return np.dtype(_m[self.value])

    @property
    def itemsize(self):
        return self.np_dtype.itemsize

float16 = Dtype.float16
float32 = Dtype.float32
float64 = Dtype.float64
int32 = Dtype.int32
int64 = Dtype.int64
