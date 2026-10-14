"""Adds NumPy array support to msgpack.

msgpack is good for (de)serializing data over a network for multiple reasons:
- msgpack is secure (as opposed to pickle/dill/etc which allow for arbitrary code execution)
- msgpack is widely used and has good cross-language support
- msgpack does not require a schema (as opposed to protobuf/flatbuffers/etc) which is convenient in dynamically typed
    languages like Python and JavaScript
- msgpack is fast and efficient (as opposed to readable formats like JSON/YAML/etc); I found that msgpack was ~4x faster
    than pickle for serializing large arrays using the below strategy

The code below is adapted from https://github.com/lebedov/msgpack-numpy. The reason not to use that library directly is
that it falls back to pickle for object arrays.
"""

import functools
import os
import zlib

import msgpack
import numpy as np

# Opt-in lossless wire compression for large uint8 arrays (camera images). The sender compresses only when
# OPENPI_WIRE_ZLIB=1; every receiver accepts both forms, so decoded arrays are byte-identical either way.
_ZLIB_MIN_BYTES = 65536


def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        data = obj.tobytes()
        if obj.dtype == np.uint8 and len(data) >= _ZLIB_MIN_BYTES and os.environ.get("OPENPI_WIRE_ZLIB") == "1":
            return {
                b"__ndarray__": True,
                b"data": zlib.compress(data, 1),
                b"zlib": True,
                b"dtype": obj.dtype.str,
                b"shape": obj.shape,
            }
        return {
            b"__ndarray__": True,
            b"data": data,
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        data = zlib.decompress(obj[b"data"]) if obj.get(b"zlib") else obj[b"data"]
        return np.ndarray(buffer=data, dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)
