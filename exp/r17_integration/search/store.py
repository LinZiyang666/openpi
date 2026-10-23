"""Versioned read-only dense/ragged NumPy store, published by READY manifest."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from exp.r17_integration.debug.validate import canonical, digest, file_sha, numeric_arrays, require_binding

SCHEMA = "r17.store.v1"


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_json(path, value):
    path = Path(path)
    part = path.with_name(path.name + ".part")
    with part.open("wb") as f:
        f.write(canonical(value) + b"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)
    fsync_directory(path.parent)


def write_store(directory, binding, rows, arrays, *, source_hashes=None, metadata=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    atomic_json(directory / "PARTIAL.json", dict(schema=SCHEMA, binding=binding, status="partial"))
    if not rows or not arrays:
        raise ValueError("empty store")
    manifest = dict(schema=SCHEMA, status="READY", binding=binding, rows=rows,
                    sources=source_hashes or {}, metadata=metadata or {}, arrays={})
    for name, values in arrays.items():
        numeric_arrays({name: np.empty(0)})  # enforce safe flat dataset names
        if len(values) != len(rows):
            raise ValueError("row/array count mismatch: " + name)
        values = [numeric_arrays({name: v})[name] for v in values]
        if any(v.dtype != values[0].dtype for v in values):
            raise ValueError("mixed dtype: " + name)
        dense = all(v.shape == values[0].shape for v in values)
        data = np.stack(values) if dense else np.concatenate([v.reshape(-1) for v in values])
        path = directory / (name + ".npy")
        with path.open("xb") as f:
            np.save(f, data, allow_pickle=False)
            f.flush()
            os.fsync(f.fileno())
        entry = dict(file=path.name, sha256=file_sha(path), dtype=data.dtype.str,
                     shape=list(data.shape), mode="dense" if dense else "ragged")
        if not dense:
            entry.update(row_shapes=[list(v.shape) for v in values],
                         offsets=np.cumsum([0] + [v.size for v in values]).tolist())
        manifest["arrays"][name] = entry
    manifest["content_digest"] = digest(manifest)
    atomic_json(directory / "READY.json", manifest)
    return Store(directory)


class Store:
    def __init__(self, directory, *, expected_binding=None):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory / "READY.json").read_text())
        sealed = dict(self.manifest)
        content_digest = sealed.pop("content_digest")
        if digest(sealed) != content_digest or sealed["schema"] != SCHEMA or sealed["status"] != "READY":
            raise ValueError("store manifest tampered or unsupported")
        if expected_binding is not None:
            require_binding(sealed["binding"], expected_binding)
        self._arrays = {}
        for name, entry in sealed["arrays"].items():
            path = self.directory / entry["file"]
            data = np.load(path, mmap_mode="r", allow_pickle=False)
            if data.dtype.str != entry["dtype"] or list(data.shape) != entry["shape"]:
                raise ValueError("store geometry mismatch: " + name)
            if entry["mode"] == "dense":
                self._arrays[name] = data
            else:
                offsets = entry["offsets"]
                self._arrays[name] = tuple(data[offsets[i]:offsets[i+1]].reshape(shape)
                                          for i, shape in enumerate(entry["row_shapes"]))

    @property
    def rows(self):
        return tuple(json.loads(canonical(self.manifest["rows"])))

    def array(self, name):
        return self._arrays[name]

    @property
    def fields(self):
        return tuple(self._arrays)
