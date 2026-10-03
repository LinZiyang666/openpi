"""Campaign-shared, stat-validated SHA256 cache; never trust stale signatures."""
import fcntl
import hashlib
import json
import logging
from pathlib import Path

import numpy as np

from .writer import atomic_json


def cache_path(directory):
    directory = Path(directory).resolve()
    for parent in directory.parents:
        if parent.name == "runs":
            return parent.parent / "catalog" / "digest_cache.json"
    return directory.parent / "catalog" / "digest_cache.json"


class DigestCache:
    """Serialize updates across ports/processes and invalidate on any stat change.

    Entries use (realpath, size, mtime_ns). File hashes and mapped-array byte
    ranges are separate digests: the latter exclude a .npy header and identify
    the actual deployed view, rather than assuming that a copied table matches
    the store file. Arrays without a mapped source are hashed directly.
    """
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.hits = self.misses = self.bytes_hashed = 0

    def _lookup(self, path, kind, compute, byte_count=None):
        path = Path(path).resolve(strict=True)
        with self.lock_path.open("a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                cache = json.loads(self.path.read_text()) if self.path.exists() else {}
                if not isinstance(cache, dict) or cache.get("schema") != "osdebug.digest.v1" or not isinstance(cache.get("files"), dict):
                    cache = dict(schema="osdebug.digest.v1", files={})
            except (ValueError, TypeError):
                logging.getLogger("osdebug").warning("invalid digest cache %s; recomputing", self.path)
                cache = dict(schema="osdebug.digest.v1", files={})
            for _ in range(3):
                before = path.stat()
                signature = dict(realpath=str(path), size=before.st_size, mtime_ns=before.st_mtime_ns)
                entry = cache["files"].get(str(path), {})
                matches = isinstance(entry, dict) and entry.get("signature") == signature
                digests = entry.get("digests", {}) if matches else {}
                saved = digests.get(kind) if isinstance(digests, dict) else None
                if isinstance(saved, str) and len(saved) == 64 and all(c in "0123456789abcdef" for c in saved):
                    self.hits += 1
                    return saved
                digest = compute(path)
                after = path.stat()
                self.bytes_hashed += before.st_size if byte_count is None else byte_count
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    continue
                updated = dict(digests) if isinstance(digests, dict) else {}
                updated[kind] = digest
                cache["files"][str(path)] = dict(signature=signature, digests=updated)
                atomic_json(self.path, cache)
                self.misses += 1
                return digest
            raise OSError("artifact changed repeatedly while hashing: " + str(path))

    def file(self, path):
        def compute(real):
            digest = hashlib.sha256()
            with real.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            return digest.hexdigest()
        return self._lookup(path, "file_sha256", compute)

    def array(self, value):
        array = np.asarray(value)
        compute = lambda path=None: hashlib.sha256(memoryview(np.ascontiguousarray(array)).cast("B")).hexdigest()
        backing, seen = value, set()
        while backing is not None and id(backing) not in seen:
            seen.add(id(backing))
            if (isinstance(backing, np.memmap) and backing.filename and backing.mode == "r"
                    and array.flags.c_contiguous and not array.flags.writeable):
                offset = int(backing.offset + array.ctypes.data - backing.ctypes.data)
                kind = "array:" + json.dumps([offset, list(array.shape), array.dtype.str], separators=(",", ":"))
                return self._lookup(backing.filename, kind, compute, array.nbytes)
            backing = getattr(backing, "base", None)
        self.bytes_hashed += array.nbytes
        self.misses += 1
        return compute()

    def stats(self):
        return dict(path=str(self.path), hits=self.hits, misses=self.misses, bytes_hashed=self.bytes_hashed,
                    key_fields=["realpath", "size", "mtime_ns"])
