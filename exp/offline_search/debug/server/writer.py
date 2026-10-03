"""Bounded, blocking background serialization with explicit capture failures."""
import atexit
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np

from ..schema import DECISION_BLOCK_SIZE, write_npz_block


def atomic_json(path, value):
    path = Path(path)
    part = path.with_name(path.name + ".part")
    with part.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(part), str(path))


def pack_arrays(items, member="arrays"):
    """Missing numeric fields get NaNs; availability is recorded in JSONL."""
    ids = [str(x["record"]["decision_id"]) for x in items]
    # Valid envelopes fit U80. Retain a malformed received ID losslessly too;
    # its error record must still join to the client's original request.
    arrays = {"decision_id": np.asarray(ids, dtype="<U{}".format(max(80, max(map(len, ids), default=0))))}
    keys = set().union(*(x[member] for x in items))
    for key in sorted(keys):
        present = [x[member][key] for x in items if key in x[member]]
        template = np.asarray(present[0])
        if any(np.asarray(a).shape != template.shape or np.asarray(a).dtype != template.dtype for a in present):
            raise ValueError("incompatible array shape or dtype: " + key)
        if template.dtype.hasobject:
            raise ValueError("object array forbidden: " + key)
        dtype = template.dtype
        missing = len(present) != len(items)
        fill = (np.full(template.shape, np.nan, dtype=dtype) if dtype.kind in "fc"
                else np.zeros(template.shape, dtype=dtype)) if missing else None
        arrays[key] = np.stack([x[member].get(key, fill) for x in items]).astype(dtype, copy=False)
        if missing:
            arrays[key + "_available"] = np.asarray([key in x[member] for x in items], dtype=bool)
    if member == "arrays":
        prompts = list(dict.fromkeys(x.get("prompt", "") for x in items))
        # Do not silently truncate an unusual prompt; record its actual width.
        arrays["prompts"] = np.asarray(prompts, dtype="<U{}".format(max(512, max(map(len, prompts), default=0))))
        arrays["prompt_idx"] = np.asarray([prompts.index(x.get("prompt", "")) for x in items], dtype="<i4")
    return arrays


def compatible(items, item):
    for member in ("arrays", "rawkeys"):
        for old in items:
            for key in old[member].keys() & item[member].keys():
                a, b = old[member][key], item[member][key]
                if a.shape != b.shape or a.dtype != b.dtype:
                    return False
    return True


class BlockWriter:
    """One process owns a directory. Queue fullness waits, never drops.

    An item larger than the queue limit is admitted only into an empty queue,
    counted in oversized_items, and immediately serialized. Pending blocks are
    independently bounded by 16 decisions. I/O failures retain explicit error
    rows where possible and make writer_stats/echo report incomplete capture.
    """
    def __init__(self, directory, queue_bytes=512 * 1024 ** 2, flush_seconds=0.25):
        self.directory = Path(directory)
        self.pid = os.getpid()
        self.decisions_path = self.directory / "decisions_{}.jsonl".format(self.pid)
        self.stats_path = self.directory / "writer_stats_{}.json".format(self.pid)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "blocks").mkdir(exist_ok=True)
        (self.directory / "rawkeys").mkdir(exist_ok=True)
        self.limit = int(queue_bytes)
        if self.limit <= 0 or flush_seconds <= 0:
            raise ValueError("queue_bytes and flush_seconds must be positive")
        self.flush_seconds = float(flush_seconds)
        self._cv = threading.Condition()
        self._queue = deque()
        self._bytes = 0
        self._closing = False
        self._closed = False
        self._active = False
        self._flush = False
        self._submit_lock = threading.Lock()
        self._error = None
        self._blk = 0
        existing = list((self.directory / "blocks").glob("d_{}_*.npz".format(self.pid)))
        if existing:
            self._blk = max(int(p.stem.rsplit("_", 1)[1]) for p in existing) + 1
        self.stats = dict(pid=self.pid, queue_high_water_bytes=0, queue_bytes=0, queue_limit_bytes=self.limit,
                          accepted=0, written=0, bytes=0, raw_bytes=0, blocks=0,
                          serialization_ms=0., queue_wait_ms=0., errors=[], oversized_items=0, drained=False)
        self._thread = threading.Thread(target=self._run, name="osdebug-writer", daemon=True)
        self._persist_stats()
        self._thread.start()
        atexit.register(self.close)

    @property
    def error(self):
        return self._error

    def put(self, record, arrays, rawkeys=None, prompt=""):
        # The observer owns these detached buffers; the queue does not reference
        # serving state. JSON serialization happens before byte accounting.
        item = dict(record=dict(record), arrays=arrays, rawkeys=rawkeys or {}, prompt=str(prompt))
        size = sum(a.nbytes for a in arrays.values()) + sum(a.nbytes for a in item["rawkeys"].values())
        size += len(json.dumps(record, allow_nan=False).encode("utf-8")) + len(str(prompt).encode("utf-8"))
        started = time.perf_counter()
        with self._submit_lock, self._cv:
            while not self._closing and self._bytes and self._bytes + size > self.limit:
                self._cv.wait()
            if self._closing:
                raise RuntimeError("capture writer is closed")
            wait_ms = (time.perf_counter() - started) * 1000
            item["record"]["queue_wait_ms"] = wait_ms
            self._queue.append((item, size))
            self._bytes += size
            self.stats["accepted"] += 1
            self.stats["raw_bytes"] += size
            self.stats["queue_wait_ms"] += wait_ms
            self.stats["oversized_items"] += int(size > self.limit)
            self.stats["queue_high_water_bytes"] = max(self.stats["queue_high_water_bytes"], self._bytes)
            self._cv.notify_all()
        return wait_ms

    def flush(self):
        with self._cv:
            self._flush = True
            self._cv.notify_all()
            while (self._queue or self._active or self._flush) and not self._closed:
                self._cv.wait()

    def close(self):
        with self._submit_lock, self._cv:
            self._closing = True
            self._cv.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join()

    def _persist_stats(self):
        with self._cv:
            self.stats["queue_bytes"] = self._bytes
            snapshot = dict(self.stats)
        atomic_json(self.stats_path, snapshot)

    def _publish(self, items, stream):
        start = time.perf_counter()
        filename = "d_{}_{:06d}.npz".format(self.pid, self._blk)
        for i, item in enumerate(items):
            item["record"].update(blk=filename, blk_i=i)
        failure = None
        try:
            path = self.directory / "blocks" / filename
            write_npz_block(path, pack_arrays(items))
            self.stats["bytes"] += path.stat().st_size
            sampled = [x for x in items if x["rawkeys"]]
            if sampled:
                rawname = "k_{}_{:06d}.npz".format(self.pid, self._blk)
                rawpath = self.directory / "rawkeys" / rawname
                write_npz_block(rawpath, pack_arrays(sampled, "rawkeys"))
                self.stats["bytes"] += rawpath.stat().st_size
                for i, item in enumerate(sampled):
                    item["record"].update(rawkeys_blk=rawname, rawkeys_blk_i=i)
        except Exception as exc:
            failure = "{}: {}".format(type(exc).__name__, exc)
            self._error = failure
            self.stats["errors"].append(dict(blk=filename, error=failure,
                                            decision_ids=[x["record"]["decision_id"] for x in items]))
            for item in items:
                item["record"].update(status="error", writer_error=failure)
        for item in items:
            stream.write(json.dumps(item["record"], allow_nan=False, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
        self._blk += 1
        self.stats["written"] += len(items)
        self.stats["blocks"] += int(failure is None)
        self.stats["serialization_ms"] += (time.perf_counter() - start) * 1000
        self._persist_stats()

    def _run(self):
        pending = []
        try:
            with self.decisions_path.open("a", encoding="utf-8") as stream:
                while True:
                    with self._cv:
                        if not self._queue and not self._closing and not self._flush:
                            self._cv.wait(self.flush_seconds)
                        if self._queue:
                            item, size = self._queue.popleft()
                            self._bytes -= size
                            self._active = True
                            self._cv.notify_all()
                        else:
                            item = None
                        stop = self._closing and not self._queue and item is None
                        flush = self._flush and not self._queue
                    if item is not None:
                        if pending and not compatible(pending, item):
                            self._publish(pending, stream)
                            pending = []
                        pending.append(item)
                    if pending and (len(pending) >= DECISION_BLOCK_SIZE or item is None or flush):
                        self._publish(pending, stream)
                        pending = []
                    with self._cv:
                        self._active = bool(pending)
                        if flush and not pending:
                            self._flush = False
                        self._cv.notify_all()
                    if stop:
                        break
        except Exception as exc:
            self._error = "{}: {}".format(type(exc).__name__, exc)
            self.stats["errors"].append(dict(error=self._error, pending=[x["record"]["decision_id"] for x in pending],
                queued=[x[0]["record"]["decision_id"] for x in self._queue]))
        finally:
            with self._cv:
                self._closing = self._closed = True
                self.stats["drained"] = not self._queue and not pending and self._error is None
                self._cv.notify_all()
            try:
                self._persist_stats()
            except Exception:
                # The caller still receives the sticky error; do not raise into
                # live inference if the filesystem itself is unavailable.
                pass
