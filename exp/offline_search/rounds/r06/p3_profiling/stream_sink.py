"""Opt-in background byte transport with bounded RAM and a durable spill fallback.

No NumPy/simulator imports. Network and durable ACK waits never occur in write().
When spilling, write() necessarily performs local disk IO; it never drops data.
Graceful close drains at episode boundaries. SIGKILL/power loss of the client can
lose its unacknowledged RAM queue; such an attempt has no completion certificate.
"""
import atexit
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
import weakref

from .stream_protocol import MAX_BODY, attempt_key, endpoint, filename, identifier, pack, receive

ACTIVE = weakref.WeakSet()


def finish_active():
    for sink in list(ACTIVE):
        sink.close()


atexit.register(finish_active)


class StreamSink:
    def __init__(self, address, token, run, arm, uid, attempt, directory,
                 queue_bytes=8*1024*1024, fail_seconds=5., timeout=1., close_seconds=10.):
        endpoint(address)
        identifier(run); identifier(arm, arm=True)
        key = attempt_key(uid, attempt)
        if not token:
            raise ValueError("stream token missing")
        if type(queue_bytes) is not int or queue_bytes < 1024 or any(
                not math.isfinite(x) or x <= 0 for x in (fail_seconds, timeout, close_seconds)):
            raise ValueError("invalid stream limits")
        self.address, self.token = address, token
        self.identity = dict(run=run, arm=arm, uid=uid, attempt=attempt, key=key)
        self.directory = Path(directory) / key
        self.limit, self.fail_seconds, self.timeout, self.close_seconds = queue_bytes, fail_seconds, timeout, close_seconds
        self.cv = threading.Condition()
        self.queue, self.inflight, self.queued_bytes = deque(), None, 0
        self.inflight_cost = 0
        self.closing = self.closed = self.spilled = False
        self.spool = self.socket = None
        self.spool_hash, self.spool_bytes = hashlib.sha256(), 0
        self.files = {}
        self.controls_hash, self.controls_bytes = hashlib.sha256(), 0
        self.stats = dict(queue_high_water=0, frames_acked=0, reconnects=0, spill=False)
        self.thread = threading.Thread(target=self._sender, name="p3-stream", daemon=True)
        ACTIVE.add(self)
        self.thread.start()

    @classmethod
    def from_environment(cls, directory, uid, attempt):
        directory = Path(directory)
        return cls(os.environ["P3_STREAM"], os.environ["P3_STREAM_TOKEN"],
                   os.environ.get("P3_STREAM_RUN", directory.parent.parent.name),
                   os.environ.get("P3_STREAM_ARM", directory.parent.name), uid, attempt, directory,
                   queue_bytes=int(os.environ.get("P3_STREAM_QUEUE_BYTES", 8*1024*1024)),
                   fail_seconds=float(os.environ.get("P3_STREAM_FAIL_S", 5)),
                   timeout=float(os.environ.get("P3_STREAM_TIMEOUT_S", 1)),
                   close_seconds=float(os.environ.get("P3_STREAM_CLOSE_S", 10)))

    def _spool_frame(self, event):
        raw = pack(*event)
        remaining = memoryview(raw)
        while remaining:
            count = self.spool.write(remaining)
            if not count:
                raise OSError("short spill write")
            remaining = remaining[count:]
        self.spool_hash.update(raw)
        self.spool_bytes += len(raw)

    def _spill_locked(self, reason):
        if self.spilled:
            return
        self.directory.mkdir(parents=True, exist_ok=True)
        self.spool = (self.directory / "stream.p3spill.part").open("xb", buffering=0)
        self.spilled = self.stats["spill"] = True
        self.stats["spill_reason"] = reason
        if self.inflight is not None:
            self._spool_frame(self.inflight)
        for event, cost in self.queue:
            self._spool_frame(event)
        self.queue.clear()
        self.queued_bytes = 0
        self.cv.notify_all()

    def _enqueue(self, header, body=b""):
        event = ({**self.identity, **header}, body)
        cost = len(body) + len(json.dumps(event[0])) + 128
        with self.cv:
            if self.closed:
                raise ValueError("write after stream close")
            occupied = self.queued_bytes + self.inflight_cost
            if not self.spilled and occupied + cost > self.limit:
                self._spill_locked("bounded_queue_full")
            if self.spilled:
                self._spool_frame(event)
            else:
                self.queue.append((event, cost))
                self.queued_bytes += cost
                self.stats["queue_high_water"] = max(self.stats["queue_high_water"], occupied+cost)
                self.cv.notify()

    def write(self, text):
        data = text.encode("utf-8")
        offset = self.controls_bytes
        self.controls_hash.update(data)
        self.controls_bytes += len(data)
        for start in range(0, len(data), MAX_BODY):
            self._enqueue(dict(op="data", file="controls.jsonl", offset=offset+start), data[start:start+MAX_BODY])
        return len(text)

    def flush(self):
        # emit() calls flush per line; normal mode has no synchronous IO here.
        pass

    def write_file(self, name, data):
        filename(name)
        if name == "controls.jsonl" or name in self.files:
            raise ValueError("duplicate/invalid snapshot")
        info = dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
        for start in range(0, len(data), MAX_BODY):
            self._enqueue(dict(op="data", file=name, offset=start), data[start:start+MAX_BODY])
        self._enqueue(dict(op="finish", file=name, **info))
        self.files[name] = info

    def _disconnect(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None

    def _send(self, event):
        if self.socket is None:
            self.socket = socket.create_connection(endpoint(self.address), timeout=self.timeout)
            self.socket.settimeout(self.timeout)
            self.stats["reconnects"] += 1
        header, body = event
        # File hashing / terminal trace validation may take longer when many
        # workers finish together. This wait is confined to the sender thread.
        self.socket.settimeout(self.timeout if header["op"] == "data" else max(self.timeout, 10.))
        self.socket.sendall(pack({**header, "token": self.token}, body))
        response = receive(self.socket.recv)
        if response is None or not response[0].get("ok"):
            raise RuntimeError("receiver rejected frame: " + str(response))
        if header["op"] == "data" and response[0].get("offset", -1) < header["offset"] + len(body):
            raise RuntimeError("receiver acknowledged a short offset")

    def _sender(self):
        failed_at = None
        try:
            while True:
                with self.cv:
                    while not self.queue and not self.closing and not self.spilled:
                        self.cv.wait()
                    if self.spilled or (not self.queue and self.closing):
                        return
                    event, cost = self.queue.popleft()
                    self.queued_bytes -= cost
                    self.inflight = event
                    self.inflight_cost = cost
                while True:
                    with self.cv:
                        if self.spilled:
                            return
                    try:
                        attempt_started = time.monotonic()
                        self._send(event)
                        failed_at = None
                        with self.cv:
                            self.stats["frames_acked"] += 1
                            self.inflight = None
                            self.inflight_cost = 0
                        break
                    except (OSError, EOFError, ValueError, RuntimeError) as exc:
                        self._disconnect()
                        failed_at = failed_at or attempt_started
                        if time.monotonic() - failed_at >= self.fail_seconds:
                            with self.cv:
                                self._spill_locked(type(exc).__name__ + ":" + str(exc)[:160])
                            return
                        with self.cv:
                            self.cv.wait(min(.1, self.fail_seconds))
        except Exception as exc:
            with self.cv:
                self._spill_locked("sender_error:" + repr(exc))
        finally:
            self._disconnect()

    def close(self):
        if self.closed:
            return
        info = dict(bytes=self.controls_bytes, sha256=self.controls_hash.hexdigest())
        self.files["controls.jsonl"] = info
        self._enqueue(dict(op="finish", file="controls.jsonl", **info))
        self._enqueue(dict(op="attempt", files=self.files.copy(), sender_stats=self.stats.copy()))
        with self.cv:
            self.closing = True
            self.cv.notify_all()
        self.thread.join(self.close_seconds)
        with self.cv:
            if self.thread.is_alive() and not self.spilled:
                self._spill_locked("episode_close_deadline")
        if self.thread.is_alive():
            # No receiver operation may still be in flight when the coordinator
            # begins replaying this spill. This wait is outside the control loop.
            sock = self.socket
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
            self.thread.join(self.timeout+1)
            if self.thread.is_alive():
                raise RuntimeError("stream sender did not stop; attempt must not be accepted")
        with self.cv:
            if self.spilled:
                self.spool.flush()
                os.fsync(self.spool.fileno())
                self.spool.close()
                final = self.directory / "stream.p3spill"
                os.replace(self.directory / "stream.p3spill.part", final)
                meta = dict(**self.identity, bytes=self.spool_bytes, sha256=self.spool_hash.hexdigest(), stats=self.stats)
                tmp = self.directory / "stream.p3spill.json.tmp"
                with tmp.open("x") as f:
                    json.dump(meta, f); f.flush(); os.fsync(f.fileno())
                os.replace(tmp, self.directory / "stream.p3spill.json")
            self.closed = True
        ACTIVE.discard(self)
