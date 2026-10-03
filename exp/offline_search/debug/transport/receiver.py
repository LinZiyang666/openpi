# Adapted from rounds/r06/p3_profiling/stream_receiver.py; offset ACK/durable receipt semantics retained.
"""Coordinator-launched receiver. Tests bind only loopback port 0."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import socketserver
import tempfile
import threading
import time

from .validation import validate_episode
from .protocol import attempt_key, filename, identifier, pack, receive


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def sync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_path(root, *components):
    current = Path(root)
    for part in components:
        if not part or part in (".", "..") or "/" in part or "\\" in part or "\0" in part:
            raise ValueError("unsafe path component")
        current = current / part
        if current.is_symlink():
            raise ValueError("symlink in telemetry path")
    return current


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, sort_keys=True)
            f.write("\n"); f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
        sync_dir(path.parent)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)




class Store:
    def __init__(self, run_root):
        self.root = Path(run_root).resolve()
        self.run = identifier(self.root.name)
        self.arms = {identifier(r["arm"], arm=True) for r in json.loads((self.root/"arms.json").read_text())}
        self.locks, self.mutex = {}, threading.Lock()

    def locations(self, h):
        if h["run"] != self.run or identifier(h["arm"], arm=True) not in self.arms:
            raise ValueError("unknown run/arm")
        key = attempt_key(h["uid"], h["attempt"])
        if h.get("key") != key:
            raise ValueError("attempt key mismatch")
        data = safe_path(self.root, "runs", h["arm"], "debug", "client", key)
        meta = safe_path(self.root, "runs", h["arm"], "debug", "receipts", key)
        return key, data, meta

    def apply(self, h, body=b""):
        if h.get("op") == "health":
            if h.get("run") != self.run or (h.get("arm") is not None and h["arm"] not in self.arms):
                raise ValueError("wrong receiver run/arm")
            atomic_json(safe_path(self.root, "state", "osdebug_stream.health.json"), dict(run=self.run, time_ns=time.time_ns()))
            return dict(ok=True, run=self.run, protocol=1, durable_write=True)
        key, directory, metadata = self.locations(h)
        with self.mutex:
            lock = self.locks.setdefault((h["arm"], key), threading.RLock())
        with lock:
            metadata.mkdir(parents=True, exist_ok=True)
            # Spill replay is a separate coordinator process. Serialize it
            # against any receiver request whose ACK was lost, as well as
            # against a restarted/overlapping receiver process.
            with safe_path(metadata, ".lock").open("a+b") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                return self._apply(h, body, directory, metadata)

    def _apply(self, h, body, directory, metadata):
        op = h["op"]
        if op == "attempt":
            files = h["files"]
            if not isinstance(files, dict) or "events.jsonl" not in files or len(files) > 2000:
                raise ValueError("missing/invalid attempt manifest")
            totals = dict(bytes=sum(info["bytes"] for info in files.values()),
                          sha256=hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                          sha256_scope="canonical sorted file manifest: names, sizes and content hashes")
            target = safe_path(metadata, "complete.json")
            if target.exists():
                old = json.loads(target.read_text())
                if (old["uid"], old["attempt"], old["files"]) != (h["uid"], h["attempt"], files):
                    raise ValueError("conflicting attempt completion")
                if any(k in old and old[k] != v for k, v in totals.items()):
                    raise ValueError("attempt aggregate receipt mismatch")
                if any(k not in old for k in totals):
                    old.update(totals)
                    atomic_json(target, old)
                # A lost completion ACK must not repeatedly parse a 35 MB
                # trace. The coordinator re-hashes all files before cleanup.
                return dict(ok=True, completed=old["status"])
            for name, info in files.items():
                filename(name)
                p = safe_path(directory, name)
                receipt = safe_path(metadata, name+".json")
                row = json.loads(receipt.read_text())
                if any(row[k] != info[k] for k in ("bytes", "sha256")) or p.stat().st_size != info["bytes"]:
                    raise ValueError("attempt file not committed")
            try:
                trace = validate_episode(directory, h["uid"], h["attempt"])
                if not set(trace["snapshots"]) <= set(files):
                    raise ValueError("snapshot missing from completion manifest")
            except (ValueError, OSError, KeyError, TypeError, IndexError) as exc:
                # Preserve and ACK every committed byte of unaccepted/error
                # prefixes. Such a receipt can NEVER admit an accepted attempt.
                trace = dict(status="error", success=None, snapshots=[],
                             validation_error=type(exc).__name__ + ":" + str(exc))
            out = dict(run=h["run"], arm=h["arm"], uid=h["uid"], attempt=h["attempt"], key=h["key"],
                       files=files, **trace, **totals, sender_stats=h.get("sender_stats", {}))
            atomic_json(target, out)
            return dict(ok=True, completed=trace["status"])
        if op not in ("data", "finish"):
            raise ValueError("unknown operation")
        name = filename(h["file"])
        target = safe_path(directory, name)
        partial = safe_path(directory, "."+name+".osdebugpart")
        receipt = safe_path(metadata, name+".json")
        directory.mkdir(parents=True, exist_ok=True)
        if op == "data":
            offset = h["offset"]
            if type(offset) is not int or not 0 <= offset <= 16*1024**3:
                raise ValueError("invalid offset")
            path = target if target.exists() else partial
            if not path.exists() and (metadata/"complete.json").exists():
                raise ValueError("new file after attempt completion")
            with path.open("r+b" if path.exists() else "w+b") as f:
                f.seek(0, 2); size = f.tell()
                if offset > size:
                    raise ValueError("offset gap")
                overlap = min(len(body), size-offset)
                f.seek(offset)
                if f.read(overlap) != body[:overlap]:
                    raise ValueError("conflicting duplicate bytes")
                if len(body) > overlap:
                    if path == target:
                        raise ValueError("append after file completion")
                    f.seek(size); f.write(body[overlap:])
                f.flush(); os.fsync(f.fileno())
                f.seek(0, 2); size = f.tell()
            # Keep the partial-file directory entry durable before ACKing.
            if offset == 0:
                sync_dir(directory)
            return dict(ok=True, offset=size)
        info = dict(bytes=h["bytes"], sha256=h["sha256"])
        if type(info["bytes"]) is not int or info["bytes"] < 0:
            raise ValueError("invalid finish size")
        path = target if target.exists() else partial
        if not path.exists() and info["bytes"] == 0:
            path.touch()
        if path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError("finish checksum mismatch")
        if path == partial:
            os.replace(partial, target); sync_dir(directory)
        row = dict(run=h["run"], arm=h["arm"], uid=h["uid"], attempt=h["attempt"],
                   relative_path=h["key"]+"/"+name, **info)
        if receipt.exists() and json.loads(receipt.read_text()) != row:
            raise ValueError("conflicting file receipt")
        atomic_json(receipt, row)
        return dict(ok=True, **info)


class Handler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(30)
        try:
            while True:
                event = receive(self.request.recv)
                if event is None:
                    return
                h, body = event
                if not secrets.compare_digest(str(h.pop("token", "")), self.server.token):
                    raise ValueError("invalid receiver token")
                if self.server.delay:
                    time.sleep(self.server.delay)
                response = self.server.store.apply(h, body)
                # Test-only fault hook: ACK loss after durable data application.
                if self.server.drop_once and h["op"] == "data":
                    self.server.drop_once = False
                    return
                self.request.sendall(pack(response))
        except (OSError, EOFError):
            return
        except Exception as exc:
            try:
                self.request.sendall(pack(dict(ok=False, error=type(exc).__name__+":"+str(exc))))
            except OSError:
                pass


class Receiver(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, address, store, token, delay=0., drop_once=False):
        self.store, self.token = store, token
        self.delay, self.drop_once = delay, drop_once
        super().__init__(address, Handler)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--bind", default="0.0.0.0")
    ap.add_argument("--token-file", type=Path)
    ap.add_argument("--ready-file", type=Path)
    a = ap.parse_args()
    store = Store(a.run_root)
    token_file = a.token_file or a.run_root/"state/osdebug_stream.token"
    token_file.parent.mkdir(parents=True, exist_ok=True)
    if not token_file.exists():
        fd = os.open(str(token_file), os.O_WRONLY|os.O_CREAT|os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            f.write(secrets.token_hex(32)+"\n"); f.flush(); os.fsync(f.fileno())
    token = token_file.read_text().strip()
    if len(token) != 64 or any(c not in "0123456789abcdef" for c in token):
        raise ValueError("invalid receiver token file")
    with Receiver((a.bind, a.port), store, token) as server:
        ready = dict(run=store.run, bind=a.bind, port=server.server_address[1], pid=os.getpid())
        if a.ready_file:
            atomic_json(a.ready_file, ready)
        print(json.dumps(ready), flush=True)
        server.serve_forever()


if __name__ == "__main__":
    main()
