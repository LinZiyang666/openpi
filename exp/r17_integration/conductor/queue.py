"""Global ready queue; reuse generic scheduler barriers, retry and fences."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import time
from openpi.conductor.scheduler import EpisodeScheduler, StageState


class CompatibleEpisodeQueue(EpisodeScheduler):
    def __init__(self, graph, *, endpoints, dispatch_path, **kwargs):
        super().__init__(graph, **kwargs)
        self.endpoints = {e.endpoint.key: e for e in endpoints}
        if len(self.endpoints) != len(endpoints) or not endpoints:
            raise ValueError("endpoint catalog must be nonempty and unique")
        self.dispatch_path = Path(dispatch_path)
        self.dispatch_path.parent.mkdir(parents=True, exist_ok=True)
        self._assigned_endpoint = {}
        if self.dispatch_path.exists():
            raw = self.dispatch_path.read_bytes()
            for line in raw.splitlines(keepends=True):
                if not line.endswith(b"\n"):
                    # Never append after a torn frame. Preserve it and require recovery.
                    raise ValueError("torn dispatch journal; coordinator must preserve/recover before resume")
                row = json.loads(line)
                uid = row["task_uid"]
                self._dispatch_gen[uid] = max(self._dispatch_gen.get(uid, 0), row["attempt"])
                self._ep_attempts[uid] = max(self._ep_attempts.get(uid, 0), row["attempt"] - 1)

    def busy(self):
        with self._lock:
            counts = dict.fromkeys(self.endpoints, 0)
            for uid in self._dispatch_ts:
                key = self._assigned_endpoint.get(uid)
                if key in counts:
                    counts[key] += 1
            return counts

    def next_task(self, server_key=None):
        with self._lock:
            if server_key is not None and server_key not in self.endpoints:
                return None
            self._refresh_activation()
            busy = self.busy()
            for sid in sorted(self._rt):
                rt = self._rt[sid]
                if rt.state != StageState.READY:
                    continue
                for uid in rt.pending:
                    ep = self._ep_index[uid][1]
                    compatible = [e for e in self.endpoints.values() if e.accepts(ep)]
                    if not compatible:
                        continue
                    # A pulling worker can connect to any compatible endpoint;
                    # EpisodeTask carries the actual target (stock WorkerLoop semantics).
                    target = min(compatible, key=lambda e: (busy[e.endpoint.key]/e.capacity, e.endpoint.key)).endpoint
                    gen = self._dispatch_gen.get(uid, 0) + 1
                    result = dataclasses.replace(ep, server_host=target.host, server_port=target.port, attempt=gen)
                    row = {"schema": "r17.dispatch.v1", "task_uid": uid, "attempt": gen,
                           "endpoint": target.key, "benchmark_protocol": ep.extra["benchmark_protocol"]}
                    # Receipt durable BEFORE changing memory or returning to transport.
                    with self.dispatch_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(row, sort_keys=True) + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                    rt.pending.remove(uid)
                    rt.dispatched.add(uid)
                    self._dispatch_gen[uid] = gen
                    self._dispatch_ts[uid] = time.monotonic()
                    self._assigned_endpoint[uid] = target.key
                    return result
            return None
