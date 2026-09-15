"""Process-level registry for online RIT curves shared across connections.

``serve_groot_libero --concurrent`` rebuilds every mutable cache component per
connection, which is right for gates and key builders but would turn the
online estimator into one independent learner per worker. This registry is
the one object a server process creates and injects into the component
factory, so every connection that serves the same bundle attaches to the same
``OnlineRiskCurves``.

Identity: an entry is keyed by ``(yaml_id, library_sha256)``. Attaching with a
different configuration fingerprint under the same key is refused, so two
arms that merely share a file stem can never pool their learning.

Atomicity: every read that feeds a verdict (``decision_snapshot``) and every
write (``record_batch``) runs under one lock per entry, so a decision's
``q_pre`` / cuts come from one revision and concurrent commits never tear.

Persistence is file-based: a full ``OnlineRiskCurves.snapshot()`` is written
every ``snapshot_every`` committed batches, on ``flush`` (the interceptor's
task-end hook) and at interpreter exit; feedback rows go to a JSONL beside
the snapshots so the aggregate can join them against the client's per-step
rows on ``(yaml_id, task_uid, attempt, decision_idx)``.

Public interface: ``CurveRegistry``, ``RegistryKey``.
Key dependencies: ``openpi.cache.components.online_rit``.
"""

from __future__ import annotations

import atexit
import dataclasses
import json
import math
import pathlib
import threading
import time
import uuid
from typing import Any, Callable, Optional

from openpi.cache.components.online_rit import (
    ContinuationFeedback,
    DecisionSnapshot,
    OnlineRiskCurves,
    seal_state,
)


@dataclasses.dataclass(frozen=True)
class RegistryKey:
    yaml_id: str
    library_sha256: str

    @property
    def dirname(self) -> str:
        return f"{self.yaml_id}__{self.library_sha256[:12]}"


@dataclasses.dataclass
class _Entry:
    curves: OnlineRiskCurves
    fingerprint: str
    lock: threading.Lock
    log_dir: Optional[pathlib.Path]
    snapshot_every: int
    batches_since_snapshot: int = 0
    n_attached: int = 0
    update_seq: int = 0
    snapshot_seq: int = 0
    invalid_reasons: list[str] = dataclasses.field(default_factory=list)


class CurveRegistry:
    """Shared, lock-protected online curves keyed by bundle identity."""

    def __init__(self, *, state_log_root: Optional[str] = None, server_instance_id: Optional[str] = None,
                 require_persistence: bool = False) -> None:
        self._entries: dict[RegistryKey, _Entry] = {}
        self._guard = threading.Lock()
        self._root = pathlib.Path(state_log_root) if state_log_root else None
        self._require_persistence = require_persistence
        self.server_instance_id = server_instance_id or uuid.uuid4().hex[:12]
        atexit.register(self._flush_all, "atexit")

    # -- lifecycle ---------------------------------------------------------

    def attach(
        self,
        *,
        yaml_id: str,
        library_sha256: str,
        fingerprint: str,
        factory: Callable[[], OnlineRiskCurves],
        snapshot_every: int = 200,
        log_dir: Optional[str] = None,
    ) -> RegistryKey:
        """Return the key of the shared entry, creating it on first attach.

        ``log_dir`` (the judge's ``state_log_dir``) wins over the registry's
        root; either way the files land under ``<root>/<yaml__sha12>/
        <server_instance_id>/`` so two server processes serving the same
        bundle never write into one directory.
        """
        if not yaml_id or not library_sha256:
            raise ValueError("online_rit needs both yaml_id and library_sha256 to attach")
        key = RegistryKey(str(yaml_id), str(library_sha256))
        with self._guard:
            entry = self._entries.get(key)
            if entry is None:
                root = pathlib.Path(log_dir) if log_dir else self._root
                if root is None and self._require_persistence:
                    raise ValueError("online serving requires judge.state_log_dir or --online-state-dir")
                log_path = None
                if root is not None:
                    log_path = root / key.dirname / self.server_instance_id
                    log_path.mkdir(parents=True, exist_ok=True)
                entry = _Entry(
                    curves=factory(),
                    fingerprint=str(fingerprint),
                    lock=threading.Lock(),
                    log_dir=log_path,
                    snapshot_every=int(snapshot_every),
                )
                self._entries[key] = entry
                self._write_snapshot_locked(key, entry, "attach")
            elif entry.fingerprint != str(fingerprint):
                raise ValueError(
                    f"online_rit registry key {key} already holds a different configuration "
                    "fingerprint; a second arm may not pool its learning with the first"
                )
            entry.n_attached += 1
        return key

    def revision(self, key: RegistryKey) -> int:
        entry = self._entries[key]
        with entry.lock:
            return entry.curves.revision

    def curves(self, key: RegistryKey) -> OnlineRiskCurves:
        """Direct handle for tests and offline tools; callers hold no lock."""
        return self._entries[key].curves

    def log_dir(self, key: RegistryKey) -> Optional[pathlib.Path]:
        return self._entries[key].log_dir

    def mark_invalid(self, key: RegistryKey, reasons: list[str], *, decision=None) -> None:
        """Record that this stream received an observation it could not use."""
        entry = self._entries[key]
        with entry.lock:
            for r in reasons:
                entry.invalid_reasons.append(f"{decision.get('decision_idx') if decision else '?'}:{r}")
            self._write_snapshot_locked(key, entry, "invalid")

    def is_invalid(self, key: RegistryKey) -> bool:
        entry = self._entries[key]
        with entry.lock:
            return bool(entry.invalid_reasons)

    # -- decision path -----------------------------------------------------

    def decision_snapshot(
        self, key: RegistryKey, decision_id: tuple, score: float, delta: float
    ) -> DecisionSnapshot:
        entry = self._entries[key]
        with entry.lock:
            c = entry.curves
            q_pre: dict[int, Optional[float]] = {}
            kinds: dict[int, str] = {}
            cuts: dict[int, Optional[float]] = {}
            avail: dict[int, bool] = {}
            for tier in c.tier_indices:
                q, kind = c.query(tier, score)
                q_pre[tier] = q
                kinds[tier] = kind
                theta, ok = c.cut(tier, delta)
                cuts[tier] = theta if ok else None
                avail[tier] = ok
            return DecisionSnapshot(
                decision_id=tuple(decision_id),
                decision_revision=c.revision,
                score=float(score),
                q_pre=q_pre,
                cuts=cuts,
                support_kind=kinds,
                cut_available=avail,
                shadowed=c.shadowed(delta),
            )

    def record_batch(
        self,
        key: RegistryKey,
        snapshot: DecisionSnapshot,
        feedback: list[ContinuationFeedback],
        *,
        n_rejected: int = 0,
        episode: Optional[dict] = None,
    ) -> dict:
        entry = self._entries[key]
        with entry.lock:
            c = entry.curves
            c.n_rejected += int(n_rejected)
            diag = c.update_batch(snapshot.score, feedback, snapshot.decision_id)
            entry.update_seq += 1
            diag["update_seq"] = entry.update_seq
            diag["update_revision_before"] = diag.pop("revision_before")
            diag["update_revision_after"] = diag.pop("revision_after")
            diag["server_instance_id"] = self.server_instance_id
            diag["flow_invalid"] = bool(entry.invalid_reasons)
            try:
                self._append_feedback_locked(entry, snapshot, feedback, diag, episode)
            except Exception as exc:
                entry.invalid_reasons.append(f"feedback write failed: {type(exc).__name__}: {exc}")
                raise
            if diag["learned"]:
                entry.batches_since_snapshot += 1
                if entry.batches_since_snapshot >= entry.snapshot_every:
                    self._write_snapshot_locked(key, entry, "periodic")
        return diag

    # -- persistence -------------------------------------------------------

    def flush(self, key: Optional[RegistryKey] = None, reason: str = "flush") -> list[pathlib.Path]:
        """Write a full snapshot of one entry (or all) now."""
        keys = [key] if key is not None else list(self._entries)
        out = []
        for k in keys:
            entry = self._entries[k]
            with entry.lock:
                path = self._write_snapshot_locked(k, entry, reason)
            if path is not None:
                out.append(path)
        return out

    def _flush_all(self, reason: str) -> None:
        try:
            self.flush(None, reason)
        except Exception:  # pragma: no cover - atexit must never raise
            pass

    def _write_snapshot_locked(self, key: RegistryKey, entry: _Entry, reason: str) -> Optional[pathlib.Path]:
        entry.batches_since_snapshot = 0
        if entry.log_dir is None:
            return None
        entry.snapshot_seq += 1
        state = entry.curves.snapshot()
        state["yaml_id"] = key.yaml_id
        state["library_sha256"] = key.library_sha256
        state["fingerprint"] = entry.fingerprint
        state["server_instance_id"] = self.server_instance_id
        state["reason"] = reason
        state["written_at"] = time.time()
        state["snapshot_seq"] = entry.snapshot_seq
        state["update_seq"] = entry.update_seq
        state["flow_invalid"] = bool(entry.invalid_reasons)
        state["invalid_reasons"] = list(entry.invalid_reasons)
        # Seal over the final document: the reader verifies exactly these bytes.
        seal_state(state)
        path = entry.log_dir / f"state_{state['n_updates']:08d}_{reason}.json"
        content = json.dumps(state, allow_nan=False) + "\n"
        try:
            for target in (path, entry.log_dir / "state_latest.json"):
                temporary = target.with_suffix(".tmp")
                temporary.write_text(content, encoding="utf-8")
                temporary.replace(target)
        except Exception as exc:
            entry.invalid_reasons.append(f"snapshot write failed: {type(exc).__name__}: {exc}")
            raise
        return path

    def _append_feedback_locked(
        self,
        entry: _Entry,
        snapshot: DecisionSnapshot,
        feedback: list[ContinuationFeedback],
        diag: dict,
        episode: Optional[dict],
    ) -> None:
        if entry.log_dir is None:
            return
        row = {
            "decision": snapshot.to_json(),
            "episode": dict(episode or {}),
            "fb": [{"tier": fb.tier_index, "d": fb.d, "source": fb.source} for fb in feedback],
            "diag": {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in diag.items()},
            "t": time.time(),
        }
        with (entry.log_dir / "feedback.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, allow_nan=False) + "\n")

    # -- introspection -----------------------------------------------------

    def keys(self) -> list[RegistryKey]:
        with self._guard:
            return list(self._entries)

    def describe(self, key: RegistryKey) -> dict[str, Any]:
        entry = self._entries[key]
        with entry.lock:
            c = entry.curves
            return {
                "yaml_id": key.yaml_id,
                "library_sha256": key.library_sha256,
                "revision": c.revision,
                "n_updates": c.n_updates,
                "n_feedback": c.n_feedback,
                "n_attached": entry.n_attached,
                "update_enabled": c.update_enabled,
                "flow_invalid": bool(entry.invalid_reasons),
                "invalid_reasons": list(entry.invalid_reasons),
            }
