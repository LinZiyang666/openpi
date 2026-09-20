"""Policy-agnostic per-decision recorder of the step-vs-warm-start line (plan §3.1).

One ``DiagRecorder`` per served arm process owns the rows file; each client connection's served
policy (``Pi05DiagInterceptor`` / ``GrootDiagPolicy``) holds its own ``DiagSession`` from
``recorder.session()``, so interleaved connections never share episode state. A session is fed
the episode identity from ``episode_start`` and, after every executed decision, the stage-2
handle plus three callables supplied by the policy adapter:

* ``sample(noise, k)``  -> ``[H, D]`` chunk: the flow-matching loop from ``noise`` with ``k`` Euler
  steps (``k == K`` is the full loop);
* ``resume(x_t, t)``    -> ``[H, D]`` chunk: the loop resumed from a cached snapshot ``x_t`` at ``t``;
* ``top1()``            -> ``(score, entry_id, intermediates)`` of the read-only retrieval winner or
  ``(None, None, None)``.

Three invariants (inherited from ``openpi.cache.shadow_teacher``): the executed action is never
touched (the recorder only observes it); the production RNG is never touched (every shadow noise
comes from a private CPU generator seeded by a content digest); a shadow failure becomes an
``error`` row and never an exception in the request path.

Evidence per episode: one JSONL metadata row per decision (``status=ok|error``) plus a
``finalize`` row, and one ``.npz`` of float32 arrays (``a_exec``, ``a_full``, ``a_k_<k>``,
``a_warm_<t>``) bound to the rows by ``arrays`` (relative path) and ``arrays_sha256``. Arm modes
without shadow sampling (plain / full / warm) write the same rows without arrays; their evidence
is the per-decision ``hit_type / start_t / executed_steps / n_stage3_calls``.

Dense decisions (``N_full = n_primary + n_dense_extra``) are chosen online by
``sha256("20260919|env_id|task|init_idx|decision_idx|dense") mod 16 == 0``; the rule contains no
arm / attempt / launch so a retried episode selects the same decisions.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import pathlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

logger = logging.getLogger("exp.step_diag.recorder")

DENSE_SALT = "20260919"
DENSE_MOD = 16
SCHEMA_VERSION = "step_diag_rows_v1"
# Keys the worker-side client proxy stamps into episode_start (exp.step_diag.worker_entry.STAMP_KEYS).
CLIENT_STAMP_KEYS = ("launch_id", "arm_id", "experiment_id", "config_sha")


def stable_digest_int(*parts: Any) -> int:
    """Deterministic 63-bit integer from the ``|``-joined string of ``parts`` (never ``hash()``)."""
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFF_FFFF_FFFF_FFFF


def is_dense_decision(env_id: str, task: str, init_idx: Any, decision_idx: int) -> bool:
    """The frozen online dense rule (plan §2 Q-A item 4): no arm / attempt / launch in the key."""
    digest = hashlib.sha256(
        f"{DENSE_SALT}|{env_id}|{task}|{init_idx}|{decision_idx}|dense".encode("utf-8")
    ).digest()
    return int.from_bytes(digest, "big") % DENSE_MOD == 0


def noise_seed(experiment_id: str, env_id: str, task: str, env_seed: Any, attempt: int, decision_idx: int,
               sample: int) -> int:
    """Private-noise identity of one ``(decision, sample)``; shared by every k of that sample."""
    return stable_digest_int(experiment_id, env_id, task, env_seed, attempt, decision_idx, sample)


def make_noise(seed: int, shape: Sequence[int]) -> torch.Tensor:
    """Standard-normal float32 CPU noise from a fresh generator seeded with ``seed``."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    return torch.randn(tuple(shape), generator=gen, dtype=torch.float32)


def _to_np(chunk: Any) -> np.ndarray:
    t = torch.as_tensor(chunk).detach()
    if t.is_inference():
        t = t.clone()
    arr = t.to("cpu", torch.float32).numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"chunk must be [H, D] (or [1, H, D]), got {arr.shape}")
    return np.ascontiguousarray(arr, dtype=np.float32)


@dataclass
class EpisodeIdentity:
    """What ``episode_start`` told us; everything the analysis joins on lives here."""

    benchmark: str
    task: str
    episode_id: int
    task_uid: str
    attempt: int
    task_id: Optional[int]
    init_idx: Optional[int]
    env_seed: Optional[int]
    lane: Optional[str] = None
    pin_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_episode_start(cls, *, experiment: str, task: str, episode_id: int,
                           extra_metadata: Optional[dict]) -> "EpisodeIdentity":
        meta = dict(extra_metadata or {})
        init_idx = meta.get("orig_init_state_idx", meta.get("init_state_idx"))
        return cls(
            benchmark=str(experiment),
            task=str(task),
            episode_id=int(episode_id),
            task_uid=str(meta.get("task_uid", f"{experiment}:{task}:{episode_id}")),
            attempt=int(meta.get("attempt", 1)),
            task_id=None if meta.get("task_id") is None else int(meta["task_id"]),
            init_idx=None if init_idx is None else int(init_idx),
            env_seed=None if meta.get("seed") is None else int(meta["seed"]),
            lane=meta.get("lane"),
            pin_id=meta.get("pin_id") or meta.get("realized_pin_id"),
            extra={k: v for k, v in meta.items() if k not in {"pinned_objects"}},
        )


@dataclass
class DiagSpec:
    """Frozen sampling contract of one served arm (written into every row)."""

    experiment_id: str
    env_id: str
    arm_id: str
    mode: str  # shadow | plain | full | warm
    k_full: int
    k_set: tuple = ()
    warm_ts: tuple = ()
    n_primary: int = 4
    n_dense_extra: int = 28
    action_shape: tuple = ()  # (H, D)
    exec_steps: Optional[int] = None  # plain / full arms: the pinned executed step count
    config_sha: str = ""

    def to_json(self) -> dict:
        return {"experiment_id": self.experiment_id, "env_id": self.env_id, "arm_id": self.arm_id,
                "mode": self.mode, "k_full": self.k_full, "k_set": list(self.k_set), "warm_ts": list(self.warm_ts),
                "n_primary": self.n_primary, "n_dense_extra": self.n_dense_extra,
                "action_shape": list(self.action_shape), "exec_steps": self.exec_steps, "config_sha": self.config_sha}


class DiagRecorder:
    """Process-wide sink of one served arm: the rows file, the arrays directory and the counters.

    Episode state is per *session* (``session()``): a served policy owns one session per client
    connection, so two connections interleaving their episodes on one server never share a
    decision counter or an arrays buffer. The recorder's own ``begin_episode / record /
    finalize_episode`` delegate to a default session for single-connection servers and tests.
    """

    def __init__(self, spec: DiagSpec, out_dir: str | os.PathLike, *, launch_id: str = "") -> None:
        self._spec = spec
        self._dir = pathlib.Path(out_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._launch_id = launch_id or time.strftime("%Y%m%dT%H%M%S")
        self._rows_path = self._dir / f"rows_{spec.arm_id}_{self._launch_id}.jsonl"
        self._io_lock = threading.Lock()
        self._n_ok = 0
        self._n_err = 0
        self._n_sessions = 0
        self._sessions: List["DiagSession"] = []
        self._default = self.session()

    # -- lifecycle -------------------------------------------------------

    @property
    def spec(self) -> DiagSpec:
        return self._spec

    @property
    def rows_path(self) -> pathlib.Path:
        return self._rows_path

    @property
    def launch_id(self) -> str:
        return self._launch_id

    def session(self) -> "DiagSession":
        """A fresh per-connection episode state sharing this recorder's sink."""
        with self._io_lock:
            self._n_sessions += 1
            sess = DiagSession(self, self._n_sessions)
            self._sessions.append(sess)
        return sess

    def begin_episode(self, identity: EpisodeIdentity) -> None:
        self._default.begin_episode(identity)

    def finalize_episode(self, success: Optional[bool], *, terminal: bool = True) -> None:
        self._default.finalize_episode(success, terminal=terminal)

    def record(self, **kwargs: Any) -> Optional[dict]:
        return self._default.record(**kwargs)

    def close(self) -> None:
        """Close every session (an open episode becomes a non-terminal finalize row)."""
        with self._io_lock:
            sessions = list(self._sessions)
        for sess in sessions:
            sess.close()

    # -- sink ------------------------------------------------------------

    def _write(self, rows: List[dict], arrays: Dict[str, np.ndarray], arrays_rel: Optional[str]) -> Optional[str]:
        """Persist one episode's arrays (sha256 returned) and append its rows; one lock, one append."""
        arrays_sha = None
        if arrays and arrays_rel:
            path = self._dir / arrays_rel
            path.parent.mkdir(parents=True, exist_ok=True)
            buf = io.BytesIO()
            np.savez(buf, **arrays)
            data = buf.getvalue()
            path.write_bytes(data)
            arrays_sha = hashlib.sha256(data).hexdigest()
        with self._io_lock:
            with self._rows_path.open("a", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, default=_json_default) + "\n")
        return arrays_sha

    def _count(self, ok: bool) -> None:
        with self._io_lock:
            if ok:
                self._n_ok += 1
            else:
                self._n_err += 1

    def _base_row(self, ep: EpisodeIdentity, idx: int, session_id: int) -> dict:
        s = self._spec
        return {
            "schema": SCHEMA_VERSION, "experiment_id": s.experiment_id, "launch_id": self._launch_id,
            "session_id": session_id,
            "arm_id": s.arm_id, "env_id": s.env_id, "mode": s.mode, "config_sha": s.config_sha,
            "benchmark": ep.benchmark, "task": ep.task, "task_id": ep.task_id, "task_uid": ep.task_uid,
            "attempt": ep.attempt, "episode_id": ep.episode_id, "init_idx": ep.init_idx, "env_seed": ep.env_seed,
            "lane": ep.lane, "pin_id": ep.pin_id, "decision_idx": idx,
            "layout": ep.extra.get("layout"), "style": ep.extra.get("style"),
            "init_pool_sha256": ep.extra.get("init_pool_sha256"),
            "k_full": s.k_full, "k_set": list(s.k_set), "warm_ts": list(s.warm_ts), "n_primary": s.n_primary,
        }

    @staticmethod
    def _client_stamp(ep: EpisodeIdentity) -> dict:
        """What the worker's client proxy stamped into episode_start (driver launch id, arm, config)."""
        return {k: ep.extra.get(k) for k in CLIENT_STAMP_KEYS}

    def _stamp_mismatch(self, ep: EpisodeIdentity) -> List[str]:
        """Stamp keys the client sent that disagree with this served arm (empty = consistent)."""
        s = self._spec
        want = {"arm_id": s.arm_id, "experiment_id": s.experiment_id, "config_sha": s.config_sha}
        return [k for k, v in want.items() if ep.extra.get(k) not in (None, "") and str(ep.extra[k]) != str(v)]

    @property
    def counts(self) -> dict:
        return {"ok": self._n_ok, "error": self._n_err}


class DiagSession:
    """Episode state of one client connection: identity, decision counter, arrays, pending rows."""

    def __init__(self, recorder: DiagRecorder, session_id: int) -> None:
        self._rec = recorder
        self._id = int(session_id)
        self._lock = threading.Lock()
        self._episode: Optional[EpisodeIdentity] = None
        self._decision_idx = 0
        self._arrays: Dict[str, np.ndarray] = {}
        self._pending_rows: List[dict] = []

    @property
    def spec(self) -> DiagSpec:
        return self._rec.spec

    @property
    def session_id(self) -> int:
        return self._id

    # -- lifecycle -------------------------------------------------------

    def begin_episode(self, identity: EpisodeIdentity) -> None:
        with self._lock:
            if self._episode is not None:
                # An unbalanced start: close the previous episode as non-terminal so it is
                # visible to the admission gate instead of silently merged.
                self._finalize_locked(success=None, terminal=False, reason="unbalanced_episode_start")
            self._episode = identity
            self._decision_idx = 0
            self._arrays = {}
            self._pending_rows = []

    def finalize_episode(self, success: Optional[bool], *, terminal: bool = True) -> None:
        with self._lock:
            self._finalize_locked(success=success, terminal=terminal, reason=None)

    def close(self) -> None:
        with self._lock:
            self._finalize_locked(success=None, terminal=False, reason="recorder_closed")

    def _finalize_locked(self, *, success: Optional[bool], terminal: bool, reason: Optional[str]) -> None:
        ep = self._episode
        if ep is None:
            return
        rec = self._rec
        arrays_rel = None
        if self._arrays:
            arrays_rel = f"arrays_{rec.spec.arm_id}_{rec.launch_id}/{_safe(ep.task_uid)}_a{ep.attempt:02d}.npz"
        rows = list(self._pending_rows)
        fin = rec._base_row(ep, self._decision_idx, self._id) | {
            "status": "finalize", "terminal": bool(terminal), "outcome": success,
            "n_decisions": self._decision_idx, "reason": reason,
            "client_stamp": rec._client_stamp(ep), "stamp_mismatch": rec._stamp_mismatch(ep),
        }
        rows.append(fin)
        arrays_sha = rec._write([], self._arrays, arrays_rel) if self._arrays else None
        for row in rows:
            row["arrays"] = arrays_rel
            row["arrays_sha256"] = arrays_sha
        rec._write(rows, {}, None)
        self._episode = None
        self._decision_idx = 0
        self._arrays = {}
        self._pending_rows = []

    # -- per decision ----------------------------------------------------

    def record(
        self,
        *,
        a_exec: Any,
        executed_steps: Optional[int],
        n_stage3_calls: Optional[int],
        hit_type: Optional[str],
        start_t: Optional[float],
        schedule_id: Optional[str],
        sample: Optional[Callable[[torch.Tensor, int], Any]] = None,
        resume: Optional[Callable[[torch.Tensor, float], Any]] = None,
        top1: Optional[Callable[[], tuple]] = None,
        extra: Optional[dict] = None,
    ) -> Optional[dict]:
        """Record one decision; shadow-sample when ``sample`` is given. Never raises."""
        with self._lock:
            ep = self._episode
            if ep is None:
                logger.warning("record() outside an episode; dropped")
                return None
            idx = self._decision_idx
            self._decision_idx += 1
            row = self._rec._base_row(ep, idx, self._id) | {
                "status": "ok", "hit_type": hit_type, "start_t": start_t, "schedule_id": schedule_id,
                "executed_steps": executed_steps, "n_stage3_calls": n_stage3_calls,
                "dense": False, "n_full": 0, "noise_ids": [], "shadow_nfe": 0,
                "top1_score": None, "top1_entry_id": None,
                "warm_status": {f"{t:.4f}": "no_candidate" for t in self.spec.warm_ts},
            }
            if extra:
                row.update(extra)
            t0 = time.perf_counter()
            try:
                self._arrays[f"a_exec_{idx:04d}"] = _to_np(a_exec)
                if sample is not None:
                    self._shadow_locked(row, ep, idx, sample, resume, top1)
                row["wall_ms"] = (time.perf_counter() - t0) * 1000.0
                self._rec._count(True)
            except Exception as exc:  # noqa: BLE001 - the episode outranks the label
                row["status"] = "error"
                row["error_reason"] = f"{type(exc).__name__}: {exc}"[:500]
                row["wall_ms"] = (time.perf_counter() - t0) * 1000.0
                self._rec._count(False)
                logger.warning("shadow failed at decision %d: %r", idx, exc)
            self._pending_rows.append(row)
            return row

    def _shadow_locked(self, row: dict, ep: EpisodeIdentity, idx: int, sample, resume, top1) -> None:
        spec = self.spec
        dense = is_dense_decision(spec.env_id, ep.task, ep.init_idx, idx)
        n_full = spec.n_primary + (spec.n_dense_extra if dense else 0)
        row["dense"] = dense
        row["n_full"] = n_full
        shape = tuple(spec.action_shape)
        got = tuple(self._arrays[f"a_exec_{idx:04d}"].shape)
        if got != shape:
            # The env table fixes [H, D]; a live model of another shape is an error row, never a
            # silent re-spec (the noise contract would no longer match the recorded arrays).
            raise ValueError(f"executed chunk shape {got} != env action_shape {shape}")
        noise_ids = []
        full = []
        per_k: Dict[int, list] = {int(k): [] for k in spec.k_set}
        nfe = 0
        for n in range(n_full):
            seed = noise_seed(spec.experiment_id, spec.env_id, ep.task,
                              (ep.env_seed, ep.init_idx, ep.extra.get("init_pool_sha256")), ep.attempt, idx, n)
            noise_ids.append(seed)
            z = make_noise(seed, shape)
            full.append(_to_np(sample(z, spec.k_full)))
            nfe += spec.k_full
            if n < spec.n_primary:
                for k in per_k:
                    per_k[k].append(_to_np(sample(z, k)))
                    nfe += k
        self._arrays[f"a_full_{idx:04d}"] = np.stack(full, axis=0)
        for k, chunks in per_k.items():
            self._arrays[f"a_k{k}_{idx:04d}"] = np.stack(chunks, axis=0)
        row["noise_ids"] = noise_ids
        # warm start from the read-only retrieval winner
        if top1 is not None and resume is not None and spec.warm_ts:
            score, entry_id, intermediates = top1()
            row["top1_score"] = None if score is None else float(score)
            row["top1_entry_id"] = entry_id
            for t in spec.warm_ts:
                key = f"{float(t):.4f}"
                if intermediates is None:
                    row["warm_status"][key] = "no_candidate"
                    continue
                x_t = intermediates.get(round(float(t), 4), intermediates.get(float(t)))
                if x_t is None:
                    row["warm_status"][key] = "no_snapshot"
                    continue
                chunk = resume(torch.as_tensor(x_t), float(t))
                self._arrays[f"a_warm_{key}_{idx:04d}"] = _to_np(chunk)
                row["warm_status"][key] = "ok"
                nfe += _remaining_steps(spec.k_full, float(t), spec.env_id)
        row["shadow_nfe"] = nfe


def _remaining_steps(k_full: int, t: float, env_id: str) -> int:
    """Euler steps a resume at ``t`` executes: descending (pi05) or ascending (groot) schedules."""
    if "groot" in env_id:
        return int(round((1.0 - t) * k_full))
    return int(round(t * k_full))


def _safe(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if torch.is_tensor(o):
        return o.detach().cpu().tolist()
    return str(o)
