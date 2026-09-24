"""HDF5 trace writer and per-connection sink (plan §7.1, §7.2, §7.4).

``TraceWriter`` is a process-level singleton per output directory: one daemon
thread owns every h5py and sidecar handle, fed by a single FIFO. ``Step``
messages are bounded by a semaphore (back-pressure only ever lands on the
inference thread that produced the step); ``Open`` / ``Close`` are control
messages that never block the caller, so the server's event loop is never
stalled by disk. ``H5TraceSink`` is the per-connection facade the interceptor
records into; it owns the episode token and the sticky failure state.

Commit protocol of one episode (plan §7.4): a reservation file is created with
``O_CREAT | O_EXCL`` next to the final path (cross-process mutual exclusion),
the episode streams into ``<final>.h5.tmp`` (and ``<stem>.trace.jsonl.tmp``),
``Close`` writes the terminal attrs, flushes + fsyncs both files, publishes the
sidecar and finally renames the H5 -- the rename is the commit point. Any
failure moves the episode to ``Failed``: the temp file becomes
``<final>.h5.failed`` (best effort) and is never renamed to ``.h5``. The
existence of ``<final>.h5`` with ``trace_closed_ok=True`` is therefore the only
success credential the auditors accept.

Public interface: ``TraceWriter.get`` / ``open_episode`` / ``write_step`` /
``close_episode`` / ``errors`` / ``drain`` / ``stop``; ``H5TraceSink``;
``write_trace_group`` (the ``step_XXXX/trace`` schema in one place).
Depends on h5py, numpy, ``openpi.collect.data_collector`` (shared legacy
schema) and ``openpi.cache.trace.types``.
"""

from __future__ import annotations

import atexit
import dataclasses
import datetime
import json
import logging
import os
import pathlib
import queue
import threading
import time
import urllib.parse
import uuid
from typing import Any, Optional

import h5py
import numpy as np

from openpi.collect.data_collector import (
    METADATA_ATTR_ALLOWLIST,
    resolve_episode_path,
    write_episode_attrs,
    write_step_group,
)
from openpi.cache.trace.types import (
    ARM_FULL_HIT,
    ARM_WARM_EXEC,
    ATTR_CLOSED_OK,
    ATTR_JSON_DATASET_MARKER,
    ATTR_JSON_MAX_BYTES,
    ATTR_NOISE_RECORDED,
    ATTR_SCHEMA_VERSION,
    ATTR_TERMINAL,
    ATTR_WRITE_ERRORS,
    NOISE_ACTION_RE,
    TRACE_GROUP,
    TRACE_SCHEMA_VERSION,
    DrainReport,
    EpisodeIdentity,
    StepTrace,
    TracePlan,
    TraceWriteError,
    TraceWriteFailed,
    warm_arm_name,
)

logger = logging.getLogger("openpi.cache.trace.h5_sink")

_STATE_OPEN = "open"
_STATE_CLOSED = "closed"
_STATE_FAILED = "failed"

_FAILURE_LOG = "_trace_failures.jsonl"


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


def encode_dataset_name(key: str) -> str:
    """Reversible percent-encoding of an arbitrary wire key as an HDF5 name."""
    return urllib.parse.quote(key, safe="")


def decode_dataset_name(name: str) -> str:
    return urllib.parse.unquote(name)


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "name") and hasattr(value, "value"):
        return value.name
    return str(value)


def _set_json_attr(grp: h5py.Group, name: str, value: Any) -> None:
    """Write a JSON attr, spilling to ``trace/json/<name>`` when it is too large."""
    if value is None:
        return
    text = value if isinstance(value, str) else _json_dumps(value)
    if len(text.encode("utf-8")) <= ATTR_JSON_MAX_BYTES:
        grp.attrs[name] = text
        return
    json_grp = grp.require_group("json")
    json_grp.create_dataset(name, data=text, dtype=h5py.string_dtype("utf-8"))
    grp.attrs[name] = ATTR_JSON_DATASET_MARKER


def read_json_attr(grp: h5py.Group, name: str) -> Any:
    """Inverse of ``_set_json_attr`` for readers (handles the spilled form)."""
    if name not in grp.attrs:
        return None
    raw = grp.attrs[name]
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if raw == ATTR_JSON_DATASET_MARKER:
        raw = grp["json"][name][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
    return json.loads(raw)


def _write_search(grp: h5py.Group, search) -> None:
    sgrp = grp.create_group("search")
    sgrp.attrs["checkpoint"] = search.checkpoint
    if search.gate_real_should_search is not None:
        sgrp.attrs["gate_real_should_search"] = bool(search.gate_real_should_search)
    sgrp.attrs["gate_twin_should_search"] = bool(search.gate_twin_should_search)
    vlen = h5py.string_dtype("utf-8")
    if search.real_topk_ids is not None:
        sgrp.create_dataset("real_topk_ids", data=list(search.real_topk_ids), dtype=vlen)
        sgrp.create_dataset(
            "real_topk_scores", data=np.asarray(search.real_topk_scores or [], dtype=np.float32)
        )
    sgrp.create_dataset("twin_topk_ids", data=list(search.twin_topk_ids), dtype=vlen)
    sgrp.create_dataset(
        "twin_topk_scores", data=np.asarray(search.twin_topk_scores, dtype=np.float32)
    )
    if search.twin_chain_scores is not None:
        sgrp.create_dataset(
            "twin_topk_chain_scores",
            data=np.asarray(search.twin_chain_scores, dtype=np.float32),
        )
    pf = search.twin_per_field
    if pf is not None:
        pgrp = sgrp.create_group("twin_topk_per_field")
        prgrp = sgrp.create_group("twin_topk_per_field_present")
        for f_idx, field_name in enumerate(pf.fields):
            pgrp.create_dataset(field_name, data=np.asarray(pf.scores[:, f_idx], dtype=np.float32))
            prgrp.create_dataset(field_name, data=np.asarray(pf.present[:, f_idx], dtype=bool))
        sgrp.attrs["twin_topk_per_field_kind"] = _json_dumps(pf.kind_by_field)
        _set_json_attr(sgrp, "twin_topk_per_field_metadata", pf.field_metadata)
        if pf.current_step_wss is not None:
            sgrp.create_dataset(
                "twin_topk_current_step_fused",
                data=np.asarray(pf.current_step_wss, dtype=np.float32),
            )
        elif pf.not_applicable_reason:
            sgrp.attrs["twin_topk_current_step_fused_not_applicable"] = pf.not_applicable_reason
    if search.twin_winner_per_field:
        wgrp = sgrp.create_group("twin_winner_per_field")
        for k, v in search.twin_winner_per_field.items():
            wgrp.create_dataset(k, data=np.float32(v))
    if search.twin_field_own_margin:
        mgrp = sgrp.create_group("twin_field_own_margin")
        for k, v in search.twin_field_own_margin.items():
            mgrp.create_dataset(k, data=np.float32(v))
    if search.twin_fused_margin is not None:
        sgrp.attrs["twin_fused_margin"] = float(search.twin_fused_margin)
    if search.twin_n_results is not None:
        sgrp.attrs["twin_n_results"] = int(search.twin_n_results)
    _set_json_attr(sgrp, "twin_retrieval_signals_json", search.twin_retrieval_signals_json)
    _set_json_attr(sgrp, "real_verdict_json", search.real_verdict_json)
    _set_json_attr(sgrp, "twin_proposed_verdict_json", search.twin_proposed_verdict_json)
    _set_json_attr(sgrp, "twin_verdict_json", search.twin_verdict_json)
    if search.twin_validation_error:
        sgrp.attrs["twin_validation_error"] = search.twin_validation_error
    if search.twin_replay_target:
        sgrp.attrs["twin_replay_target"] = search.twin_replay_target
    if search.top1_entry_id:
        sgrp.attrs["top1_entry_id"] = search.top1_entry_id


def write_trace_group(step_grp: h5py.Group, step: StepTrace, plan: TracePlan) -> None:
    """Write ``step_XXXX/trace`` for one decision (plan §7.2).

    The legacy keys are written by ``write_step_group`` on ``step_grp`` itself;
    this function only ever creates the ``trace`` subgroup, and refuses any
    dataset name inside it that would look like a library snapshot.
    """
    tg = step_grp.create_group(TRACE_GROUP)
    tg.attrs["executed_arm"] = step.executed_arm
    for k, v in step.verdict.items():
        if v is None or isinstance(v, (dict, list)):
            continue
        if isinstance(v, bool):
            tg.attrs[k] = bool(v)
        elif isinstance(v, (int, float, str)):
            tg.attrs[k] = v
    _set_json_attr(tg, "verdict_json", step.verdict)
    _set_json_attr(tg, "tier_status_json", step.tier_status)
    _set_json_attr(tg, "error_proxies_json", step.error_proxies)
    _set_json_attr(tg, "timing_json", step.timing_ms)
    _set_json_attr(tg, "warm_index_map_json", {str(k): v for k, v in step.warm_index_map.items()})
    _set_json_attr(tg, "cp3_twin_json", step.cp3_twin_json)
    _set_json_attr(tg, "real_continuation_json", step.real_continuation_json)
    _set_json_attr(tg, "twin_continuation_json", step.twin_continuation_json)
    if step.prompt is not None:
        tg.attrs["prompt"] = step.prompt

    if step.raw_images:
        rgrp = tg.create_group("raw_images")
        mapping = {}
        for key, img in step.raw_images.items():
            name = encode_dataset_name(key)
            mapping[name] = key
            rgrp.create_dataset(name, data=np.asarray(img), compression="lzf")
        rgrp.attrs["wire_keys_json"] = _json_dumps(mapping)
    if step.raw_state is not None:
        ds = tg.create_dataset("raw_state", data=np.asarray(step.raw_state, dtype=np.float32))
        if step.raw_state_layout:
            ds.attrs["layout_json"] = _json_dumps([list(kv) for kv in step.raw_state_layout])
    if step.model_images:
        mgrp = tg.create_group("model_images")
        for key, img in step.model_images.items():
            mgrp.create_dataset(encode_dataset_name(key), data=np.asarray(img), compression="lzf")
        if step.image_mask:
            mgrp.attrs["image_mask_json"] = _json_dumps(step.image_mask)
    if step.tokenized_prompt is not None:
        tg.create_dataset(
            "tokenized_prompt", data=np.asarray(step.tokenized_prompt, dtype=np.int64)
        )
    if step.query_keys:
        qgrp = tg.create_group("query_keys")
        for name, vec in step.query_keys.items():
            qgrp.create_dataset(name, data=np.asarray(vec, dtype=np.float32))
    if step.search is not None:
        _write_search(tg, step.search)

    agrp = tg.create_group("actions")
    if step.action_full_hit is not None:
        agrp.create_dataset(ARM_FULL_HIT, data=np.asarray(step.action_full_hit, dtype=np.float32))
    for idx, chunk in step.action_warm.items():
        if chunk is None:
            continue
        agrp.create_dataset(warm_arm_name(idx), data=np.asarray(chunk, dtype=np.float32))
    if step.action_warm_exec is not None:
        agrp.create_dataset(ARM_WARM_EXEC, data=np.asarray(step.action_warm_exec, dtype=np.float32))
    agrp.create_dataset(
        "full_inference", data=np.asarray(step.legacy.clean_action, dtype=np.float32)
    )
    agrp.create_dataset("executed", data=np.asarray(step.action_executed, dtype=np.float32))

    def _guard(name: str, obj: Any) -> None:
        leaf = name.rsplit("/", 1)[-1]
        if NOISE_ACTION_RE.match(leaf):
            raise ValueError(
                f"trace group must not contain a snapshot-looking dataset: {name}"
            )

    tg.visititems(_guard)


def _sidecar_row(step_idx: int, step: StepTrace) -> dict[str, Any]:
    search = step.search
    row: dict[str, Any] = {
        "step": step_idx,
        "executed_arm": step.executed_arm,
        "hit_type": step.verdict.get("hit_type"),
        "start_t": step.verdict.get("start_t"),
        "winner_id": step.verdict.get("winner_id"),
        "score": step.verdict.get("score"),
        "searched": step.verdict.get("searched"),
        "tier_status": step.tier_status,
        "error_proxies": step.error_proxies,
        "timing_ms": step.timing_ms,
    }
    if search is not None:
        row["top1_entry_id"] = search.top1_entry_id
        row["gate_twin_should_search"] = search.gate_twin_should_search
        row["twin_topk"] = [
            [i, float(s)] for i, s in zip(search.twin_topk_ids, search.twin_topk_scores)
        ]
    return row


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class _Job:
    token: str
    identity: EpisodeIdentity
    plan: TracePlan
    final_path: pathlib.Path
    tmp_path: pathlib.Path
    reservation: pathlib.Path
    sidecar_final: Optional[pathlib.Path]
    sidecar_tmp: Optional[pathlib.Path]
    h5: Optional[h5py.File] = None
    sidecar: Any = None
    n_steps: int = 0
    state: str = _STATE_OPEN
    prompt_written: bool = False


class TraceWriter:
    """One writer thread per output directory (plan §7.1 / §7.4)."""

    _instances: dict[str, "TraceWriter"] = {}
    _instances_lock = threading.Lock()

    @classmethod
    def get(cls, out_dir: str | os.PathLike, *, queue_steps: int = 256) -> "TraceWriter":
        key = str(pathlib.Path(out_dir).expanduser().resolve())
        with cls._instances_lock:
            inst = cls._instances.get(key)
            if inst is None:
                inst = cls(key, queue_steps=queue_steps)
                cls._instances[key] = inst
            return inst

    @classmethod
    def all_instances(cls) -> list["TraceWriter"]:
        with cls._instances_lock:
            return list(cls._instances.values())

    def __init__(self, out_dir: str, *, queue_steps: int = 256) -> None:
        if queue_steps < 1:
            raise ValueError("queue_steps must be >= 1")
        self.out_dir = pathlib.Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue = queue.Queue()
        self._sem = threading.BoundedSemaphore(queue_steps)
        self._queue_steps = queue_steps
        self._lock = threading.Lock()
        self._jobs: dict[str, _Job] = {}
        self._errors: list[TraceWriteError] = []
        self._errors_by_token: dict[str, TraceWriteError] = {}
        self._fatal: Optional[BaseException] = None
        self._stopping = False
        self._stop_enqueued = False
        self._abort_reason: Optional[str] = None
        self._shutdown_deadline: Optional[float] = None
        self.build_mode = False
        self._idle = threading.Condition(self._lock)
        self._pending = 0  # messages enqueued and not yet processed
        self._thread = threading.Thread(
            target=self._run, name=f"trace-writer:{self.out_dir.name}", daemon=True
        )
        self._thread.start()
        atexit.register(self._atexit_drain)

    # -- health -------------------------------------------------------------

    @property
    def healthy(self) -> bool:
        return self._thread.is_alive() and self._fatal is None and not self._stopping

    def errors(self) -> list[TraceWriteError]:
        with self._lock:
            return list(self._errors)

    def error_for(self, token: str) -> Optional[TraceWriteError]:
        with self._lock:
            return self._errors_by_token.get(token)

    def queue_depth(self) -> int:
        with self._lock:
            return self._pending

    # -- producer API (any thread) -----------------------------------------

    def _put(self, msg: tuple) -> None:
        with self._lock:
            if not self.healthy:
                raise TraceWriteFailed(
                    f"trace writer for {self.out_dir} is not running "
                    f"(fatal={self._fatal!r}, stopping={self._stopping})"
                )
            self._pending += 1
            self._queue.put(msg)

    def open_episode(self, token: str, identity: EpisodeIdentity, plan: TracePlan) -> None:
        if plan.fail_loud:
            self.build_mode = True
        self._put(("open", token, identity, plan))

    def write_step(self, token: str, step_idx: int, step: StepTrace) -> None:
        """Enqueue one step; blocks the caller (only) while the queue is full."""
        while not self._sem.acquire(timeout=0.5):
            if not self.healthy:
                raise TraceWriteFailed(
                    f"trace writer for {self.out_dir} died while the queue was full"
                )
        try:
            self._put(("step", token, step_idx, step))
        except BaseException:
            self._sem.release()
            raise

    def close_episode(self, token: str, *, success: bool, terminal: bool) -> None:
        self._put(("close", token, bool(success), bool(terminal)))

    def fail_episode(self, token: str, reason: str) -> None:
        self._put(("fail", token, reason))

    def abort(self, reason: str) -> None:
        """Revoke commit permission immediately; only the writer closes handles."""
        with self._lock:
            self._abort_reason = self._abort_reason or reason
            self._stopping = True

    def drain(self, timeout: Optional[float] = 30.0) -> DrainReport:
        """Wait until every enqueued message is processed (or ``timeout``)."""
        deadline = None if timeout is None else time.monotonic() + timeout
        timed_out = False
        with self._idle:
            while self._pending > 0 and self._thread.is_alive():
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    timed_out = True
                    break
                self._idle.wait(timeout=remaining if remaining is not None else 1.0)
            unfinished = tuple(
                job.token for job in self._jobs.values() if job.state == _STATE_OPEN
            )
            errors = tuple(self._errors)
            fatal_error = repr(self._fatal) if self._fatal is not None else None
        return DrainReport(
            unfinished=unfinished,
            errors=errors,
            timed_out=timed_out,
            thread_alive=self._thread.is_alive(),
            fatal_error=fatal_error,
        )

    def stop(self, timeout: Optional[float] = 30.0) -> DrainReport:
        """Drain, fail every still-open episode, then stop the thread.

        A stopped writer leaves the process registry: ``all_instances`` lists
        live writers only, so a later shutdown sweep never re-reports one.
        """
        deadline = None if timeout is None else time.monotonic() + max(0.0, timeout)
        with self._lock:
            self._stopping = True
            if deadline is not None:
                self._shutdown_deadline = (
                    deadline if self._shutdown_deadline is None
                    else min(deadline, self._shutdown_deadline)
                )
        report = self.drain(timeout)
        if report.timed_out:
            self.abort("shutdown deadline exceeded")
        with self._lock:
            if not self._stop_enqueued:
                self._stop_enqueued = True
                self._queue.put(("stop",))
        remaining = None if deadline is None else max(0.0, deadline - time.monotonic())
        self._thread.join(timeout=remaining)
        if not self._thread.is_alive():
            with type(self)._instances_lock:
                for key, inst in list(type(self)._instances.items()):
                    if inst is self:
                        del type(self)._instances[key]
        with self._lock:
            unfinished = tuple(
                job.token for job in self._jobs.values() if job.state == _STATE_OPEN
            )
            errors = tuple(self._errors)
            fatal_error = repr(self._fatal) if self._fatal is not None else None
        return DrainReport(
            unfinished=unfinished,
            errors=errors,
            timed_out=report.timed_out or self._thread.is_alive(),
            thread_alive=self._thread.is_alive(),
            fatal_error=fatal_error,
        )

    def _atexit_drain(self) -> None:
        # Best effort only: the entry point's own shutdown path is the one
        # that decides the exit status (plan §7.4-5).
        try:
            self.stop(timeout=10.0)
        except Exception:  # noqa: BLE001 - interpreter is going away
            pass

    # -- writer thread --------------------------------------------------------

    def _run(self) -> None:
        while True:
            msg = self._queue.get()
            kind = msg[0]
            if kind == "stop":
                with self._lock:
                    open_jobs = [j for j in self._jobs.values() if j.state == _STATE_OPEN]
                for job in open_jobs:
                    try:
                        self._fail_job(job, "stop", "writer stopped with episode open")
                    except BaseException as exc:  # noqa: BLE001 - keep closing other jobs
                        with self._lock:
                            self._fatal = exc
                            self._abort_reason = self._abort_reason or repr(exc)
                        logger.exception("trace writer: stop cleanup failed")
                return
            try:
                if kind == "open":
                    self._handle_open(*msg[1:])
                elif kind == "step":
                    try:
                        self._handle_step(*msg[1:])
                    finally:
                        self._sem.release()
                elif kind == "close":
                    self._handle_close(*msg[1:])
                elif kind == "fail":
                    job = self._jobs.get(msg[1])
                    if job is not None:
                        self._fail_job(job, "lifecycle", msg[2])
            except BaseException as exc:  # noqa: BLE001 - never let the thread die silently
                logger.exception("trace writer: unexpected failure on %s", kind)
                with self._lock:
                    self._fatal = exc
                    self._abort_reason = self._abort_reason or repr(exc)
                try:
                    token = msg[1]
                    job = self._jobs.get(token)
                    if job is not None:
                        self._fail_job(job, kind, repr(exc))
                    else:
                        identity = msg[2] if kind == "open" else None
                        self._record_token_error(token, identity, kind, repr(exc))
                except BaseException:  # noqa: BLE001 - the recovery itself failed
                    # ``_fatal`` / ``_abort_reason`` are already set, so the
                    # drain report and the health check carry the failure;
                    # keep the thread alive to serve the remaining queue.
                    logger.exception("trace writer: failure recovery on %s raised", kind)
            finally:
                with self._idle:
                    self._pending -= 1
                    if self._pending == 0:
                        self._idle.notify_all()

    # -- episode lifecycle (writer thread only) -------------------------------

    def _record_error(self, job: _Job, stage: str, message: str) -> None:
        self._record_token_error(job.token, job.identity, stage, message)

    def _record_token_error(
        self, token: str, identity: Optional[EpisodeIdentity], stage: str, message: str
    ) -> None:
        err = TraceWriteError(
            token=token,
            episode=(f"{identity.experiment}/{identity.episode_name or identity.episode_id}"
                     if identity is not None else token),
            stage=stage,
            message=message,
            timestamp=time.time(),
        )
        with self._lock:
            self._errors.append(err)
            self._errors_by_token[token] = err
        logger.error("trace writer: %s failed at %s: %s", err.episode, stage, message)
        try:
            with open(self.out_dir / _FAILURE_LOG, "a", encoding="utf-8") as fh:
                fh.write(_json_dumps(dataclasses.asdict(err)) + "\n")
        except OSError:
            logger.error("trace writer: failure log %s is not writable either", _FAILURE_LOG)

    def _fail_job(self, job: _Job, stage: str, message: str) -> None:
        if job.state == _STATE_FAILED:
            return
        job.state = _STATE_FAILED
        try:
            self._record_error(job, stage, message)
        finally:
            # Failure reporting is best effort; it must not prevent the
            # writer thread from releasing handles and cleaning temporary files.
            for handle_name in ("h5", "sidecar"):
                handle = getattr(job, handle_name)
                if handle is not None:
                    try:
                        handle.close()
                    except Exception:  # noqa: BLE001 - already failing
                        pass
                    setattr(job, handle_name, None)
            try:
                if job.tmp_path.exists():
                    os.replace(job.tmp_path, job.final_path.with_suffix(".h5.failed"))
            except OSError:
                pass
            if job.sidecar_tmp is not None:
                try:
                    job.sidecar_tmp.unlink(missing_ok=True)
                except OSError:
                    pass
            self._release_reservation(job)

    def _release_reservation(self, job: _Job) -> None:
        try:
            job.reservation.unlink(missing_ok=True)
        except OSError as exc:
            logger.error("trace writer: reservation cleanup failed for %s: %r", job.final_path, exc)

    def _handle_open(self, token: str, identity: EpisodeIdentity, plan: TracePlan) -> None:
        if self._abort_reason is not None:
            self._record_token_error(token, identity, "open", self._abort_reason)
            return
        if token in self._jobs:
            logger.error("trace writer: duplicate open for token %s ignored", token)
            return
        final_path = resolve_episode_path(
            self.out_dir,
            identity.experiment,
            identity.episode_id,
            identity.episode_name,
            pid_suffix=True,
        )
        tmp_path = final_path.with_suffix(".h5.tmp")
        reservation = final_path.with_suffix(".h5.reserved")
        stem = final_path.with_suffix("")
        sidecar_final = stem.parent / f"{stem.name}.trace.jsonl" if plan.sidecar_jsonl else None
        sidecar_tmp = stem.parent / f"{stem.name}.trace.jsonl.tmp" if plan.sidecar_jsonl else None
        job = _Job(
            token=token,
            identity=identity,
            plan=plan,
            final_path=final_path,
            tmp_path=tmp_path,
            reservation=reservation,
            sidecar_final=sidecar_final,
            sidecar_tmp=sidecar_tmp,
        )
        with self._lock:
            self._jobs[token] = job
        if final_path.exists():
            job.state = _STATE_FAILED
            self._record_error(job, "open", f"final file already exists: {final_path}")
            return
        owns_reservation = False
        try:
            fd = os.open(reservation, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            owns_reservation = True
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(
                    _json_dumps(
                        {
                            "token": token,
                            "pid": os.getpid(),
                            "experiment": identity.experiment,
                            "episode_id": identity.episode_id,
                            "episode_name": identity.episode_name,
                        }
                    )
                )
        except FileExistsError:
            job.state = _STATE_FAILED
            self._record_error(job, "open", f"path is reserved by another writer: {reservation}")
            return
        except OSError as exc:
            job.state = _STATE_FAILED
            self._record_error(job, "open", f"cannot create reservation: {exc!r}")
            if owns_reservation:
                self._release_reservation(job)
            return
        try:
            # Another process may have committed and released its reservation
            # after the early exists() check. We now own the path exclusively.
            existing = [final_path, tmp_path, final_path.with_suffix(".h5.failed")]
            existing.extend(p for p in (sidecar_tmp, sidecar_final) if p is not None)
            if any(p.exists() for p in existing):
                job.state = _STATE_FAILED
                self._record_error(job, "open", f"episode file already exists: {final_path}")
                self._release_reservation(job)
                return
            job.h5 = h5py.File(tmp_path, "w")
            self._write_preliminary_attrs(job)
            if sidecar_tmp is not None:
                job.sidecar = open(sidecar_tmp, "w", encoding="utf-8")
                job.sidecar.write(_json_dumps({"_episode": self._episode_header(job)}) + "\n")
        except Exception as exc:  # noqa: BLE001
            self._fail_job(job, "open", repr(exc))

    def _episode_header(self, job: _Job) -> dict[str, Any]:
        ident = job.identity
        return {
            "experiment": ident.experiment,
            "task": ident.task,
            "episode_id": ident.episode_id,
            "episode_name": ident.episode_name,
            "task_uid": ident.task_uid,
            "attempt": ident.attempt,
            "model": job.plan.model,
            "schedule_id": job.plan.schedule.schedule_id,
            "checkpoint": job.plan.checkpoint,
            "warm_tiers": list(job.plan.warm_tiers),
            "noise_actions_recorded": job.plan.record_noise_actions,
        }

    def _write_preliminary_attrs(self, job: _Job) -> None:
        f = job.h5
        ident = job.identity
        plan = job.plan
        episode_attrs = {
            k: ident.extra_metadata[k] for k in METADATA_ATTR_ALLOWLIST if k in ident.extra_metadata
        }
        write_episode_attrs(
            f,
            experiment=ident.experiment,
            task=ident.task,
            episode_id=ident.episode_id,
            num_steps=0,
            success=False,
            episode_attrs=episode_attrs,
        )
        f.attrs["denoise_schedule_id"] = plan.schedule.schedule_id
        f.attrs["denoising_num_steps"] = int(plan.schedule.num_steps)
        f.attrs[ATTR_SCHEMA_VERSION] = TRACE_SCHEMA_VERSION
        f.attrs[ATTR_NOISE_RECORDED] = bool(plan.record_noise_actions)
        f.attrs["trace_denoise_schedule_id"] = plan.schedule.schedule_id
        f.attrs["trace_denoising_num_steps"] = int(plan.schedule.num_steps)
        f.attrs["trace_model"] = plan.model
        f.attrs["trace_checkpoint"] = plan.checkpoint or ""
        f.attrs["trace_concurrent"] = bool(plan.concurrent)
        f.attrs["trace_rng_isolation"] = plan.rng_isolation
        f.attrs["trace_warm_tiers"] = _json_dumps(list(plan.warm_tiers))
        f.attrs["trace_bundle_id"] = plan.bundle_id
        f.attrs["trace_yaml_id"] = plan.yaml_id
        f.attrs["trace_yaml_sha256"] = plan.yaml_sha256
        f.attrs["trace_connection_id"] = plan.connection_id
        f.attrs["trace_library_sha256"] = plan.library_sha256 or ""
        f.attrs["trace_key_builder_type"] = plan.key_builder_type or ""
        f.attrs["trace_task_uid"] = ident.task_uid
        f.attrs["trace_attempt"] = int(ident.attempt)
        f.attrs["trace_has_conductor_identity"] = bool(ident.has_conductor_identity)
        f.attrs["trace_extra_metadata_json"] = _json_dumps(ident.extra_metadata)
        f.attrs[ATTR_TERMINAL] = False
        f.attrs[ATTR_CLOSED_OK] = False
        f.attrs[ATTR_WRITE_ERRORS] = 0

    def _handle_step(self, token: str, step_idx: int, step: StepTrace) -> None:
        job = self._jobs.get(token)
        if job is None:
            # A step for a token the writer never opened (or already retired)
            # is a caller bug; it is logged and dropped rather than allowed
            # to kill the writer for every other connection.
            logger.error("trace writer: step %d for unknown token %s dropped", step_idx, token)
            return
        if job.state != _STATE_OPEN:
            return  # failed or already closed: drop (the error is already recorded)
        if self._abort_reason is not None:
            self._fail_job(job, "stop", self._abort_reason)
            return
        if step_idx != job.n_steps:
            self._fail_job(
                job, "step", f"step index {step_idx} is not the next index {job.n_steps}"
            )
            return
        try:
            f = job.h5
            grp = f.create_group(f"step_{step_idx:04d}")
            write_step_group(grp, step.legacy)
            write_trace_group(grp, step, job.plan)
            if not job.prompt_written and step.prompt is not None:
                f.attrs["prompt"] = str(step.prompt)
                job.prompt_written = True
            if job.sidecar is not None:
                job.sidecar.write(_json_dumps(_sidecar_row(step_idx, step)) + "\n")
            job.n_steps += 1
        except Exception as exc:  # noqa: BLE001
            self._fail_job(job, "step", repr(exc))

    def _handle_close(self, token: str, success: bool, terminal: bool) -> None:
        job = self._jobs.get(token)
        if job is None:
            logger.error("trace writer: close for unknown or already closed token %s ignored", token)
            return
        if job.state == _STATE_FAILED:
            with self._lock:
                self._jobs.pop(token, None)
            return
        try:
            f = job.h5
            f.attrs["num_steps"] = int(job.n_steps)
            f.attrs["success"] = bool(success)
            f.attrs["timestamp"] = datetime.datetime.now().isoformat()
            f.attrs[ATTR_TERMINAL] = bool(terminal)
            f.attrs[ATTR_WRITE_ERRORS] = 0
            f.attrs[ATTR_CLOSED_OK] = True
            f.flush()
            f.close()
            job.h5 = None
            _fsync_path(job.tmp_path)
            if job.sidecar is not None:
                job.sidecar.flush()
                os.fsync(job.sidecar.fileno())
                job.sidecar.close()
                job.sidecar = None
            # Linearize commit permission with stop/abort. A close that was
            # blocked in I/O cannot publish after the shutdown deadline.
            with self._lock:
                if self._abort_reason is not None:
                    raise TraceWriteFailed(self._abort_reason)
                if self._shutdown_deadline is not None and time.monotonic() >= self._shutdown_deadline:
                    raise TraceWriteFailed("shutdown deadline exceeded before commit")
                if job.sidecar_tmp is not None:
                    os.replace(job.sidecar_tmp, job.sidecar_final)
                # Commit point: only a fully flushed file becomes ``.h5``.
                os.replace(job.tmp_path, job.final_path)
                job.state = _STATE_CLOSED
        except Exception as exc:  # noqa: BLE001
            self._fail_job(job, "close", repr(exc))
            return
        finally:
            with self._lock:
                self._jobs.pop(token, None)
        self._release_reservation(job)
        logger.info(
            "trace writer: episode %s written -> %s (%d steps, success=%s, terminal=%s)",
            job.identity.episode_id,
            job.final_path,
            job.n_steps,
            success,
            terminal,
        )


def _fsync_path(path: pathlib.Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Per-connection sink
# ---------------------------------------------------------------------------


class H5TraceSink:
    """Per-connection facade over ``TraceWriter`` (implements ``TraceSink``).

    Owns the episode token, the step counter and the sticky failure state:
    with ``fail_loud`` (build-cache) a writer failure on any earlier episode of
    this connection makes ``on_episode_start`` and ``record_step`` raise
    ``TraceWriteFailed``; in diagnostic mode the failure is only recorded.
    """

    def __init__(
        self,
        out_dir: str | os.PathLike,
        *,
        plan: TracePlan,
        writer: Optional[TraceWriter] = None,
        queue_steps: int = 256,
    ) -> None:
        self._writer = writer if writer is not None else TraceWriter.get(out_dir, queue_steps=queue_steps)
        self._plan = plan
        self._fail_loud = bool(plan.fail_loud)
        self._token: Optional[str] = None
        self._identity: Optional[EpisodeIdentity] = None
        self._step_idx = 0
        self._in_flight = 0
        self._tokens: list[str] = []
        self._lock = threading.Lock()

    @property
    def writer(self) -> TraceWriter:
        return self._writer

    @property
    def identity(self) -> Optional[EpisodeIdentity]:
        return self._identity

    @property
    def step_idx(self) -> int:
        return self._step_idx

    def sticky_error(self) -> Optional[TraceWriteError]:
        """The first writer failure on any episode this sink has opened."""
        for token in self._tokens:
            err = self._writer.error_for(token)
            if err is not None:
                return err
        return None

    def _check_sticky(self, where: str) -> None:
        if not self._fail_loud:
            return
        err = self.sticky_error()
        if err is not None:
            raise TraceWriteFailed(
                f"trace build-cache writer failed earlier on this connection "
                f"({err.episode} at {err.stage}: {err.message}); refusing {where}"
            )
        if not self._writer.healthy:
            raise TraceWriteFailed(f"trace writer is not running; refusing {where}")

    def on_episode_start(self, identity: EpisodeIdentity) -> None:
        with self._lock:
            if self._token is not None:
                # A start without an end: the previous episode is closed as a
                # half episode so its token can never be written to again.
                self._writer.close_episode(self._token, success=False, terminal=False)
                self._token = None
            self._check_sticky("episode_start")
            token = uuid.uuid4().hex
            self._tokens.append(token)
            self._token = token
            self._identity = identity
            self._step_idx = 0
            self._in_flight = 0
            self._writer.open_episode(token, identity, self._plan)

    def begin_step(self) -> None:
        with self._lock:
            self._in_flight += 1

    def finish_step(self) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def record_step(self, step: StepTrace) -> None:
        with self._lock:
            self._check_sticky("record_step")
            token = self._token
            if token is None:
                raise RuntimeError("H5TraceSink.record_step called outside an episode")
            idx = self._step_idx
            self._step_idx += 1
        self._writer.write_step(token, idx, step)

    def on_episode_end(self, success: bool) -> None:
        with self._lock:
            if self._token is None:
                return
            if self._in_flight != 0:
                token = self._token
                self._token = None
                self._writer.fail_episode(token, "episode_end with an in-flight step")
                raise RuntimeError(
                    "H5TraceSink.on_episode_end with an in-flight step: the server "
                    "processes one connection sequentially, this cannot happen"
                )
            token = self._token
            self._token = None
            self._writer.close_episode(token, success=bool(success), terminal=True)

    def on_task_end(self) -> None:
        with self._lock:
            if self._token is None:
                return
            token = self._token
            self._token = None
            self._writer.close_episode(token, success=False, terminal=False)

    def close(self) -> None:
        self.on_task_end()
