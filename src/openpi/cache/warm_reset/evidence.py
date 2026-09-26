"""Server-side evidence of warm reset decisions and its admission checker.

The standard LIBERO / RoboCasa workers copy ``__hit_meta__`` into their
per-step rows through a whitelist, so the ``warm_reset`` wire field never
reaches them (plan ``logs/warm_continuation_first_class_plan.log.md`` §3.4-F2).
The authoritative record is therefore written on the server by an outer
wrapper around the cache interceptor:

* one ``decision`` row per request, its index assigned at the request entry
  (an exception writes a ``status: error`` row and is re-raised);
* one ``finalize`` row per episode, committed in the same ``write`` as the
  episode's decision rows (``PerStepWriter.flush_episode``), so a truncated
  tail always loses the ``finalize`` first and fails closed.

``episode_problems`` admits an episode only against **trusted** expectations
(``ExpectedEpisode``, built by the exp analysis layer from the journal's
accepted terminal, the driver-stamped per-step rows and the dispatched yaml),
never against values read off the rows it checks. It reproduces the step_diag
admission rules (equal continuation NFE, self-start proof, K + N pricing) plus
the completeness / closure / identity rules of the production evidence.

The same wrapper and closure rules serve two sibling arm kinds: a ``miss``
arm (plain / full; ``MissSpec``) whose decision rows are MISS verdicts
carrying the executed ``miss_nfe``, and a library-free self start
(``trigger: always``) whose decision rows carry ``hit_type: SELF_ONLY`` and
the block's ``start_t``.

Public interface: ``WarmResetEvidencePolicy`` (Pi0.5, wraps ``infer``),
``GrootWarmResetEvidencePolicy`` (GR00T, wraps ``get_action``),
``ExpectedEpisode``, ``decision_problems``, ``episode_problems``,
``EVIDENCE_SCHEMA``.
Depends on ``openpi.serving.per_step_recorder.PerStepWriter`` and
``openpi.cache.warm_reset.{types,runtime}``; jax-free.
"""

from __future__ import annotations

import collections
import dataclasses
import math
import os
import pathlib
import re
import socket
import struct
import time
from typing import Any, Callable, Mapping, Optional

from openpi.cache.types import DIRECTION_ASC, PI05_V1, schedule_from_id
from openpi.cache.warm_reset.runtime import META_SCHEMA, WarmResetSession
from openpi.cache.warm_reset.types import (
    HIT_SELF_ONLY,
    MissSpec,
    WarmResetSpec,
    _is_int,
    native_grid,
    resolve_plan,
)
from openpi.serving.per_step_recorder import PerStepWriter

#: Schema tag of every evidence row.
EVIDENCE_SCHEMA = "warm_reset_evidence_v1"
ROW_DECISION = "decision"
ROW_FINALIZE = "finalize"

_SELF_KEYS = ("self_start", "self_seed", "self_direct_nfe")


def _token(value: Any) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", str(value))


# ------------------------------------------------------------------
# Evidence wrappers
# ------------------------------------------------------------------


class _EvidencePolicy:
    """Shared body of the two wrappers: session driving and row writing.

    The lifecycle hooks are implemented explicitly and forwarded; everything
    else is delegated through ``__getattr__``, so the server's ``hasattr``
    probes see exactly the wrapped interceptor's surface.
    """

    def __init__(
        self,
        inner: Any,
        *,
        session: WarmResetSession,
        family: str,
        bundle_id: str,
        yaml_id: Optional[str],
        yaml_sha256: Optional[str],
        schedule_id: str,
        k: int,
    ) -> None:
        self._inner = inner
        self._session = session
        self._spec_digest = session.spec.digest()
        self._family = family
        self._bundle_id = bundle_id
        self._yaml_id = yaml_id
        self._yaml_sha256 = yaml_sha256
        self._schedule_id = schedule_id
        self._k = k
        name = (
            f"warm_reset_{_token(yaml_id or 'none')}_{_token(socket.gethostname())}_"
            f"{os.getpid()}_{session.conn_id}.jsonl"
        )
        self._path = pathlib.Path(session.spec.evidence_dir) / name
        self._writer: Optional[PerStepWriter] = None

    @property
    def evidence_path(self) -> pathlib.Path:
        """The JSONL file this connection writes (one file per connection)."""
        return self._path

    # -- rows ------------------------------------------------------------

    def _open_writer(self) -> PerStepWriter:
        if self._writer is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._writer = PerStepWriter(self._path, stamp_success=True)
        return self._writer

    def _common(self, row_kind: str) -> dict:
        session = self._session
        identity = dict(session.identity)
        return {
            "schema": EVIDENCE_SCHEMA,
            "row_kind": row_kind,
            "conn_id": session.conn_id,
            "episode_seq": session.episode_seq,
            "bundle_id": self._bundle_id,
            "yaml_id": self._yaml_id,
            "yaml_sha256": self._yaml_sha256,
            "spec_digest": self._spec_digest,
            "schedule_id": self._schedule_id,
            "k": self._k,
            "family": self._family,
            "experiment": identity["experiment"],
            "task": identity["task"],
            "episode_id": identity["episode_id"],
            "task_uid": identity.get("task_uid"),
            "attempt": identity.get("attempt"),
            "identity": identity,
        }

    def _write_decision(
        self, idx: int, meta: Any, *, status: str, error: Optional[str], started: float
    ) -> None:
        meta = meta if isinstance(meta, dict) else {}
        warm = meta.get("warm_reset")
        row = self._common(ROW_DECISION)
        row.update(
            decision_idx=idx,
            status=status,
            hit_type=meta.get("hit_type"),
            start_t=meta.get("start_t"),
            winner_id=meta.get("winner_id"),
            warm_reset=dict(warm) if isinstance(warm, dict) else None,
            error=error,
            wall_ms=(time.perf_counter() - started) * 1000.0,
        )
        if "miss_nfe" in meta:
            # A ``miss`` arm's executed MISS step count; absent on every other row.
            row["miss_nfe"] = meta["miss_nfe"]
        self._writer.write_row(row)

    def _close_episode(self, *, terminal: bool, outcome: Optional[bool]) -> None:
        row = self._common(ROW_FINALIZE)
        row.update(n_decisions=self._session.end_episode(), terminal=terminal, outcome=outcome)
        self._writer.write_row(row)

    def _decide(self, call: Callable[[], Any]) -> Any:
        """Run one request as one decision row (index assigned at the entry)."""
        idx = self._session.begin_decision()
        started = time.perf_counter()
        try:
            out = call()
        except Exception as exc:
            self._write_decision(
                idx, None, status="error", error=f"{type(exc).__name__}: {exc}", started=started
            )
            raise
        meta = out.get("__hit_meta__") if isinstance(out, dict) else None
        self._write_decision(idx, meta, status="ok", error=None, started=started)
        return out

    # -- lifecycle -------------------------------------------------------

    def on_episode_start(
        self,
        experiment: str = "",
        task: str = "",
        episode_id: int = -1,
        episode_name: str = "",
        extra_metadata: dict | None = None,
    ) -> None:
        """Open the evidence episode, then forward. A still-open previous episode
        (no ``episode_end``) is first closed non-terminal."""
        writer = self._open_writer()
        if self._session.in_episode:
            self._close_episode(terminal=False, outcome=None)
            writer.flush_episode(None)
        self._session.begin_episode(
            experiment=experiment, task=task, episode_id=episode_id, extra_metadata=extra_metadata
        )
        writer.begin_episode()
        hook = getattr(self._inner, "on_episode_start", None)
        if hook is not None:
            hook(
                experiment=experiment,
                task=task,
                episode_id=episode_id,
                episode_name=episode_name,
                extra_metadata=extra_metadata,
            )

    def on_episode_end(self, success: bool) -> None:
        """Write the terminal ``finalize`` and commit the episode, then forward."""
        try:
            if self._session.in_episode:
                self._close_episode(terminal=True, outcome=bool(success))
                self._writer.flush_episode(bool(success))
        finally:
            hook = getattr(self._inner, "on_episode_end", None)
            if hook is not None:
                hook(success)

    def on_task_end(self) -> None:
        """Close a still-open episode non-terminal, close the file, then forward."""
        try:
            if self._session.in_episode:
                self._close_episode(terminal=False, outcome=None)
            if self._writer is not None:
                self._writer.close()
                self._writer = None
        finally:
            hook = getattr(self._inner, "on_task_end", None)
            if hook is not None:
                hook()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class WarmResetEvidencePolicy(_EvidencePolicy):
    """Evidence wrapper around a Pi0.5 ``InferenceInterceptor`` (``infer``)."""

    def infer(self, obs: dict, **kwargs: Any) -> dict:
        """One decision: the interceptor's ``infer`` plus its evidence row."""
        return self._decide(lambda: self._inner.infer(obs, **kwargs))


class GrootWarmResetEvidencePolicy(_EvidencePolicy):
    """Evidence wrapper around a ``GrootCacheInterceptor`` (``get_action``)."""

    def get_action(self, observations: dict) -> dict:
        """One decision: the interceptor's ``get_action`` plus its evidence row."""
        return self._decide(lambda: self._inner.get_action(observations))


# ------------------------------------------------------------------
# Admission (trusted expectations only)
# ------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExpectedEpisode:
    """What one accepted episode must look like, from trusted sources only.

    ``task_uid`` / ``attempt`` / ``outcome``: the run's single accepted
    terminal. ``n_decisions``: the matching driver-stamped per-step decision
    rows of that run. ``yaml_id`` / ``bundle_id``: the dispatched EpisodeTask.
    ``spec`` / ``spec_digest`` / ``yaml_sha256``: computed from the yaml the
    caller dispatched. ``schedule_id`` / ``k`` / ``start_t``: the arm
    definition. ``identity``: the episode_start identity rebuilt from the
    dispatched task. None of it is read from the rows being checked.

    ``spec`` is a ``WarmResetSpec`` (a warm reset arm; ``trigger: always``
    expects ``SELF_ONLY`` decisions at the block's ``start_t``) or a
    ``MissSpec`` (a plain / full arm: ``MISS`` decisions of ``num_steps``
    executed steps, ``start_t`` ``None``).
    """

    task_uid: str
    attempt: int
    outcome: bool
    n_decisions: int
    yaml_id: Optional[str]
    bundle_id: str
    spec: WarmResetSpec | MissSpec
    spec_digest: str
    yaml_sha256: Optional[str]
    schedule_id: str
    k: int
    start_t: Optional[float]
    identity: Mapping[str, Any]


def _same(value: Any, want: Any) -> bool:
    """Equal and of the same type: no bool-for-int or str-for-number coercion."""
    if type(value) is not type(want):
        return False
    if isinstance(want, dict):
        return value.keys() == want.keys() and all(_same(value[k], v) for k, v in want.items())
    if isinstance(want, (list, tuple)):
        return len(value) == len(want) and all(_same(v, w) for v, w in zip(value, want))
    return value == want


def _is_miss(expected: ExpectedEpisode) -> bool:
    return isinstance(expected.spec, MissSpec)


def _expected_hit_type(expected: ExpectedEpisode) -> str:
    """The ``hit_type`` every decision of the expected arm must carry."""
    if _is_miss(expected):
        return "MISS"
    return HIT_SELF_ONLY if expected.spec.always else "WARM_START"


def _expected_steps(expected: ExpectedEpisode) -> tuple[list[str], Optional[int]]:
    """``(problems, N)`` of the trusted expectation itself (N = MISS steps on a ``miss`` arm)."""
    if not isinstance(expected, ExpectedEpisode) or not isinstance(expected.spec, (WarmResetSpec, MissSpec)):
        return ["invalid_expected"], None
    if not isinstance(expected.identity, Mapping):
        return ["invalid_expected"], None
    identity, spec = expected.identity, expected.spec
    if _is_miss(expected):
        start_ok = expected.start_t is None and _is_int(spec.num_steps) and spec.num_steps >= 1
    else:
        start_ok = (
            isinstance(expected.start_t, float)
            and math.isfinite(expected.start_t)
            and (not spec.always or spec.start_t == expected.start_t)
        )
    ok = (
        _is_int(expected.attempt)
        and expected.attempt >= 0
        and _is_int(expected.n_decisions)
        and expected.n_decisions >= 0
        and _is_int(expected.k)
        and isinstance(expected.outcome, bool)
        and start_ok
        and isinstance(expected.task_uid, str) and bool(expected.task_uid)
        and isinstance(expected.bundle_id, str) and bool(expected.bundle_id)
        and (expected.yaml_id is None or isinstance(expected.yaml_id, str))
        and (expected.yaml_sha256 is None or isinstance(expected.yaml_sha256, str))
        and isinstance(expected.schedule_id, str)
        and isinstance(identity.get("experiment"), str)
        and isinstance(identity.get("task"), str)
        and _is_int(identity.get("episode_id"))
        and all(isinstance(key, str) for key in identity)
        and all(key not in identity or _same(identity[key], want)
                for key, want in (("task_uid", expected.task_uid), ("attempt", expected.attempt)))
        and isinstance(spec.evidence_dir, str) and bool(spec.evidence_dir)
    )
    if not ok:
        return ["invalid_expected"], None
    if _is_miss(expected):
        try:
            if expected.spec_digest != spec.digest():
                return ["invalid_expected"], None
            schedule = schedule_from_id(expected.schedule_id)
        except (ValueError, TypeError, OverflowError):
            return ["invalid_expected"], None
        return ([] if schedule.num_steps == expected.k else ["invalid_expected"]), spec.num_steps
    if spec.self_start:
        if not (
            isinstance(spec.seed_namespace, str) and spec.seed_namespace
            and isinstance(spec.seed_keys, tuple) and spec.seed_keys
            and all(isinstance(key, str) and key and key != "task_uid" and key in identity
                    for key in spec.seed_keys)
        ):
            return ["invalid_expected"], None
    elif spec.seed_namespace is not None or spec.seed_keys != ():
        return ["invalid_expected"], None
    try:
        if expected.spec_digest != spec.digest():
            return ["invalid_expected"], None
        schedule = schedule_from_id(expected.schedule_id)
        n_steps = resolve_plan(spec, schedule, expected.start_t).n_steps
    except (ValueError, TypeError, OverflowError):
        return ["invalid_expected"], None
    if schedule.num_steps != expected.k:
        ok = False
    return ([] if ok else ["invalid_expected"]), n_steps


def _common_problems(row: dict, expected: ExpectedEpisode) -> list[str]:
    """Checks every evidence row must pass, whatever its kind."""
    out: set[str] = set()
    if row.get("schema") != EVIDENCE_SCHEMA:
        out.add("schema_mismatch")
    required = {
        "schema", "row_kind", "conn_id", "episode_seq", "bundle_id", "yaml_id",
        "yaml_sha256", "spec_digest", "schedule_id", "k", "family", "experiment",
        "task", "episode_id", "task_uid", "attempt", "identity", "success",
    }
    if not required.issubset(row):
        out.add("invalid_field")
    if not isinstance(row.get("conn_id"), str) or not row.get("conn_id") or not (
        _is_int(row.get("episode_seq")) and row["episode_seq"] >= 0
    ):
        out.add("invalid_field")
    for key, want in (
        ("task_uid", expected.task_uid),
        ("attempt", expected.attempt),
        ("yaml_id", expected.yaml_id),
        ("bundle_id", expected.bundle_id),
    ):
        if not _same(row.get(key), want):
            out.add("identity_mismatch")
    if expected.yaml_sha256 is not None and row.get("yaml_sha256") != expected.yaml_sha256:
        out.add("identity_mismatch")
    if row.get("yaml_sha256") is not None and not isinstance(row["yaml_sha256"], str):
        out.add("invalid_field")
    family = "pi05" if expected.schedule_id == PI05_V1.schedule_id else "groot"
    if row.get("family") != family:
        out.add("schedule_mismatch")
    for key in ("experiment", "task", "episode_id"):
        if not _same(row.get(key), expected.identity.get(key)):
            out.add("identity_mismatch")
    identity = row.get("identity")
    if not isinstance(identity, dict) or any(
        key not in identity or not _same(identity[key], want)
        for key, want in expected.identity.items()
    ):
        out.add("identity_mismatch")
    if row.get("spec_digest") != expected.spec_digest:
        out.add("spec_mismatch")
    if row.get("schedule_id") != expected.schedule_id or not _same(row.get("k"), expected.k):
        out.add("schedule_mismatch")
    if not _same(row.get("success"), expected.outcome):
        out.add("outcome_mismatch")
    return sorted(out)


def _decision_head_problems(row: dict, expected: ExpectedEpisode) -> set[str]:
    """Index, required fields, wall time, status, hit type and start_t of one decision row."""
    out: set[str] = set()
    idx = row.get("decision_idx")
    if not (_is_int(idx) and idx >= 0):
        out.add("invalid_field")
    required = {"decision_idx", "status", "hit_type", "start_t", "winner_id", "warm_reset", "error", "wall_ms"}
    if not required.issubset(row):
        out.add("invalid_field")
    wall_ms = row.get("wall_ms")
    if (isinstance(wall_ms, bool) or not isinstance(wall_ms, (int, float))
            or not math.isfinite(wall_ms) or wall_ms < 0):
        out.add("invalid_field")
    if row.get("status") != "ok" or row.get("error") is not None:
        out.add("decision_error")
    if row.get("hit_type") != _expected_hit_type(expected):
        out.add("hit_type_mismatch")
    if not _same(row.get("start_t"), expected.start_t):
        out.add("schedule_mismatch")
    return out


def _miss_decision_problems(row: dict, expected: ExpectedEpisode, n_steps: int) -> list[str]:
    """A ``miss`` arm's decision: a MISS with ``n_steps`` executed steps and no continuation."""
    out = _decision_head_problems(row, expected)
    if row.get("warm_reset") is not None:
        out.add("spec_mismatch")
    if "miss_nfe" not in row:
        out.add("miss_nfe_missing")
    elif not (_is_int(row["miss_nfe"]) and row["miss_nfe"] >= 0):
        out.add("invalid_field")
    elif row["miss_nfe"] != n_steps:
        out.add("steps_mismatch")
    return sorted(out)


def _decision_content_problems(row: dict, expected: ExpectedEpisode, n_steps: int) -> list[str]:
    if _is_miss(expected):
        return _miss_decision_problems(row, expected, n_steps)
    out = _decision_head_problems(row, expected)
    idx = row.get("decision_idx")
    inner = row.get("warm_reset")
    if inner is None:
        out.add("warm_reset_missing")
        return sorted(out)
    if not isinstance(inner, dict):
        out.add("invalid_field")
        return sorted(out)
    if inner.get("schema") != META_SCHEMA:
        out.add("schema_mismatch")
    if inner.get("spec_digest") != expected.spec_digest:
        out.add("spec_mismatch")
    plan = resolve_plan(expected.spec, schedule_from_id(expected.schedule_id), expected.start_t)
    for key in ("source", "point", "kind", "level"):
        if key not in inner:
            out.add("invalid_field")
        if not _same(inner.get(key), getattr(plan, key)):
            out.add("spec_mismatch")
    if plan.direction == DIRECTION_ASC:
        taus, dt = native_grid(plan)
        if not _same(inner.get("tau"), list(taus)):
            out.add("schedule_mismatch")
        buckets = inner.get("bucket")
        if not (isinstance(buckets, list) and len(buckets) == n_steps
                and all(_is_int(b) and b >= 0 for b in buckets)):
            out.add("invalid_field")
    else:
        # The executor creates dt as float32; preserve that exact rounding.
        dt = struct.unpack("f", struct.pack("f", -plan.level / n_steps))[0]
    if not _same(inner.get("t"), list(plan.flow_times())) or not _same(inner.get("dt"), dt):
        out.add("schedule_mismatch")
    if (
        inner.get("schedule_id") != expected.schedule_id
        or not _same(inner.get("k"), expected.k)
        or not _same(inner.get("start_t"), expected.start_t)
    ):
        out.add("schedule_mismatch")
    counts = ("continuation_nfe", "n_stage3_calls", "self_start_calls", "decision_nfe", "n_steps")
    if any(not (_is_int(inner.get(name)) and inner[name] >= 0) for name in counts):
        out.add("invalid_field")
    if not (_same(inner.get("continuation_nfe"), n_steps) and _same(inner.get("n_steps"), n_steps)):
        out.add("steps_mismatch")
    if not _same(inner.get("n_stage3_calls"), 1):
        out.add("extra_stage3_calls")
    if expected.spec.self_start:
        if inner.get("self_start") is not True:
            out.add("self_start_missing")
        if not _same(inner.get("self_start_calls"), 1):
            out.add("extra_self_start_calls")
        want_seed = (
            expected.spec.seed_policy().seed(expected.identity, idx) if _is_int(idx) else None
        )
        if not _is_int(inner.get("self_seed")) or inner["self_seed"] != want_seed:
            out.add("self_seed_mismatch")
        if not _same(inner.get("self_direct_nfe"), expected.k):
            out.add("self_direct_nfe_mismatch")
        self_nfe = inner.get("self_direct_nfe")
    else:
        if any(key in inner for key in _SELF_KEYS) or not _same(inner.get("self_start_calls"), 0):
            out.add("self_start_on_cache_arm")
        self_nfe = 0
    cont, total = inner.get("continuation_nfe"), inner.get("decision_nfe")
    if not (
        _is_int(cont) and _is_int(total) and _is_int(self_nfe) and total == cont + self_nfe
    ):
        out.add("decision_nfe_mismatch")
    return sorted(out)


def _finalize_problems(row: dict, expected: ExpectedEpisode) -> list[str]:
    out = set(_common_problems(row, expected))
    if row.get("terminal") is not True or not isinstance(row.get("outcome"), bool):
        out.add("non_terminal")
    if not _same(row.get("outcome"), expected.outcome):
        out.add("outcome_mismatch")
    if not _same(row.get("n_decisions"), expected.n_decisions):
        out.add("decision_count_mismatch")
    return sorted(out)


def decision_problems(row: dict, *, expected: ExpectedEpisode) -> list[str]:
    """Problem codes of one ``decision`` row against the trusted expectation."""
    exp_problems, n_steps = _expected_steps(expected)
    if exp_problems:
        return exp_problems
    if not isinstance(row, dict):
        return ["invalid_row"]
    out = set(_common_problems(row, expected))
    if row.get("row_kind") != ROW_DECISION:
        out.add("invalid_row")
        return sorted(out)
    out.update(_decision_content_problems(row, expected, n_steps))
    return sorted(out)


def episode_problems(rows: list[dict], *, expected: ExpectedEpisode) -> dict:
    """Admission of one (task_uid, attempt) episode.

    ``rows`` must be every evidence row of that attempt within the caller's
    run-scoped evidence directory. Returns ``{problems, continuation_nfe,
    self_start_nfe, total_nfe}``; ``problems`` is a ``Counter`` and any
    non-zero code rejects the episode, in which case the three totals are
    ``None`` (a partial sum is never a usable episode cost). A ``miss`` arm
    also returns ``miss_nfe`` (= ``total_nfe``, the summed executed MISS
    steps) and ``continuation_nfe`` ``None``.
    """
    problems: collections.Counter = collections.Counter()
    none = {"continuation_nfe": None, "self_start_nfe": None, "total_nfe": None}
    if isinstance(expected, ExpectedEpisode) and _is_miss(expected):
        none["miss_nfe"] = None
    exp_problems, n_steps = _expected_steps(expected)
    if exp_problems:
        problems.update(exp_problems)
        return {"problems": problems, **none}
    if not rows:
        problems["server_evidence_missing"] += 1
        return {"problems": problems, **none}
    if expected.n_decisions < 1:
        problems["no_decisions"] += 1
    sessions = set()
    decisions: list[dict] = []
    finals = 0
    for row in rows:
        if not isinstance(row, dict):
            problems["invalid_row"] += 1
            continue
        conn_id, episode_seq = row.get("conn_id"), row.get("episode_seq")
        if isinstance(conn_id, str) and _is_int(episode_seq) and episode_seq >= 0:
            sessions.add((conn_id, episode_seq))
        kind = row.get("row_kind")
        if kind == ROW_DECISION:
            decisions.append(row)
            problems.update(_common_problems(row, expected))
            problems.update(_decision_content_problems(row, expected, n_steps))
        elif kind == ROW_FINALIZE:
            finals += 1
            problems.update(_finalize_problems(row, expected))
        else:
            problems["invalid_row"] += 1
            problems.update(_common_problems(row, expected))
    if len(sessions) > 1:
        problems["duplicate_session"] += 1
    if finals == 0:
        problems["finalize_missing"] += 1
    elif finals > 1:
        problems["duplicate_finalize"] += 1
    indices = collections.Counter(
        d.get("decision_idx") for d in decisions if _is_int(d.get("decision_idx"))
    )
    if any(count > 1 for count in indices.values()):
        problems["duplicate_decision"] += 1
    if set(indices) != set(range(expected.n_decisions)):
        problems["decision_gap"] += 1
    if any(problems.values()):
        return {"problems": problems, **none}
    if _is_miss(expected):
        miss_total = sum(d["miss_nfe"] for d in decisions)
        return {"problems": problems, "continuation_nfe": None, "self_start_nfe": 0,
                "miss_nfe": miss_total, "total_nfe": miss_total}
    continuation = sum(d["warm_reset"]["continuation_nfe"] for d in decisions)
    self_start = (
        sum(d["warm_reset"]["self_direct_nfe"] for d in decisions)
        if expected.spec.self_start
        else 0
    )
    return {
        "problems": problems,
        "continuation_nfe": continuation,
        "self_start_nfe": self_start,
        "total_nfe": continuation + self_start,
    }


__all__ = [
    "EVIDENCE_SCHEMA",
    "ExpectedEpisode",
    "GrootWarmResetEvidencePolicy",
    "WarmResetEvidencePolicy",
    "decision_problems",
    "episode_problems",
]
