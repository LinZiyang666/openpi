"""Generalized per-step JSONL recorder for observability / gate-research collection.

Generalizes the verdict_factor_judge per-step writer (originally
``exp/verdict_factor_judge/common/per_step_log_writer.py``, now a thin re-export
shim over this module) into a schema-agnostic, two-mode recorder.

Modes (constructor flag ``stamp_success``):

* ``stamp_success=False`` (shim mode) — ``flush_episode()`` never injects a
  ``success`` key, so the row bytes are identical to the original writer. The
  verdict_factor_judge experiment re-exports this module and stays unchanged.
* ``stamp_success=True`` (gate mode) — ``flush_episode(success)`` stamps
  ``row["success"]=success`` on every buffered row; ``close()`` tail-flushes an
  in-flight (crash-interrupted) episode with ``success=None`` — explicit
  unknown, never ``False``.

Row schema is caller-defined. For gate-research collection each row carries the
verdict fields + identity + ``searched`` + an inline ``collect`` dict whose
values are already plain Python lists (converted from ``np.ndarray`` at the
client boundary, since conductor's ``msgpack.packb`` and ``json.dumps`` cannot
encode ndarrays).

Public interface:
  - :class:`PerStepWriter` — per-worker episode-batched writer.
  - :class:`PerStepWriterPool` — per-worker registry + sorted merge.
  - :func:`filter_searched` — drop gate-skipped rows (selection bias C5).
  - :func:`summarize_gate_log` — general eval-phase hit_type tally.
  - :func:`install_atexit` — opt-in ``pool.finalize`` at process exit.
"""

from __future__ import annotations

import atexit
import json
import pathlib
import threading
from typing import Any, Iterable

# Default merge sort key. Callers may override per-experiment via the pool.
_DEFAULT_SORT_KEYS = ("task_id", "subset_init_state_idx", "episode_id", "step_idx")

# Sentinel distinguishing "no success argument" (shim mode: do not inject a
# ``success`` key) from an explicit ``success=None`` (gate mode in-flight:
# stamp null). A plain default of ``None`` could not tell these apart.
_MISSING = object()


# ----------------------------------------------------------------------
# Writer
# ----------------------------------------------------------------------
class PerStepWriter:
    """Episode-batched JSONL writer owned by a single worker thread.

    ``write_row`` buffers in memory; ``flush_episode`` commits one batch in a
    single ``write`` call. Per-worker file isolation removes lock contention.
    A crash between two episodes loses at most the in-flight one.
    """

    def __init__(self, path, stamp_success: bool = False) -> None:
        self._path = pathlib.Path(path)
        # Block-buffered; ``flush_episode`` / ``close`` are the only commit points.
        self._fh = open(self._path, "a", encoding="utf-8")
        self._closed = False
        self._stamp_success = bool(stamp_success)
        self._buffer: list[dict] = []
        self._episode_open = False

    @property
    def path(self) -> pathlib.Path:
        """Filesystem path this writer is appending to."""
        return self._path

    def begin_episode(self) -> None:
        """Mark an episode start.

        Only meaningful in gate mode: it arms the ``close()`` tail-flush so a
        crash mid-episode stamps ``success=None`` instead of leaving the batch
        unlabelled. A no-op in shim mode.
        """
        self._episode_open = True

    def write_row(self, row: dict) -> None:
        """Append ``row`` to the in-memory episode buffer (no disk touch)."""
        if self._closed:
            raise RuntimeError(f"PerStepWriter for {self._path} is closed")
        self._episode_open = True
        self._buffer.append(row)

    def flush_episode(self, success: Any = _MISSING) -> int:
        """Serialize the buffer in one batch and ``write`` to disk.

        Gate mode: pass ``success`` to stamp every buffered row's ``success``
        key before committing. Shim mode (or gate mode called without a
        ``success`` argument) injects no key — byte-identical output.

        ``sort_keys=True`` keeps line bytes deterministic across reruns;
        ``allow_nan=False`` surfaces contract violations rather than emitting
        non-standard ``NaN`` literals. Returns the number of rows committed.
        """
        if self._closed:
            raise RuntimeError(f"PerStepWriter for {self._path} is closed")
        if self._stamp_success and success is not _MISSING:
            for row in self._buffer:
                row["success"] = success
        n = self._commit()
        self._episode_open = False
        return n

    def _commit(self) -> int:
        if not self._buffer:
            return 0
        chunk = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in self._buffer
        )
        n = len(self._buffer)
        self._buffer.clear()
        self._fh.write(chunk)
        self._fh.flush()
        return n

    def close(self) -> None:
        """Flush residual buffer + close the file. Idempotent.

        Gate mode: an in-flight episode (buffered rows, no clean
        ``flush_episode``) is incomplete — its rows are stamped
        ``success=None`` (never ``False``) before the tail flush.
        """
        if self._closed:
            return
        try:
            if self._buffer:
                if self._stamp_success and self._episode_open:
                    for row in self._buffer:
                        row.setdefault("success", None)
                try:
                    self._commit()
                except Exception:  # noqa: BLE001
                    # close() runs from finalize / atexit paths; raising would
                    # mask the original error.
                    pass
            self._fh.flush()
        finally:
            self._fh.close()
            self._closed = True


# ----------------------------------------------------------------------
# Pool
# ----------------------------------------------------------------------
class PerStepWriterPool:
    """Per-yaml writer registry + final merge.

    Owns one temp file per worker, exposes :meth:`writer_for`, and on
    :meth:`finalize` merges every temp file into ``<out_dir>/<yaml_id>.jsonl``
    sorted by ``sort_keys`` then deletes the temp files. Idempotent.
    """

    def __init__(
        self,
        out_dir,
        yaml_id: str,
        num_workers: int,
        stamp_success: bool = False,
        sort_keys: tuple[str, ...] = _DEFAULT_SORT_KEYS,
        filter_searched: bool = False,
    ) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        if not yaml_id:
            raise ValueError("yaml_id must be a non-empty string")
        self._out_dir = pathlib.Path(out_dir)
        self._yaml_id = yaml_id
        self._num_workers = int(num_workers)
        self._stamp_success = bool(stamp_success)
        self._sort_keys = tuple(sort_keys)
        # Gate mode: drop gate-skipped (searched=False) rows at merge time so
        # the merged file is a clean, selection-bias-free (C5) training set.
        self._filter_searched = bool(filter_searched)
        self._out_dir.mkdir(parents=True, exist_ok=True)

        # Truncate leftover temp files so reruns are idempotent (append mode
        # would double rows from a previously crashed run).
        self._temp_paths: list[pathlib.Path] = [
            self._out_dir / f"_{yaml_id}.worker_{wid}.jsonl"
            for wid in range(self._num_workers)
        ]
        for p in self._temp_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8"):
                pass

        self._writers: dict[int, PerStepWriter] = {}
        self._lock = threading.Lock()
        self._finalized = False
        self._merged_path = self._out_dir / f"{yaml_id}.jsonl"

    @property
    def yaml_id(self) -> str:
        """The yaml id this pool was constructed for."""
        return self._yaml_id

    @property
    def num_workers(self) -> int:
        """Number of worker slots allocated."""
        return self._num_workers

    @property
    def merged_path(self) -> pathlib.Path:
        """Final ``<out_dir>/<yaml_id>.jsonl`` path (may not exist yet)."""
        return self._merged_path

    def writer_for(self, worker_id: int) -> PerStepWriter:
        """Return the writer pinned to ``worker_id``, creating it on first use."""
        if worker_id < 0 or worker_id >= self._num_workers:
            raise IndexError(
                f"worker_id {worker_id} out of range [0, {self._num_workers})"
            )
        with self._lock:
            if self._finalized:
                raise RuntimeError(
                    f"PerStepWriterPool({self._yaml_id}) already finalized"
                )
            writer = self._writers.get(worker_id)
            if writer is None:
                writer = PerStepWriter(
                    self._temp_paths[worker_id], stamp_success=self._stamp_success
                )
                self._writers[worker_id] = writer
            return writer

    def finalize(self) -> pathlib.Path:
        """Merge worker temp files into the canonical sorted output. Idempotent."""
        with self._lock:
            if self._finalized:
                return self._merged_path

            for writer in self._writers.values():
                writer.close()

            merged_rows: list[dict] = []
            for path in self._temp_paths:
                if not path.exists():
                    continue
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        merged_rows.append(json.loads(line))

            if self._filter_searched:
                merged_rows = filter_searched(merged_rows)
            merged_rows.sort(key=lambda r: tuple(r.get(k, 0) for k in self._sort_keys))

            with open(self._merged_path, "w", encoding="utf-8") as fh:
                for row in merged_rows:
                    fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

            for path in self._temp_paths:
                if path.exists():
                    path.unlink()

            self._finalized = True
            return self._merged_path


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def filter_searched(rows: Iterable[dict]) -> list[dict]:
    """Drop gate-skipped rows (``searched=False``) for C5 selection bias.

    Gate-skipped steps must not be used as gate-training labels. Rows without a
    ``searched`` key (non-collect runs) are kept.
    """
    return [r for r in rows if r.get("searched", True) is not False]


# The three legal per-step gate verdicts. n_eval_verdicts counts ONLY these, so
# it stays == n_full_hit + n_warm_start + n_miss and non-verdict rows (the per-
# episode ``_kind=="episode_summary"`` provenance row, which also carries
# phase=="eval") never inflate the inference-ratio denominator downstream.
_VERDICT_HIT_TYPES = {"FULL_HIT", "WARM_START", "MISS"}


def summarize_gate_log(gate_dir: str, eval_yaml_id: str) -> dict:
    """Tally eval-phase gate ``hit_type`` counts in a gate/per-step JSONL.

    The general gate-analysis replacement for the verdict_factor-specific
    ``_summarize_per_step_log`` (plan §2.11 R1): schema-agnostic, it reads
    ``<gate_dir>/<eval_yaml_id>.jsonl`` and counts eval-phase VERDICT rows by
    ``hit_type``. ``n_eval_verdicts`` increments only for rows whose ``hit_type``
    is a real verdict (``FULL_HIT``/``WARM_START``/``MISS``); the per-episode
    ``episode_summary`` provenance row (no ``hit_type``) is excluded so the
    downstream inference-ratio denominator is not inflated. Best-effort —
    returns zero counts when the dir/file is absent (e.g. collection disabled).
    """
    counts = {
        "n_eval_verdicts": 0,
        "n_full_hit": 0,
        "n_warm_start": 0,
        "n_miss": 0,
    }
    if not gate_dir:
        return counts
    path = pathlib.Path(gate_dir) / f"{eval_yaml_id}.jsonl"
    if not path.exists():
        return counts
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("phase") != "eval":
            continue
        ht = (row.get("hit_type") or "").upper()
        if ht not in _VERDICT_HIT_TYPES:
            continue  # skip episode_summary / any non-verdict row
        counts["n_eval_verdicts"] += 1
        if ht == "FULL_HIT":
            counts["n_full_hit"] += 1
        elif ht == "WARM_START":
            counts["n_warm_start"] += 1
        elif ht == "MISS":
            counts["n_miss"] += 1
    return counts


def install_atexit(pool: PerStepWriterPool) -> None:
    """Register ``pool.finalize`` to run at process exit.

    Explicit helper (not registered in ``__init__``) so callers — especially
    tests, where pytest reuses the interpreter — opt in deliberately instead of
    leaking a callback per transient pool.
    """
    atexit.register(pool.finalize)
