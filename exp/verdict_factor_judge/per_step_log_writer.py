"""Per-step JSONL writer + per-yaml merge for verdict_factor_judge B1.

This module owns the client-side observability path described in
``logs/verdict_factor_judge_dedicated_runner_plan.log.md`` §7. It exposes:

* :class:`PerStepWriter` — append-only line-buffered JSONL writer pinned to a
  single worker thread, so concurrent writes from N workers never need a
  shared lock.
* :class:`PerStepWriterPool` — registry that allocates one writer per worker
  and merges all temp files into a single sorted ``<yaml_id>.jsonl`` on
  :meth:`PerStepWriterPool.finalize`.
* :func:`install_atexit` — opt-in helper that wires ``pool.finalize`` into the
  process ``atexit`` chain. Kept out of ``__init__`` because pytest reuses
  processes and a transient pool registering itself would leave stale
  callbacks across test cases.

Wire schema (must match plan §7.2 verbatim — downstream analysis joins on
field names)::

    {
      "yaml_id": str, "task_id": int, "subset_init_state_idx": int,
      "orig_init_state_idx": int, "episode_id": int, "step_idx": int,
      "phase": "warmup"|"eval",
      "hit_type": str, "start_t": float|null,
      "winner_id": str|null, "cp1_score": float|null
    }
"""

from __future__ import annotations

import atexit
import json
import pathlib
import threading
from typing import Optional


# Sort key used for the merged output. Centralised so tests and the merge
# itself agree on tuple order.
_SORT_KEYS = ("task_id", "subset_init_state_idx", "episode_id", "step_idx")


class PerStepWriter:
    """Append-only line-buffered JSONL writer owned by a single worker thread.

    Per-worker file isolation removes lock contention; ``buffering=1`` flushes
    on every newline so a crash mid-episode preserves all completed rows
    (plan §7.3).
    """

    def __init__(self, path: pathlib.Path) -> None:
        """Open ``path`` in append mode with line buffering.

        Caller is responsible for ensuring the parent directory exists; the
        owning :class:`PerStepWriterPool` mkdirs ahead of time.
        """
        self._path = pathlib.Path(path)
        # buffering=1 → line-buffered text mode; every ``\n`` flushes to the
        # OS. Combined with a worker-private file this gives crash-safety
        # without an explicit fsync per row.
        self._fh = open(self._path, "a", buffering=1, encoding="utf-8")
        self._closed = False

    @property
    def path(self) -> pathlib.Path:
        """Filesystem path this writer is appending to."""
        return self._path

    def write_row(self, row: dict) -> None:
        """Serialise ``row`` to JSON and append one line.

        ``ensure_ascii=False`` keeps task descriptions readable when the
        merged file is grepped manually; ``sort_keys=True`` keeps line bytes
        deterministic so two reruns of the same episode produce diff-able
        artefacts.
        """
        if self._closed:
            raise RuntimeError(f"PerStepWriter for {self._path} is closed")
        self._fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def close(self) -> None:
        """Flush + close the underlying file. Idempotent."""
        if self._closed:
            return
        try:
            self._fh.flush()
        finally:
            self._fh.close()
            self._closed = True


class PerStepWriterPool:
    """Per-yaml writer registry + final merge.

    Owns N temp files (one per worker), exposes :meth:`writer_for`, and on
    :meth:`finalize` merges every temp file into
    ``<out_dir>/<yaml_id>.jsonl`` sorted by
    ``(task_id, subset_init_state_idx, episode_id, step_idx)`` then deletes
    the temp files. Idempotent — calling :meth:`finalize` twice returns the
    canonical path without re-merging.
    """

    def __init__(
        self,
        out_dir: pathlib.Path,
        yaml_id: str,
        num_workers: int,
    ) -> None:
        if num_workers < 1:
            raise ValueError(f"num_workers must be >= 1, got {num_workers}")
        if not yaml_id:
            raise ValueError("yaml_id must be a non-empty string")
        self._out_dir = pathlib.Path(out_dir)
        self._yaml_id = yaml_id
        self._num_workers = int(num_workers)
        self._out_dir.mkdir(parents=True, exist_ok=True)

        # Pre-create one temp path per worker. Truncating leftover files is
        # the safest way to make reruns idempotent — append mode would double
        # rows if a previous crashed run left content behind.
        self._temp_paths: list[pathlib.Path] = [
            self._out_dir / f"_{yaml_id}.worker_{wid}.jsonl"
            for wid in range(self._num_workers)
        ]
        for p in self._temp_paths:
            # Truncate-on-open via mode="w" then immediately close — the
            # actual append-mode handle is created lazily inside writer_for.
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
        """Return the writer pinned to ``worker_id``, creating it on first use.

        Lazy creation keeps the file handle count equal to the number of
        active workers; serial mode pays for one open file even when the
        pool was sized to ``num_workers > 1`` (plan §7.3).
        """
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
                writer = PerStepWriter(self._temp_paths[worker_id])
                self._writers[worker_id] = writer
            return writer

    def finalize(self) -> pathlib.Path:
        """Merge worker temp files into the canonical sorted output.

        Thread-safe and idempotent — repeated calls short-circuit and return
        the same merged path.
        """
        with self._lock:
            if self._finalized:
                return self._merged_path

            # Close any open writers first so all rows are flushed to disk
            # before we read them back.
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

            merged_rows.sort(key=_row_sort_key)

            with open(self._merged_path, "w", encoding="utf-8") as fh:
                for row in merged_rows:
                    fh.write(
                        json.dumps(row, ensure_ascii=False, sort_keys=True)
                        + "\n"
                    )

            for path in self._temp_paths:
                # Removing temp files only after the merged file is fully
                # written prevents data loss if the merge raises mid-flight.
                if path.exists():
                    path.unlink()

            self._finalized = True
            return self._merged_path


def _row_sort_key(row: dict) -> tuple:
    """Build the merge sort key with defensive ``0`` fallback per field.

    Spec marks every key as mandatory but partial rows (e.g. a future
    pre-flush crash) should still merge in a stable order rather than crash
    the whole pool. Default-to-zero keeps merging robust without masking
    schema drift in tests, which assert on real values.
    """
    return tuple(row.get(k, 0) for k in _SORT_KEYS)


def install_atexit(pool: PerStepWriterPool) -> None:
    """Register ``pool.finalize`` to run at process exit.

    Provided as an explicit helper rather than registering inside
    ``PerStepWriterPool.__init__`` so callers (especially tests, where
    pytest reuses the interpreter process) opt in deliberately. Otherwise
    every transient pool would leak a callback into ``atexit`` and pile up
    across test runs.
    """
    atexit.register(pool.finalize)
