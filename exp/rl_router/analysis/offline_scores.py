#!/usr/bin/env python3
"""X15 U1 — recompute retrieval scores from the frozen library and prove parity.

    python offline_scores.py --dump <batch dir> --library <pkl> --arm-yaml <yaml> \
        --out feats.jsonl [--parity-report parity.json]

Every X15 feature is a function of the retrieval scores, so the pipeline that
builds training rows must reproduce the scores the judge saw live. It does that
by **actually re-running the search**: the frozen library is loaded into the
same ``InMemoryBackend``, the arm yaml supplies the same fusion weights and
normalisation, and the dumped query keys are replayed through
``search_with_diagnostics``.

That independence is the point. An earlier revision of this module compared the
dump against a field the dump itself carried, so a row that agreed with itself
reported perfect parity — a gate that cannot fail is not a gate. The recomputed
side must come from the library, never from the file being checked.

The gate, and what failing it means::

    fused-score MAE <= 1e-3, top-1 identical >= 99.5%, top-5 set overlap >= 99%

Failing means the fp16 dump cannot stand in for the live query; the fix is to
record pre-normalisation raw keys, not to proceed with scores that disagree.
Output is written to a temporary file and promoted only after the gate passes,
so a failed run leaves no half-trusted training data behind.

Memory is the other hard constraint: ziyang10 runs under a 32 GiB cgroup whose
OOM killer takes the whole pod, tether agent included. Batches stream one row at
a time under an explicit RSS budget.

Key dependencies: ``InMemoryBackend`` (the same fusion the server runs),
``CacheStorage``, and the dump schema written by ``DumpingJudge``.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import pickle
import resource
import tempfile
from typing import Any, Iterator, Optional

import torch
import yaml

DEFAULT_RSS_BUDGET_GB = 8.0

# Parity thresholds — the plan's frozen numbers, not tunables.
MAX_FUSED_MAE = 1e-3
MIN_TOP1_AGREEMENT = 0.995
MIN_TOP5_OVERLAP = 0.99


class ParityError(RuntimeError):
    """Offline scores disagree with the online ones beyond the frozen gate."""


class RssBudgetExceeded(RuntimeError):
    """The streaming loop grew past its memory budget."""


# ------------------------------------------------------------------
# Memory
# ------------------------------------------------------------------


def current_rss_gb() -> float:
    """Resident set size of this process, in GiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def check_rss(budget_gb: float = DEFAULT_RSS_BUDGET_GB, *, where: str = "") -> None:
    """Fail loudly rather than let the cgroup OOM killer take the pod.

    The killer does not stop at this process: it takes the jupyter pod and the
    tether agent with it, which is why the loop polices itself.
    """
    rss = current_rss_gb()
    if rss > budget_gb:
        raise RssBudgetExceeded(
            f"RSS {rss:.2f} GiB exceeds the {budget_gb:.1f} GiB budget"
            f"{f' at {where}' if where else ''}; the pod's 32 GiB cgroup OOM "
            "killer would take the tether agent too. Stream smaller batches."
        )


# ------------------------------------------------------------------
# Recomputation
# ------------------------------------------------------------------


class OfflineScorer:
    """Replay dumped query keys against the frozen library.

    Holds the library backend for the lifetime of one run. The search path is
    the production one (``search_with_diagnostics``), because a parallel
    reimplementation would be checking this module against itself rather than
    against what the server does.
    """

    def __init__(self, library_path: str | pathlib.Path, arm_yaml: str | pathlib.Path) -> None:
        from openpi.cache.backends.in_memory_backend import InMemoryBackend

        # Load through the production contract, not by inserting entries by
        # hand. The real builder stores query keys and action chunks as NumPy
        # arrays (``_detach_entries``); ``load_artifact`` is what converts them
        # back to tensors, backfills trajectory/outcome fields on older
        # artifacts and validates vector_dims. Bypassing it worked only against
        # hand-built tensor fixtures and died on the first real library with
        # ``TypeError: expected Tensor ... got numpy.ndarray``.
        dims = _artifact_vector_dims(library_path)
        self._backend = InMemoryBackend(dims)
        self._backend.load_artifact(str(library_path))
        self._n_entries = len(self._backend._entries)

        cfg = yaml.safe_load(pathlib.Path(arm_yaml).read_text(encoding="utf-8"))
        cp1 = cfg["checkpoints"]["cp1"]
        ss = cp1["search_strategy"]
        self._top_k = int(ss.get("top_k", 5))
        self._fusion_weights = {
            name: float(spec["weight"])
            for name, spec in cfg.get("keys", {}).items()
            if spec.get("enabled")
        }
        self._field_similarity = ss.get("field_similarity")
        self._score_normalization = ss.get("score_normalization")
        self._checkpoint_id = _cp1_id()

    @property
    def n_entries(self) -> int:
        return self._n_entries

    def score(self, query_keys: dict[str, torch.Tensor]) -> dict:
        """Recompute this step's diagnostics from the library."""
        from openpi.cache.storage_types import QuerySpec

        spec = QuerySpec(
            query_keys=query_keys,
            top_k=self._top_k,
            checkpoint_id=self._checkpoint_id,
            fusion_weights=self._fusion_weights,
            fusion_method="weighted_score_sum",
            field_similarity=self._field_similarity,
            score_normalization=self._score_normalization,
        )
        _, diagnostics = self._backend.search_with_diagnostics(spec)
        return {
            "fused_topk": [[i, float(s)] for i, s in diagnostics.fused_topk],
            "winner_per_field": {k: float(v) for k, v in diagnostics.winner_per_field.items()},
            "field_own_margin": {k: float(v) for k, v in diagnostics.field_own_margin.items()},
            "fused_margin": float(diagnostics.fused_margin),
            "n_results": int(diagnostics.n_results),
        }


def _artifact_vector_dims(path: str | pathlib.Path) -> dict[str, int]:
    """Read just the dims header so the backend can be sized before loading.

    ``load_artifact`` validates the dims it finds against the backend's own, so
    the backend has to be constructed with the artifact's dims first.
    """
    with open(path, "rb") as fh:
        blob = pickle.load(fh)
    for key in ("entries", "vector_dims"):
        if key not in blob:
            raise ValueError(f"library {path} lacks {key!r}")
    return blob["vector_dims"]


def _cp1_id():
    from openpi.cache.types import CheckpointID

    return CheckpointID.CP1


def dumped_query_keys(row: dict) -> Optional[dict[str, torch.Tensor]]:
    """Reconstruct the query keys a dump row carries, if it carries them."""
    raw = row.get("query_keys")
    if not raw:
        return None
    return {name: torch.tensor(values, dtype=torch.float32) for name, values in raw.items()}


# ------------------------------------------------------------------
# Dump reading (streaming)
# ------------------------------------------------------------------


def iter_dump_rows(batch_dir: str | pathlib.Path) -> Iterator[dict]:
    """Yield per-decision rows from one batch directory, one file at a time.

    Deliberately a generator over files: a batch is hundreds of MB of fp16
    features and materialising it whole is what the RSS budget exists to stop.
    """
    batch = pathlib.Path(batch_dir)
    for path in sorted(batch.glob("*.jsonl")):
        if path.name == "manifest.jsonl":
            continue
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)


# ------------------------------------------------------------------
# Parity
# ------------------------------------------------------------------


def compare_parity(online: dict, offline: dict) -> dict:
    """Per-decision agreement between the recorded and RECOMPUTED scores.

    ``offline`` must come from :meth:`OfflineScorer.score`; passing the dump's
    own field back in would make every comparison trivially perfect.
    """
    on_topk = [(str(i), float(s)) for i, s in online.get("fused_topk", [])]
    off_topk = [(str(i), float(s)) for i, s in offline.get("fused_topk", [])]
    if not on_topk or not off_topk:
        return {"comparable": False}

    width = min(len(on_topk), len(off_topk))
    mae = sum(abs(on_topk[i][1] - off_topk[i][1]) for i in range(width)) / width
    top1_same = on_topk[0][0] == off_topk[0][0]
    on_ids = {i for i, _ in on_topk[:5]}
    off_ids = {i for i, _ in off_topk[:5]}
    overlap = len(on_ids & off_ids) / max(1, len(on_ids))
    return {
        "comparable": True,
        "fused_mae": mae,
        "top1_same": top1_same,
        "top5_overlap": overlap,
    }


def summarise_parity(per_decision: list[dict]) -> dict:
    """Aggregate per-decision comparisons into the gate's inputs."""
    usable = [c for c in per_decision if c.get("comparable")]
    if not usable:
        return {"n_compared": 0, "fused_mae": 0.0,
                "top1_agreement": 0.0, "top5_overlap": 0.0}
    n = len(usable)
    return {
        "n_compared": n,
        "fused_mae": sum(c["fused_mae"] for c in usable) / n,
        "top1_agreement": sum(1 for c in usable if c["top1_same"]) / n,
        "top5_overlap": sum(c["top5_overlap"] for c in usable) / n,
    }


def assert_parity(report: dict) -> None:
    """Enforce the frozen gate; raise with the numbers that failed it."""
    if report["n_compared"] == 0:
        raise ParityError(
            "no decision could be recomputed and compared, so parity is "
            "unproven; check that the dump carries query_keys and online "
            "step_features"
        )
    problems = []
    if report["fused_mae"] > MAX_FUSED_MAE:
        problems.append(f"fused MAE {report['fused_mae']:.2e} > {MAX_FUSED_MAE:.0e}")
    if report["top1_agreement"] < MIN_TOP1_AGREEMENT:
        problems.append(
            f"top-1 agreement {report['top1_agreement']:.4f} < {MIN_TOP1_AGREEMENT}"
        )
    if report["top5_overlap"] < MIN_TOP5_OVERLAP:
        problems.append(
            f"top-5 overlap {report['top5_overlap']:.4f} < {MIN_TOP5_OVERLAP}"
        )
    if problems:
        raise ParityError(
            "recomputed scores disagree with the online ones: "
            + "; ".join(problems)
            + ". The fp16 dump cannot stand in for the live query; record "
            "pre-normalisation raw keys instead of proceeding."
        )


# ------------------------------------------------------------------
# Driver
# ------------------------------------------------------------------


def run(
    dump_dir: str,
    scorer: OfflineScorer,
    out_path: str,
    *,
    rss_budget_gb: float = DEFAULT_RSS_BUDGET_GB,
) -> dict:
    """Recompute, gate, and publish atomically.

    Rows land in a temporary file and are promoted to ``out_path`` only after
    the parity gate passes — a failed run must not leave training data that
    looks usable.
    """
    per_decision: list[dict] = []
    written = 0
    tmp_dir = pathlib.Path(out_path).parent
    tmp_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=str(tmp_dir), suffix=".partial")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            for row in iter_dump_rows(dump_dir):
                query_keys = dumped_query_keys(row)
                online = row.get("step_features")
                if query_keys is not None:
                    recomputed = scorer.score(query_keys)
                    row = {**row, "offline_features": recomputed}
                    if online is not None:
                        per_decision.append(compare_parity(online, recomputed))
                out.write(json.dumps(row) + "\n")
                written += 1
                if written % 500 == 0:
                    check_rss(rss_budget_gb, where=f"row {written}")

        report = summarise_parity(per_decision)
        report["n_rows"] = written
        report["n_library_entries"] = scorer.n_entries
        report["peak_rss_gb"] = current_rss_gb()
        assert_parity(report)                    # raises before promotion
        os.replace(tmp_path, out_path)
        return report
    except BaseException:
        with contextlib_suppress():
            os.unlink(tmp_path)
        raise


class contextlib_suppress:
    """Tiny suppressor so cleanup never masks the original failure."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: Any) -> bool:
        return True


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dump", required=True, help="batch dump directory")
    ap.add_argument("--library", required=True, help="frozen library pkl")
    ap.add_argument("--arm-yaml", required=True, help="the arm yaml that produced it")
    ap.add_argument("--out", required=True, help="feature/label jsonl to write")
    ap.add_argument("--parity-report", help="write the parity numbers as json")
    ap.add_argument("--rss-budget-gb", type=float, default=DEFAULT_RSS_BUDGET_GB)
    args = ap.parse_args()

    scorer = OfflineScorer(args.library, args.arm_yaml)
    try:
        report = run(args.dump, scorer, args.out, rss_budget_gb=args.rss_budget_gb)
    except ParityError:
        raise
    finally:
        pass
    if args.parity_report:
        pathlib.Path(args.parity_report).write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
