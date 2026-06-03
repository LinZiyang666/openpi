"""ROUND 1 equivalence comparison — PrebuiltMatrixBackend vs src InMemoryBackend.

Loads the real libero_10 cp1_spatial_pool_16 library once (via the same
``build_cache_components`` the server uses), shares its ``_entries`` into a
``PrebuiltMatrixBackend``, and runs leave-one-out search over BOTH backends through the
full ``backend.search(spec)`` dispatch (covering ``_filter_entries`` ordering vs the
prebuilt id->row map). LOO is implemented by popping the query entry from the shared
``_entries`` dict for the duration of one search and restoring it afterwards.

Hard verdict (blocking): 3-way verdict flips == 0 AND winner-id mismatches == 0 across
all LOO queries. Soft monitors: top fused score abs-diff (max/median/p99) and
near-threshold margin counts. Numbers match the 3-way judge in the round1 yaml
(FULL=0.997697 / WARM=0.997403).

Usage::
    uv run python exp/cache_latency_bench/opt/compare_equivalence.py \\
        --cache-config exp/cache_latency_bench/config/round1/cp1_libero10.yaml \\
        --out exp/cache_latency_bench/data/round1/parity_result.json
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import statistics

import torch
import tyro

from openpi.cache.config import build_cache_components, load_cache_config
from openpi.cache.storage_types import QueryFilter, QuerySpec

from exp.cache_latency_bench.opt.prebuilt_matrix_backend import PrebuiltMatrixBackend

# 3-way thresholds (round1 yaml judge: threshold / warm_tiers[0].threshold).
FULL = 0.997697
WARM = 0.997403

# Search params mirroring config/round1/cp1_libero10.yaml exactly.
W = {"vision_0": 0.25, "vision_1": 0.4375, "robot_state": 0.3125}
FIELD_SIM = {
    "vision_0": {"type": "cosine"},
    "vision_1": {"type": "cosine"},
    "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 1.0}},
}
SCORE_NORM = {
    "type": "per_field",
    "fields": {
        "vision_0": {"method": "zscore", "params": {"mu": 0.9739899923664463, "sigma": 0.0061831533438692935, "squash": "tanh"}},
        "vision_1": {"method": "zscore", "params": {"mu": 0.9659078322399228, "sigma": 0.006527797454113087, "squash": "tanh"}},
        "robot_state": {"method": "zscore", "params": {"mu": -1.9584325681212513, "sigma": 0.7484941685797242, "squash": "tanh"}},
    },
}


def verdict3(score) -> str:
    if score is None:
        return "MISS"
    if score >= FULL:
        return "FULL"
    if score >= WARM:
        return "WARM"
    return "MISS"


@dataclasses.dataclass
class Args:
    """Arguments for the ROUND 1 equivalence comparison."""

    # Round1 cache yaml (defines the library pkl to load).
    cache_config: str = "exp/cache_latency_bench/config/round1/cp1_libero10.yaml"
    # Output parity json.
    out: str = "exp/cache_latency_bench/data/round1/parity_result.json"
    # Optional per-task cap on LOO queries (0 = all). Use a small value for a smoke run.
    limit_per_task: int = 0
    # If True, keep the query entry in the library (self-included): candidates == whole
    # bucket in matrix-row order, which exercises the ZERO-COPY fast path. Default False
    # = leave-one-out (exercises the index_select gather path).
    self_included: bool = False


def main(args: Args) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.set_grad_enabled(False)

    config = load_cache_config(args.cache_config)
    components = build_cache_components(config)
    storage = components["storage"]
    old = storage._backend  # InMemoryBackend with the library loaded
    new = PrebuiltMatrixBackend(old.vector_dims)
    new.__dict__.update(old.__dict__)  # share the SAME _entries dict reference
    new._prebuild_task_matrices()
    assert new._entries is old._entries

    shared = old._entries  # mutating this dict affects BOTH backends (shared ref)

    task_ids: dict[tuple, list] = collections.defaultdict(list)
    for e in shared.values():
        task_ids[(e.checkpoint_id, e.payload.task_key)].append(e.id)

    n_q = 0
    winner_mismatch = 0
    verdict_flips = 0
    diffs: list[float] = []
    near_full = 0
    near_warm = 0
    flip_examples: list[dict] = []
    diff_examples: list[dict] = []

    for (ck, tk), ids in task_ids.items():
        use_ids = ids if args.limit_per_task <= 0 else ids[: args.limit_per_task]
        for qid in use_ids:
            q_entry = shared[qid]
            if not args.self_included:
                shared.pop(qid)  # LOO: remove from the shared library
            # Mimic the real keybuilder: only enabled fields are queried (prompt_emb /
            # vision_2 are disabled in the yaml, so they must not enter active fields —
            # _iter_active_fields defaults an unlisted weight to 1.0 and would over-include).
            query_keys = {f: q_entry.query_keys[f] for f in W if f in q_entry.query_keys}
            spec = QuerySpec(
                query_keys=query_keys,
                top_k=1,
                checkpoint_id=ck,
                filters=QueryFilter(task_key=tk),
                fusion_weights=W,
                fusion_method="weighted_score_sum",
                field_similarity=FIELD_SIM,
                score_normalization=SCORE_NORM,
            )
            ro = old.search(spec)
            rn = new.search(spec)
            if not args.self_included:
                shared[qid] = q_entry  # restore

            io = ro[0].id if ro else None
            so = ro[0].score if ro else None
            i_n = rn[0].id if rn else None
            sn = rn[0].score if rn else None

            n_q += 1
            if io != i_n:
                winner_mismatch += 1
                if len(flip_examples) < 8:
                    flip_examples.append({"task": tk[:30], "qid": qid, "old_id": io, "new_id": i_n,
                                          "old_score": so, "new_score": sn})
            vo, vn = verdict3(so), verdict3(sn)
            if vo != vn:
                verdict_flips += 1
            if so is not None and sn is not None:
                d = abs(so - sn)
                diffs.append(d)
                if d > 0 and len(diff_examples) < 8:
                    diff_examples.append({"task": tk[:30], "qid": qid, "diff": d, "old": so, "new": sn})
            if so is not None:
                if abs(so - FULL) <= 1e-4:
                    near_full += 1
                if abs(so - WARM) <= 1e-4:
                    near_warm += 1
        print(f"[parity] done task N(LOO)={len(use_ids)} {tk[:40]}", flush=True)

    diffs_sorted = sorted(diffs)
    out = {
        "n_q": n_q,
        "verdict_flips_3way": verdict_flips,         # HARD, must be 0
        "winner_id_mismatch": winner_mismatch,        # HARD, must be 0
        "score_abs_diff_max": (diffs_sorted[-1] if diffs_sorted else 0.0),
        "score_abs_diff_median": (statistics.median(diffs_sorted) if diffs_sorted else 0.0),
        "score_abs_diff_p99": (diffs_sorted[min(len(diffs_sorted) - 1, int(0.99 * len(diffs_sorted)))] if diffs_sorted else 0.0),
        "n_diff_nonzero": sum(1 for d in diffs if d > 0),
        "near_full_boundary_1e-4": near_full,
        "near_warm_boundary_1e-4": near_warm,
        "new_fast_path_hits": new._fast_hits,      # must be > 0 (proves fast path fired)
        "new_fallbacks": new._fallbacks,
        "flip_examples": flip_examples,
        "diff_examples": diff_examples,
        "PASS": (verdict_flips == 0 and winner_mismatch == 0),
        "cache_config": args.cache_config,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k not in ("flip_examples", "diff_examples")}, indent=2))
    print(f"[parity] {'PASS' if out['PASS'] else 'FAIL'} -> {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
