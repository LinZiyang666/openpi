"""ROUND-RRF R1 equivalence — weighted_rrf winner-id parity (PrebuiltMatrix/PrenormDot vs stock).

weighted_rrf shares the SAME ``_compute_field_scores`` hot kernel as weighted_sum, so the
shipped R1 (PrebuiltMatrix) / R2 (PrenormDot) backends apply to RRF automatically. But RRF
fuses by ``argsort`` rank (not normalize+sum), so prenorm's ~2e-6 error can swap a near-tie
rank → change the RRF score → change the winner. Equivalence is therefore proven by
**WINNER-ID PARITY (HARD)**, NOT by a geometric margin (which fails for RRF — discrete ties).

- R1 (``--backend prebuilt``): original ``F.cosine_similarity`` → argsort order bit-identical
  → RRF score bit-identical → winner bit-identical (expect rrf_score_abs_diff == 0).
- R2 (``--backend prenorm``): GEMV cosine ~2e-6 → winner-id parity expected, rrf_score_abs_diff
  ~1e-4 monitored, winner must hold (0 mismatch).

Usage::
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/compare_rrf_equivalence.py \\
        --backend prebuilt --threads 4 --out exp/cache_latency_bench/data/round_rrf/parity_r1.json
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

from exp.cache_latency_bench.opt.inject import attach_lean_rrf, attach_prebuilt, attach_prenorm_dot

W = {"vision_0": 0.0625, "vision_1": 0.5, "robot_state": 0.4375}
FIELD_SIM = {
    "vision_0": {"type": "cosine"}, "vision_1": {"type": "cosine"},
    "robot_state": {"type": "l2", "to_similarity": {"type": "exp", "tau": 1.0}},
}
RRF_K = 60


@dataclasses.dataclass
class Args:
    """Arguments for the weighted_rrf winner-id equivalence harness."""

    cache_config: str = "exp/cache_latency_bench/config/round_rrf/cp1_libero10_rrf_kin.yaml"
    out: str = "exp/cache_latency_bench/data/round_rrf/parity_r1.json"
    backend: str = "prebuilt"   # "prebuilt" (R1 bit-equal) | "prenorm" (R2 winner parity)
    breadth: int = 15
    self_included: bool = False
    threads: int = 4


def main(args: Args) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)

    cfg = load_cache_config(args.cache_config)
    comp = build_cache_components(cfg)
    storage = comp["storage"]
    old = storage._backend  # stock InMemoryBackend (reference)
    if args.backend == "prenorm":
        new = attach_prenorm_dot(storage)
    elif args.backend == "lean_rrf":
        new = attach_lean_rrf(storage)
    else:
        new = attach_prebuilt(storage)
    assert new._entries is old._entries
    shared = old._entries

    task_ids = collections.defaultdict(list)
    for e in shared.values():
        task_ids[(e.checkpoint_id, e.payload.task_key)].append(e.id)
    max_n = max(len(v) for v in task_ids.values())
    subset = set()
    for (ck, tk), ids in task_ids.items():
        for i in ids[: args.breadth]:
            subset.add(i)
        if len(ids) == max_n:  # biggest bucket fully (N=399 boundary)
            subset.update(ids)

    id_meta = {e.id: (e.checkpoint_id, e.payload.task_key) for e in shared.values()}
    n_q = 0
    winner_mismatch = 0
    diffs: list[float] = []

    for qid in subset:
        ck, tk = id_meta[qid]
        qe = shared[qid]
        if not args.self_included:
            shared.pop(qid)
        qk = {f: qe.query_keys[f] for f in W}
        spec = QuerySpec(
            query_keys=qk, top_k=1, checkpoint_id=ck, filters=QueryFilter(task_key=tk),
            fusion_weights=W, fusion_method="weighted_rrf", field_similarity=FIELD_SIM,
            backend_hints={"rrf_k": RRF_K},
        )
        ro = old.search(spec)
        rn = new.search(spec)
        if not args.self_included:
            shared[qid] = qe
        io = ro[0].id if ro else None
        so = ro[0].score if ro else None
        i_n = rn[0].id if rn else None
        sn = rn[0].score if rn else None
        n_q += 1
        if io != i_n:
            winner_mismatch += 1
        if so is not None and sn is not None:
            diffs.append(abs(so - sn))

    diffs.sort()
    if args.backend == "lean_rrf":
        hits = new._lean_rrf_hits if args.self_included else new._prenorm_hits
    elif args.backend == "prenorm":
        hits = new._prenorm_hits
    else:
        hits = new._fast_hits
    assert hits and hits > 0, "fast path never fired — 0-diff would be a false positive"
    out = {
        "n_q": n_q,
        "backend": args.backend,
        "winner_id_mismatch": winner_mismatch,             # HARD, must be 0
        "rrf_score_abs_diff_max": diffs[-1] if diffs else 0.0,    # R1 expect 0, R2 ~1e-4
        "rrf_score_abs_diff_median": statistics.median(diffs) if diffs else 0.0,
        "n_diff_nonzero": sum(1 for d in diffs if d > 0),
        "new_fast_or_prenorm_hits": hits,
        "PASS": winner_mismatch == 0,
        "path": "self_included" if args.self_included else "LOO",
        "threads": args.threads,
        "cache_config": args.cache_config,
    }
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[rrf-parity-{args.backend}] {'PASS' if out['PASS'] else 'FAIL'} (winner_mismatch={winner_mismatch}) -> {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
