"""ROUND 2 equivalence — PrenormDotBackend vs src InMemoryBackend on a near-threshold subset.

Round 2 changed the numeric path (cosine -> prenorm GEMV), so equivalence is NOT
bit-identical; it is proven by 3-way verdict + winner parity. Per owner, this runs a
near-threshold SUBSET (not full 2640): only queries whose reference fused score sits within
the amplified error band of a threshold can flip, so the subset = near-boundary queries
(the only flip-eligible set) ∪ per-task breadth ∪ the hardest bucket (N=399).

Two-pass: PASS-1 cheaply scores every LOO query with the prenorm GEMV (≈cosine, fast) to
pick the subset; PASS-2 runs the subset through the real ``backend.search(spec)`` for BOTH
old (InMemoryBackend, F.cosine_similarity) and new (PrenormDotBackend, mv), comparing the
3-way verdict, winner id, score diff, and a geometric safety margin
(actual_fused_err - boundary_gap, must be < 0). ``--threads`` pins set_num_threads so the
verified config matches the shipped runner; run it at several thread counts to confirm the
0-flip result is not schedule-specific. Asserts new._prenorm_hits>0 (else a 0-diff is a
false positive — both sides fell back to the parent).

Usage::
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/compare_prenorm_equivalence.py \\
        --near-threshold --threads 4 --out exp/cache_latency_bench/data/round2/parity_subset_loo_t4.json
"""

from __future__ import annotations

import collections
import dataclasses
import json
import os
import statistics

import torch
import torch.nn.functional as F
import tyro

from openpi.cache.components.score_normalizers import build_field_normalizers
from openpi.cache.config import build_cache_components, load_cache_config
from openpi.cache.storage_types import QueryFilter, QuerySpec

from exp.cache_latency_bench.opt.inject import attach_prenorm_dot

FULL = 0.997697
WARM = 0.997403
BAND = 1e-5  # 10x the measured fused-err ceiling (9.54e-7, G2-verified) — covers every
# flip-eligible query with a 10x margin while keeping the subset small (~70-120 vs 1309).

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
_ACTIVE = [(f, W[f], FIELD_SIM[f]) for f in ("vision_0", "vision_1", "robot_state")]


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
    """Arguments for the ROUND 2 near-threshold equivalence comparison."""

    cache_config: str = "exp/cache_latency_bench/config/round2/cp1_libero10.yaml"
    task_pack: str = "exp/cache_latency_bench/data/opt_bench/task_pack.pt"
    out: str = "exp/cache_latency_bench/data/round2/parity_subset.json"
    # Restrict to the near-threshold subset (default True). False = every query (slow).
    near_threshold: bool = True
    # Per-task breadth (queries per task always included for distribution coverage).
    breadth: int = 5
    # Pin CPU threads so the verified config matches the shipped runner.
    threads: int = 4
    # self-included (zero-copy resident-norm path) vs LOO (index_select path).
    self_included: bool = False
    # Which optimized backend to verify vs the reference: "prenorm" or "lean".
    # "lean" only fires on whole-bucket candidates, so pair it with --self-included.
    backend: str = "prenorm"


def _select_subset(pack, breadth):
    """PASS-1: pick the near-threshold subset via the cheap prenorm GEMV reference."""
    norm = build_field_normalizers(_ACTIVE, SCORE_NORM, FIELD_SIM)

    def fuse(c0, c1, l2):
        return W["vision_0"] * norm["vision_0"](c0) + W["vision_1"] * norm["vision_1"](c1) + W["robot_state"] * norm["robot_state"](l2)

    subset: set = set()
    for tk, d in pack.items():
        v0, v1, rs, ids = d["vision_0"], d["vision_1"], d["robot_state"], d["ids"]
        n = v0.shape[0]
        v0n = F.normalize(v0, dim=1)
        v1n = F.normalize(v1, dim=1)
        for i in range(min(breadth, n)):  # per-task breadth (sanity / distribution)
            subset.add(ids[i])
        for i in range(n):  # near-threshold population — the ONLY flip-eligible set
            sel = torch.ones(n, dtype=torch.bool)
            sel[i] = False
            c0 = v0n[sel].mv(v0n[i])
            c1 = v1n[sel].mv(v1n[i])
            l2 = torch.norm(rs[i].unsqueeze(0) - rs[sel], p=2, dim=1)
            ts = float(fuse(c0, c1, l2).max())
            if min(abs(ts - FULL), abs(ts - WARM)) < BAND:
                subset.add(ids[i])
    return subset


def main(args: Args) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.set_grad_enabled(False)
    torch.set_num_threads(args.threads)

    config = load_cache_config(args.cache_config)
    components = build_cache_components(config)
    storage = components["storage"]
    old = storage._backend  # InMemoryBackend reference (F.cosine_similarity)
    if args.backend == "lean":
        from exp.cache_latency_bench.opt.inject import attach_lean_search
        new = attach_lean_search(storage)  # LeanSearchBackend (whole-bucket lean path)
    else:
        new = attach_prenorm_dot(storage)  # PrenormDotBackend (mv); shares the same _entries
    assert new._entries is old._entries
    shared = old._entries

    # PASS-1 subset selection (cheap).
    if args.near_threshold:
        pack = torch.load(args.task_pack, map_location="cpu", weights_only=False)
        subset_ids = _select_subset(pack, args.breadth)
    else:
        subset_ids = set(shared.keys())

    id_to_meta = {e.id: (e.checkpoint_id, e.payload.task_key) for e in shared.values()}
    todo = [qid for qid in subset_ids if qid in id_to_meta]

    n_q = winner_mismatch = verdict_flips = 0
    diffs: list[float] = []
    safety_max = -1e9  # max(actual_err - boundary_gap); must stay < 0
    min_margin = 1e9
    flip_examples: list[dict] = []

    for qid in todo:
        ck, tk = id_to_meta[qid]
        q_entry = shared[qid]
        if not args.self_included:
            shared.pop(qid)
        query_keys = {f: q_entry.query_keys[f] for f in W if f in q_entry.query_keys}
        spec = QuerySpec(
            query_keys=query_keys, top_k=1, checkpoint_id=ck,
            filters=QueryFilter(task_key=tk), fusion_weights=W,
            fusion_method="weighted_score_sum", field_similarity=FIELD_SIM,
            score_normalization=SCORE_NORM,
        )
        ro = old.search(spec)
        rn = new.search(spec)
        if not args.self_included:
            shared[qid] = q_entry

        io = ro[0].id if ro else None
        so = ro[0].score if ro else None
        i_n = rn[0].id if rn else None
        sn = rn[0].score if rn else None

        n_q += 1
        if io != i_n:
            winner_mismatch += 1
            if len(flip_examples) < 10:
                flip_examples.append({"task": tk[:30], "qid": qid, "old_id": io, "new_id": i_n, "old": so, "new": sn})
        if verdict3(so) != verdict3(sn):
            verdict_flips += 1
        if so is not None and sn is not None:
            err = abs(so - sn)
            diffs.append(err)
            gap = min(abs(so - FULL), abs(so - WARM))
            min_margin = min(min_margin, gap)
            safety_max = max(safety_max, err - gap)  # < 0 => error can't cross a threshold

    diffs.sort()
    if args.backend == "lean" and args.self_included:
        assert new._lean_hits > 0, "lean path never fired — 0-diff would be a false positive"
    else:
        assert new._prenorm_hits > 0, "prenorm GEMV never fired — 0-diff would be a false positive"
    out = {
        "n_q": n_q,
        "subset_size": len(todo),
        "threads": args.threads,
        "path": "self_included(zero-copy)" if args.self_included else "LOO(index_select)",
        "verdict_flips_3way": verdict_flips,      # HARD, must be 0
        "winner_id_mismatch": winner_mismatch,     # HARD, must be 0
        "score_abs_diff_max": (diffs[-1] if diffs else 0.0),       # SOFT, expect ~1e-4
        "score_abs_diff_median": (statistics.median(diffs) if diffs else 0.0),
        "score_abs_diff_p99": (diffs[min(len(diffs) - 1, int(0.99 * len(diffs)))] if diffs else 0.0),
        "geometric_safety_max": safety_max,        # max(err - margin); < 0 => provably no flip
        "min_boundary_margin": min_margin,
        "new_prenorm_hits": new._prenorm_hits,
        "new_fallbacks": new._fallbacks,
        "new_lean_hits": getattr(new, "_lean_hits", None),
        "new_lean_fallbacks": getattr(new, "_lean_fallbacks", None),
        "backend": args.backend,
        "band": BAND,
        "PASS": (verdict_flips == 0 and winner_mismatch == 0),
        "cache_config": args.cache_config,
    }
    with open(args.out, "w") as fh:
        json.dump({**out, "flip_examples": flip_examples}, fh, indent=2)
    print(json.dumps(out, indent=2))
    print(f"[prenorm-parity] {'PASS' if out['PASS'] else 'FAIL'}  threads={args.threads}  "
          f"path={out['path']}  safety_max={safety_max:.2e}  -> {args.out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
