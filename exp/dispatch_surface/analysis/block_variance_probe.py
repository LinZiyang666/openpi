"""Empirical block variance -- the evidence that retired the cost bench.

HISTORICAL. The isolated cost bench, its block design and its power simulation
were deleted on 2026-08-27; cost is now computed analytically from the
precheck's own per_step journal (see analyze_precheck.py, and sections 9-10 of
logs/dispatch_surface_cost_axis_change.md). This probe is kept because it is
the measurement those decisions rested on, and it still reproduces its own
figures -- the few constants it needed from the deleted power module are
inlined below.

The quantity: the SD, across blocks, of the RELATIVE difference between two
arms' per-decision cost, which is what the block design's power simulation
consumed. Nothing in the frozen pipeline derived that number -- plan section
4.6 said to take it from the E0 latency bench, but E0 is an in-process
microbenchmark of single forward calls and has no block structure at all, so
it cannot supply a block SD.

This probe measures the quantity directly on the gate-threshold-pareto sweep,
which is the only dataset with the right shape: 32 arms evaluated on one
shared A-pool, so arms are paired within an init set exactly as the cost bench
pairs them, with a real per-episode verdict mix.

**Cost axis = GPU inference ratio.** Per decision:

    FULL_HIT    stage1
    WARM_START  stage1 + stage2 + start_t * stage3   (start_t=0.3 -> 30%)
    MISS        stage1 + stage2 + stage3

with the CUDA-graph stage costs (the optimised build). Retrieval occupies no
GPU time and is not part of the cost. The verdict mix is measured; E0 supplies
the unit costs.

The sweep's own ``client_timing.infer_ms`` is carried only as a contrast, never
as the cost: gtp ran 4 server replicas against 64 client workers, so its
per-decision figure (~1.8 s against E0's ~0.07 s CUDA-graph forward) is
dominated by queueing that the serial, single-worker cost bench will not have.
It is reported because its block SD lands close to the synthesised one, which
is itself the finding -- queueing is not what drives block variance, episode
composition is, and that source does not disappear with the queue.

Sections:
  1. paired arm-pair SD over all pairs (the quantity sigma names)
  2. noise floor: one arm against itself on disjoint init sets
  3. scaling with block size
  4. the R the frozen gate would need at the measured sigma

Usage:
  uv run python -m exp.dispatch_surface.analysis.block_variance_probe \
      --per-step exp/gate_threshold_pareto/data/eval/libero_spatial/per_step.jsonl \
      --out exp/dispatch_surface/analysis/block_variance_probe.json
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import math
import pathlib
import statistics

# E0 latency_bench, RTX 4090, CUDA-graph build -- the optimised system, per
# owner instruction. The eager and default rows describe a system that has not
# been optimised and would understate how cheap a FULL_HIT is relative to a
# MISS, which is exactly the ratio this probe turns on.
STAGE1_MS = 10.26
STAGE2_MS = 27.69
STAGE3_MS = 29.57

# Frozen gate constants, mirrored from power_sim_cost_blocks.GATES so this
# probe reports the R those exact gates would need.
LATENCY_EFFECT = 0.02   # |true effect| against a 0% threshold
COMPUTE_MARGIN = 0.05   # -10% true effect against a -5% threshold
Z_GATE1 = 1.6449        # p95
Z_GATE2 = 1.9600        # p97.5
Z_POWER = 0.8416        # 80%

# A' pool budget. run_cost_bench.materialize_block_pools requires DISTINCT
# inits per block and reserves the last A' position for warmup, so a design
# using k inits per task per block admits at most floor(USABLE_INITS / k)
# blocks. This is a hard ceiling on R that no amount of episodes can lift.
APRIME_INITS_PER_TASK = 30
USABLE_INITS_PER_TASK = APRIME_INITS_PER_TASK - 1


def decision_compute_ms(hit_type: str | None, start_t) -> float:
    """Per-decision GPU cost implied by one verdict.

    FULL_HIT pays stage1 only. WARM_START resumes the flow at start_t and
    steps down to 0, so it runs ``start_t`` of stage3 -- ``run_stage3_from``
    executes ``round(start_t * num_steps)`` steps, i.e. start_t=0.3 runs 3 of
    10 and saves 70% (pi0_pytorch.py:691). MISS, and any step the gate never
    probed, pays all three stages.
    """
    if hit_type == "FULL_HIT":
        return STAGE1_MS
    if hit_type == "WARM_START":
        frac = float(start_t if start_t is not None else 1.0)
        return STAGE1_MS + STAGE2_MS + frac * STAGE3_MS
    return STAGE1_MS + STAGE2_MS + STAGE3_MS




def load_episodes(per_step_path: pathlib.Path) -> list[dict]:
    """Reduce a per_step journal to one cost row per (arm, task, init)."""
    verdict_counts: dict[str, collections.Counter] = collections.defaultdict(
        collections.Counter
    )
    compute_sum: dict[str, float] = collections.defaultdict(float)
    timing: dict[str, dict] = {}
    for line in open(per_step_path):
        row = json.loads(line)
        if row.get("_kind") == "client_timing":
            timing[row["task_uid"]] = row
            continue
        uid = row.get("task_uid")
        if uid is None:
            continue
        verdict_counts[uid][row.get("hit_type") or "UNPROBED"] += 1
        compute_sum[uid] += decision_compute_ms(row.get("hit_type"), row.get("start_t"))

    episodes = []
    for uid, counts in sorted(verdict_counts.items()):
        t = timing.get(uid)
        if t is None:
            continue
        arm, _phase, task_id, init_idx = uid.split(":")
        decisions = sum(counts.values())
        infers = int(t.get("infers") or 0)
        if decisions == 0 or infers == 0:
            continue
        episodes.append({
            "arm": arm,
            "task_id": int(task_id),
            "init_idx": int(init_idx),
            "compute_ms": compute_sum[uid] / decisions,
            "measured_client_ms": float(t["infer_ms"]) / infers,
            "success": bool(t.get("success")),
        })
    return episodes


class Panel:
    """Arms x (task, init) cost panel with block construction."""

    def __init__(self, episodes: list[dict]):
        self.by_arm: dict[str, dict] = collections.defaultdict(dict)
        for e in episodes:
            self.by_arm[e["arm"]][(e["task_id"], e["init_idx"])] = e
        self.arms = sorted(self.by_arm)
        self.cells = sorted(set.intersection(*(set(d) for d in self.by_arm.values())))
        self.tasks = sorted({t for t, _ in self.cells})
        self.per_task = {
            t: sorted(i for (tt, i) in self.cells if tt == t) for t in self.tasks
        }
        self.max_inits = min(len(v) for v in self.per_task.values())

    def blocks(self, inits_per_task: int = 1, offset: int = 0, stride: int = 1):
        """Task-stratified blocks, the construction run_cost_bench uses."""
        out = []
        k = 0
        while True:
            cells = []
            for j in range(inits_per_task):
                idx = offset + (k * inits_per_task + j) * stride
                if any(idx >= len(self.per_task[t]) for t in self.tasks):
                    return out
                cells.extend((t, self.per_task[t][idx]) for t in self.tasks)
            out.append(cells)
            k += 1

    def mean(self, arm: str, cells, field: str) -> float:
        return statistics.fmean(self.by_arm[arm][c][field] for c in cells)

    def rel_diffs(self, a: str, b: str, blocks, field: str) -> list[float]:
        out = []
        for cells in blocks:
            ca, cb = self.mean(a, cells, field), self.mean(b, cells, field)
            out.append((ca - cb) / cb)
        return out


def needed_r(sigma: float, effect: float, z_gate: float) -> int:
    """Blocks for 80% power under the normal approximation to the bootstrap."""
    return math.ceil(((z_gate + Z_POWER) * sigma / effect) ** 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--per-step",
        default="exp/gate_threshold_pareto/data/eval/libero_spatial/per_step.jsonl",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    episodes = load_episodes(pathlib.Path(args.per_step))
    panel = Panel(episodes)
    blocks = panel.blocks()
    report: dict = {
        "source": args.per_step,
        "arms": len(panel.arms),
        "shared_cells": len(panel.cells),
        "tasks": len(panel.tasks),
        "blocks_at_10ep": len(blocks),
        "e0_stage_ms": {"stage1": STAGE1_MS, "stage2": STAGE2_MS, "stage3": STAGE3_MS},
    }
    print(f"arms={report['arms']} shared cells={report['shared_cells']} "
          f"blocks(10 ep)={report['blocks_at_10ep']}")

    # 1. Arm-pair SD ---------------------------------------------------------
    report["arm_pair_sd"] = {}
    pair_stats = {}
    for field in ("compute_ms", "measured_client_ms"):
        sds = []
        for a, b in itertools.combinations(panel.arms, 2):
            d = panel.rel_diffs(a, b, blocks, field)
            sds.append((statistics.stdev(d), abs(statistics.fmean(d)), a, b))
        sds.sort()
        vals = [s for s, _, _, _ in sds]
        q = statistics.quantiles(vals, n=100)
        report["arm_pair_sd"][field] = {
            "pairs": len(sds), "min": vals[0], "p25": q[24], "median": q[49],
            "p75": q[74], "p90": q[89], "p95": q[94], "max": vals[-1],
        }
        pair_stats[field] = sds
        print(f"  {field}: pair SD median={q[49]:.4f} p90={q[89]:.4f} max={vals[-1]:.4f}")

    # Near-identical pairs: the closest analogue to the nested SV-vs-S0 test,
    # where the true difference is near zero and only noise remains.
    near = sorted(pair_stats["compute_ms"], key=lambda x: x[1])[:5]
    report["near_identical_pairs"] = [
        {"a": a, "b": b, "mean_abs_rel_gap": gap, "block_sd_compute": sd}
        for sd, gap, a, b in near
    ]

    # 2. Noise floor ---------------------------------------------------------
    half = panel.max_inits // 2
    even = panel.blocks(offset=0, stride=2)[:half]
    odd = panel.blocks(offset=1, stride=2)[:half]
    report["noise_floor_same_arm"] = {}
    for field in ("compute_ms", "measured_client_ms"):
        sds = []
        for arm in panel.arms:
            diffs = [(panel.mean(arm, x, field) - panel.mean(arm, y, field))
                     / panel.mean(arm, y, field) for x, y in zip(even, odd)]
            sds.append(statistics.stdev(diffs))
        report["noise_floor_same_arm"][field] = {
            "min": min(sds), "median": statistics.median(sds), "max": max(sds),
        }
        print(f"  noise floor {field}: median={statistics.median(sds):.4f}")

    # 3. Block-size scaling --------------------------------------------------
    scaling = []
    probe_pairs = [(a, b) for _sd, _gap, a, b in near]
    for ipt in (1, 2, 3, 4, 6, 8, 12):
        row = {"block_episodes": ipt * len(panel.tasks)}
        for field in ("compute_ms", "measured_client_ms"):
            blks = panel.blocks(inits_per_task=ipt)
            sds = [statistics.stdev(panel.rel_diffs(a, b, blks, field))
                   for a, b in probe_pairs if len(blks) >= 3]
            row[f"sigma_{field}"] = statistics.median(sds) if sds else None
            row["blocks"] = len(blks)
        scaling.append(row)
    report["block_size_scaling"] = scaling

    # 4. R the gate would need ----------------------------------------------
    at10 = scaling[0]
    sc = at10["sigma_compute_ms"]
    report["required_r_at_frozen_block"] = {
        "sigma_compute": sc,
        "gate1_compute": needed_r(sc, COMPUTE_MARGIN, Z_GATE1),
        "gate2_compute": needed_r(sc, COMPUTE_MARGIN, Z_GATE2),
        "frozen_r_candidates": [5, 10, 15],
        "note": "cost axis is the GPU inference ratio",
    }
    print("\nR needed on the GPU-inference-ratio axis, 10-episode block:")
    for k in ("gate1_compute", "gate2_compute"):
        print(f"  {k:15s} {report['required_r_at_frozen_block'][k]}")

    # 5. Feasible (block, R) combinations ------------------------------------
    # The frozen design fixes a block at 10 episodes and R in {5,10,15}; the
    # block size is the other lever, so report which pairs actually clear the
    # 80% bar and what each costs in episodes.
    feasible = []
    for row in scaling:
        s_c = row["sigma_compute_ms"]
        if s_c is None:
            continue
        need1 = needed_r(s_c, COMPUTE_MARGIN, Z_GATE1)
        need2 = needed_r(s_c, COMPUTE_MARGIN, Z_GATE2)
        need = max(need1, need2)
        feasible.append({
            "block_episodes": row["block_episodes"],
            "sigma_compute": s_c,
            "required_r": need,
            "episodes_total": 5 * need * row["block_episodes"] * 2,
            "within_frozen_r": need <= 15,
        })
    report["feasible_designs"] = feasible
    print("\nblock size vs required R (5 arms, 2 passes):")
    for row in feasible:
        mark = "  <= frozen R" if row["within_frozen_r"] else ""
        print(f"  block={row['block_episodes']:3d} ep  sigma={row['sigma_compute']:.4f}  "
              f"R>={row['required_r']:3d}  total={row['episodes_total']:6d} ep{mark}")

    # 6. Robustness: compute is synthesised, so its sigma depends on the
    # assumed FULL_HIT share of a MISS (stage1 / all three stages). The
    # CUDA-graph build puts that at STAGE1/(STAGE1+STAGE2+STAGE3); the
    # conclusion has to hold across the plausible range rather than at one
    # point, since the production build was never measured (E0 caveat).
    probe_pairs = [(a, b) for _sd, _gap, a, b in near]
    cuda_share = STAGE1_MS / (STAGE1_MS + STAGE2_MS + STAGE3_MS)
    stage1_share_scan = []
    for share in sorted({0.10, 0.15, round(cuda_share, 4), 0.20, 0.237, 0.30, 0.45}):
        sds = []
        for a, b in probe_pairs:
            diffs = []
            for cells in blocks:
                means = []
                for arm in (a, b):
                    vals = []
                    for cell in cells:
                        e = panel.by_arm[arm][cell]
                        # e["compute_ms"] is a blend of STAGE1_MS and the full
                        # cost; invert to the FULL_HIT fraction, re-blend at
                        # the scanned share.
                        lo, hi = STAGE1_MS, STAGE1_MS + STAGE2_MS + STAGE3_MS
                        frac_hit = (hi - e["compute_ms"]) / (hi - lo)
                        vals.append(frac_hit * share + (1 - frac_hit) * 1.0)
                    means.append(statistics.fmean(vals))
                diffs.append((means[0] - means[1]) / means[1])
            sds.append(statistics.stdev(diffs))
        stage1_share_scan.append({
            "stage1_share_of_miss": share,
            "sigma_compute": statistics.median(sds),
            "required_r": needed_r(statistics.median(sds), COMPUTE_MARGIN, Z_GATE2),
            "is_cuda_graph_build": abs(share - cuda_share) < 1e-6,
        })
    report["stage1_share_sensitivity"] = stage1_share_scan
    report["cuda_graph_stage1_share"] = cuda_share

    print(f"\nstage1 share of a MISS: CUDA-graph build = {cuda_share:.3f}")
    for row in stage1_share_scan:
        mark = "  <- CUDA-graph" if row["is_cuda_graph_build"] else ""
        print(f"  share={row['stage1_share_of_miss']:.3f} -> "
              f"sigma={row['sigma_compute']:.4f}  R>={row['required_r']}{mark}")

    # 7. Feasibility inside the A' pool --------------------------------------
    # The normal approximation above says how many blocks the gate needs; this
    # says how many the pool can actually supply, and runs the FROZEN simulator
    # at that ceiling so the answer is the one the real power record would give.
    import numpy as np

    # Inlined from the deleted power_sim_cost_blocks (the block design it
    # served was removed with the cost bench). Kept verbatim so this probe --
    # the evidence behind that removal -- still reproduces its own figures.
    POWER_SEED, POWER_N_SIM, POWER_N_BOOT, POWER_TARGET = 20260827, 2000, 500, 0.80
    compute_gates = [
        ("gate1_compute", -0.10, -0.05, 0.95, "compute"),
        ("gate2_compute", 0.00, 0.05, 0.975, "compute"),
    ]

    def gate_power(r, true_effect, threshold, quantile, sigma, rng, n_sim, n_boot):
        passes = 0
        for _ in range(n_sim):
            diffs = true_effect + rng.normal(0.0, sigma, size=r)
            idx = rng.integers(0, r, size=(n_boot, r))
            if np.quantile(diffs[idx].mean(axis=1), quantile) <= threshold:
                passes += 1
        return passes / n_sim

    def powers_at(r: int, sigma: float) -> dict:
        rng = np.random.default_rng(POWER_SEED)
        return {n: gate_power(r, eff, thr, q, sigma, rng, POWER_N_SIM, POWER_N_BOOT)
                for n, eff, thr, q, _a in compute_gates}

    feasibility = []
    for row in scaling:
        sigma_row = row["sigma_compute_ms"]
        if sigma_row is None:
            continue
        k = row["block_episodes"] // len(panel.tasks)
        r_ceiling = USABLE_INITS_PER_TASK // k
        if r_ceiling < 2:
            continue
        at_ceiling = powers_at(r_ceiling, sigma_row)
        # Smallest R that clears the bar, ignoring the pool ceiling.
        r_needed = None
        for r in range(2, 80):
            if all(p >= POWER_TARGET for p in powers_at(r, sigma_row).values()):
                r_needed = r
                break
        feasibility.append({
            "block_episodes": row["block_episodes"],
            "inits_per_task_per_block": k,
            "sigma_compute": sigma_row,
            "r_ceiling_from_pool": r_ceiling,
            "power_at_ceiling": at_ceiling,
            "fits_in_pool": all(p >= POWER_TARGET for p in at_ceiling.values()),
            "r_needed": r_needed,
            "inits_per_task_needed": None if r_needed is None else k * r_needed,
            "inits_per_task_short": (
                None if r_needed is None
                else max(0, k * r_needed - USABLE_INITS_PER_TASK)
            ),
        })
    report["aprime_feasibility"] = {
        "aprime_inits_per_task": APRIME_INITS_PER_TASK,
        "usable_inits_per_task": USABLE_INITS_PER_TASK,
        "power_target": POWER_TARGET,
        "rows": feasibility,
        "any_design_fits": any(r["fits_in_pool"] for r in feasibility),
    }

    print(f"\nA' pool feasibility ({USABLE_INITS_PER_TASK} usable inits/task, "
          f"blocks need distinct inits):")
    print(f"  {'block':>6s} {'k':>3s} {'R_max':>6s} {'gate1':>7s} {'gate2':>7s} "
          f"{'R_need':>7s} {'short':>6s}")
    for row in feasibility:
        pw = row["power_at_ceiling"]
        short = row["inits_per_task_short"]
        print(f"  {row['block_episodes']:6d} {row['inits_per_task_per_block']:3d} "
              f"{row['r_ceiling_from_pool']:6d} {pw['gate1_compute']:7.3f} "
              f"{pw['gate2_compute']:7.3f} {str(row['r_needed']):>7s} "
              f"{'-' if not short else f'+{short}':>6s}")
    if not report["aprime_feasibility"]["any_design_fits"]:
        best = min((r for r in feasibility if r["inits_per_task_short"] is not None),
                   key=lambda r: r["inits_per_task_short"])
        print(f"  NO design reaches {POWER_TARGET:.0%} inside the pool; closest is "
              f"block={best['block_episodes']} ep needing "
              f"{best['inits_per_task_needed']} inits/task "
              f"(+{best['inits_per_task_short']} over the pool)")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
