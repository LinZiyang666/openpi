#!/usr/bin/env python3
"""Does a trained router beat the best FIXED arm ratio?

    python router_vs_fixed.py --art-root <dir> --sweep <sweep.json> \
        --run l10_ts_lam1_s0 --run l10_ts_lam1_s0_knee [...]

This is the question M6 exists to answer, and it is easy to answer wrongly by
comparing two numbers that were not measured the same way. Two instruments,
both on the same pool, carry different confounds:

* the sweep -- one constant policy per ``p``, every point on the SAME
  ``sample_batch(batch_idx=0, seed=0)`` draw, so differences ACROSS p are
  paired and the pair composition cancels; but its absolute level is whatever
  that one subsample is worth.
* the runs  -- 4000 episodes over 40 different subsamples, so the level is the
  pool mean, but the share varies only because the policy drifted, and drift is
  confounded with training time.

So three readings are printed, and the caveat is printed with them rather than
left to the reader:

1. runs binned by realized share, against the sweep interpolated to that share;
2. whole-run mean against the sweep at the run's own mean share;
3. the BEST window of any run against the BEST measured fixed ratio -- the most
   generous form of the question. If state-dependence can win at all, the
   router's best stretch beats the best constant. A negative result *here* is
   the one that settles it.

Reading the level difference as "the router is worse" over-claims: the sweep's
absolute level carries a subsample offset the runs do not share. What survives
that is parity.
"""
import argparse
import json
import math
import pathlib
import sys


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (centre - half, centre + half)


def load_run(art_root: pathlib.Path, rid: str, arm: str) -> list[tuple[float, float]]:
    """(share of every arm except ``arm``, mean success) per batch."""
    path = art_root / rid / "metrics.jsonl"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        m = json.loads(line)
        out.append((1.0 - m["arm_executed_rate"].get(arm, float("nan")),
                    m["mean_success"]))
    return out


def interp(sweep: list[dict], p: float) -> float:
    xs = [r["p_realized"] for r in sweep]
    ys = [r["success_rate"] for r in sweep]
    if p <= xs[0]:
        return ys[0]
    if p >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if p <= xs[i]:
            t = (p - xs[i - 1]) / (xs[i] - xs[i - 1])
            return ys[i - 1] + t * (ys[i] - ys[i - 1])
    return ys[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--art-root", required=True, help="dir holding <run_id>/metrics.jsonl")
    ap.add_argument("--sweep", required=True, help="sweep_mixture.py's sweep.json")
    ap.add_argument("--run", action="append", required=True, help="run id; repeat")
    ap.add_argument("--cheap-arm", default="student",
                    help="the arm the sweep's p is measured AGAINST (p = 1 - share)")
    ap.add_argument("--episodes-per-batch", type=int, default=100)
    ap.add_argument("--window", type=int, default=10, help="batches in the 'best window' test")
    ap.add_argument("--bin-edges", default="0.15,0.25,0.30,0.35,0.40,0.45,0.60")
    args = ap.parse_args()

    art = pathlib.Path(args.art_root)
    sweep = json.loads(pathlib.Path(args.sweep).read_text(encoding="utf-8"))
    sweep = [r for r in sweep if r.get("success_rate") is not None]
    sweep.sort(key=lambda r: r["p_realized"])
    epb = args.episodes_per_batch

    print("FIXED-RATIO CURVE (constant policy; every point on the same (task,init) draw)")
    print(f"{'p_real':>8}{'n':>6}{'SR':>9}   95% CI")
    for r in sweep:
        lo, hi = wilson(r["successes"], r["episodes"])
        print(f"{r['p_realized']:>8.3f}{r['episodes']:>6}{r['success_rate']:>9.4f}"
              f"   [{lo:.3f}, {hi:.3f}]")
    best = max(sweep, key=lambda r: r["success_rate"])
    print(f"best fixed ratio: p={best['p_realized']:.3f}  SR={best['success_rate']:.4f}\n")

    runs = {rid: load_run(art, rid, args.cheap_arm) for rid in args.run}

    edges = [float(x) for x in args.bin_edges.split(",")]
    print("RUNS BINNED BY REALIZED SHARE  (vs the sweep at that share)")
    print(f"{'bin':>14}{'n_ep':>7}{'SR_router':>11}{'95% CI':>18}{'SR_fixed':>10}{'diff':>9}")
    allpts = [pt for v in runs.values() for pt in v]
    for lo_e, hi_e in zip(edges[:-1], edges[1:]):
        sel = [(t, s) for t, s in allpts if lo_e <= t < hi_e]
        if not sel:
            continue
        n = len(sel) * epb
        k = round(sum(s for _, s in sel) * epb)
        sr, lo, hi = k / n, *wilson(k, n)
        fixed = interp(sweep, sum(t for t, _ in sel) / len(sel))
        print(f"  [{lo_e:.2f},{hi_e:.2f}){n:>7}{sr:>11.4f}   [{lo:.3f}, {hi:.3f}]"
              f"{fixed:>10.4f}{sr - fixed:>+9.4f}")

    print("\nWHOLE-RUN MEAN vs FIXED RATIO AT THE SAME MEAN SHARE")
    print(f"{'run':>24}{'mean p':>9}{'SR_router':>11}{'95% CI':>18}{'SR_fixed':>10}{'diff':>9}")
    for rid, v in runs.items():
        n = len(v) * epb
        k = round(sum(s for _, s in v) * epb)
        sr, lo, hi = k / n, *wilson(k, n)
        pm = sum(t for t, _ in v) / len(v)
        fixed = interp(sweep, pm)
        print(f"{rid:>24}{pm:>9.4f}{sr:>11.4f}   [{lo:.3f}, {hi:.3f}]"
              f"{fixed:>10.4f}{sr - fixed:>+9.4f}")

    w = args.window
    print(f"\nBEST {w}-BATCH ({w * epb} ep) WINDOW OF ANY RUN  vs  BEST FIXED RATIO")
    bw = None
    for rid, v in runs.items():
        for i in range(len(v) - w + 1):
            seg = v[i:i + w]
            sr = sum(s for _, s in seg) / w
            if bw is None or sr > bw[0]:
                bw = (sr, rid, i, sum(t for t, _ in seg) / w)
    if bw is None:
        print("  (no run long enough for the window)")
        return 1
    sr, rid, i, pm = bw
    n = w * epb
    lo, hi = wilson(round(sr * n), n)
    blo, bhi = wilson(best["successes"], best["episodes"])
    se = math.sqrt(sr * (1 - sr) / n
                   + best["success_rate"] * (1 - best["success_rate"]) / best["episodes"])
    d = sr - best["success_rate"]
    print(f"  router best window : {rid} b{i:04d}-b{i + w - 1:04d}  p={pm:.3f}  "
          f"SR={sr:.4f}  [{lo:.3f}, {hi:.3f}]")
    print(f"  best fixed ratio   : p={best['p_realized']:.3f}  "
          f"SR={best['success_rate']:.4f}  [{blo:.3f}, {bhi:.3f}]")
    print(f"  difference         : {d:+.4f} +/- {se:.4f}  (z = {d / se:+.2f})")

    print("\nread: only a positive, significant difference in the third block would show "
          "state-dependence buying what a constant cannot. Parity means the router is an "
          "expensive constant. Do NOT read the level differences as 'the router is worse' "
          "-- the sweep's absolute level carries a subsample offset the runs do not share; "
          "what survives that is parity.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
