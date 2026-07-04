"""Experiment A: raw-score distribution anatomy + percentile-collapse mechanics.

Per (suite, builder, field) this quantifies, on exact all-pairs data:
  (1) calibration-pool mismatch: legacy random-pair pool (includes
      intra-trajectory near-duplicates) vs the LOEO serving pool — KS distance,
      percentile shifts, and the intra-trajectory share of the right tail;
  (2) the legacy percentile band fitted both ways (p5/p95/denom) and what it
      does to the serving distribution: saturation fractions, output std,
      same/cross separation, tie-aware AUC, and top-10 ceiling-tie rate;
  (3) the same diagnostics for zscore+tanh, plus local gain (slope) at the
      serving median for both maps — the amplify-vs-saturate asymmetry.

Output: expA_results.json (+ human-readable dump to stdout).

Usage:
    uv run exp/weighted_sum/analysis/fusion_theory/expA_distributions.py \
        --cache-dir <scores-cache> --out exp/weighted_sum/analysis/fusion_theory/data
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fusion_theory_common as C  # noqa: E402


def ks_stat(a: np.ndarray, b: np.ndarray, grid: int = 4096) -> float:
    """Two-sample KS statistic on a quantile grid (exact enough at n>=1e5)."""
    qs = np.linspace(0.0, 1.0, grid)
    xs = np.unique(np.concatenate([np.quantile(a, qs), np.quantile(b, qs)]))
    fa = np.searchsorted(np.sort(a), xs, side="right") / len(a)
    fb = np.searchsorted(np.sort(b), xs, side="right") / len(b)
    return float(np.abs(fa - fb).max())


def tie_aware_auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC with tie correction (ties contribute 1/2)."""
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    # average ranks for ties
    i = 0
    r = np.arange(1, len(allv) + 1, dtype=np.float64)
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[i : j + 1] = r[i : j + 1].mean()
        i = j + 1
    ranks[order] = r
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def norm_output_diag(out_loeo: np.ndarray, same_task_loeo: np.ndarray) -> dict:
    s = out_loeo[same_task_loeo]
    c = out_loeo[~same_task_loeo]
    return {
        "std": float(out_loeo.std()),
        "sat0": float(np.mean(out_loeo <= 0.0)),
        "sat1": float(np.mean(out_loeo >= 1.0)),
        "interior": float(np.mean((out_loeo > 0.0) & (out_loeo < 1.0))),
        "sep_same_minus_cross": float(s.mean() - c.mean()) if len(s) and len(c) else 0.0,
        "auc_tie_aware": tie_aware_auc(s, c) if len(s) and len(c) else 0.5,
    }


def top10_ceiling_tie_rate(raw: np.ndarray, out: np.ndarray, same_traj: np.ndarray, sim_type: str) -> float:
    """Among each query's top-10 raw-score candidates (LOEO), the fraction whose
    normalized score sits at the clip ceiling 1.0 — saturated exactly where the
    argmax decision is made."""
    x = C.orient(raw, sim_type).copy()
    x[same_traj] = -np.inf
    top = np.argsort(-x, axis=1)[:, :10]
    rows = np.arange(raw.shape[0])[:, None]
    vals = out[rows, top]
    return float(np.mean(vals >= 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for suite, builder in C.COMBOS:
        art = C.load_artifact(suite, builder, Path(args.cache_dir))
        same_traj = art.same_traj
        same_task = art.same_task
        loeo_mask = ~same_traj
        offdiag = ~np.eye(art.n, dtype=bool)
        intra = same_traj & offdiag

        combo_key = f"{suite}/{builder}"
        results[combo_key] = {}
        for f in C.FIELDS:
            raw = art.raw[f]
            st = C.SIM_TYPE[f]
            rp = raw[offdiag]              # legacy calibration pool (has intra-traj)
            lo = raw[loeo_mask]            # serving pool
            st_loeo = same_task[loeo_mask]

            # -- pool anatomy in legacy-mapped [0,1] space --------------------
            rp_m = C.map_legacy(rp, st)
            lo_m = C.map_legacy(lo, st)
            intra_m = C.map_legacy(raw[intra], st)
            p95_rp = float(np.percentile(rp_m, 95))
            tail_rp = rp_m > p95_rp
            # share of the calibration right tail occupied by intra-traj pairs
            n_tail = int(tail_rp.sum())
            intra_share_tail = float(np.sum(intra_m > p95_rp)) / max(n_tail, 1)

            pools = {
                "rp_mapped": {q: float(np.percentile(rp_m, q)) for q in (1, 5, 50, 95, 99)},
                "loeo_mapped": {q: float(np.percentile(lo_m, q)) for q in (1, 5, 50, 95, 99)},
                "ks_rp_vs_loeo_mapped": ks_stat(rp_m, lo_m),
                "intra_traj_pair_frac": float(intra.sum() / offdiag.sum()),
                "intra_share_of_rp_top5pct_tail": intra_share_tail,
                "loeo_mean_oriented": float(C.orient(lo, st).mean()),
                "loeo_std_oriented": float(C.orient(lo, st).std()),
                "cohens_d_same_cross_oriented": float(
                    (C.orient(lo[st_loeo], st).mean() - C.orient(lo[~st_loeo], st).mean())
                    / C.orient(lo, st).std()
                ),
                "auc_raw": tie_aware_auc(C.orient(lo[st_loeo], st), C.orient(lo[~st_loeo], st)),
            }

            # -- four calibration x family cells on the serving pool ----------
            cells = {}
            fit_leg, app_leg = C.NORMALIZERS["legacy_percentile"]
            fit_z, app_z = C.NORMALIZERS["zscore"]
            params = {
                "legacy@RP": fit_leg(rp, st),
                "legacy@LOEO": fit_leg(lo, st),
                "zscore@RP": fit_z(rp, st),
                "zscore@LOEO": fit_z(lo, st),
            }
            for name, p in params.items():
                app = app_leg if name.startswith("legacy") else app_z
                out_full = app(raw, st, p)
                d = norm_output_diag(out_full[loeo_mask], st_loeo)
                d["params"] = {k: (v if isinstance(v, float) else v) for k, v in p.items()}
                if name.startswith("legacy"):
                    d["denom"] = p["p95"] - p["p5"]
                    d["loeo_frac_below_band"] = float(np.mean(C.map_legacy(lo, st) < p["p5"]))
                    d["loeo_frac_above_band"] = float(np.mean(C.map_legacy(lo, st) > p["p95"]))
                d["top10_ceiling_tie_rate"] = top10_ceiling_tie_rate(raw, out_full, same_traj, st)
                cells[name] = d

            # -- local gains at the serving median ----------------------------
            med_or = float(np.median(C.orient(lo, st)))
            pz = params["zscore@LOEO"]
            zmed = (med_or - pz["mu"]) / pz["sigma"]
            gain_z = 0.5 * (1 - np.tanh(zmed) ** 2) / pz["sigma"]
            pl = params["legacy@RP"]
            if st == "cosine":
                gain_leg = 0.5 / max(pl["p95"] - pl["p5"], 1e-12)  # d s0/d cos = 1/2
            else:
                d_med = -med_or
                gain_leg = (np.exp(-d_med / C.LEGACY_TAU) / C.LEGACY_TAU) / max(pl["p95"] - pl["p5"], 1e-12)
            gains = {
                "zscore_tanh_dsdx_at_median": float(gain_z),
                "legacy_RP_dsdx_at_median_inband": float(gain_leg),
                "z_of_serving_median_under_LOEO_fit": float(zmed),
            }

            results[combo_key][f] = {"pools": pools, "cells": cells, "gains": gains}
            print(f"[{combo_key}] {f}: KS(RP,LOEO)={pools['ks_rp_vs_loeo_mapped']:.3f} "
                  f"denomRP={cells['legacy@RP']['denom']:.5f} "
                  f"sat1={cells['legacy@RP']['sat1']:.3f} "
                  f"tie@top10={cells['legacy@RP']['top10_ceiling_tie_rate']:.3f} "
                  f"| zscore sat1={cells['zscore@LOEO']['sat1']:.4f} "
                  f"std={cells['zscore@LOEO']['std']:.3f}")

    (outdir / "expA_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {outdir / 'expA_results.json'}")


if __name__ == "__main__":
    main()
