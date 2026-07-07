"""Experiment E: why library-relative rank fusion (RRF / Borda) trails
z-score+tanh even though the frozen empirical CDF matches it.

The class boundary under test is NOT "rank vs score": the frozen eCDF is a
*fixed pointwise monotone map* (score fusion), whereas RRF/Borda replace each
score by its *rank within the current query's candidate set* — a relative
order statistic that is blind to margins by construction.

Parts:
  (1) Retrieval metrics for RRF over a k sweep (1..1000) plus the Borda
      (k -> inf) limit, against zscore+tanh / frozen-eCDF references;
  (2) weight-simplex sweep (153 points) for rrf60 / borda / zscore — the
      "is it just weight tuning?" test: compare maxima over the simplex;
  (3) conflict analysis: queries where zscore and rrf60 pick different top-1;
      McNemar-style asymmetry + exact binomial p;
  (4) margin diagnostics: P(top-1 correct | fused margin) monotonicity, and
      the per-conflict margin asymmetry (rank majority backed by small
      z-margins vs a single field with a large z-margin).

Usage:
    uv run exp/weighted_sum/analysis/fusion_theory/expE_rrf.py \
        --cache-dir <scores-cache> --out exp/weighted_sum/analysis/fusion_theory/results
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fusion_theory_common as C  # noqa: E402

RRF_KS = [1, 5, 10, 20, 60, 240, 1000]


def rank_matrices(art) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """Per-field per-row ranks (1 = best) over LOEO-allowed candidates.

    Own-trajectory entries are pushed to -inf so allowed candidates occupy
    ranks 1..m_row exactly; the disallowed tail ranks are never used because
    fusion re-masks before argmax.
    """
    allowed = ~art.same_traj
    m_row = allowed.sum(axis=1).astype(np.float64)
    ranks = {}
    for f in C.FIELDS:
        x = C.orient(art.raw[f].astype(np.float64), C.SIM_TYPE[f])
        x = np.where(allowed, x, -np.inf)
        order = np.argsort(-x, axis=1, kind="stable")
        r = np.empty_like(x)
        rows = np.arange(art.n)[:, None]
        r[rows, order] = np.arange(1, art.n + 1, dtype=np.float64)[None, :]
        ranks[f] = r
    return ranks, m_row


def fuse_rank(ranks, m_row, w, phi: str, k: float = 60.0) -> np.ndarray:
    if phi == "rrf":
        return sum(w[f] / (k + ranks[f]) for f in C.FIELDS)
    if phi == "borda":
        denom = np.maximum(m_row - 1.0, 1.0)[:, None]
        return sum(w[f] * (1.0 - (ranks[f] - 1.0) / denom) for f in C.FIELDS)
    raise ValueError(phi)


def top1_and_metrics(art, S, full: bool = False):
    Sm = np.where(art.same_traj, -np.inf, S)
    top1 = Sm.argmax(axis=1)
    rows = np.arange(art.n)
    out = {"top1": top1, "top1_same_task": art.same_task[rows, top1].astype(np.float64)}
    if full:
        out["metrics"] = C.retrieval_metrics(art, S)
    return out


def binom_two_sided_p(k: int, n: int) -> float:
    """Exact two-sided binomial test p-value against p=0.5."""
    if n == 0:
        return 1.0
    pk = [math.comb(n, i) * 0.5 ** n for i in range(n + 1)]
    p_obs = pk[k]
    return float(min(1.0, sum(p for p in pk if p <= p_obs + 1e-15)))


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
        key = f"{suite}/{builder}"
        results[key] = {"ksweep": {}, "sweep": {}, "conflict": {}, "margin": {}}
        print(f"\n===== {key} (N={art.n}) =====")

        # ---- references: fixed pointwise maps -----------------------------
        zfields, zs_norm, ec_norm = {}, {}, {}
        for f in C.FIELDS:
            raw = art.raw[f].astype(np.float64)
            stype = C.SIM_TYPE[f]
            pool = C.pool_loeo(raw, art.same_traj).astype(np.float64)
            p = C.fit_zscore(pool, stype)
            sigma = p["sigma"] if p["sigma"] > 1e-12 else 1.0
            zfields[f] = (C.orient(raw, stype) - p["mu"]) / sigma
            zs_norm[f] = 0.5 * (np.tanh(zfields[f]) + 1.0)
            fit_e, app_e = C.NORMALIZERS["ecdf"]
            ec_norm[f] = app_e(raw, stype, fit_e(pool, stype))

        ranks, m_row = rank_matrices(art)

        # ---- Part 1: k sweep + references ---------------------------------
        for wname, w in (("w_prod", C.W_PROD), ("w_unif", C.W_UNIF)):
            ref = top1_and_metrics(art, C.fuse(zs_norm, w), full=True)
            ecr = top1_and_metrics(art, C.fuse(ec_norm, w), full=True)
            row = {
                "zscore": {"top1_same_task": float(ref["top1_same_task"].mean()),
                           "action_regret": float(ref["metrics"]["action_regret@1"].mean())},
                "ecdf": {"top1_same_task": float(ecr["top1_same_task"].mean()),
                         "action_regret": float(ecr["metrics"]["action_regret@1"].mean())},
            }
            for k in RRF_KS:
                rr = top1_and_metrics(art, fuse_rank(ranks, m_row, w, "rrf", k), full=True)
                d, lo, hi = C.paired_cluster_bootstrap(
                    rr["top1_same_task"], ref["top1_same_task"], art.traj, 2000, 5)
                row[f"rrf{k}"] = {
                    "top1_same_task": float(rr["top1_same_task"].mean()),
                    "action_regret": float(rr["metrics"]["action_regret@1"].mean()),
                    "delta_vs_zscore": d, "lo": lo, "hi": hi,
                }
            bo = top1_and_metrics(art, fuse_rank(ranks, m_row, w, "borda"), full=True)
            d, lo, hi = C.paired_cluster_bootstrap(
                bo["top1_same_task"], ref["top1_same_task"], art.traj, 2000, 5)
            row["borda"] = {
                "top1_same_task": float(bo["top1_same_task"].mean()),
                "action_regret": float(bo["metrics"]["action_regret@1"].mean()),
                "delta_vs_zscore": d, "lo": lo, "hi": hi,
            }
            results[key]["ksweep"][wname] = row
            print(f"  [{wname}] zscore={row['zscore']['top1_same_task']:.4f} "
                  f"ecdf={row['ecdf']['top1_same_task']:.4f} "
                  + " ".join(f"rrf{k}={row[f'rrf{k}']['top1_same_task']:.4f}" for k in RRF_KS)
                  + f" borda={row['borda']['top1_same_task']:.4f}")

        # ---- Part 2: weight-simplex maxima ---------------------------------
        step = 16
        grid = [(i / step, j / step, (step - i - j) / step)
                for i in range(step + 1) for j in range(step + 1 - i)]
        for cname in ("zscore", "rrf60", "borda"):
            vals = []
            for w0, w1, w2 in grid:
                w = {"vision_0": w0, "vision_1": w1, "robot_state": w2}
                if cname == "zscore":
                    S = C.fuse(zs_norm, w)
                elif cname == "rrf60":
                    S = fuse_rank(ranks, m_row, w, "rrf", 60)
                else:
                    S = fuse_rank(ranks, m_row, w, "borda")
                vals.append(float(top1_and_metrics(art, S)["top1_same_task"].mean()))
            vals = np.array(vals)
            results[key]["sweep"][cname] = {
                "max": float(vals.max()), "mean": float(vals.mean()),
                "argmax_w": list(grid[int(vals.argmax())]),
            }
            print(f"  sweep {cname:>7s}: max={vals.max():.4f} mean={vals.mean():.4f} "
                  f"argmax_w={grid[int(vals.argmax())]}")

        # ---- Part 3: conflict analysis (w_prod, rrf60 vs zscore) ------------
        z_res = top1_and_metrics(art, C.fuse(zs_norm, C.W_PROD))
        r_res = top1_and_metrics(art, fuse_rank(ranks, m_row, C.W_PROD, "rrf", 60))
        conflict = z_res["top1"] != r_res["top1"]
        cq = np.where(conflict)[0]
        z_ok = z_res["top1_same_task"][cq].astype(bool)
        r_ok = r_res["top1_same_task"][cq].astype(bool)
        n01 = int(np.sum(z_ok & ~r_ok))   # zscore right, rrf wrong
        n10 = int(np.sum(~z_ok & r_ok))   # rrf right, zscore wrong
        results[key]["conflict"] = {
            "n_queries": int(art.n), "n_conflict": int(len(cq)),
            "conflict_frac": float(len(cq) / art.n),
            "zscore_right_rrf_wrong": n01, "rrf_right_zscore_wrong": n10,
            "both_right": int(np.sum(z_ok & r_ok)), "both_wrong": int(np.sum(~z_ok & ~r_ok)),
            "mcnemar_p": binom_two_sided_p(min(n01, n10), n01 + n10),
        }
        print(f"  conflict: {len(cq)}/{art.n} ({len(cq)/art.n:.3f}) "
              f"z-right/r-wrong={n01} r-right/z-wrong={n10} "
              f"p={results[key]['conflict']['mcnemar_p']:.2e}")

        # ---- Part 4: margin diagnostics ------------------------------------
        # (a) P(top-1 same-task | fused zscore margin) by quintile
        Sz = np.where(art.same_traj, -np.inf, C.fuse(zs_norm, C.W_PROD))
        part = np.partition(Sz, -2, axis=1)
        margin = part[:, -1] - part[:, -2]
        qs = np.quantile(margin, [0.2, 0.4, 0.6, 0.8])
        bins = np.digitize(margin, qs)
        acc_by_bin = [float(z_res["top1_same_task"][bins == b].mean()) for b in range(5)]
        med_by_bin = [float(np.median(margin[bins == b])) for b in range(5)]
        results[key]["margin"]["acc_by_margin_quintile"] = acc_by_bin
        results[key]["margin"]["median_margin_by_quintile"] = med_by_bin
        print(f"  P(correct|margin quintile): {['%.3f' % a for a in acc_by_bin]}")

        # (b) per-conflict margin asymmetry in z units
        a_idx, b_idx = z_res["top1"][cq], r_res["top1"][cq]
        dz = np.stack([zfields[f][cq, a_idx] - zfields[f][cq, b_idx] for f in C.FIELDS])
        n_fields_favor_a = (dz > 0).sum(axis=0)
        best_for_a = dz.max(axis=0)          # largest z-margin backing the zscore pick
        worst_for_a = (-dz).max(axis=0)      # largest z-margin backing the rrf pick
        results[key]["margin"]["conflict_fields_favoring_zscore_pick_hist"] = [
            int(np.sum(n_fields_favor_a == c)) for c in range(4)]
        results[key]["margin"]["median_best_dz_for_zscore_pick"] = float(np.median(best_for_a))
        results[key]["margin"]["median_best_dz_for_rrf_pick"] = float(np.median(worst_for_a))
        print(f"  conflicts: fields favoring z-pick hist={results[key]['margin']['conflict_fields_favoring_zscore_pick_hist']} "
              f"med best dz(z-pick)={np.median(best_for_a):.2f}σ vs dz(rrf-pick)={np.median(worst_for_a):.2f}σ")

    (outdir / "expE_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {outdir / 'expE_results.json'}")


if __name__ == "__main__":
    main()
