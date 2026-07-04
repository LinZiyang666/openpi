"""Experiment C: which squash on top of z-scoring — and which of its properties
(boundedness, strict monotonicity, smoothness) actually pay.

Protocols:
  C1 on-distribution: retrieval metrics for every squash in SQUASHES with the
     same LOEO-fitted (mu, sigma). Hypothesis: all smooth strictly-monotone
     sigmoids are statistically indistinguishable; hard clips lose via interior
     censoring/ties; identity matches on clean data (affine equivalence).
  C2 calibration shift: (mu, sigma) fitted only on a 5-task split; queries from
     the other 5 tasks. Measures sensitivity of each squash to a misplaced
     operating point (saturation fraction, retrieval delta).
  C3 bounded influence (single-field veto): corrupt vision_0 of each query's
     current top-1 correct candidate by delta*sigma downward; a bounded squash
     caps the score damage at w_f * range, an unbounded one lets a single field
     veto the candidate. Reports survival@top1 and same-task retention vs delta.

Usage:
    uv run exp/weighted_sum/analysis/fusion_theory/expC_squash.py \
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

SQUASH_ORDER = [
    "tanh", "logistic", "probit", "arctan", "softsign",
    "hardclip1", "hardclip2", "hardclip3", "identity",
]


def zmats(art, pool_mask=None) -> dict[str, np.ndarray]:
    """Standardized z matrices per field; (mu, sigma) from the LOEO pool
    (optionally restricted to pool_mask rows/cols for the shift protocol)."""
    out = {}
    for f in C.FIELDS:
        raw = art.raw[f].astype(np.float64)
        stype = C.SIM_TYPE[f]
        m = ~art.same_traj
        if pool_mask is not None:
            m = m & pool_mask
        pool = raw[m]
        p = C.fit_zscore(pool, stype)
        sigma = p["sigma"] if p["sigma"] > 1e-12 else 1.0
        out[f] = (C.orient(raw, stype) - p["mu"]) / sigma
    return out


def fuse_squash(z: dict[str, np.ndarray], squash, w: dict[str, float]) -> np.ndarray:
    return sum(w[f] * squash(z[f]) for f in C.FIELDS)


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
        results[key] = {"C1": {}, "C2": {}, "C3": {}}
        print(f"\n===== {key} =====")

        # ---------------- C1: on-distribution ----------------------------
        z = zmats(art)
        base_top1 = None
        for name in SQUASH_ORDER:
            S = fuse_squash(z, C.SQUASHES[name], C.W_PROD)
            m = C.retrieval_metrics(art, S)
            mean, lo95, hi95 = C.cluster_bootstrap(m["top1_same_task"], art.traj, 1000, 1)
            sat = float(np.mean([np.mean(np.abs(C.SQUASHES[name](z[f])[~art.same_traj]
                                  - np.clip(C.SQUASHES[name](z[f])[~art.same_traj], 1e-12, 1 - 1e-12)) > 0)
                                 for f in C.FIELDS])) if name != "identity" else 0.0
            if name == "tanh":
                base_top1 = m["top1_same_task"]
                delta, dlo, dhi = 0.0, 0.0, 0.0
            else:
                delta, dlo, dhi = C.paired_cluster_bootstrap(
                    m["top1_same_task"], base_top1, art.traj, 2000, 2)
            results[key]["C1"][name] = {
                "top1_same_task": mean, "lo": lo95, "hi": hi95,
                "delta_vs_tanh": delta, "delta_lo": dlo, "delta_hi": dhi,
                "action_regret": float(m["action_regret@1"].mean()),
                "sat_frac": sat,
            }
            print(f"  C1 {name:>10s}: top1={mean:.4f} [{lo95:.3f},{hi95:.3f}] "
                  f"d_vs_tanh={delta:+.4f} [{dlo:+.4f},{dhi:+.4f}] regret={m['action_regret@1'].mean():.3f}")

        # ---------------- C1b: clip half-width dose-response --------------
        # Retrieval quality as a function of the censoring band half-width k
        # (in sigma units); tanh is the k -> inf, censoring-free limit.
        halfwidths = [0.5, 1.0, 1.5, 1.645, 2.0, 2.5, 3.0, 4.0, 5.0]
        c1b = []
        for k in halfwidths:
            def sq(zz, kk=k):
                return np.clip(0.5 + zz / (2.0 * kk), 0.0, 1.0)
            S = fuse_squash(z, sq, C.W_PROD)
            m = C.retrieval_metrics(art, S)
            cens = float(np.mean([np.mean(np.abs(z[f][~art.same_traj]) > k) for f in C.FIELDS]))
            c1b.append({"k": k, "top1_same_task": float(m["top1_same_task"].mean()),
                        "censored_frac": cens})
        results[key]["C1b"] = c1b
        print("  C1b halfwidth->top1:", [(p["k"], round(p["top1_same_task"], 3)) for p in c1b])

        # ---------------- C2: calibration split shift --------------------
        tasks = np.unique(art.task)
        rng = np.random.default_rng(7)
        split_a = rng.choice(tasks, size=len(tasks) // 2, replace=False)
        in_a = np.isin(art.task, split_a)
        # pool restricted to A-queries x A-candidates; queries from B only
        pool_mask = in_a[:, None] & in_a[None, :]
        z_shift = zmats(art, pool_mask=pool_mask)
        qsel = ~in_a
        shift_stats = {}
        for f in C.FIELDS:
            raw = art.raw[f].astype(np.float64)
            stype = C.SIM_TYPE[f]
            pa = C.fit_zscore(raw[(~art.same_traj) & pool_mask], stype)
            pb = C.fit_zscore(raw[(~art.same_traj) & (~in_a[:, None] & ~in_a[None, :])], stype)
            shift_stats[f] = {
                "mu_offset_in_sigmaA": (pb["mu"] - pa["mu"]) / max(pa["sigma"], 1e-12),
                "sigma_ratio_B_over_A": pb["sigma"] / max(pa["sigma"], 1e-12),
            }
        results[key]["C2"]["_shift_stats"] = shift_stats
        print("  C2 shift:", {f: round(s["mu_offset_in_sigmaA"], 2) for f, s in shift_stats.items()})
        for name in SQUASH_ORDER:
            S_full = fuse_squash(z, C.SQUASHES[name], C.W_PROD)
            S_shift = fuse_squash(z_shift, C.SQUASHES[name], C.W_PROD)
            m_full = C.retrieval_metrics(art, S_full)["top1_same_task"][qsel]
            m_shift = C.retrieval_metrics(art, S_shift)["top1_same_task"][qsel]
            # saturation of B-queries' scores under shifted params
            satB = float(np.mean([
                np.mean((C.SQUASHES[name](z_shift[f][qsel][:, ~in_a]) >= 1 - 1e-9)
                        | (C.SQUASHES[name](z_shift[f][qsel][:, ~in_a]) <= 1e-9))
                for f in C.FIELDS])) if name != "identity" else 0.0
            d, dlo, dhi = C.paired_cluster_bootstrap(m_shift, m_full, art.traj[qsel], 2000, 3)
            results[key]["C2"][name] = {
                "top1_B_full_calib": float(m_full.mean()),
                "top1_B_shift_calib": float(m_shift.mean()),
                "delta": d, "lo": dlo, "hi": dhi, "satB": satB,
            }
            print(f"  C2 {name:>10s}: full={m_full.mean():.4f} shifted={m_shift.mean():.4f} "
                  f"delta={d:+.4f} satB={satB:.3f}")

        # ---------------- C3: single-field veto --------------------------
        deltas = [2.0, 5.0, 10.0, 30.0]
        for wname, w in (("w_prod", C.W_PROD), ("w_unif", C.W_UNIF)):
            S0 = fuse_squash(z, C.SQUASHES["tanh"], w)
            Sm0 = np.where(art.same_traj, -np.inf, S0)
            top1_0 = Sm0.argmax(axis=1)
            rows = np.arange(art.n)
            qidx = np.where(art.same_task[rows, top1_0])[0]
            results[key]["C3"][wname] = {}
            for name in SQUASH_ORDER:
                surv = []
                still_same = []
                for dlt in deltas:
                    zc = {f: z[f][qidx].copy() for f in C.FIELDS}
                    # corrupt vision_0 of each query's own top-1 candidate
                    zc["vision_0"][np.arange(len(qidx)), top1_0[qidx]] -= dlt
                    Sq = sum(w[f] * C.SQUASHES[name](zc[f]) for f in C.FIELDS)
                    Sq[art.same_traj[qidx]] = -np.inf
                    new_top1 = Sq.argmax(axis=1)
                    surv.append(float(np.mean(new_top1 == top1_0[qidx])))
                    still_same.append(float(np.mean(art.same_task[qidx, new_top1])))
                # bounded-influence signature: median final rank of the
                # corrupted candidate (plateaus for bounded squashes,
                # sinks to the bottom for identity as delta grows)
                med_rank = []
                for dlt in deltas:
                    zc = {f: z[f][qidx].copy() for f in C.FIELDS}
                    zc["vision_0"][np.arange(len(qidx)), top1_0[qidx]] -= dlt
                    Sq = sum(w[f] * C.SQUASHES[name](zc[f]) for f in C.FIELDS)
                    Sq[art.same_traj[qidx]] = -np.inf
                    tgt = Sq[np.arange(len(qidx)), top1_0[qidx]]
                    rank = (Sq > tgt[:, None]).sum(axis=1) + 1
                    med_rank.append(float(np.median(rank)))
                results[key]["C3"][wname][name] = {
                    "deltas": deltas, "survival": surv, "still_same_task": still_same,
                    "median_rank_of_corrupted": med_rank,
                }
                print(f"  C3[{wname}] {name:>10s}: survival={['%.3f' % s for s in surv]} "
                      f"same_task={['%.3f' % s for s in still_same]} "
                      f"med_rank={med_rank}")

        # ---------------- C3b: decoy attack (upward corruption) ----------
        # Inflate vision_0 of one random cross-task candidate per query by
        # +delta sigma. Unbounded squashes hand the decoy an unbounded gain
        # (false hit -> wrong task's actions replayed); bounded squashes cap
        # the gain at w_f * (1 - s_decoy).
        rng3 = np.random.default_rng(13)
        decoy = np.array([
            rng3.choice(np.where(~art.same_task[q] & ~art.same_traj[q])[0])
            for q in range(art.n)
        ])
        results[key]["C3b"] = {}
        for name in SQUASH_ORDER:
            wins = []
            for dlt in deltas:
                zc = {f: z[f].copy() for f in C.FIELDS}
                zc["vision_0"][np.arange(art.n), decoy] += dlt
                Sq = sum(C.W_PROD[f] * C.SQUASHES[name](zc[f]) for f in C.FIELDS)
                Sq[art.same_traj] = -np.inf
                wins.append(float(np.mean(Sq.argmax(axis=1) == decoy)))
            results[key]["C3b"][name] = {"deltas": deltas, "decoy_win_rate": wins}
            print(f"  C3b {name:>10s}: decoy_wins={['%.3f' % v for v in wins]}")

    (outdir / "expC_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {outdir / 'expC_results.json'}")


if __name__ == "__main__":
    main()
