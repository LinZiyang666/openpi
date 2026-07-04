"""Experiment D: absolute-score (threshold) semantics — magnitude-faithful vs
rank-based fusion when the library may contain nothing relevant.

Regimes per query:
  HIT  — standard LOEO library (own trajectory removed, same task present);
  MISS — the query's whole task removed from the library (nothing relevant).

A cache must abstain in the MISS regime, which requires the fused top-1 score
to carry absolute meaning. Configs compared: zscore+tanh, probit, hardclip3,
identity(raw-z), legacy percentile, frozen-pool ecdf, and per-query rank
normalization (s = 1 - (r-1)/n, the RRF-style library-relative rank).

Reported per config:
  - AUC separating HIT vs MISS top-1 fused scores;
  - false-hit rate at the threshold achieving 90% hit-recall;
  - threshold transfer: tau fitted on a 5-task split, FHR/recall on the rest;
  - library-size scaling of the MISS top-1 score (extreme-value behaviour),
    with the 1 - 1/(n+1) uniform-max law overlaid for rank normalization.

Usage:
    uv run exp/weighted_sum/analysis/fusion_theory/expD_missdetect.py \
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


def build_field_scores(art) -> dict[str, dict[str, np.ndarray]]:
    """Per-config per-field normalized [N,N] matrices (rank_perquery handled
    separately because it depends on the candidate set)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    z = {}
    for f in C.FIELDS:
        raw = art.raw[f].astype(np.float64)
        stype = C.SIM_TYPE[f]
        pool = C.pool_loeo(raw, art.same_traj).astype(np.float64)
        p = C.fit_zscore(pool, stype)
        sigma = p["sigma"] if p["sigma"] > 1e-12 else 1.0
        z[f] = (C.orient(raw, stype) - p["mu"]) / sigma
    out["zscore_tanh"] = {f: 0.5 * (np.tanh(z[f]) + 1.0) for f in C.FIELDS}
    out["probit"] = {f: C.SQUASHES["probit"](z[f]) for f in C.FIELDS}
    out["hardclip3"] = {f: C.SQUASHES["hardclip3"](z[f]) for f in C.FIELDS}
    out["identity_z"] = {f: z[f] for f in C.FIELDS}
    leg = {}
    ecdf = {}
    for f in C.FIELDS:
        raw = art.raw[f].astype(np.float64)
        stype = C.SIM_TYPE[f]
        rp = C.pool_random_pair(raw).astype(np.float64)
        loeo = C.pool_loeo(raw, art.same_traj).astype(np.float64)
        fit_l, app_l = C.NORMALIZERS["legacy_percentile"]
        leg[f] = app_l(raw, stype, fit_l(rp, stype))
        fit_e, app_e = C.NORMALIZERS["ecdf"]
        ecdf[f] = app_e(raw, stype, fit_e(loeo, stype))
    out["legacy_percentile"] = leg
    out["ecdf_frozen"] = ecdf
    return out


def top1_scores(ns: dict[str, np.ndarray], art, lib_mask: np.ndarray, w) -> np.ndarray:
    """Top-1 fused score per query over candidates allowed by lib_mask [N,N]."""
    S = C.fuse(ns, w)
    S = np.where(lib_mask, S, -np.inf)
    return S.max(axis=1)


def rank_perquery_top1(art, lib_mask: np.ndarray, w) -> np.ndarray:
    """Per-query rank normalization: within the allowed candidate set, each
    field's scores are replaced by 1 - (rank-1)/n; fused top-1 returned."""
    n = art.n
    out = np.empty(n)
    oriented = {f: C.orient(art.raw[f].astype(np.float64), C.SIM_TYPE[f]) for f in C.FIELDS}
    for q in range(n):
        allowed = np.where(lib_mask[q])[0]
        if len(allowed) == 0:
            out[q] = np.nan
            continue
        S = np.zeros(len(allowed))
        m = len(allowed)
        for f in C.FIELDS:
            x = oriented[f][q, allowed]
            r = np.empty(m)
            r[np.argsort(-x, kind="stable")] = np.arange(m)
            S += w[f] * (1.0 - r / max(m - 1, 1))
        out[q] = S.max()
    return out


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(pos > neg) + 0.5 P(=) via rank sum."""
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks on ties
    sv = allv[order]
    i = 0
    r = np.arange(1, len(allv) + 1, dtype=np.float64)
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            r[i:j + 1] = r[i:j + 1].mean()
        i = j + 1
    ranks[order] = r
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


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
        results[key] = {"configs": {}, "scaling": {}}
        print(f"\n===== {key} =====")

        hit_mask = ~art.same_traj
        miss_mask = ~art.same_task  # whole task left out
        w = C.W_PROD

        cfgs = build_field_scores(art)
        s_hit_all, s_miss_all = {}, {}
        for cname, ns in cfgs.items():
            s_hit_all[cname] = top1_scores(ns, art, hit_mask, w)
            s_miss_all[cname] = top1_scores(ns, art, miss_mask, w)
        s_hit_all["rank_perquery"] = rank_perquery_top1(art, hit_mask, w)
        s_miss_all["rank_perquery"] = rank_perquery_top1(art, miss_mask, w)

        for cname in list(cfgs) + ["rank_perquery"]:
            sh, sm = s_hit_all[cname], s_miss_all[cname]
            a = auc(sh, sm)
            tau90 = float(np.percentile(sh, 10))  # 90% hit-recall threshold
            fhr = float(np.mean(sm >= tau90))
            # threshold transfer across a task split
            rng = np.random.default_rng(11)
            tasks = np.unique(art.task)
            ta = rng.choice(tasks, size=len(tasks) // 2, replace=False)
            qa = np.isin(art.task, ta)
            tau_a = float(np.percentile(sh[qa], 10))
            recall_b = float(np.mean(sh[~qa] >= tau_a))
            fhr_b = float(np.mean(sm[~qa] >= tau_a))
            results[key]["configs"][cname] = {
                "auc_hit_vs_miss": a,
                "mean_top1_hit": float(sh.mean()), "mean_top1_miss": float(sm.mean()),
                "tau@90recall": tau90, "fhr@90recall": fhr,
                "transfer_recall_B": recall_b, "transfer_fhr_B": fhr_b,
            }
            print(f"  {cname:>17s}: AUC={a:.3f} hit_mean={sh.mean():.3f} miss_mean={sm.mean():.3f} "
                  f"FHR@90={fhr:.3f} | transfer recall_B={recall_b:.3f} fhr_B={fhr_b:.3f}")

        # ---- library-size scaling of the MISS top-1 score ---------------
        sizes = [2 ** k for k in range(5, 12) if 2 ** k < art.n] + [None]
        rng = np.random.default_rng(23)
        scal = {c: [] for c in ("zscore_tanh", "ecdf_frozen", "rank_perquery", "identity_z")}
        qsub = rng.choice(art.n, size=min(400, art.n), replace=False)
        for sz in sizes:
            sub_masks = np.zeros((len(qsub), art.n), dtype=bool)
            for qi, q in enumerate(qsub):
                allowed = np.where(miss_mask[q])[0]
                if sz is not None and len(allowed) > sz:
                    allowed = rng.choice(allowed, size=sz, replace=False)
                sub_masks[qi, allowed] = True
            for cname in scal:
                if cname == "rank_perquery":
                    vals = []
                    oriented = {f: C.orient(art.raw[f].astype(np.float64), C.SIM_TYPE[f])
                                for f in C.FIELDS}
                    for qi, q in enumerate(qsub):
                        allowed = np.where(sub_masks[qi])[0]
                        m = len(allowed)
                        S = np.zeros(m)
                        for f in C.FIELDS:
                            x = oriented[f][q, allowed]
                            r = np.empty(m)
                            r[np.argsort(-x, kind="stable")] = np.arange(m)
                            S += w[f] * (1.0 - r / max(m - 1, 1))
                        vals.append(S.max())
                    scal[cname].append(float(np.mean(vals)))
                else:
                    ns = cfgs[cname]
                    S = C.fuse({f: ns[f][qsub] for f in C.FIELDS}, w)
                    S = np.where(sub_masks, S, -np.inf)
                    scal[cname].append(float(S.max(axis=1).mean()))
        results[key]["scaling"] = {
            "sizes": [s if s is not None else int(np.median(miss_mask.sum(axis=1))) for s in sizes],
            **scal,
            "uniform_max_law": [1.0 - 1.0 / (s + 1) if s else None for s in
                                [s if s is not None else int(np.median(miss_mask.sum(axis=1)))
                                 for s in sizes]],
        }
        print(f"  scaling sizes={results[key]['scaling']['sizes']}")
        for cname in scal:
            print(f"    {cname:>15s}: {['%.3f' % v for v in scal[cname]]}")

    (outdir / "expD_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {outdir / 'expD_results.json'}")


if __name__ == "__main__":
    main()
