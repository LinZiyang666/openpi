"""Experiment B: 2x2 factorial (calibration pool x normalizer family) + full
family head-to-head on exact LOEO retrieval, plus weight-simplex response.

Parts:
  (0) prompt_emb historical probe — reconstructs the legacy 4-field regime where
      a task-constant field drives p5~p95 (the denom<=0 -> 0.5 folklore).
  (1) Retrieval metrics for {legacy,zscore} x {RP,LOEO} and the wider family
      (affine_clip, ecdf, direction-unify, norm2/logit/power mixes, raw-z,
      single-field baselines) under production and uniform weights, with
      trajectory-cluster bootstrap CIs, paired deltas vs zscore@LOEO, and
      argmax tie diagnostics.
  (2) Weight-simplex sweep (step 1/16, 153 points) for four representative
      families — measures how much of the simplex is 'live' under each
      normalizer (weights-inert pathology).
  (3) Decision-region diagnostics: z-location of per-field top-10 candidates,
      tanh gain retained there, per-field score std among fused top-50.

Usage:
    uv run exp/weighted_sum/analysis/fusion_theory/expB_factorial.py \
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

DIRUNIFY_TAU = 1.0  # production YAML to_similarity.tau for the type:none path


# ------------------------------------------------------------------
# Config construction: per-field (method, pool) -> normalized [N,N]
# ------------------------------------------------------------------
def get_pool(art, f: str, which: str) -> np.ndarray:
    raw = art.raw[f]
    if which == "RP":
        return C.pool_random_pair(raw)
    if which == "LOEO":
        return C.pool_loeo(raw, art.same_traj)
    raise ValueError(which)


def build_norm_scores(art, spec: dict[str, tuple[str, str]]) -> dict[str, np.ndarray]:
    """spec[field] = (method, pool). method 'dirunify' and 'rawz' are special."""
    out = {}
    for f, (method, pool_name) in spec.items():
        raw = art.raw[f].astype(np.float64)
        stype = C.SIM_TYPE[f]
        if method == "dirunify":
            out[f] = C.map_legacy(raw, stype, tau=DIRUNIFY_TAU)
            continue
        pool = get_pool(art, f, pool_name).astype(np.float64)
        if method == "rawz":
            p = C.fit_zscore(pool, stype)
            sigma = p["sigma"] if p["sigma"] > 1e-12 else 1.0
            out[f] = (C.orient(raw, stype) - p["mu"]) / sigma
            continue
        fit, app = C.NORMALIZERS[method]
        out[f] = app(raw, stype, fit(pool, stype))
    return out


CONFIGS: dict[str, dict[str, tuple[str, str]]] = {
    "legacy@RP": {f: ("legacy_percentile", "RP") for f in C.FIELDS},
    "legacy@LOEO": {f: ("legacy_percentile", "LOEO") for f in C.FIELDS},
    "zscore@RP": {f: ("zscore", "RP") for f in C.FIELDS},
    "zscore@LOEO": {f: ("zscore", "LOEO") for f in C.FIELDS},
    "affine_clip@LOEO": {f: ("affine_clip", "LOEO") for f in C.FIELDS},
    "ecdf@LOEO": {f: ("ecdf", "LOEO") for f in C.FIELDS},
    "dirunify": {f: ("dirunify", "-") for f in C.FIELDS},
    "rawz@LOEO": {f: ("rawz", "LOEO") for f in C.FIELDS},
    "norm2_mix@LOEO": {
        "vision_0": ("neg_log_one_minus", "LOEO"),
        "vision_1": ("neg_log_one_minus", "LOEO"),
        "robot_state": ("affine_clip", "LOEO"),
    },
    "logit_mix@LOEO": {
        "vision_0": ("logit", "LOEO"),
        "vision_1": ("logit", "LOEO"),
        "robot_state": ("exp_l2", "LOEO"),
    },
    "power_mix@LOEO": {
        "vision_0": ("power", "LOEO"),
        "vision_1": ("power", "LOEO"),
        "robot_state": ("exp_l2", "LOEO"),
    },
}

SWEEP_CONFIGS = ["legacy@RP", "zscore@LOEO", "ecdf@LOEO", "affine_clip@LOEO"]


def tie_diag(S: np.ndarray, same_traj: np.ndarray) -> tuple[float, float]:
    Sm = np.where(same_traj, -np.inf, S)
    mx = Sm.max(axis=1, keepdims=True)
    ties = (Sm == mx).sum(axis=1)
    return float(np.mean(ties > 1)), float(ties.mean())


def top1_metrics(S, art):
    Sm = np.where(art.same_traj, -np.inf, S)
    top1 = Sm.argmax(axis=1)
    rows = np.arange(art.n)
    return {
        "top1_same_task": art.same_task[rows, top1].astype(np.float64),
        "action_mse@1": art.action_d2[rows, top1],
    }


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
        results[key] = {"prompt_probe": {}, "metrics": {}, "sweep": {}, "decision_region": {}}
        print(f"\n===== {key} (N={art.n}) =====")

        # ---- Part 0: prompt_emb historical probe -----------------------
        praw = art.raw["prompt_emb"].astype(np.float64)
        offdiag = ~np.eye(art.n, dtype=bool)
        rp = praw[offdiag]
        st_off = art.same_task[offdiag]
        p_leg = C.fit_legacy_percentile(rp, "cosine")
        same_at_one = float(np.mean(rp[st_off] > 1 - 1e-6))
        results[key]["prompt_probe"] = {
            "p5": p_leg["p5"], "p95": p_leg["p95"], "denom": p_leg["p95"] - p_leg["p5"],
            "same_task_frac_cos_eq_1": same_at_one,
            "same_task_pair_frac": float(st_off.mean()),
        }
        print(f"prompt_emb: p5={p_leg['p5']:.6f} p95={p_leg['p95']:.6f} "
              f"denom={p_leg['p95'] - p_leg['p5']:.6f} same-task cos==1 frac={same_at_one:.3f}")

        # ---- Part 1: retrieval metrics ----------------------------------
        weightings = {"w_prod": C.W_PROD, "w_unif": C.W_UNIF}
        norm_cache: dict[str, dict[str, np.ndarray]] = {}
        per_query_cache: dict[tuple[str, str], np.ndarray] = {}
        for cname, spec in CONFIGS.items():
            ns = build_norm_scores(art, spec)
            norm_cache[cname] = ns if cname in SWEEP_CONFIGS else None
            for wname, w in weightings.items():
                S = C.fuse(ns, w)
                m = C.retrieval_metrics(art, S)
                tie_frac, tie_size = tie_diag(S, art.same_traj)
                agg = {}
                for mk, vec in m.items():
                    mean, lo95, hi95 = C.cluster_bootstrap(vec, art.traj, n_boot=1000, seed=1)
                    agg[mk] = {"mean": mean, "lo": lo95, "hi": hi95}
                agg["tie_frac"] = tie_frac
                agg["tie_size_mean"] = tie_size
                results[key]["metrics"][f"{cname}|{wname}"] = agg
                per_query_cache[(cname, wname)] = m["top1_same_task"]
                print(f"  {cname:>18s}|{wname}: top1_task={agg['top1_same_task']['mean']:.3f} "
                      f"[{agg['top1_same_task']['lo']:.3f},{agg['top1_same_task']['hi']:.3f}] "
                      f"regret={agg['action_regret@1']['mean']:.3f} "
                      f"mrr={agg['mrr_same_task']['mean']:.3f} tie_frac={tie_frac:.3f} "
                      f"tie_sz={tie_size:.1f}")
            del ns

        # single-field raw baselines
        for f in C.FIELDS:
            x = C.orient(art.raw[f].astype(np.float64), C.SIM_TYPE[f])
            m = C.retrieval_metrics(art, x)
            agg = {mk: dict(zip(("mean", "lo", "hi"), C.cluster_bootstrap(v, art.traj, 1000, 1)))
                   for mk, v in m.items()}
            results[key]["metrics"][f"single_{f}"] = agg
            print(f"  single_{f}: top1_task={agg['top1_same_task']['mean']:.3f} "
                  f"regret={agg['action_regret@1']['mean']:.3f}")

        # paired deltas vs zscore@LOEO (production), production weights
        base = per_query_cache[("zscore@LOEO", "w_prod")]
        deltas = {}
        for cname in CONFIGS:
            if cname == "zscore@LOEO":
                continue
            d, lo95, hi95 = C.paired_cluster_bootstrap(
                per_query_cache[(cname, "w_prod")], base, art.traj, n_boot=2000, seed=2)
            deltas[cname] = {"delta_top1": d, "lo": lo95, "hi": hi95}
        results[key]["paired_delta_vs_zscoreLOEO"] = deltas

        # ---- Part 2: weight-simplex sweep --------------------------------
        step = 16
        grid = [(i / step, j / step, (step - i - j) / step)
                for i in range(step + 1) for j in range(step + 1 - i)]
        for cname in SWEEP_CONFIGS:
            ns = norm_cache[cname]
            A = ns["vision_0"]
            B = ns["vision_1"]
            R = ns["robot_state"]
            pts = []
            for w0, w1, w2 in grid:
                S = w0 * A + w1 * B + w2 * R
                mm = top1_metrics(S, art)
                tie_frac, tie_size = tie_diag(S, art.same_traj)
                pts.append({
                    "w": [w0, w1, w2],
                    "top1_same_task": float(mm["top1_same_task"].mean()),
                    "action_mse@1": float(mm["action_mse@1"].mean()),
                    "tie_frac": tie_frac,
                })
            vals = np.array([p["top1_same_task"] for p in pts])
            results[key]["sweep"][cname] = {
                "points": pts,
                "max": float(vals.max()), "mean": float(vals.mean()),
                "min": float(vals.min()),
                "range": float(vals.max() - vals.min()),
                "argmax_w": pts[int(vals.argmax())]["w"],
            }
            print(f"  sweep {cname:>18s}: top1_task max={vals.max():.3f} "
                  f"mean={vals.mean():.3f} range={vals.max() - vals.min():.3f} "
                  f"argmax_w={pts[int(vals.argmax())]['w']}")
            norm_cache[cname] = None

        # ---- Part 3: decision-region diagnostics -------------------------
        dr = {}
        ns_prod = build_norm_scores(art, CONFIGS["zscore@LOEO"])
        S_prod = C.fuse(ns_prod, C.W_PROD)
        Sm = np.where(art.same_traj, -np.inf, S_prod)
        top50 = np.argsort(-Sm, axis=1)[:, :50]
        rows = np.arange(art.n)[:, None]
        for f in C.FIELDS:
            raw = art.raw[f].astype(np.float64)
            stype = C.SIM_TYPE[f]
            pool = get_pool(art, f, "LOEO").astype(np.float64)
            p = C.fit_zscore(pool, stype)
            x = C.orient(raw, stype)
            z = (x - p["mu"]) / p["sigma"]
            zm = np.where(art.same_traj, -np.inf, z)
            ztop10 = -np.sort(-zm, axis=1)[:, :10]
            zmed = float(np.median(ztop10))
            dr[f] = {
                "z_top10_median": zmed,
                "z_top10_p90": float(np.percentile(ztop10, 90)),
                "tanh_gain_at_zmed": float(1 - np.tanh(zmed) ** 2),
                "std_norm_in_fused_top50": float(ns_prod[f][rows, top50].std()),
            }
            print(f"  decision region {f}: z_top10 med={zmed:.2f} "
                  f"tanh'={dr[f]['tanh_gain_at_zmed']:.4f} "
                  f"std(s|top50)={dr[f]['std_norm_in_fused_top50']:.4f}")
        results[key]["decision_region"] = dr

    (outdir / "expB_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nwritten: {outdir / 'expB_results.json'}")


if __name__ == "__main__":
    main()
