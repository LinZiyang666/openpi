"""Export the CP2 arm ladder: IR-addressed threshold cuts (GST K=1) -> YAML + arm matrix.

For each tier (``n0`` = ActionCache N_hit=0, FULL_HIT@CP2; ``n1`` = N_hit=1,
WARM_START@0.1) the shadow table's raw cosines are inverted into one cut per
target inference ratio: ``IR(theta) = [n(s>=theta)*c_tier + n(s<theta)*c_miss]
/ (N*c_miss)`` over the shadow rows, the candidate cuts are the observed
scores (a percentile rule, GST K=1) and the nearest attainable cut is kept
with its predicted IR and gap (targets whose gap exceeds ``--max-gap`` are
dropped). Two fixed reference arms use ActionCache's published default
``T_hit = 0.85`` (raw cosine).

Every YAML is built from a single programmatic template (no file template),
then loaded back through ``load_cache_config`` and asserted field by field
against the export record, so the deployed judge / tier / normalisation match
what the record claims. ``theta_raw`` (cosine) and ``theta_norm =
(theta_raw+1)/2`` (the ``affine_clip`` score the judge compares) are both
recorded.

Usage:
  uv run python -m exp.actioncache_baseline.export_arms \\
      --suite libero_spatial --lib-tag lib50 --shadow-table <shadow.jsonl> \\
      --library-pkl </abs/path/cp2.pkl> --out-dir <dir> [--targets 60,65,...,95]
"""

from __future__ import annotations

import argparse
import json
import pathlib

from typing import Sequence

import numpy as np
import yaml

from exp.actioncache_baseline import libs
from openpi.cache.config import load_cache_config

DEFAULT_TARGETS = tuple(range(60, 100, 5))
DEFAULT_REF_THETA = 0.85  # ActionCache Table 5 default T_hit for pi0.5
DEFAULT_MAX_GAP = 1.0     # IR points
TIER_TARGET_CAP = libs.TIER_TARGET_CAP   # n0 <= 8, n1 <= 7 target arms (+1 reference each)
GROUP_ARM_CAP = libs.GROUP_ARM_CAP       # <= 17 arms per (suite, library) group


# ------------------------------------------------------------------
# YAML template
# ------------------------------------------------------------------


def cp2_arm_yaml(*, preload_path: str, projection: libs.ProjectionArgs, tier: str,
                 theta_raw: float) -> dict:
    """The deployed CP2 arm config for one (tier, cut)."""
    if tier not in libs.TIERS:
        raise ValueError(f"unknown tier {tier!r}")
    tn = libs.theta_norm(theta_raw)
    if tier == "n0":
        judge = {"type": "threshold", "threshold": float(tn)}
    else:
        # FULL threshold above the score range so only the warm tier can fire.
        judge = {"type": "threshold", "threshold": libs.N1_FULL_THRESHOLD,
                 "warm_tiers": [{"threshold": float(tn), "start_t": float(libs.WARM_START_T)}]}
    off = {"enabled": False, "weight": 0.0}
    return {
        "enabled": True,
        "timer": {"enabled": False},
        "keys": {
            "vision_0": dict(off), "vision_1": dict(off), "vision_2": dict(off),
            "prompt_emb": dict(off), "robot_state": dict(off),
            libs.FIELD: {"enabled": True, "weight": 1.0},
        },
        "key_builder": {
            "type": libs.KEY_BUILDER_TYPE,
            "cp2_vlm": {"seed": int(projection.seed), "d": int(projection.d),
                        "p": float(projection.p), "input_dim": int(projection.input_dim)},
        },
        "checkpoints": {
            "cp2": {
                "enabled": True,
                "gate": {"type": "always_search"},
                "judge": judge,
                "search_strategy": {
                    "type": "weighted_score_sum_knn",
                    "top_k": 1,
                    "step_filter": "all",
                    "task_scoped": False,
                    "field_similarity": {libs.FIELD: {"type": "cosine"}},
                    "score_normalization": {
                        "type": "per_field",
                        "fields": {libs.FIELD: {"method": "affine_clip",
                                                "params": {"lo": -1.0, "hi": 1.0}}},
                    },
                },
            }
        },
        "backend": {
            "type": "in_memory",
            "vector_dims": {libs.FIELD: int(projection.d)},
            "in_memory": {"preload_path": str(preload_path), "index_type": "brute_force"},
        },
        "write_policy": {"type": "never"},
    }


def assert_arm_yaml(path: str | pathlib.Path, *, tier: str, theta_raw: float,
                    projection: libs.ProjectionArgs, preload_path: str) -> None:
    """Load-and-assert: the deployed config must say exactly what the record says.

    The protocol clauses (single-step top-1, whole-suite, cosine + affine_clip,
    N_hit judge shapes, write never, ...) are the shared
    ``libs.cp2_contract_problems`` — the same function the runner applies
    before issuing rollouts — plus the record-specific bindings below.
    """
    cfg = load_cache_config(path)
    problems = libs.cp2_contract_problems(cfg)
    if problems:
        raise AssertionError(f"{path}: " + "; ".join(problems))
    kb = cfg.key_builder.cp2_vlm
    if (kb.seed, kb.d, kb.p, kb.input_dim) != (projection.seed, projection.d, projection.p, projection.input_dim):
        raise AssertionError(f"{path}: projection params differ from record")
    if dict(cfg.backend.vector_dims) != {libs.FIELD: projection.d}:
        raise AssertionError(f"{path}: backend vector_dims {dict(cfg.backend.vector_dims)}")
    if str(cfg.backend.in_memory.preload_path) != str(preload_path):
        raise AssertionError(f"{path}: preload_path {cfg.backend.in_memory.preload_path!r}")
    if libs.cp2_tier_of_config(cfg) != tier:
        raise AssertionError(f"{path}: judge shape {libs.cp2_tier_of_config(cfg)!r} != tier {tier!r}")
    cp2 = cfg.checkpoints["cp2"]
    tn = libs.theta_norm(theta_raw)
    if tier == "n0":
        if cp2.judge.warm_tiers or abs(cp2.judge.threshold - tn) > 1e-12:
            raise AssertionError(f"{path}: n0 judge {cp2.judge.threshold} tiers={cp2.judge.warm_tiers}")
    else:
        wt = cp2.judge.warm_tiers or []
        if abs(cp2.judge.threshold - libs.N1_FULL_THRESHOLD) > 1e-12 or len(wt) != 1 \
                or abs(wt[0]["threshold"] - tn) > 1e-12 \
                or abs(wt[0]["start_t"] - libs.WARM_START_T) > 1e-9:
            raise AssertionError(f"{path}: n1 judge {cp2.judge.threshold} tiers={wt}")
    if cfg.write_policy.type != "never":
        raise AssertionError(f"{path}: write_policy {cfg.write_policy.type}")


# ------------------------------------------------------------------
# IR addressing (GST K=1)
# ------------------------------------------------------------------


def ir_percent(s: np.ndarray, theta: float, tier: str, table: str = libs.DEFAULT_COST_TABLE) -> float:
    """Predicted IR (percent) of admitting every shadow row with ``s >= theta``.

    ``-inf`` admits everything (the tier's floor), ``+inf`` admits nothing (100).
    """
    hit_type, start_t = libs.TIERS[tier]
    c = libs.cp2_tier_cost(hit_type, start_t, table)
    m = libs.miss_cost(table)
    admit = float(np.mean(np.asarray(s, dtype=np.float64) >= theta))
    return 100.0 * (admit * c + (1.0 - admit) * m) / m


def attainable_range(s: np.ndarray, tier: str, table: str = libs.DEFAULT_COST_TABLE) -> tuple[float, float]:
    """(all admitted, none admitted) IR in percent."""
    return ir_percent(s, -np.inf, tier, table), 100.0


def invert_ir(s: np.ndarray, tier: str, target: float, *, table: str = libs.DEFAULT_COST_TABLE,
              max_gap: float = DEFAULT_MAX_GAP) -> dict | None:
    """Nearest attainable cut for ``target`` IR (percent) on the shadow scores.

    Candidate cuts are the observed scores (admit = ``s >= theta``) plus
    ``+inf`` (nothing admitted). Ties prefer the higher cut (fewer admits).
    Returns None when the nearest IR is farther than ``max_gap`` points.
    """
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0:
        raise ValueError("empty shadow table")
    hit_type, start_t = libs.TIERS[tier]
    c = libs.cp2_tier_cost(hit_type, start_t, table)
    m = libs.miss_cost(table)
    ss = np.sort(s)  # ascending
    n = ss.size
    # theta = ss[i] admits the n - i rows with s >= ss[i] (duplicates: use the
    # first index of each unique value so admit counts are exact).
    uniq, first = np.unique(ss, return_index=True)
    admit = (n - first) / n
    ir = 100.0 * (admit * c + (1.0 - admit) * m) / m
    thetas = list(uniq) + [np.inf]
    irs = list(ir) + [100.0]
    gaps = [abs(v - target) for v in irs]
    best = min(range(len(thetas)), key=lambda i: (gaps[i], -thetas[i] if np.isfinite(thetas[i]) else -np.inf))
    if gaps[best] > max_gap:
        return None
    return {"theta_raw": float(thetas[best]), "predicted_ir": float(irs[best]),
            "ir_gap": float(irs[best] - target), "admit_frac": float(0.0 if not np.isfinite(thetas[best]) else (n - int(np.searchsorted(ss, thetas[best], side="left"))) / n)}


def plan_tier_targets(s: np.ndarray, tier: str, targets: Sequence[float], *, table: str,
                      max_gap: float, skipped: list[dict], cap: int) -> list[tuple[str, dict]]:
    """IR targets -> ``[(label, cut)]`` under the frozen per-tier budget.

    A target is omitted, with its reason appended to ``skipped``, when it
    lies below the tier's attainable floor (``below_tier_floor`` — the
    admit-everything IR, e.g. 60 % for N_hit=1 whose floor is 60.6 %), when no
    observed cut lands within ``max_gap`` IR points (``no_cut_within_max_gap``),
    when it resolves to the same cut as an already planned target
    (``duplicate_cut``), or — only if the survivors still exceed ``cap`` — as
    the lowest-IR targets (``tier_budget``, poorest resolution end).
    """
    lo, hi = attainable_range(s, tier, table)
    plan: list[tuple[str, dict]] = []
    seen_cuts: set[float] = set()
    for target in sorted(float(t) for t in targets):
        label = f"ir{int(round(target)):02d}"
        if target < lo:
            skipped.append({"tier": tier, "target_ir": target, "reason": "below_tier_floor",
                            "attainable": [lo, hi]})
            continue
        sol = invert_ir(s, tier, target, table=table, max_gap=max_gap)
        if sol is None:
            skipped.append({"tier": tier, "target_ir": target, "reason": "no_cut_within_max_gap",
                            "attainable": [lo, hi], "max_gap": max_gap})
            continue
        if sol["theta_raw"] in seen_cuts:
            skipped.append({"tier": tier, "target_ir": target, "reason": "duplicate_cut",
                            "theta_raw": sol["theta_raw"]})
            continue
        seen_cuts.add(sol["theta_raw"])
        plan.append((label, {"target_ir": target, **sol}))
    while len(plan) > cap:
        label, sol = plan.pop(0)
        skipped.append({"tier": tier, "target_ir": sol["target_ir"], "reason": "tier_budget",
                        "cap": cap, "theta_raw": sol["theta_raw"]})
    return plan


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------


def load_shadow(path: str | pathlib.Path) -> tuple[np.ndarray, dict]:
    rows = [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    s = np.asarray([r["s_raw"] for r in rows], dtype=np.float64)
    rec_path = pathlib.Path(path).with_suffix(".record.json")
    rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
    return s, rec


def export(args: argparse.Namespace) -> dict:
    s, shadow_rec = load_shadow(args.shadow_table)
    lib_meta = {k: v for k, v in libs.load_pickle(args.library_pkl).items() if k not in ("entries", "library_stats")}
    proj_meta = lib_meta["projection"]
    projection = libs.ProjectionArgs(seed=int(proj_meta["seed"]), d=int(proj_meta["d"]),
                                     p=float(proj_meta["p"]), input_dim=int(proj_meta["D"]))
    library_sha = libs.sha256_file(args.library_pkl)
    if shadow_rec and shadow_rec.get("library_sha256") not in (None, library_sha):
        raise SystemExit("shadow table was built against a different library than --library-pkl")
    preload_path = str(pathlib.Path(args.library_pkl).resolve()) if args.deploy_library_path == "" else args.deploy_library_path
    out_dir = pathlib.Path(args.out_dir)
    (out_dir / "arms").mkdir(parents=True, exist_ok=True)
    targets = [float(t) for t in args.targets.split(",") if t.strip()]
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]

    arms: dict[str, dict] = {}
    skipped: list[dict] = []
    rows_out: list[dict] = []
    for tier in tiers:
        lo, hi = attainable_range(s, tier, args.cost_table)
        plan = plan_tier_targets(s, tier, targets, table=args.cost_table, max_gap=args.max_gap,
                                 skipped=skipped, cap=TIER_TARGET_CAP[tier])
        ref = {"target_ir": None, "theta_raw": float(args.ref_theta),
               "predicted_ir": ir_percent(s, args.ref_theta, tier, args.cost_table), "ir_gap": None,
               "admit_frac": float(np.mean(s >= args.ref_theta))}
        plan.append((f"ref{int(round(args.ref_theta * 1000)):03d}", ref))
        if len(plan) > TIER_TARGET_CAP[tier] + 1:
            raise SystemExit(f"tier {tier}: {len(plan)} arms exceed the frozen cap {TIER_TARGET_CAP[tier]} + reference")
        for label, sol in plan:
            arm = libs.arm_name(args.suite, args.lib_tag, tier, label)
            doc = cp2_arm_yaml(preload_path=preload_path, projection=projection, tier=tier,
                               theta_raw=sol["theta_raw"])
            path = out_dir / "arms" / f"{arm}.yaml"
            path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
            assert_arm_yaml(path, tier=tier, theta_raw=sol["theta_raw"], projection=projection,
                            preload_path=preload_path)
            arms[arm] = {
                "arm_id": arm, "tier": tier, "hit_type": libs.TIERS[tier][0],
                "start_t": libs.TIERS[tier][1], **sol,
                "theta_norm": libs.theta_norm(sol["theta_raw"]),
                "yaml": str(path.resolve()), "yaml_sha256": libs.sha256_file(path),
            }
            rows_out.append({"arm": arm, "yaml": str(path.resolve()), "suite": args.suite})
    if len(rows_out) > GROUP_ARM_CAP:
        raise SystemExit(f"{len(rows_out)} arms exceed the frozen group cap {GROUP_ARM_CAP}")
    matrix = {"protocol": libs.PROTOCOL, "suite": args.suite, "rule": f"acb_{args.lib_tag}",
              "checkpoint": "cp2", "arms": rows_out}
    (out_dir / "arm_matrix.yaml").write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    record = {
        "protocol": libs.PROTOCOL, "suite": args.suite, "lib_tag": args.lib_tag,
        "cost_table": args.cost_table, "targets": targets, "tiers": tiers, "max_gap": args.max_gap,
        "ref_theta_raw": args.ref_theta,
        "shadow_table": str(pathlib.Path(args.shadow_table).resolve()),
        "shadow_record": shadow_rec, "n_rows": int(s.size),
        "library_pkl": str(pathlib.Path(args.library_pkl).resolve()), "library_sha256": library_sha,
        "deploy_library_path": preload_path, "projection": proj_meta,
        "attainable_ir": {t: list(attainable_range(s, t, args.cost_table)) for t in tiers},
        "budget": {"tier_target_cap": dict(TIER_TARGET_CAP), "group_arm_cap": GROUP_ARM_CAP,
                   "arms_per_tier": {t: sum(1 for a in arms.values() if a["tier"] == t) for t in tiers},
                   "total_arms": len(rows_out)},
        "arms": arms, "skipped": skipped,
        "arm_matrix": str((out_dir / "arm_matrix.yaml").resolve()),
        "git_commit": libs.git_commit(),
    }
    libs.dump_json(out_dir / "export_record.json", record)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=sorted(libs.SUITE_TAGS))
    ap.add_argument("--lib-tag", required=True, help="library regime tag, e.g. lib50 / libS6")
    ap.add_argument("--shadow-table", required=True)
    ap.add_argument("--library-pkl", required=True, help="local CP2 library (metadata + digest source)")
    ap.add_argument("--deploy-library-path", default="",
                    help="absolute path of the same library on the server host (defaults to --library-pkl)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--targets", default=",".join(str(t) for t in DEFAULT_TARGETS))
    ap.add_argument("--tiers", default="n0,n1")
    ap.add_argument("--ref-theta", type=float, default=DEFAULT_REF_THETA)
    ap.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP)
    ap.add_argument("--cost-table", default=libs.DEFAULT_COST_TABLE, choices=sorted(libs.COST_TABLES))
    args = ap.parse_args()
    rec = export(args)
    for arm, r in rec["arms"].items():
        print(f"{arm}: theta_raw={r['theta_raw']:.4f} pred_ir={r['predicted_ir']:.2f} gap={r['ir_gap']}")
    print(f"{len(rec['arms'])} arms, {len(rec['skipped'])} targets skipped -> {rec['arm_matrix']}")


if __name__ == "__main__":
    main()
