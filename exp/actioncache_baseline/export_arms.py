"""Export the CP2 arm ladder: IR-addressed threshold cuts (GST K=1) -> YAML + arm matrix.

For each tier (``n0`` = ActionCache N_hit=0, FULL_HIT@CP2; ``n1`` = N_hit=1,
WARM_START at the teacher's last snapshot) the shadow table's raw cosines are
inverted into one cut per target inference ratio::

    IR(theta) = [n(s>=theta)*c_tier + n(s<theta)*c_miss] / (N*M)

where ``c_tier`` / ``c_miss`` are the CP2 verdict costs and ``M`` the no-cache
teacher's cost (``libs.cp2_verdict_cost`` / ``teacher_forward_cost``). For the
GR00T arm the extra key-encoder forward ``E`` is inside every ``c_*`` and not
in ``M``, so an all-MISS arm prices above 100 %. Candidate cuts are the
observed scores (a percentile rule, GST K=1); the nearest attainable cut is
kept with its predicted IR and gap.

Target selection. The Pi0.5 line keeps its frozen ladder (targets on the CLI,
per-tier caps, drop reasons). The GR00T line (plan §3.9) asks for exactly
four targets per tier: the preferred ``{45,60,75,90}`` when all four resolve
to distinct finite cuts within ``max_gap``, otherwise the tier falls back to
the predefined shadow-only rule -- the distinct finite candidate cuts are
ranked by predicted IR, both ends are taken, then the candidates nearest 1/3
and 2/3 of that span (ties -> the higher cut); fewer than four distinct
candidates fails the export. GR00T targets are labelled ``t01..t04`` in both
selection paths; fallback targets carry the cut's predicted IR as ``target_ir``. One fixed reference arm per
tier uses the paper's default ``T_hit`` for the teacher (Pi0.5 0.85, GR00T
0.65). Every group is therefore 2 x (4 + 1) = 10 arms.

Preflight gate (GR00T, plan §3.11): the exporter refuses to emit without this
suite's certified encoder-cost record (``E``) and a passing decision-overhead
record from ``bench_cp2_overhead_groot.py``.

Every YAML is built from a single programmatic template, loaded back through
``load_cache_config`` and asserted field by field against the export record.

Usage:
  uv run python -m exp.actioncache_baseline.export_arms \\
      --suite libero_spatial --lib-tag lib50 --shadow-table <shadow.jsonl> \\
      --library-pkl </abs/path/cp2.pkl> --out-dir <dir> [--targets 60,65,...,95]
  uv run python -m exp.actioncache_baseline.export_arms --teacher groot_libero \\
      --suite libero_spatial --lib-tag w13s3 --shadow-table <shadow.jsonl> \\
      --library-pkl <cp2.pkl> --cost-record <cost_groot_libero_measured.json> \\
      --encoder-cost-record <cost_groot_cp2_encoded_<suite>.json> \\
      --preflight-record <overhead.json> --out-dir <dir>
"""

from __future__ import annotations

import argparse
import json
import math
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
#: GR00T (plan §3.9): exactly four targets per tier, preferred ladder first.
GROOT_PREFERRED_TARGETS = (45.0, 60.0, 75.0, 90.0)
GROOT_TARGETS_PER_TIER = 4
#: Overhead verdicts that let the emitter proceed (bench_cp2_overhead*).
PREFLIGHT_OK_VERDICTS = ("ok_report", "report_with_caption")


def _cost(table, profile: libs.TeacherProfile) -> libs.CostRecord:
    """A CostRecord from either a Pi0.5 table name or a record."""
    if isinstance(table, libs.CostRecord):
        return table
    if profile.name != libs.PI05.name:
        raise ValueError(f"{profile.name} pricing needs a CostRecord, got table {table!r}")
    return libs.pi05_cost_record(table)


# ------------------------------------------------------------------
# YAML template
# ------------------------------------------------------------------


def cp2_arm_yaml(*, preload_path: str, projection: libs.ProjectionArgs, tier: str,
                 theta_raw: float, profile: libs.TeacherProfile = libs.PI05) -> dict:
    """The deployed CP2 arm config for one (tier, cut)."""
    if tier not in profile.tiers:
        raise ValueError(f"unknown tier {tier!r}")
    tn = libs.theta_norm(theta_raw)
    if tier == "n0":
        judge = {"type": "threshold", "threshold": float(tn)}
    else:
        # FULL threshold above the score range so only the warm tier can fire.
        judge = {"type": "threshold", "threshold": libs.N1_FULL_THRESHOLD,
                 "warm_tiers": [{"threshold": float(tn), "start_t": float(profile.warm_start_t)}]}
    off = {"enabled": False, "weight": 0.0}
    doc = {
        "enabled": True,
        "timer": {"enabled": False},
        "keys": {
            "vision_0": dict(off), "vision_1": dict(off), "vision_2": dict(off),
            "prompt_emb": dict(off), "robot_state": dict(off),
            libs.FIELD: {"enabled": True, "weight": 1.0},
        },
        "key_builder": {
            "type": profile.builder,
            profile.projection_block: projection.builder_block(profile),
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
    if profile.denoise_schedule is not None:
        doc["denoise_schedule"] = profile.denoise_schedule
    return doc


def assert_arm_yaml(path: str | pathlib.Path, *, tier: str, theta_raw: float,
                    projection: libs.ProjectionArgs, preload_path: str,
                    profile: libs.TeacherProfile = libs.PI05) -> None:
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
    if cfg.key_builder.type != profile.builder:
        raise AssertionError(f"{path}: key_builder.type {cfg.key_builder.type!r} != {profile.builder!r}")
    kb = getattr(cfg.key_builder, profile.projection_block)
    want = projection.builder_block(profile)
    got = {k: getattr(kb, k) for k in want}
    if got != want:
        raise AssertionError(f"{path}: projection params {got} differ from record {want}")
    if dict(cfg.backend.vector_dims) != {libs.FIELD: projection.d}:
        raise AssertionError(f"{path}: backend vector_dims {dict(cfg.backend.vector_dims)}")
    if str(cfg.backend.in_memory.preload_path) != str(preload_path):
        raise AssertionError(f"{path}: preload_path {cfg.backend.in_memory.preload_path!r}")
    if libs.cp2_tier_of_config(cfg, profile) != tier:
        raise AssertionError(f"{path}: judge shape {libs.cp2_tier_of_config(cfg, profile)!r} != tier {tier!r}")
    cp2 = cfg.checkpoints["cp2"]
    tn = libs.theta_norm(theta_raw)
    if tier == "n0":
        if cp2.judge.warm_tiers or abs(cp2.judge.threshold - tn) > 1e-12:
            raise AssertionError(f"{path}: n0 judge {cp2.judge.threshold} tiers={cp2.judge.warm_tiers}")
    else:
        wt = cp2.judge.warm_tiers or []
        if abs(cp2.judge.threshold - libs.N1_FULL_THRESHOLD) > 1e-12 or len(wt) != 1 \
                or abs(wt[0]["threshold"] - tn) > 1e-12 \
                or abs(wt[0]["start_t"] - profile.warm_start_t) > 1e-9:
            raise AssertionError(f"{path}: n1 judge {cp2.judge.threshold} tiers={wt}")
    if cfg.write_policy.type != "never":
        raise AssertionError(f"{path}: write_policy {cfg.write_policy.type}")


# ------------------------------------------------------------------
# IR addressing (GST K=1)
# ------------------------------------------------------------------


def _unit_costs(tier: str, table, profile: libs.TeacherProfile) -> tuple[float, float, float]:
    """(c_tier, c_miss, M) for one tier under one pricing."""
    rec = _cost(table, profile)
    hit_type, start_t = profile.tiers[tier]
    return (libs.cp2_verdict_cost(rec, hit_type, start_t),
            libs.cp2_verdict_cost(rec, "MISS", None),
            libs.teacher_forward_cost(rec))


def ir_percent(s: np.ndarray, theta: float, tier: str, table=libs.DEFAULT_COST_TABLE,
               *, profile: libs.TeacherProfile = libs.PI05) -> float:
    """Predicted IR (percent) of admitting every shadow row with ``s >= theta``.

    ``-inf`` admits everything (the tier's floor), ``+inf`` admits nothing
    (100 % for Pi0.5; ``100*(M+E)/M`` for a teacher with an encoder cost).
    """
    c, m_miss, m = _unit_costs(tier, table, profile)
    admit = float(np.mean(np.asarray(s, dtype=np.float64) >= theta))
    return 100.0 * (admit * c + (1.0 - admit) * m_miss) / m


def attainable_range(s: np.ndarray, tier: str, table=libs.DEFAULT_COST_TABLE,
                     *, profile: libs.TeacherProfile = libs.PI05) -> tuple[float, float]:
    """(all admitted, none admitted) IR in percent."""
    return ir_percent(s, -np.inf, tier, table, profile=profile), ir_percent(s, np.inf, tier, table, profile=profile)


def candidate_cuts(s: np.ndarray, tier: str, table=libs.DEFAULT_COST_TABLE,
                   *, profile: libs.TeacherProfile = libs.PI05) -> list[dict]:
    """Every distinct finite cut the shadow scores can realise, with its IR.

    ``theta = s_(i)`` admits the rows with ``s >= theta``; duplicates collapse
    to the first index of each unique value so the admit counts are exact.
    Sorted by predicted IR ascending (== theta ascending).
    """
    s = np.asarray(s, dtype=np.float64)
    if s.size == 0:
        raise ValueError("empty shadow table")
    c, m_miss, m = _unit_costs(tier, table, profile)
    ss = np.sort(s)
    n = ss.size
    uniq, first = np.unique(ss, return_index=True)
    admit = (n - first) / n
    ir = 100.0 * (admit * c + (1.0 - admit) * m_miss) / m
    return [{"theta_raw": float(t), "predicted_ir": float(v), "admit_frac": float(a)}
            for t, v, a in zip(uniq, ir, admit)]


def invert_ir(s: np.ndarray, tier: str, target: float, *, table=libs.DEFAULT_COST_TABLE,
              max_gap: float = DEFAULT_MAX_GAP, profile: libs.TeacherProfile = libs.PI05) -> dict | None:
    """Nearest attainable cut for ``target`` IR (percent) on the shadow scores.

    Candidate cuts are the observed scores (admit = ``s >= theta``) plus
    ``+inf`` (nothing admitted). Ties prefer the higher cut (fewer admits).
    Returns None when the nearest IR is farther than ``max_gap`` points.
    """
    cands = candidate_cuts(s, tier, table, profile=profile)
    _, m_miss, m = _unit_costs(tier, table, profile)
    thetas = [c["theta_raw"] for c in cands] + [np.inf]
    irs = [c["predicted_ir"] for c in cands] + [100.0 * m_miss / m]
    admits = [c["admit_frac"] for c in cands] + [0.0]
    gaps = [abs(v - target) for v in irs]
    best = min(range(len(thetas)), key=lambda i: (gaps[i], -thetas[i] if np.isfinite(thetas[i]) else -np.inf))
    if gaps[best] > max_gap:
        return None
    return {"theta_raw": float(thetas[best]), "predicted_ir": float(irs[best]),
            "ir_gap": float(irs[best] - target), "admit_frac": float(admits[best])}


def plan_tier_targets(s: np.ndarray, tier: str, targets: Sequence[float], *, table,
                      max_gap: float, skipped: list[dict], cap: int,
                      profile: libs.TeacherProfile = libs.PI05) -> list[tuple[str, dict]]:
    """IR targets -> ``[(label, cut)]`` under the frozen per-tier budget.

    A target is omitted, with its reason appended to ``skipped``, when it
    lies below the tier's attainable floor (``below_tier_floor`` — the
    admit-everything IR, e.g. 60 % for N_hit=1 whose floor is 60.6 %), when no
    observed cut lands within ``max_gap`` IR points (``no_cut_within_max_gap``),
    when it resolves to the same cut as an already planned target
    (``duplicate_cut``), or — only if the survivors still exceed ``cap`` — as
    the lowest-IR targets (``tier_budget``, poorest resolution end).
    """
    lo, hi = attainable_range(s, tier, table, profile=profile)
    plan: list[tuple[str, dict]] = []
    seen_cuts: set[float] = set()
    for target in sorted(float(t) for t in targets):
        label = f"ir{int(round(target)):02d}"
        if target < lo:
            skipped.append({"tier": tier, "target_ir": target, "reason": "below_tier_floor",
                            "attainable": [lo, hi]})
            continue
        sol = invert_ir(s, tier, target, table=table, max_gap=max_gap, profile=profile)
        if sol is None:
            skipped.append({"tier": tier, "target_ir": target, "reason": "no_cut_within_max_gap",
                            "attainable": [lo, hi], "max_gap": max_gap})
            continue
        if sol["theta_raw"] in seen_cuts or not np.isfinite(sol["theta_raw"]):
            skipped.append({"tier": tier, "target_ir": target,
                            "reason": "duplicate_cut" if sol["theta_raw"] in seen_cuts else "infinite_cut",
                            "theta_raw": sol["theta_raw"]})
            continue
        seen_cuts.add(sol["theta_raw"])
        plan.append((label, {"target_ir": target, **sol}))
    while len(plan) > cap:
        label, sol = plan.pop(0)
        skipped.append({"tier": tier, "target_ir": sol["target_ir"], "reason": "tier_budget",
                        "cap": cap, "theta_raw": sol["theta_raw"]})
    return plan


def fallback_tier_targets(s: np.ndarray, tier: str, *, table, profile: libs.TeacherProfile,
                          n_targets: int = GROOT_TARGETS_PER_TIER) -> list[tuple[str, dict]]:
    """Plan §3.9 shadow-only fallback: both ends of the realisable IR span, then
    the unselected candidates nearest 1/3 and 2/3 of it (ties -> higher cut).

    ``target_ir`` of every fallback arm is the chosen cut's own predicted IR
    (there is no external target to miss). Fewer than ``n_targets`` distinct
    finite candidates is a failure, never padded with duplicates.
    """
    cands = candidate_cuts(s, tier, table, profile=profile)
    if len(cands) < n_targets:
        raise SystemExit(
            f"tier {tier}: only {len(cands)} distinct finite cuts in the shadow table; "
            f"{n_targets} are required (no duplicate arms are emitted)"
        )
    chosen: list[dict] = [cands[0], cands[-1]]
    lo, hi = cands[0]["predicted_ir"], cands[-1]["predicted_ir"]
    for frac in (1.0 / 3.0, 2.0 / 3.0):
        goal = lo + frac * (hi - lo)
        pool = [c for c in cands if c not in chosen]
        best = min(pool, key=lambda c: (abs(c["predicted_ir"] - goal), -c["theta_raw"]))
        chosen.append(best)
    chosen.sort(key=lambda c: c["predicted_ir"])
    plan: list[tuple[str, dict]] = []
    for i, c in enumerate(chosen, start=1):
        plan.append((f"t{i:02d}", {"target_ir": c["predicted_ir"], "theta_raw": c["theta_raw"],
                                    "predicted_ir": c["predicted_ir"], "ir_gap": 0.0,
                                    "admit_frac": c["admit_frac"]}))
    return plan


def plan_groot_tier(s: np.ndarray, tier: str, *, table: libs.CostRecord, max_gap: float,
                    preferred: Sequence[float] = GROOT_PREFERRED_TARGETS) -> tuple[list[tuple[str, dict]], dict]:
    """Preferred four targets, else the whole tier falls back (plan §3.9)."""
    skipped: list[dict] = []
    plan = plan_tier_targets(s, tier, preferred, table=table, max_gap=max_gap, skipped=skipped,
                             cap=GROOT_TARGETS_PER_TIER, profile=libs.GROOT_LIBERO)
    if len(plan) == GROOT_TARGETS_PER_TIER:
        return [(f"t{i:02d}", sol) for i, (_, sol) in enumerate(plan, 1)], {
                      "selection_rule": "preferred_targets", "preferred": list(preferred),
                      "dropped": skipped}
    fallback = fallback_tier_targets(s, tier, table=table, profile=libs.GROOT_LIBERO)
    cands = candidate_cuts(s, tier, table, profile=libs.GROOT_LIBERO)
    return fallback, {
        "selection_rule": "fallback_quartiles", "preferred": list(preferred), "dropped": skipped,
        "preferred_resolved": [{"label": lbl, **sol} for lbl, sol in plan],
        "candidate_range": {"n_candidates": len(cands), "ir_lo": cands[0]["predicted_ir"],
                            "ir_hi": cands[-1]["predicted_ir"]},
        "rule": "both ends of the distinct-cut IR span, then nearest unselected to 1/3 and 2/3 "
                "of the span (ties -> higher cut); target_ir = the cut's predicted IR",
    }


# ------------------------------------------------------------------
# Export
# ------------------------------------------------------------------


def load_shadow(path: str | pathlib.Path) -> tuple[np.ndarray, dict]:
    """``(s_raw scores, sidecar record)`` of a shadow table; the record is ``{}`` when the sidecar is absent."""
    rows = [json.loads(line) for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    s = np.asarray([r["s_raw"] for r in rows], dtype=np.float64)
    rec_path = pathlib.Path(path).with_suffix(".record.json")
    rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
    return s, rec


PREFLIGHT_CORE_SEGMENTS = ("cp2_encode", "cp2_collect", "cp2_build", "cp2_search", "cp2_judge")
PREFLIGHT_COLD = 50


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _shadow_row_problems(path: pathlib.Path, n_rows: int) -> list[str]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    episodes: dict[tuple[int, int], dict] = {}
    instructions: dict[int, str] = {}
    names: dict[str, tuple[int, int]] = {}
    if len(rows) != n_rows:
        return ["shadow row count changed while checking the table"]
    for i, row in enumerate(rows):
        if not all(isinstance(row.get(k), int) and not isinstance(row[k], bool) and row[k] >= 0
                   for k in ("task_id", "subset", "orig", "step_idx")):
            return [f"row {i}: missing or invalid episode / decision identity"]
        task, subset, orig, step = (row[k] for k in ("task_id", "subset", "orig", "step_idx"))
        key = (task, subset)
        if not (task < 10 and subset < 15 and orig < 50):
            return [f"row {i}: task / subset / original state is outside the frozen cohort"]
        if not _finite(row.get("s_raw")) or not -1.00001 <= row["s_raw"] <= 1.00001:
            return [f"row {i}: invalid cosine score"]
        if (not isinstance(row.get("task"), str) or not row["task"].strip()
                or not isinstance(row.get("episode"), str) or not row["episode"]
                or not isinstance(row.get("success"), bool)):
            return [f"row {i}: missing instruction / episode / success identity"]
        if instructions.setdefault(task, row["task"]) != row["task"]:
            return [f"row {i}: inconsistent instruction for task {task}"]
        if names.setdefault(row["episode"], key) != key:
            return [f"row {i}: episode name reused across cohort identities"]
        identity = (orig, row["episode"], row["success"])
        episode = episodes.setdefault(key, {"identity": identity, "steps": set()})
        if episode["identity"] != identity or step in episode["steps"]:
            return [f"row {i}: conflicting episode identity or duplicate decision"]
        episode["steps"].add(step)
    expected = {(task, subset) for task in range(10) for subset in range(15)}
    if episodes.keys() != expected:
        return [f"actual shadow coverage is {len(episodes)} episodes, expected all 10 x 15 identities"]
    for key, episode in episodes.items():
        if episode["steps"] != set(range(len(episode["steps"]))):
            return [f"episode {key}: shadow decision indices are not contiguous from zero"]
    for task in range(10):
        if len({episodes[task, subset]["identity"][0] for subset in range(15)}) != 15:
            return [f"task {task}: original states are not distinct"]
    return []


def check_shadow_record(rec: dict, *, shadow_path: pathlib.Path, n_rows: int, suite: str, library_sha256: str,
                        projection: dict, library_model_digest: str | None) -> dict:
    """The GR00T shadow table's provenance a frozen export requires (G2-B5).

    The sidecar written by ``build_shadow_table_groot.py`` must exist, name
    this teacher / suite / library / projection / schedule / model, cover the
    *complete* accepted cohort (``cohort_episodes == cohort_expected``,
    ``complete``, not ``limited``), and its ``n_rows`` / ``out_jsonl_sha256``
    must match the table actually being read -- a table copied, truncated or
    produced by a debug ``--limit-episodes`` run is not a frozen cohort.
    Returns the binding the export record stores.
    """
    prof = libs.GROOT_LIBERO
    problems: list[str] = []
    if not rec:
        raise SystemExit(f"{shadow_path}: no .record.json sidecar; a GR00T export needs the shadow table's provenance")
    for name in ("teacher", "suite", "library_sha256", "accepted_manifest_sha256", "cohort_manifest_sha256",
                 "cohort_episodes", "cohort_expected", "complete", "limited", "n_rows", "out_jsonl_sha256",
                 "projection", "schedule_id", "stage1_path", "model", "task_map_sha256"):
        if name not in rec:
            problems.append(f"missing field {name!r}")
    if problems:
        raise SystemExit(f"{shadow_path}: shadow record rejected: " + "; ".join(problems))
    if rec["teacher"] != prof.name:
        problems.append(f"teacher {rec['teacher']!r} != {prof.name!r}")
    if rec["suite"] != suite:
        problems.append(f"suite {rec['suite']!r} != {suite!r}")
    if rec["library_sha256"] != library_sha256:
        problems.append("library_sha256 differs from --library-pkl")
    if rec["projection"] != projection:
        problems.append("projection differs from the library's")
    if rec["schedule_id"] != prof.denoise_schedule or rec["stage1_path"] not in prof.stage1_paths:
        problems.append(f"schedule_id / stage1_path {rec['schedule_id']!r} / {rec['stage1_path']!r} are not the profile's")
    if rec["limited"] or not rec["complete"] or rec["cohort_episodes"] != rec["cohort_expected"]:
        problems.append(f"cohort is not complete (episodes {rec['cohort_episodes']} / {rec['cohort_expected']}, "
                        f"limited={rec['limited']}, complete={rec['complete']})")
    if int(rec["cohort_expected"]) != libs.SHADOW_COHORT_EPISODES:
        problems.append(f"cohort_expected {rec['cohort_expected']} != {libs.SHADOW_COHORT_EPISODES}")
    if int(rec["n_rows"]) != n_rows or n_rows <= 0:
        problems.append(f"record n_rows {rec['n_rows']} != {n_rows} rows read")
    if rec["out_jsonl_sha256"] != libs.sha256_file(shadow_path):
        problems.append("out_jsonl_sha256 differs from the table being read (copied or truncated after recording)")
    for name in ("accepted_manifest_sha256", "cohort_manifest_sha256", "task_map_sha256"):
        value = rec[name]
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            problems.append(f"{name} must be a full sha256")
    problems.extend(_shadow_row_problems(shadow_path, n_rows))
    model = rec["model"] if isinstance(rec["model"], dict) else {}
    lib_model = (model.get("library_model") or {}).get("weights_digest") or model.get("weights_digest")
    if not library_model_digest or lib_model != library_model_digest or not model.get("bound_to_library"):
        problems.append("shadow model binding does not name the library's weights_digest")
    if problems:
        raise SystemExit(f"{shadow_path}: shadow record rejected: " + "; ".join(problems))
    return {"path": str(shadow_path.resolve()), "sha256": rec["out_jsonl_sha256"], "n_rows": n_rows,
            "accepted_manifest_sha256": rec["accepted_manifest_sha256"],
            "cohort_manifest_sha256": rec["cohort_manifest_sha256"], "cohort_episodes": rec["cohort_episodes"],
            "task_map_sha256": rec.get("task_map_sha256")}


def check_preflight_record(path: str | pathlib.Path, *, suite: str, library_sha256: str,
                           accepted_manifest_sha256: str | None = None, projection: dict | None = None,
                           library_model_digest: str | None = None) -> dict:
    """The decision-overhead gate (plan §3.11) a GR00T export must have passed.

    Binds the record to this suite / library / accepted cohort / projection /
    model and *recomputes* the verdict from the warm total P95 (finite, from
    ``> 0`` warm decisions after exactly ``PREFLIGHT_COLD`` cold ones, with
    every core probe sampled on every decision) -- the stored ``verdict``
    label must agree and lie in ``PREFLIGHT_OK_VERDICTS``. Absent or
    over-budget measurements fail before any arm file is written (G2-B5).
    """
    p = pathlib.Path(path)
    rec = json.loads(p.read_text(encoding="utf-8"))
    problems = []
    if rec.get("record_kind") != "cp2_decision_overhead" or rec.get("teacher") != libs.GROOT_LIBERO.name:
        problems.append(f"not a {libs.GROOT_LIBERO.name} cp2_decision_overhead record")
    if rec.get("suite") != suite:
        problems.append(f"suite {rec.get('suite')!r} != {suite!r}")
    if rec.get("library_sha256") != library_sha256:
        problems.append("library_sha256 differs from --library-pkl")
    if accepted_manifest_sha256 is not None and rec.get("accepted_manifest_sha256") != accepted_manifest_sha256:
        problems.append("accepted_manifest_sha256 differs from the shadow table's cohort")
    if projection is not None and rec.get("projection") != projection:
        problems.append("projection differs from the library's")
    if library_model_digest is not None:
        bound = ((rec.get("model") or {}).get("bound") or {}).get("weights_digest")
        if bound != library_model_digest:
            problems.append("model binding does not name the library's weights_digest")
    if rec.get("schedule_id") != libs.GROOT_SCHEDULE_ID:
        problems.append(f"schedule_id {rec.get('schedule_id')!r} != {libs.GROOT_SCHEDULE_ID!r}")
    if rec.get("stage1_path") not in libs.GROOT_LIBERO.stage1_paths:
        problems.append("stage1_path is not the GR00T reconstructed-template path")
    if not rec.get("timer_enabled", False):
        problems.append("timer was not enabled")
    n = rec.get("n_decisions")
    cold_n = rec.get("cold_decisions")
    warm = rec.get("warm") or {}
    warm_n, p95 = warm.get("count"), warm.get("p95")
    if not isinstance(n, int) or n <= PREFLIGHT_COLD:
        problems.append(f"n_decisions {n!r} is not > {PREFLIGHT_COLD}")
    if cold_n != PREFLIGHT_COLD:
        problems.append(f"cold_decisions {cold_n!r} != {PREFLIGHT_COLD}")
    if not isinstance(warm_n, int) or warm_n <= 0 or (isinstance(n, int) and cold_n == PREFLIGHT_COLD and warm_n != n - cold_n):
        problems.append(f"warm.count {warm_n!r} is not n_decisions - {PREFLIGHT_COLD} > 0")
    if not _finite(p95) or p95 < 0:
        problems.append(f"warm.p95 {p95!r} is not a finite nonnegative measurement")
    seg = rec.get("per_segment") or {}
    for name in PREFLIGHT_CORE_SEGMENTS:
        segment = seg.get(name) or {}
        count = segment.get("count")
        if not isinstance(count, int) or count <= 0 or (isinstance(n, int) and count != n):
            problems.append(f"segment {name}: {count!r} samples for {n!r} decisions")
        for stat in ("median", "p95"):
            if not _finite(segment.get(stat)) or segment[stat] < 0:
                problems.append(f"segment {name}: {stat} must be finite and nonnegative")
    recomputed = libs.preflight_verdict(p95)
    if rec.get("verdict") != recomputed:
        problems.append(f"stored verdict {rec.get('verdict')!r} != recomputed {recomputed!r} from warm P95 {p95!r}")
    if recomputed not in PREFLIGHT_OK_VERDICTS:
        problems.append(f"verdict {recomputed!r} is not one of {PREFLIGHT_OK_VERDICTS}")
    if problems:
        raise SystemExit(f"{p}: preflight record rejected: " + "; ".join(problems))
    return {"path": str(p.resolve()), "sha256": libs.sha256_file(p), "verdict": recomputed,
            "warm_total_p95_ms": float(p95), "n_decisions": n}


def export(args: argparse.Namespace) -> dict:
    """Address the arms on the shadow distribution and write the yamls, matrix and export record.

    Pi0.5: the frozen ladder with the analytic tables. GR00T (``--teacher
    groot_libero``): ten arms per group, priced with the bound cost records,
    behind the shadow-provenance, encoder-cost and decision-overhead gates --
    every one fail-closed before the first arm file is written.
    """
    prof = libs.profile(args.teacher)
    s, shadow_rec = load_shadow(args.shadow_table)
    lib_meta = {k: v for k, v in libs.load_pickle(args.library_pkl).items() if k not in ("entries", "library_stats")}
    if lib_meta.get("key_builder_type") != prof.builder:
        raise SystemExit(f"--library-pkl was built by {lib_meta.get('key_builder_type')!r}, not {prof.builder!r}")
    proj_meta = lib_meta["projection"]
    projection = libs.ProjectionArgs.from_projection_meta(proj_meta)
    library_sha = libs.sha256_file(args.library_pkl)
    library_model_digest = (lib_meta.get("model") or {}).get("weights_digest")
    if shadow_rec and shadow_rec.get("library_sha256") not in (None, library_sha):
        raise SystemExit("shadow table was built against a different library than --library-pkl")
    shadow_binding = None
    if prof.name != libs.PI05.name:
        if lib_meta.get("teacher") != prof.name or lib_meta.get("schedule_id") != prof.denoise_schedule:
            raise SystemExit(f"--library-pkl is not a {prof.name} CP2 library stamped {prof.denoise_schedule!r}")
        if not library_model_digest:
            raise SystemExit("--library-pkl carries no model.weights_digest")
        shadow_binding = check_shadow_record(
            shadow_rec, shadow_path=pathlib.Path(args.shadow_table), n_rows=int(s.size), suite=args.suite,
            library_sha256=library_sha, projection=proj_meta, library_model_digest=library_model_digest)
    preload_path = str(pathlib.Path(args.library_pkl).resolve()) if args.deploy_library_path == "" else args.deploy_library_path
    out_dir = pathlib.Path(args.out_dir)
    tiers = [t.strip() for t in args.tiers.split(",") if t.strip()]
    ref_theta = prof.reference_theta_raw if args.ref_theta is None else float(args.ref_theta)
    if prof.name == libs.GROOT_LIBERO.name:
        if sorted(tiers) != ["n0", "n1"]:
            raise SystemExit("a frozen GR00T group requires exactly both distinct tiers n0,n1")
        if ref_theta != prof.reference_theta_raw:
            raise SystemExit("the frozen GR00T reference threshold is 0.65")
        if not _finite(args.max_gap) or not 0 <= args.max_gap <= DEFAULT_MAX_GAP:
            raise SystemExit("GR00T max_gap must be finite and within [0,1] IR points")

    # Pricing: Pi0.5 tables by name; GR00T from the two bound cost records,
    # behind the decision-overhead preflight gate.
    preflight = None
    if prof.name == libs.PI05.name:
        cost: libs.CostRecord | str = args.cost_table
        cost_rec = libs.pi05_cost_record(args.cost_table)
        targets = [float(t) for t in args.targets.split(",") if t.strip()]
    else:
        if not (args.cost_record and args.encoder_cost_record and args.preflight_record):
            raise SystemExit(f"{prof.name} export needs --cost-record, --encoder-cost-record and --preflight-record")
        cost_rec = libs.groot_cost_record(args.cost_record, args.encoder_cost_record, suite=args.suite)
        enc_layout = cost_rec.provenance.get("layout")
        if enc_layout != proj_meta.get("layout"):
            raise SystemExit(f"encoder cost record layout {enc_layout} != library layout {proj_meta.get('layout')}")
        enc_model = (cost_rec.provenance.get("model") or {}).get("weights_digest")
        if enc_model != library_model_digest:
            raise SystemExit(f"encoder cost record was measured on model {str(enc_model)[:16]}..., the library was "
                             f"built from {str(library_model_digest)[:16]}...")
        preflight = check_preflight_record(
            args.preflight_record, suite=args.suite, library_sha256=library_sha,
            accepted_manifest_sha256=shadow_binding["accepted_manifest_sha256"], projection=proj_meta,
            library_model_digest=library_model_digest)
        cost = cost_rec
        targets = [float(t) for t in args.targets.split(",") if t.strip()] if args.targets else list(GROOT_PREFERRED_TARGETS)
        if len(targets) != GROOT_TARGETS_PER_TIER or not all(_finite(t) for t in targets):
            raise SystemExit("GR00T addressing requires four finite preferred targets")

    arms: dict[str, dict] = {}
    skipped: list[dict] = []
    rows_out: list[dict] = []
    selection: dict[str, dict] = {}
    pending: dict[str, dict] = {}
    for tier in tiers:
        lo, hi = attainable_range(s, tier, cost, profile=prof)
        if prof.name == libs.PI05.name:
            plan = plan_tier_targets(s, tier, targets, table=cost, max_gap=args.max_gap,
                                     skipped=skipped, cap=TIER_TARGET_CAP[tier], profile=prof)
            selection[tier] = {"selection_rule": "ladder_with_caps", "targets": targets}
        else:
            plan, sel = plan_groot_tier(s, tier, table=cost_rec, max_gap=args.max_gap, preferred=targets)
            selection[tier] = sel
            skipped.extend(sel["dropped"])
        ref = {"target_ir": None, "theta_raw": float(ref_theta),
               "predicted_ir": ir_percent(s, ref_theta, tier, cost, profile=prof), "ir_gap": None,
               "admit_frac": float(np.mean(s >= ref_theta))}
        for _label, sol in plan:
            if abs(sol["theta_raw"] - ref_theta) < 1e-12:
                selection[tier]["reference_duplicates_target"] = True
        plan.append((f"ref{int(round(ref_theta * 1000)):03d}", ref))
        if len(plan) > TIER_TARGET_CAP[tier] + 1:
            raise SystemExit(f"tier {tier}: {len(plan)} arms exceed the frozen cap {TIER_TARGET_CAP[tier]} + reference")
        for label, sol in plan:
            arm = libs.arm_name(args.suite, args.lib_tag, tier, label)
            if arm in pending:
                raise SystemExit(f"duplicate arm id {arm}; refusing to overwrite a planned arm")
            doc = cp2_arm_yaml(preload_path=preload_path, projection=projection, tier=tier,
                               theta_raw=sol["theta_raw"], profile=prof)
            path = out_dir / "arms" / f"{arm}.yaml"
            pending[arm] = doc
            arms[arm] = {
                "arm_id": arm, "tier": tier, "hit_type": prof.tiers[tier][0],
                "start_t": prof.tiers[tier][1], **sol,
                "theta_norm": libs.theta_norm(sol["theta_raw"]),
                "yaml": str(path.resolve()),
            }
            rows_out.append({"arm": arm, "yaml": str(path.resolve()), "suite": args.suite})
    if len(rows_out) > GROUP_ARM_CAP:
        raise SystemExit(f"{len(rows_out)} arms exceed the frozen group cap {GROUP_ARM_CAP}")
    if prof.name != libs.PI05.name and len(arms) != 10:
        raise SystemExit(f"{prof.name}: expected 10 unique arms, got {len(arms)}")
    (out_dir / "arms").mkdir(parents=True, exist_ok=True)
    for arm, doc in pending.items():
        record_arm = arms[arm]
        path = pathlib.Path(record_arm["yaml"])
        path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
        assert_arm_yaml(path, tier=record_arm["tier"], theta_raw=record_arm["theta_raw"], projection=projection,
                        preload_path=preload_path, profile=prof)
        record_arm["yaml_sha256"] = libs.sha256_file(path)
    matrix = {"protocol": libs.PROTOCOL, "suite": args.suite, "rule": f"acb_{args.lib_tag}",
              "checkpoint": "cp2", "teacher": prof.name, "arms": rows_out}
    (out_dir / "arm_matrix.yaml").write_text(yaml.safe_dump(matrix, sort_keys=False), encoding="utf-8")
    record = {
        "protocol": libs.PROTOCOL, "suite": args.suite, "lib_tag": args.lib_tag, "teacher": prof.name,
        "cost": libs.cost_record_summary(cost_rec, prof),
        "cost_table": cost_rec.table, "targets": targets, "tiers": tiers, "max_gap": args.max_gap,
        "ref_theta_raw": ref_theta, "selection": selection, "preflight": preflight,
        "shadow_table": str(pathlib.Path(args.shadow_table).resolve()), "shadow_binding": shadow_binding,
        "shadow_record": shadow_rec, "n_rows": int(s.size),
        "library_pkl": str(pathlib.Path(args.library_pkl).resolve()), "library_sha256": library_sha,
        "deploy_library_path": preload_path, "projection": proj_meta,
        "library_model": lib_meta.get("model"), "schedule_id": lib_meta.get("schedule_id"),
        "attainable_ir": {t: list(attainable_range(s, t, cost, profile=prof)) for t in tiers},
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
    ap.add_argument("--teacher", default=libs.PI05.name, choices=sorted(libs.PROFILES))
    ap.add_argument("--suite", required=True, choices=sorted(libs.SUITE_TAGS))
    ap.add_argument("--lib-tag", required=True, help="library tag in the arm id, e.g. lib50 / s6 / w13s3")
    ap.add_argument("--shadow-table", required=True)
    ap.add_argument("--library-pkl", required=True, help="the CP2 library (local path, hashed)")
    ap.add_argument("--deploy-library-path", default="",
                    help="preload_path written into the yamls (server-side path); default = resolved --library-pkl")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--targets", default="",
                    help="comma list of IR targets; default = 60..95 step 5 (pi05) / 45,60,75,90 (groot)")
    ap.add_argument("--tiers", default="n0,n1")
    ap.add_argument("--ref-theta", type=float, default=None,
                    help="reference arm raw cosine; default = the paper's T_hit for the teacher")
    ap.add_argument("--max-gap", type=float, default=DEFAULT_MAX_GAP)
    ap.add_argument("--cost-table", default=libs.DEFAULT_COST_TABLE, choices=sorted(libs.COST_TABLES),
                    help="pi05 only")
    ap.add_argument("--cost-record", default="", help="groot: the measured teacher cost JSON")
    ap.add_argument("--encoder-cost-record", default="", help="groot: this suite's encoder cost JSON (E)")
    ap.add_argument("--preflight-record", default="", help="groot: overhead.json from bench_cp2_overhead_groot")
    args = ap.parse_args()
    if not args.targets and args.teacher == libs.PI05.name:
        args.targets = ",".join(str(t) for t in DEFAULT_TARGETS)
    rec = export(args)
    print(json.dumps({"arms": len(rec["arms"]), "skipped": rec["skipped"], "selection": rec["selection"]}, default=str))


if __name__ == "__main__":
    main()
