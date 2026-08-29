"""Precheck adjudicator: one joint replicate, endpoint-clamped frontier, gates.

Implements plan section 4.6 as approved on 2026-08-27 (cost-axis change,
Review Authority conditional approval -- see
``logs/dispatch_surface_cost_axis_change.md`` sections 9 and 10), with the
discipline rules as hard code paths (refusals, not warnings):

  * cost is ANALYTIC, not measured. Per decision it is the verdict weighted by
    the frozen CUDA-graph stage costs; an arm's cost is the decision-weighted
    ratio-of-sums ``sum_d c(h_d) / N_decisions``, re-formed inside every
    bootstrap replicate from the resampled init clusters. Averaging
    per-episode means over unequal-length episodes would estimate the cost of
    a random EPISODE and is not what this line claims.
  * ONE resample per replicate: task-stratified, init-clustered, shared by
    every arm, with SR, cost and the frontier interpolation all computed
    inside it. Gate 2's intersection-union test depends on the joint
    distribution, so the two axes must never be sampled separately.
  * refuses input without a frozen primary delta (fit_record.delta_star must
    equal the primary artifact's delta);
  * refuses arm sets whose journals disagree on the paired (task, init) grid,
    and requires the grid to cover the pre-registered arm x task x init
    lattice exactly -- one accepted episode per cell, no gaps, no duplicates;
  * refuses per_step input whose decision count disagrees with the episode's
    inference count, whose ``hit_type`` is outside the three verdicts, or
    whose WARM_START rows carry a start_t other than the pinned tier. Missing
    or unknown verdicts are never silently billed as MISS and never silently
    drop an episode;
  * every replicate produces exactly one endpoint-clamped record
    {branch, D_sr, D_c} -- low replicates clamp to the argmin-SR cell and STAY
    in the cost distribution (no sample deletion path exists);
  * gate order and the stop-after-Gate-2 rule are hard-coded; there is no
    multi-point selection path (SV+/- never enter a gate).

Outputs verdict JSON + descriptive tables. Exit code 0 always (the verdict
field carries the outcome); refusals raise SystemExit.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib

import numpy as np

from exp.dispatch_surface.run_precheck import (
    FROZEN_LAUNCH_KEYS,
    LAYER_PRIMARY,
    LAYER_SECONDARY,
    PROTOCOL,
    SECONDARY_CORE_ARMS,
    official_test_inits,
    validate_matrix_artifacts,
)

B_REPLICATES = 10_000
CORE_T_ARMS = ("dsp_t_fh30_ws20", "dsp_t_fh50_ws20", "dsp_t_fh70_ws10")
ARM_SV, ARM_S0 = "dsp_sv", "dsp_s0"
COMPUTE_GATE = -0.05        # Gate 1: D_c upper quantile must be <= this
# Rev 1 Gate 2. The old rule required SV to RAISE SR while merely not costing
# more (<= +5%). That tests the wrong direction: SV's structural edge over S0 is
# that v lets it move steps from WARM to FULL, so it buys cost at roughly equal
# safety -- true Pareto dominance (same SR, clearly cheaper) would have FAILED
# the old gate. Rev 1 asks for SR non-inferiority and a real cost saving, the
# same shape as Gate 1.
GATE2_COMPUTE_GATE = -0.05  # Gate 2: relative cost saving SV must show over S0
# Upper-quantile levels. Gate 1 keeps its p95; Gate 2's cost condition is a
# ONE-SIDED 95% upper bound (ruling 9.3): the previous p97.5 spent a third of
# the gate's power on a guard, while a point estimate would have had no
# type-I control at all -- at a true +5% it releases with probability 0.5.
GATE1_UPPER_Q = 0.95
GATE1_LOWER_Q = 0.05
GATE2_UPPER_Q = 0.95
# Gate 2's SR condition is frozen at the ruling's one-sided q0.05 lower bound.
GATE2_SR_LOWER_Q = 0.05

# Frozen per-stage GPU cost, E0 latency_bench CUDA-graph three-stage cell
# (pi05_compile_ro_3stage.json). The eager and default cells describe an
# unoptimised system and must not be used.
STAGE1_MS = 10.260266
STAGE2_MS = 27.686469
STAGE3_MS = 29.571860
PINNED_START_T_WS = 0.3
VERDICTS = ("FULL_HIT", "WARM_START", "MISS")
EXPECTED_TASKS = 10
EXPECTED_TRIALS = 30        # frozen A' quota per task (plan 4.2)


def _is_json_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def unit_cost(hit_type: str, start_t) -> float:
    """Analytic per-decision GPU inference cost of one verdict.

    FULL_HIT pays stage1 only. WARM_START resumes the flow at ``start_t`` and
    steps down to 0, running ``round(start_t * num_steps)`` of the stage-3
    steps -- start_t=0.3 runs 3 of 10 and saves 70% (pi0_pytorch.py:691).
    MISS pays all three stages. Anything else is a refusal, never a default.
    """
    if hit_type == "FULL_HIT":
        return STAGE1_MS
    if hit_type == "WARM_START":
        if start_t is None or float(start_t) != PINNED_START_T_WS:
            raise SystemExit(
                f"WARM_START row carries start_t={start_t!r}, expected the pinned "
                f"{PINNED_START_T_WS} — refusing to price an untested warm tier"
            )
        return STAGE1_MS + STAGE2_MS + PINNED_START_T_WS * STAGE3_MS
    if hit_type == "MISS":
        return STAGE1_MS + STAGE2_MS + STAGE3_MS
    raise SystemExit(
        f"unknown hit_type {hit_type!r}; only {VERDICTS} may be priced. Missing "
        "or unprobed verdicts are never silently billed as MISS"
    )


# ------------------------------------------------------------------
# Frontier / replicate statistics (pure functions, unit-tested)
# ------------------------------------------------------------------


def frontier_record(t_arms: list[tuple[str, float, float]],
                    sv: tuple[float, float]) -> dict:
    """One replicate's endpoint-clamped record.

    Args:
        t_arms: [(arm_id, SR, cost)] for the three baselines.
        sv: (SR, cost) of the surface arm.

    Returns {branch, D_sr, D_c}. All branches produce D_c; low clamps to the
    argmin-SR cell, high to the argmax-SR cell.
    """
    pts = sorted(t_arms, key=lambda t: (t[1], t[0]))  # stable (SR, arm_id) order
    srs = [p[1] for p in pts]
    sr_sv, c_sv = sv
    if sr_sv < srs[0]:
        branch, c_m = "low", pts[0][2]
    elif sr_sv > srs[-1]:
        branch, c_m = "high", pts[-1][2]
    else:
        branch = "bracket"
        hi = next(i for i in range(len(srs)) if srs[i] >= sr_sv)
        if srs[hi] == sr_sv or hi == 0:
            c_m = pts[hi][2]
        else:
            lo = hi - 1
            span = srs[hi] - srs[lo]
            t = 0.0 if span <= 0 else (sr_sv - srs[lo]) / span
            c_m = pts[lo][2] + t * (pts[hi][2] - pts[lo][2])
    if not (np.isfinite(c_m) and c_m > 0):
        raise SystemExit(f"comparator has non-positive cost (c={c_m})")
    sr_m = min(max(sr_sv, srs[0]), srs[-1])
    return {"branch": branch, "D_sr": sr_sv - sr_m, "D_c": (c_sv - c_m) / c_m}


def gate1(records: list[dict]) -> dict:
    """SV vs the endpoint-clamped threshold frontier: SR floor + cost saving."""
    d_sr = np.array([r["D_sr"] for r in records])
    d_c = np.array([r["D_c"] for r in records])
    out = {
        "d_sr_p5": float(np.quantile(d_sr, GATE1_LOWER_Q)),
        "d_sr_lower_quantile": GATE1_LOWER_Q,
        "d_c_p95": float(np.quantile(d_c, GATE1_UPPER_Q)),
        "d_c_upper_quantile": GATE1_UPPER_Q,
        "cost_gate": COMPUTE_GATE,
        "branch_shares": {b: float(np.mean([r["branch"] == b for r in records]))
                          for b in ("bracket", "high", "low")},
    }
    out["pass"] = bool(out["d_sr_p5"] >= 0.0 and out["d_c_p95"] <= COMPUTE_GATE)
    return out


# Fields the retired Gate 2 wrote. Their presence means the record was produced
# under the pre-Rev-1 rule (SR strictly up, cost merely not up), whose verdict
# is not comparable with this one; refuse rather than silently re-adjudicate.
RETIRED_GATE2_FIELDS = ("gate2_compute_slack", "dc_upper_slack")


def reject_retired_gate2(record: dict, where: str) -> None:
    """Fail closed on an artefact carrying the pre-Rev-1 Gate 2 encoding."""
    stale = [f for f in RETIRED_GATE2_FIELDS if f in record]
    if stale:
        raise SystemExit(
            f"{where} carries retired Gate 2 field(s) {stale}; Rev 1 Gate 2 is "
            "SR non-inferiority plus a >=5% cost saving and cannot re-use them"
        )
    slack = record.get("gate2", {}).get("cost_gate") if isinstance(record.get("gate2"), dict) else None
    if slack is not None and slack > 0:
        raise SystemExit(
            f"{where}: gate2.cost_gate={slack} is a cost-increase ceiling, the "
            "retired rule; Rev 1 requires a negative saving gate"
        )


def gate2(d_sr: np.ndarray, d_c: np.ndarray) -> dict:
    """SV vs S0: does ``v`` buy cost at no cost in success rate?

    Intersection-union on ONE set of paired draws, so the size is bounded by the
    larger component alpha and neither side is Bonferroni-corrected.

    Both components are one-sided in the direction the offline verdict mix
    predicts: SR non-inferior (``>= 0``, not ``> 0`` -- equal SR with a real cost
    saving IS the win being claimed) and a cost saving of at least 5%.
    """
    out = {
        "dsr_lower": float(np.quantile(d_sr, GATE2_SR_LOWER_Q)),
        "dsr_lower_quantile": GATE2_SR_LOWER_Q,
        "dc_upper": float(np.quantile(d_c, GATE2_UPPER_Q)),
        "dc_upper_quantile": GATE2_UPPER_Q,
        "cost_gate": GATE2_COMPUTE_GATE,
    }
    out["sr_pass"] = bool(out["dsr_lower"] >= 0.0)
    out["cost_pass"] = bool(out["dc_upper"] <= GATE2_COMPUTE_GATE)
    out["pass"] = bool(out["sr_pass"] and out["cost_pass"])
    return out


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def parse_task_uid(task_uid: str) -> tuple[str, int, int]:
    """(yaml_id, task_id, subset_idx) from ``<arm>:<phase>:<task>:<episode>``."""
    parts = str(task_uid).rsplit(":", 3)
    if len(parts) != 4 or parts[1] != "eval":
        raise SystemExit(f"malformed task_uid {task_uid!r}")
    try:
        return parts[0], int(parts[2]), int(parts[3])
    except ValueError as exc:
        raise SystemExit(f"malformed task_uid {task_uid!r}") from exc


def official_by_task(split_manifest_path: str, trials: int) -> dict[int, list[int]]:
    """task -> official init indices held by A', in materialised order.

    ``split_init_pools.materialize_pool`` writes ``states[sorted(indices)]``,
    so subset position i is the i-th smallest official index of that task. The
    adjudicated grid is derived from this frozen manifest, never hard-coded
    (G2 B3): the episode's own ``orig_init_state_idx`` is then cross-checked
    against it rather than trusted.
    """
    return official_test_inits(split_manifest_path, trials)


def load_accepted_episodes(
    journal_path: str, arms: list[str], expected_grid: set[tuple[int, int]],
) -> dict[str, dict]:
    """arm -> {(task, subset): {success, task_uid, attempt, run_id}}.

    Only ``accepted`` terminal records count. The conductor journals stale and
    fenced attempts the same way as the live one and discriminates them by
    (accepted, attempt, run_id), so an adjudicator that ignores those fields
    would mix a requeued episode's outcome with the one the scheduler actually
    took (G2 B2). Exactly one accepted record per cell is required.
    """
    out: dict[str, dict] = {a: {} for a in arms}
    for line in open(journal_path):
        row = json.loads(line)
        arm = row.get("yaml_id")
        if arm not in out:
            continue
        if not isinstance(row.get("accepted"), bool):
            raise SystemExit(f"arm {arm}: journal accepted must be a boolean")
        if row.get("accepted") is not True:
            continue
        if row.get("phase") != "eval":
            raise SystemExit(f"arm {arm}: accepted journal row is not eval phase")
        uid_arm, task, subset = parse_task_uid(row["task_uid"])
        if uid_arm != arm:
            raise SystemExit(
                f"journal row yaml_id={arm} disagrees with its task_uid {row['task_uid']}"
            )
        key = (task, subset)
        if key not in expected_grid:
            raise SystemExit(
                f"arm {arm}: accepted episode {key} is outside the pre-registered grid"
            )
        if key in out[arm]:
            raise SystemExit(
                f"arm {arm}: two accepted records for cell {key} — refusing "
                "ambiguous attribution"
            )
        if (not _is_json_int(row.get("attempt")) or row["attempt"] < 1
                or not isinstance(row.get("run_id"), str) or not row["run_id"]):
            raise SystemExit(
                f"arm {arm} cell {key}: accepted record lacks attempt/run_id — "
                "cannot bind its per-step evidence"
            )
        if row.get("status") not in ("done", "failed") or not isinstance(
            row.get("success"), bool
        ) or ((row["status"] == "done") != row["success"]):
            raise SystemExit(
                f"arm {arm} cell {key}: accepted status/success terminal schema is invalid"
            )
        out[arm][key] = {
            "success": row["success"],
            "task_uid": row["task_uid"],
            "attempt": row["attempt"],
            "run_id": row["run_id"],
        }
    for a in arms:
        got = set(out[a])
        if got != expected_grid:
            missing = sorted(expected_grid - got)[:3]
            raise SystemExit(
                f"arm {a}: {len(expected_grid - got)} cells have no accepted episode "
                f"(e.g. {missing}) — refusing incomplete/unpaired data"
            )
    return out


def load_analytic_cost(
    per_step_path: str, arms: list[str], accepted: dict[str, dict],
    officials: dict[int, list[int]],
) -> tuple[dict[str, dict], dict]:
    """arm -> {(task, subset): (cost_sum_ms, n_decisions)} plus a summary.

    The SUM and the COUNT are kept apart on purpose: the arm's cost is a
    ratio-of-sums that must be re-formed after resampling, so a per-cell mean
    computed here would already be the wrong estimand.

    Rows are joined to the ACCEPTED (task_uid, attempt, run_id) triple. Stale
    attempts, fenced reports and rows from another run are excluded and
    counted, never billed (G2 B2). Verdict rows additionally cross-check their
    ``orig_init_state_idx`` against the official index the frozen split
    manifest assigns to that subset position (G2 B3), and every accepted
    episode must contribute exactly one ``client_timing`` row whose ``infers``
    equals the number of priced decisions (G2 B1).
    """
    cost_sum: dict[str, dict] = {a: collections.defaultdict(float) for a in arms}
    n_dec: dict[str, dict] = {a: collections.defaultdict(int) for a in arms}
    verdict_counts: dict[str, collections.Counter] = {a: collections.Counter() for a in arms}
    infers_reported: dict[str, dict] = {a: {} for a in arms}
    step_indices: dict[str, dict] = {
        a: collections.defaultdict(set) for a in arms
    }
    excluded = collections.Counter()

    def _is_accepted(arm, row):
        required = ("task_uid", "task_id", "subset_init_state_idx",
                    "attempt", "run_id", "accepted", "success")
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise SystemExit(
                f"arm {arm}: per_step row lacks identity fields {missing}"
            )
        uid_arm, task, subset = parse_task_uid(row["task_uid"])
        if uid_arm != arm:
            raise SystemExit(
                f"per_step yaml_id={arm} disagrees with task_uid {row['task_uid']}"
            )
        rec = accepted[arm].get((task, subset))
        if rec is None:
            raise SystemExit(
                f"arm {arm}: per_step row for {(task, subset)} has no accepted "
                "journal episode — refusing rows outside the adjudicated grid"
            )
        if row["task_uid"] != rec["task_uid"]:
            raise SystemExit(f"arm {arm} cell {(task, subset)}: task_uid identity mismatch")
        if (not _is_json_int(row["task_id"])
                or not _is_json_int(row["subset_init_state_idx"])
                or row["task_id"] != task or row["subset_init_state_idx"] != subset):
            raise SystemExit(
                f"arm {arm} cell {(task, subset)}: duplicated task/subset fields disagree"
            )
        if (not _is_json_int(row["attempt"]) or row["attempt"] < 1
                or not isinstance(row["run_id"], str) or not row["run_id"]):
            raise SystemExit(f"arm {arm} cell {(task, subset)}: invalid attempt/run_id")
        if not isinstance(row["accepted"], bool):
            raise SystemExit(f"arm {arm} cell {(task, subset)}: accepted must be boolean")
        live = (row["attempt"] == rec["attempt"]
                and str(row["run_id"]) == rec["run_id"]
                and row["accepted"] is True)
        if live and (not isinstance(row["success"], bool)
                     or row["success"] != rec["success"]):
            raise SystemExit(
                f"arm {arm} cell {(task, subset)}: per_step success != journal outcome"
            )
        return (task, subset), live

    for line in open(per_step_path):
        row = json.loads(line)
        arm = row.get("yaml_id")
        if arm not in cost_sum:
            continue
        key, live = _is_accepted(arm, row)
        if not live:
            excluded[arm] += 1
            continue
        if row.get("_kind") == "client_timing":
            infers = row.get("infers")
            if not _is_json_int(infers) or infers < 0:
                raise SystemExit(
                    f"arm {arm} cell {key}: client_timing infers must be a nonnegative integer"
                )
            if key in infers_reported[arm]:
                raise SystemExit(f"arm {arm}: duplicate client_timing row for cell {key}")
            infers_reported[arm][key] = infers
            continue
        if row.get("_kind") is not None:
            raise SystemExit(
                f"arm {arm} cell {key}: unsupported per_step row kind "
                f"{row.get('_kind')!r}"
            )
        task, subset = key
        if row.get("phase") != "eval":
            raise SystemExit(f"arm {arm} cell {key}: verdict row is not eval phase")
        step_idx = row.get("step_idx")
        if not _is_json_int(step_idx) or step_idx < 0:
            raise SystemExit(f"arm {arm} cell {key}: invalid step_idx {step_idx!r}")
        if step_idx in step_indices[arm][key]:
            raise SystemExit(f"arm {arm} cell {key}: duplicate verdict step_idx {step_idx}")
        step_indices[arm][key].add(step_idx)
        canonical_episode = task * len(officials[task]) + subset
        if not _is_json_int(row.get("episode_id")) or row["episode_id"] != canonical_episode:
            raise SystemExit(
                f"arm {arm} cell {key}: episode_id {row.get('episode_id')!r} != "
                f"canonical {canonical_episode}"
            )
        official = row.get("orig_init_state_idx")
        if not _is_json_int(official) or official != officials[task][subset]:
            raise SystemExit(
                f"arm {arm} cell {key}: row claims official init {official!r} but the "
                f"frozen split manifest assigns {officials[task][subset]} to that "
                "subset position — identity mismatch"
            )
        hit = row.get("hit_type")
        cost_sum[arm][key] += unit_cost(hit, row.get("start_t"))
        n_dec[arm][key] += 1
        verdict_counts[arm][hit] += 1

    out: dict[str, dict] = {}
    for a in arms:
        expected = set(accepted[a])
        cells = set(n_dec[a])
        if cells != expected:
            missing = sorted(expected - cells)[:3]
            raise SystemExit(
                f"arm {a}: per_step covers {len(cells)} cells, expected {len(expected)} "
                f"(missing e.g. {missing}) — refusing incomplete cost data"
            )
        for key, n in n_dec[a].items():
            reported = infers_reported[a].get(key)
            if reported is None:
                raise SystemExit(
                    f"arm {a} cell {key}: accepted episode has no client_timing row — "
                    "the decision count cannot be verified against the episode's own "
                    "inference count"
                )
            if reported != n:
                raise SystemExit(
                    f"arm {a} cell {key}: {n} priced decisions but the episode reports "
                    f"{reported} inferences — refusing inconsistent per_step"
                )
        out[a] = {k: (cost_sum[a][k], n_dec[a][k]) for k in n_dec[a]}

    summary = {
        "per_step_sha256": _file_sha256(pathlib.Path(per_step_path)),
        "unit_cost_ms": {"stage1": STAGE1_MS, "stage2": STAGE2_MS, "stage3": STAGE3_MS,
                         "FULL_HIT": unit_cost("FULL_HIT", None),
                         "WARM_START": unit_cost("WARM_START", PINNED_START_T_WS),
                         "MISS": unit_cost("MISS", None)},
        "verdict_counts": {a: dict(verdict_counts[a]) for a in arms},
        "decisions": {a: int(sum(n for _s, n in out[a].values())) for a in arms},
        "excluded_stale_rows": dict(excluded),
    }
    return out, summary


def arm_cost(cells: dict, keys: list[tuple[int, int]]) -> float:
    """Decision-weighted ratio-of-sums over the given cells (ruling 9.4)."""
    num = den = 0.0
    for k in keys:
        s, n = cells[k]
        num += s
        den += n
    if den <= 0:
        raise SystemExit("cost denominator is zero — no decisions in the resample")
    return num / den


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _load_arm_artifact(matrix: dict, arm: str):
    import yaml as _yaml

    from openpi.cache.components.surface_judge import load_surface_artifact

    doc = _yaml.safe_load(open(matrix["arms"][arm]))
    return load_surface_artifact(
        doc["checkpoints"]["cp1"]["judge"]["surface_artifact_path"]
    )


def check_discipline(args, matrix: dict) -> dict:
    """Refuse anything not provably from ONE frozen experiment (G2-B2/B6)."""
    validate_matrix_artifacts(matrix)
    layer = matrix.get("layer")
    if layer not in {LAYER_PRIMARY, LAYER_SECONDARY}:
        raise SystemExit("arm matrix has no valid Rev 1 analysis layer")
    fit_record = json.loads(pathlib.Path(args.fit_record).read_text())
    recorded_fit_path = pathlib.Path((matrix.get("fit_record_paths") or {}).get("sv", ""))
    if pathlib.Path(args.fit_record).resolve() != recorded_fit_path.resolve():
        raise SystemExit("--fit-record is not the SV record frozen in the arm matrix")
    if _file_sha256(pathlib.Path(args.fit_record)) != matrix["fit_record_sha256"]["sv"]:
        raise SystemExit("SV fit record content drifted after arm-matrix emission")
    reject_retired_gate2(fit_record, "fit record")
    if "delta_star" not in fit_record:
        raise SystemExit("fit_record carries no frozen delta_star — refusing")
    delta_star = fit_record["delta_star"]

    sv_art = _load_arm_artifact(matrix, ARM_SV)
    if sv_art.delta != delta_star:
        raise SystemExit(
            f"primary artifact delta {sv_art.delta} != frozen delta_star {delta_star}"
        )
    fp = sv_art.retrieval_contract.get("policy_fingerprint")
    lib_sha = sv_art.retrieval_contract.get("library_sha256")
    sv_inputs = sv_art.meta.get("input_digests")
    if not sv_inputs:
        raise SystemExit("SV artifact carries no fitting-input digests")
    if layer == LAYER_PRIMARY:
        s0_art = _load_arm_artifact(matrix, ARM_S0)
        if s0_art.delta != delta_star:
            raise SystemExit(
                f"S0 artifact delta {s0_art.delta} != frozen delta_star {delta_star} — "
                "the nested ablation must share ONE delta (G2-B2)"
            )
        if s0_art.uses_disagreement or not sv_art.uses_disagreement:
            raise SystemExit("arm artifacts have swapped uses_disagreement roles")
        if s0_art.retrieval_contract.get("policy_fingerprint") != fp or \
                s0_art.retrieval_contract.get("library_sha256") != lib_sha:
            raise SystemExit("SV and S0 artifacts bind different policy/library")
        if s0_art.meta.get("input_digests") != sv_inputs:
            raise SystemExit(
                "SV and S0 artifacts were fitted from different calibration inputs — "
                "the nested ablation requires byte-identical table/cohort/weights"
            )

    expected_core = (
        set(CORE_T_ARMS) | {ARM_S0, ARM_SV}
        if layer == LAYER_PRIMARY else set(SECONDARY_CORE_ARMS)
    )
    matrix_arms = dict(matrix.get("arms") or {})
    if set(matrix.get("core_arms") or []) != expected_core:
        raise SystemExit(
            f"arm matrix does not contain exactly the frozen {layer} core arms"
        )
    if set(matrix.get("descriptive_arms") or []) != set(matrix_arms) - expected_core:
        raise SystemExit("arm matrix descriptive arm roster is inconsistent")
    if matrix.get("library_sha256") != lib_sha:
        raise SystemExit("arm matrix library_sha256 != artifact contract")

    # Recompute every matrix YAML once. A resume may execute only unfinished
    # arms, but it may not reinterpret any arm or change the experiment roster.
    recorded_yaml_shas = matrix.get("arm_yaml_sha256")
    if not isinstance(recorded_yaml_shas, dict) or set(recorded_yaml_shas) != set(matrix_arms):
        raise SystemExit("arm matrix does not freeze exactly one digest per arm")
    arm_yaml_sha: dict[str, str] = {}
    for arm, yaml_name in matrix_arms.items():
        yaml_path = pathlib.Path(yaml_name)
        if not yaml_path.is_file():
            raise SystemExit(f"arm {arm}: yaml missing on disk: {yaml_path}")
        actual = _file_sha256(yaml_path)
        if recorded_yaml_shas[arm] != actual:
            raise SystemExit(
                f"arm {arm}: yaml content drifted since emit "
                f"(recorded {recorded_yaml_shas[arm][:12]}…, actual {actual[:12]}…)"
            )
        arm_yaml_sha[arm] = actual

    actual_matrix_sha = _file_sha256(pathlib.Path(args.arm_matrix))
    actual_split_sha = _file_sha256(pathlib.Path(args.split_manifest))
    split = json.loads(pathlib.Path(args.split_manifest).read_text())

    # Launch ledger of the run(s) that supplied BOTH axes. Experiment-wide
    # fields are immutable; executed arms are per-run so a strict-subset resume
    # remains valid and is still attributable (G2 B4 follow-up).
    ledger = json.loads(pathlib.Path(args.launch_manifest).read_text())
    if (ledger.get("schema_version") != 2
            or not isinstance(ledger.get("launches"), list)
            or not ledger["launches"]):
        raise SystemExit(
            "launch manifest is not a v2 ledger — refusing: without the frozen "
            "arm-matrix/yaml digests and run ids the episodes cannot be bound "
            "to the configuration that produced them"
        )
    launches = ledger["launches"]
    first = launches[0]
    for key in FROZEN_LAUNCH_KEYS:
        if first.get(key) is None:
            raise SystemExit(f"launch ledger entry lacks {key} — refusing")
        for other in launches[1:]:
            if other.get(key) != first.get(key):
                raise SystemExit(
                    f"launch ledger entries disagree on {key}: the accumulated "
                    "per-step rows come from different configurations"
                )
    run_ids = [entry.get("run_id") for entry in launches]
    if (any(not isinstance(r, str) or not r for r in run_ids)
            or len(set(run_ids)) != len(run_ids)):
        raise SystemExit("launch ledger has missing or duplicated run ids — refusing")
    if args.trials != EXPECTED_TRIALS or first.get("trials_per_task") != EXPECTED_TRIALS:
        raise SystemExit(
            f"trials must be the frozen {EXPECTED_TRIALS}/task: analyzer got "
            f"{args.trials}, launch recorded {first.get('trials_per_task')}"
        )
    if first.get("protocol") != PROTOCOL or first.get("layer") != layer:
        raise SystemExit("launch protocol/layer does not match the arm matrix")
    if first.get("artifact_sha256") != matrix.get("artifact_sha256") \
            or first.get("fit_record_sha256") != matrix.get("fit_record_sha256"):
        raise SystemExit("launch artifact/fit-record content bindings differ from matrix")
    if first["arm_matrix_sha256"] != actual_matrix_sha:
        raise SystemExit(
            "the arm matrix being adjudicated is not the one the run executed"
        )
    if first["split_manifest_sha256"] != actual_split_sha:
        raise SystemExit(
            "the split manifest being adjudicated is not the one the run executed"
        )
    if first.get("aprime_content_sha256") is None:
        raise SystemExit("precheck launch lacks an A' content digest — refusing")
    if not _is_json_int(first.get("env_seed")):
        raise SystemExit("precheck launch env_seed must be an integer")
    if first.get("replan_steps") != sv_art.h_exec:
        raise SystemExit("precheck launch replan_steps != primary artifact h_exec")

    executed_arms_by_run: dict[str, set[str]] = {}
    split_state_sha = {
        name: info.get("sha256")
        for name, info in ((split.get("pool_digests") or {}).get("test_aprime") or {}).items()
    }
    for idx, launch in enumerate(launches):
        if launch.get("suite") != split.get("suite"):
            raise SystemExit(f"launch ledger entry {idx} suite != split manifest")
        if set(launch.get("core_arms") or []) != expected_core:
            raise SystemExit(f"launch ledger entry {idx} core arm set != arm matrix")
        if set(launch.get("descriptive_arms") or []) != set(matrix_arms) - expected_core:
            raise SystemExit(f"launch ledger entry {idx} descriptive arm set != arm matrix")
        if launch.get("library_sha256") != lib_sha:
            raise SystemExit(f"launch ledger entry {idx} library != artifact contract")
        binding = launch.get("contract_binding") or {}
        if launch.get("policy_fingerprint") != fp or binding.get("policy_fingerprint") != fp:
            raise SystemExit(
                f"launch ledger entry {idx} policy fingerprint != artifact contract"
            )
        if binding.get("h_exec") != launch.get("replan_steps"):
            raise SystemExit(f"launch ledger entry {idx} h_exec/replan mismatch")
        pool = launch.get("pool") or {}
        if (pool.get("suite") != launch.get("suite")
                or pool.get("total_inits") != EXPECTED_TASKS * EXPECTED_TRIALS
                or pool.get("rollup_sha256") != launch.get("aprime_content_sha256")
                or pool.get("split_manifest_sha256") != launch.get("split_manifest_sha256")
                or pool.get("state_content_sha256") != split_state_sha):
            raise SystemExit(f"launch ledger entry {idx} A' pool attestation is inconsistent")
        frozen_yaml = launch.get("frozen_yaml_sha256")
        if frozen_yaml != arm_yaml_sha:
            raise SystemExit(f"launch ledger entry {idx} frozen YAML map != arm matrix")
        executed = launch.get("executed_arms")
        executed_sha = launch.get("executed_yaml_sha256")
        if not isinstance(executed, list) or not executed:
            raise SystemExit(f"launch ledger entry {idx} has no executed arms")
        if set(executed) != set(executed_sha or {}):
            raise SystemExit(f"launch ledger entry {idx} executed arm/YAML keys disagree")
        if not set(executed).issubset(matrix_arms):
            raise SystemExit(f"launch ledger entry {idx} executed unknown arms")
        for arm in executed:
            if executed_sha[arm] != arm_yaml_sha[arm]:
                raise SystemExit(
                    f"launch ledger entry {idx} executed a different YAML for {arm}"
                )
        executed_arms_by_run[str(launch["run_id"])] = set(executed)

    return {
        # Carried out so the cross-suite finalizer can key on it; already
        # cross-validated against the split manifest and the A' pool above.
        "suite": first.get("suite"),
        "protocol": PROTOCOL,
        "layer": layer,
        "delta_star": delta_star,
        "policy_fingerprint": fp,
        "library_sha256": lib_sha,
        "aprime_content_sha256": first["aprime_content_sha256"],
        "arm_matrix_sha256": actual_matrix_sha,
        "split_manifest_sha256": actual_split_sha,
        "arm_yaml_sha256": arm_yaml_sha,
        "artifact_sha256": matrix["artifact_sha256"],
        "fit_record_sha256": matrix["fit_record_sha256"],
        "launch_run_ids": run_ids,
        "executed_arms_by_run": {
            run_id: sorted(arms) for run_id, arms in executed_arms_by_run.items()
        },
        "cost_model": "analytic GPU inference cost (model-forward compute proxy); "
                      "retrieval CPU excluded; not a measured end-to-end latency",
    }


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--per-step", required=True,
                    help="run_precheck per_step.jsonl — supplies the analytic cost")
    ap.add_argument("--arm-matrix", required=True)
    ap.add_argument("--fit-record", required=True)
    ap.add_argument("--launch-manifest", required=True,
                    help="run_precheck <per-step-out>.launch.json (v2 ledger)")
    ap.add_argument("--split-manifest", required=True,
                    help="frozen split manifest; defines the adjudicated grid")
    ap.add_argument("--trials", type=int, required=True,
                    help="episodes per task in the run (A' subset indices 0..trials-1)")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    matrix = json.loads(pathlib.Path(args.arm_matrix).read_text())
    discipline = check_discipline(args, matrix)
    layer = discipline["layer"]
    core = list(CORE_T_ARMS) + ([ARM_S0, ARM_SV] if layer == LAYER_PRIMARY else [ARM_SV])

    officials = official_by_task(args.split_manifest, args.trials)
    expected_grid = {(t, j) for t in officials for j in range(args.trials)}
    accepted = load_accepted_episodes(args.journal, core, expected_grid)
    executed_by_run = {
        run_id: set(arms)
        for run_id, arms in discipline["executed_arms_by_run"].items()
    }
    for a in core:
        for key, rec in accepted[a].items():
            if a not in executed_by_run.get(rec["run_id"], set()):
                raise SystemExit(
                    f"arm {a} cell {key}: accepted episode carries run_id "
                    f"{rec['run_id']} whose launch did not execute that arm"
                )
    cost_cells, cost_summary = load_analytic_cost(
        args.per_step, core, accepted, officials
    )
    discipline["cost_inputs"] = cost_summary
    discipline["official_init_by_subset"] = {str(t): officials[t] for t in officials}

    outcomes = {a: {k: v["success"] for k, v in accepted[a].items()} for a in core}
    keys = sorted(outcomes[core[0]])
    tasks = sorted({k[0] for k in keys})
    inits_by_task = {t: sorted({k[1] for k in keys if k[0] == t}) for t in tasks}
    grid = {a: {k: outcomes[a][k] for k in keys} for a in core}

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    g2_sr, g2_c = [], []
    for _ in range(B_REPLICATES):
        # ONE task-stratified, init-clustered resample per replicate, shared by
        # every arm and by BOTH axes. Gate 2 is an intersection-union test, so
        # SR and cost must come from the same draw.
        chosen = {t: rng.choice(inits_by_task[t], size=len(inits_by_task[t]),
                                replace=True) for t in tasks}
        picked = [(t, int(i)) for t in tasks for i in chosen[t]]
        sr = {a: float(np.mean([grid[a][k] for k in picked])) for a in core}
        cost = {a: arm_cost(cost_cells[a], picked) for a in core}

        records.append(frontier_record(
            [(a, sr[a], cost[a]) for a in CORE_T_ARMS],
            (sr[ARM_SV], cost[ARM_SV]),
        ))
        if layer == LAYER_PRIMARY:
            g2_sr.append(sr[ARM_SV] - sr[ARM_S0])
            g2_c.append(cost[ARM_SV] / cost[ARM_S0] - 1.0)

    all_keys = sorted(keys)
    result = {
        "protocol": PROTOCOL,
        "analysis_layer": layer,
        "confirmatory": layer == LAYER_PRIMARY,
        # The cross-suite finalizer keys on this; it is taken from the launch
        # manifest, which check_discipline has already cross-validated against
        # the split manifest and the A' pool.
        "suite": discipline.get("suite"),
        "discipline": discipline,
        "point_estimates": {
            a: {"sr": float(np.mean(list(grid[a].values()))),
                "cost_ms_per_decision": arm_cost(cost_cells[a], all_keys)}
            for a in core
        },
        "gate_schema": {
            "gate1": {
                "d_sr_lower_quantile": GATE1_LOWER_Q,
                "d_c_upper_quantile": GATE1_UPPER_Q,
                "cost_gate": COMPUTE_GATE,
            },
            "gate2": {
                "d_sr_lower_quantile": GATE2_SR_LOWER_Q,
                "d_c_upper_quantile": GATE2_UPPER_Q,
                "cost_gate": GATE2_COMPUTE_GATE,
            },
        },
    }
    if layer == LAYER_SECONDARY:
        result["frontier_descriptive"] = {
            "d_sr_p5": float(np.quantile([r["D_sr"] for r in records], 0.05)),
            "d_sr_median": float(np.quantile([r["D_sr"] for r in records], 0.5)),
            "d_c_p95": float(np.quantile([r["D_c"] for r in records], 0.95)),
            "d_c_median": float(np.quantile([r["D_c"] for r in records], 0.5)),
            "branch_shares": {
                branch: float(np.mean([r["branch"] == branch for r in records]))
                for branch in ("bracket", "high", "low")
            },
        }
        result["verdict"] = "secondary_descriptive_complete"
        result["note"] = (
            "Production-gate layer is descriptive external-validity evidence; "
            "it does not run or alter either confirmatory gate."
        )
        pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
        print(json.dumps({"verdict": result["verdict"], "layer": layer}, indent=2))
        return

    g1 = gate1(records)
    result["gate1"] = g1
    # Fixed sequence: Gate 2 is evaluated ONLY after Gate 1 passes, and the
    # confirmatory chain stops at the first failure. There is no Gate 3.
    if g1["pass"]:
        g2 = gate2(np.array(g2_sr), np.array(g2_c))
        result["gate2"] = g2
        # Rev 1 Gate 2 proves SR NON-INFERIORITY (q0.05(dSR) >= 0) and a COST
        # SAVING (q0.95(dC) <= -5%). Neither component licenses the words "SR
        # gain" or "cost non-inferiority": the first would claim an improvement
        # the gate never tests for, the second would describe the cost side as
        # a floor when it is a required saving.
        if g2["pass"]:
            result["verdict"] = "surface_wins_v_confirmed"
            result["gate2_note"] = (
                "SR non-inferiority and a cost saving of at least 5% both hold "
                "for SV over S0."
            )
        elif g2["sr_pass"]:
            result["verdict"] = "surface_v_sr_noninferior_cost_saving_unconfirmed"
            result["gate2_note"] = (
                "SR non-inferiority holds, but the required cost saving is not "
                "established; this does NOT establish that v is ineffective, and "
                "it is NOT evidence of an SR gain."
            )
        elif g2["cost_pass"]:
            result["verdict"] = "surface_v_cost_saving_sr_noninferiority_unconfirmed"
            result["gate2_note"] = (
                "The cost saving holds, but SR non-inferiority is not "
                "established; the saving may have been bought with success rate."
            )
        else:
            result["verdict"] = "surface_wins_v_unconfirmed"
            result["gate2_note"] = (
                "Neither SR non-inferiority nor the required cost saving is "
                "established for SV over S0."
            )
            result["s0_descriptive_note"] = (
                "s-only deployment choice is descriptive only; no confirmatory "
                "S0-vs-threshold test exists in this design."
            )
    else:
        result["verdict"] = "line_demoted"
    pathlib.Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps({"verdict": result["verdict"], "gate1": g1}, indent=2))


if __name__ == "__main__":
    main()
