"""Precheck adjudicator: joint replicates, endpoint-clamped frontier, gates.

Implements plan section 4.6 verbatim, with the discipline rules as hard code
paths (refusals, not warnings):

  * refuses input without a frozen primary delta (fit_record.delta_star must
    equal the primary artifact's delta);
  * refuses arm sets whose journals disagree on the paired (task, init) grid
    or whose launch manifests carry different pool digests;
  * refuses cost data with missing blocks / warmup markers, mismatched
    arm-order manifests between the two passes, or wrong monitor levels;
  * every joint replicate produces exactly one endpoint-clamped record
    {branch, D_sr, D_c, D_l} — low replicates clamp to the argmin-SR cell and
    STAY in the cost distributions (no sample deletion path exists);
  * gate order and the stop-after-Gate-2 rule are hard-coded; there is no
    multi-point selection path (SV+/- never enter a gate).

Outputs verdict JSON + descriptive tables. Exit code 0 always (the verdict
field carries the outcome); refusals raise SystemExit.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import numpy as np

B_REPLICATES = 10_000
CORE_T_ARMS = ("dsp_t_fh30_ws20", "dsp_t_fh50_ws20", "dsp_t_fh70_ws10")
ARM_SV, ARM_S0 = "dsp_sv", "dsp_s0"
COMPUTE_GATE = -0.05     # D_c upper quantile must be <= this
LATENCY_GATE = 0.0
GATE2_COMPUTE_SLACK = 0.05
# Upper-quantile levels of the two gates. The power simulation MUST use the
# same levels (it imports these constants), or its power estimates describe a
# different test than the one adjudicated here (G2R2-B4).
GATE1_UPPER_Q = 0.95
GATE2_UPPER_Q = 0.975
STAGE_PROBES = ("stage1_vision", "stage2_llm", "stage3_flow", "stage3_warm")
EXPECTED_TRIALS = 30     # frozen A' quota per task (plan 4.2)


# ------------------------------------------------------------------
# Frontier / replicate statistics (pure functions, unit-tested)
# ------------------------------------------------------------------


def frontier_record(t_arms: list[tuple[str, float, float, float]],
                    sv: tuple[float, float, float]) -> dict:
    """One replicate's endpoint-clamped record.

    Args:
        t_arms: [(arm_id, SR, compute, latency)] for the three baselines.
        sv: (SR, compute, latency) of the surface arm.

    Returns {branch, D_sr, D_c, D_l}. All three branches produce D_c/D_l;
    low clamps to the argmin-SR cell, high to the argmax-SR cell.
    """
    pts = sorted(t_arms, key=lambda t: (t[1], t[0]))  # stable (SR, arm_id) order
    srs = [p[1] for p in pts]
    sr_sv, c_sv, l_sv = sv
    if sr_sv < srs[0]:
        branch, (c_m, l_m) = "low", (pts[0][2], pts[0][3])
    elif sr_sv > srs[-1]:
        branch, (c_m, l_m) = "high", (pts[-1][2], pts[-1][3])
    else:
        branch = "bracket"
        hi = next(i for i in range(len(srs)) if srs[i] >= sr_sv)
        if srs[hi] == sr_sv or hi == 0:
            c_m, l_m = pts[hi][2], pts[hi][3]
        else:
            lo = hi - 1
            span = srs[hi] - srs[lo]
            t = 0.0 if span <= 0 else (sr_sv - srs[lo]) / span
            c_m = pts[lo][2] + t * (pts[hi][2] - pts[lo][2])
            l_m = pts[lo][3] + t * (pts[hi][3] - pts[lo][3])
    if not (np.isfinite(c_m) and c_m > 0 and np.isfinite(l_m) and l_m > 0):
        raise SystemExit(f"comparator has non-positive cost (c={c_m}, l={l_m})")
    sr_m = min(max(sr_sv, srs[0]), srs[-1])
    return {
        "branch": branch,
        "D_sr": sr_sv - sr_m,
        "D_c": (c_sv - c_m) / c_m,
        "D_l": (l_sv - l_m) / l_m,
    }


def gate1(records: list[dict]) -> dict:
    d_sr = np.array([r["D_sr"] for r in records])
    d_c = np.array([r["D_c"] for r in records])
    d_l = np.array([r["D_l"] for r in records])
    out = {
        "d_sr_p5": float(np.quantile(d_sr, 1 - GATE1_UPPER_Q)),
        "d_c_p95": float(np.quantile(d_c, GATE1_UPPER_Q)),
        "d_l_p95": float(np.quantile(d_l, GATE1_UPPER_Q)),
        "branch_shares": {b: float(np.mean([r["branch"] == b for r in records]))
                          for b in ("bracket", "high", "low")},
    }
    out["pass"] = bool(
        out["d_sr_p5"] >= 0.0
        and out["d_c_p95"] <= COMPUTE_GATE
        and out["d_l_p95"] <= LATENCY_GATE
    )
    return out


def gate2(d_sr: np.ndarray, d_c: np.ndarray, d_l: np.ndarray) -> dict:
    out = {
        "dsr_p2_5": float(np.quantile(d_sr, 1 - GATE2_UPPER_Q)),
        "dc_p97_5": float(np.quantile(d_c, GATE2_UPPER_Q)),
        "dl_p97_5": float(np.quantile(d_l, GATE2_UPPER_Q)),
    }
    out["pass"] = bool(
        out["dsr_p2_5"] > 0.0
        and out["dc_p97_5"] <= GATE2_COMPUTE_SLACK
        and out["dl_p97_5"] <= LATENCY_GATE
    )
    return out


# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def load_sr_outcomes(
    journal_path: str, arms: list[str], *, expected_grid: set[tuple[int, int]] | None = None,
) -> dict[str, dict]:
    """arm -> {(task, official init): 0/1}, accepted rows only.

    Refusals (G2-B6): more than one accepted row for the same (arm, cell) is
    an attribution ambiguity, not a tie to break silently; every arm must
    cover EXACTLY the expected grid — an equally-incomplete pair of arms is
    not paired evidence.
    """
    outcomes: dict[str, dict] = {a: {} for a in arms}
    for line in open(journal_path):
        row = json.loads(line)
        if not row.get("accepted"):
            continue
        arm = row.get("yaml_id")
        if arm not in outcomes:
            continue
        parts = str(row["task_uid"]).split(":")
        key = (int(parts[-2]), int(parts[-1]))
        if key in outcomes[arm]:
            raise SystemExit(
                f"arm {arm}: duplicate accepted rows for cell {key} — refusing "
                "ambiguous attribution"
            )
        outcomes[arm][key] = 1 if row.get("status") == "done" and row.get("success") else 0
    ref = expected_grid if expected_grid is not None else set(outcomes[arms[0]])
    for a in arms:
        got = set(outcomes[a])
        if got != ref:
            missing, extra = sorted(ref - got)[:3], sorted(got - ref)[:3]
            raise SystemExit(
                f"arm {a}: grid mismatch (missing {len(ref - got)} e.g. {missing}; "
                f"extra {len(got - ref)} e.g. {extra}) — refusing incomplete/unpaired data"
            )
    return outcomes


def compute_unit_cost(timing_rows: list[dict], tag: str) -> float:
    """Per-decision stage GPU-time from official SystemTimer rows (G2-B3).

    Row schema is ``{timestamp, task_id, name, elapsed_ms, ...}``; any stage
    row missing a finite ``elapsed_ms`` is a schema breach and refused (the
    legacy ``dur_ms`` field never silently reads as zero again). The
    denominator is the number of DECISIONS (``total_inference`` rows), and the
    numerator sums whatever stage rows each decision actually produced —
    FULL_HIT (stage1 only), WARM_START (stage1+2+warm) and MISS mixes are
    aggregated correctly without inventing absent probes.
    """
    n_decisions = 0
    stage_sum = 0.0
    saw_stage = False
    for row in timing_rows:
        name = row.get("name")
        if name is None:
            raise SystemExit(f"unit {tag}: timing row without 'name': {row}")
        if name == "total_inference":
            n_decisions += 1
            continue
        if name in STAGE_PROBES:
            if "elapsed_ms" not in row or not np.isfinite(float(row["elapsed_ms"])):
                raise SystemExit(
                    f"unit {tag}: stage row {name!r} lacks a finite 'elapsed_ms' "
                    f"(official SystemTimer schema): {row}"
                )
            stage_sum += float(row["elapsed_ms"])
            saw_stage = True
    if n_decisions == 0:
        raise SystemExit(f"unit {tag}: no total_inference rows — cannot count decisions")
    if not saw_stage:
        raise SystemExit(f"unit {tag}: no stage timing rows")
    return stage_sum / n_decisions


def load_cost_blocks(raw_path: str, arms: list[str], pass_name: str,
                     blocks: int) -> dict[str, np.ndarray]:
    """arm -> per-block mean cost array of length `blocks`."""
    units = json.loads(pathlib.Path(raw_path).read_text())
    per: dict[str, dict[int, float]] = {a: {} for a in arms}
    for unit in units:
        arm, block = unit["arm"], int(unit["block"])
        if arm not in per:
            continue
        if block in per[arm]:
            raise SystemExit(f"arm {arm}: duplicate cost unit for block {block}")
        if pass_name == "compute":
            per[arm][block] = compute_unit_cost(unit.get("timing_rows") or [],
                                                unit.get("tag", f"{arm}/b{block}"))
        else:
            per_step = pathlib.Path(unit["per_step_path"])
            total_ms = total_n = 0.0
            for line in open(per_step):
                row = json.loads(line)
                if "infer_ms" in row:
                    total_ms += float(row["infer_ms"])
                    total_n += float(row.get("infers", 1))
            if total_n == 0:
                raise SystemExit(f"unit {unit.get('tag')}: no client_timing rows")
            per[arm][block] = total_ms / total_n
    out = {}
    for a, vals in per.items():
        if sorted(vals) != list(range(blocks)):
            raise SystemExit(f"arm {a}: cost blocks {sorted(vals)} != 0..{blocks - 1}")
        out[a] = np.array([vals[b] for b in range(blocks)])
    return out


def _load_arm_artifact(matrix: dict, arm: str):
    from openpi.cache.components.surface_judge import load_surface_artifact

    import yaml as _yaml

    doc = _yaml.safe_load(open(matrix["arms"][arm]))
    return load_surface_artifact(
        doc["checkpoints"]["cp1"]["judge"]["surface_artifact_path"]
    )


def check_discipline(args, matrix: dict) -> dict:
    """Refuse anything not provably from ONE frozen experiment (G2-B2/B6)."""
    fit_record = json.loads(pathlib.Path(args.fit_record).read_text())
    if "delta_star" not in fit_record:
        raise SystemExit("fit_record carries no frozen delta_star — refusing")
    delta_star = fit_record["delta_star"]

    sv_art = _load_arm_artifact(matrix, ARM_SV)
    s0_art = _load_arm_artifact(matrix, ARM_S0)
    if sv_art.delta != delta_star:
        raise SystemExit(
            f"primary artifact delta {sv_art.delta} != frozen delta_star {delta_star}"
        )
    if s0_art.delta != delta_star:
        raise SystemExit(
            f"S0 artifact delta {s0_art.delta} != frozen delta_star {delta_star} — "
            "the nested ablation must share ONE delta (G2-B2)"
        )
    if s0_art.uses_disagreement or not sv_art.uses_disagreement:
        raise SystemExit("arm artifacts have swapped uses_disagreement roles")
    fp = sv_art.retrieval_contract.get("policy_fingerprint")
    lib_sha = sv_art.retrieval_contract.get("library_sha256")
    if s0_art.retrieval_contract.get("policy_fingerprint") != fp or \
            s0_art.retrieval_contract.get("library_sha256") != lib_sha:
        raise SystemExit("SV and S0 artifacts bind different policy/library")
    sv_inputs = sv_art.meta.get("input_digests")
    if not sv_inputs or s0_art.meta.get("input_digests") != sv_inputs:
        raise SystemExit(
            "SV and S0 artifacts were fitted from different calibration inputs — "
            "the nested ablation requires byte-identical table/cohort/weights (G2R2-B6)"
        )

    # Primary launch manifest of the SR run.
    launch = json.loads(pathlib.Path(args.launch_manifest).read_text())
    if launch.get("library_sha256") != lib_sha:
        raise SystemExit("precheck launch library_sha256 != artifact contract")
    attested = (launch.get("contract_binding") or {}).get("policy_fingerprint")
    if attested != fp:
        raise SystemExit("precheck launch attested policy_fingerprint != artifact contract")
    if set(launch.get("core_arms") or []) != set(matrix["core_arms"]):
        raise SystemExit("precheck launch core arm set != arm matrix")
    if args.trials != EXPECTED_TRIALS or launch.get("trials_per_task") != EXPECTED_TRIALS:
        raise SystemExit(
            f"trials must be the frozen {EXPECTED_TRIALS}/task: analyzer got "
            f"{args.trials}, launch recorded {launch.get('trials_per_task')}"
        )

    man_a = json.loads(pathlib.Path(args.cost_dir, "manifest_compute.json").read_text())
    man_b = json.loads(pathlib.Path(args.cost_dir, "manifest_latency.json").read_text())
    if man_a["arm_orders"] != man_b["arm_orders"] or man_a["blocks"] != man_b["blocks"]:
        raise SystemExit("cost passes disagree on (block, arm-order) manifest — refusing")
    for field in ("block_pool_digest", "aprime_content_sha256", "arm_matrix_sha256"):
        if man_a.get(field) != man_b.get(field) or man_a.get(field) is None:
            raise SystemExit(f"cost passes lack a shared {field} — refusing")
    # Self-reported equality is not evidence (G2R3-B3): recompute the digests
    # against the ACTUAL files this adjudication is running on.
    import hashlib as _hashlib

    actual_matrix_sha = _hashlib.sha256(
        pathlib.Path(args.arm_matrix).read_bytes()
    ).hexdigest()
    if man_a["arm_matrix_sha256"] != actual_matrix_sha:
        raise SystemExit(
            "cost manifests reference a different arm matrix than the one being "
            "adjudicated — refusing"
        )
    recorded_yaml_shas = matrix.get("arm_yaml_sha256") or {}
    for arm in matrix["core_arms"]:
        yaml_path = pathlib.Path(matrix["arms"][arm])
        if not yaml_path.is_file():
            raise SystemExit(f"arm {arm}: yaml missing on disk: {yaml_path}")
        actual = _hashlib.sha256(yaml_path.read_bytes()).hexdigest()
        if recorded_yaml_shas.get(arm) != actual:
            raise SystemExit(
                f"arm {arm}: yaml content drifted since emit "
                f"(recorded {str(recorded_yaml_shas.get(arm))[:12]}…, actual {actual[:12]}…)"
            )
    if launch.get("aprime_content_sha256") is None or \
            launch.get("aprime_content_sha256") != man_a.get("aprime_content_sha256"):
        raise SystemExit(
            "cost bench A' content digest does not match the primary launch pool — "
            "the two experiments sampled different init contents"
        )
    if man_a.get("seed") != man_b.get("seed"):
        raise SystemExit("cost passes ran with different seeds — pairing broken")
    if not man_a.get("gpu_name") or man_a.get("gpu_name") != man_b.get("gpu_name"):
        raise SystemExit("cost passes lack a shared non-empty gpu_name attestation")
    if man_a["monitor_level"] != "SNAPSHOT" or man_b["monitor_level"] != "OFF":
        raise SystemExit("cost pass monitor levels are wrong — refusing")
    if man_a.get("cuda_available") is not True:
        raise SystemExit(
            "compute pass lacks cuda_available=True attestation — CPU-fallback "
            "probes must never be adjudicated as GPU compute (G2R2-B5)"
        )
    required_stages = {"stage1", "stage2", "stage3"}
    compute_backends = man_a.get("stage_probe_backends")
    if not isinstance(compute_backends, dict) or set(compute_backends) != required_stages \
            or any(v != "cuda" for v in compute_backends.values()):
        raise SystemExit(
            "compute cost manifest does not attest exact CUDA-event backends "
            "for stage1/stage2/stage3"
        )
    compute_devices = man_a.get("stage_devices")
    latency_devices = man_b.get("stage_devices")
    if not isinstance(compute_devices, dict) or set(compute_devices) != required_stages \
            or any(not isinstance(v, str) or not v.startswith("cuda")
                   for v in compute_devices.values()):
        raise SystemExit("compute cost manifest effective stage devices are not all CUDA")
    if latency_devices != compute_devices:
        raise SystemExit("compute/latency passes used different effective stage devices")
    for man, name in ((man_a, "compute"), (man_b, "latency")):
        if man.get("policy_fingerprint") != fp:
            raise SystemExit(f"cost {name} pass policy_fingerprint != artifact contract")
        if man.get("library_sha256") != lib_sha:
            raise SystemExit(f"cost {name} pass library_sha256 != artifact contract")
    if man_a["blocks"] < args.blocks:
        raise SystemExit(f"cost blocks {man_a['blocks']} < required {args.blocks}")

    from exp.dispatch_surface.power_sim_cost_blocks import (
        record_digest as _power_digest,
        validate_power_record,
    )

    power = json.loads(pathlib.Path(args.power_record).read_text())
    frozen_r = validate_power_record(power)  # digest + constants + derived R
    if frozen_r != args.blocks:
        raise SystemExit(
            f"--blocks {args.blocks} != frozen power-simulation R {frozen_r}"
        )
    for man, name in ((man_a, "compute"), (man_b, "latency")):
        if man.get("power_record_digest") != _power_digest(power):
            raise SystemExit(
                f"cost {name} pass ran against a different power record — refusing"
            )
    return {"delta_star": delta_star, "blocks": man_a["blocks"],
            "policy_fingerprint": fp, "library_sha256": lib_sha}


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--journal", required=True)
    ap.add_argument("--arm-matrix", required=True)
    ap.add_argument("--fit-record", required=True)
    ap.add_argument("--launch-manifest", required=True,
                    help="run_precheck <per-step-out>.launch.json")
    ap.add_argument("--cost-dir", required=True)
    ap.add_argument("--power-record", required=True,
                    help="frozen power-simulation record fixing R")
    ap.add_argument("--blocks", type=int, required=True)
    ap.add_argument("--trials", type=int, required=True,
                    help="episodes per task in the SR run (A' subset indices 0..trials-1)")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    matrix = json.loads(pathlib.Path(args.arm_matrix).read_text())
    core = list(CORE_T_ARMS) + [ARM_S0, ARM_SV]
    discipline = check_discipline(args, matrix)

    expected_grid = {(t, j) for t in range(10) for j in range(args.trials)}
    outcomes = load_sr_outcomes(args.journal, core, expected_grid=expected_grid)
    cost_c = load_cost_blocks(str(pathlib.Path(args.cost_dir, "raw_compute.json")),
                              core, "compute", args.blocks)
    cost_l = load_cost_blocks(str(pathlib.Path(args.cost_dir, "raw_latency.json")),
                              core, "latency", args.blocks)

    keys = sorted(outcomes[core[0]])
    tasks = sorted({k[0] for k in keys})
    inits_by_task = {t: sorted({k[1] for k in keys if k[0] == t}) for t in tasks}
    grid = {a: {k: outcomes[a][k] for k in keys} for a in core}

    rng = np.random.default_rng(args.seed)
    records: list[dict] = []
    g2_sr, g2_c, g2_l = [], [], []
    for _ in range(B_REPLICATES):
        # SR: task-stratified, init-level cluster resample (shared across arms).
        sr = {}
        chosen = {t: rng.choice(inits_by_task[t], size=len(inits_by_task[t]),
                                replace=True) for t in tasks}
        for a in core:
            vals = [grid[a][(t, int(i))] for t in tasks for i in chosen[t]]
            sr[a] = float(np.mean(vals))
        # Cost: block resample shared across arms and both channels.
        bidx = rng.integers(0, args.blocks, size=args.blocks)
        comp = {a: float(cost_c[a][bidx].mean()) for a in core}
        lat = {a: float(cost_l[a][bidx].mean()) for a in core}

        records.append(frontier_record(
            [(a, sr[a], comp[a], lat[a]) for a in CORE_T_ARMS],
            (sr[ARM_SV], comp[ARM_SV], lat[ARM_SV]),
        ))
        g2_sr.append(sr[ARM_SV] - sr[ARM_S0])
        g2_c.append((comp[ARM_SV] - comp[ARM_S0]) / comp[ARM_S0])
        g2_l.append((lat[ARM_SV] - lat[ARM_S0]) / lat[ARM_S0])

    g1 = gate1(records)
    result = {
        "discipline": discipline,
        "point_estimates": {
            a: {"sr": float(np.mean(list(grid[a].values()))),
                "compute": float(cost_c[a].mean()),
                "latency": float(cost_l[a].mean())}
            for a in core
        },
        "gate1": g1,
    }
    # Fixed sequence: Gate 2 is evaluated ONLY after Gate 1 passes, and the
    # confirmatory chain stops at the first failure. There is no Gate 3.
    if g1["pass"]:
        g2 = gate2(np.array(g2_sr), np.array(g2_c), np.array(g2_l))
        result["gate2"] = g2
        if g2["pass"]:
            result["verdict"] = "surface_wins_v_confirmed"
        else:
            result["verdict"] = "surface_wins_v_unconfirmed"
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
