"""Machine-checkable launch gates and the M4 smoke exit criteria (§1, §6, §3.10).

Nothing here is advisory. Every function returns violations that the caller
turns into a hard stop, because each guards a failure that is invisible once a
multi-day unattended run is under way:

  - **G-launch-1** — reward parameters and capacity are real measurements, not
    placeholders. A run started with a placeholder cost or an unsized dump
    produces numbers that look fine and mean nothing.
  - **G-launch-2** — the owner's D1-D8 rulings are frozen in a tracked run
    matrix, and the run about to start matches it field for field.
  - **M4 smoke** — five assertions over a real 20-episode batch that prove the
    loop's contracts hold end to end before 4,000 episodes are spent on them.

The interaction ledger also lives here: the headline curve's x-axis is
"episodes spent to obtain this router", so the warm-start pass and the λ pilot
are charged in full to every variant of the suite rather than amortised away.

Usage::

    uv run exp/rl_router/launch_gates.py check --matrix exp/rl_router/config/run_matrix.yaml \\
        --run-id l10_ts_lam1_s0 --artifacts exp/rl_router/data/artifacts.json
    uv run exp/rl_router/launch_gates.py smoke --batch-dir <out>/b0000 --dump-root <root>
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Optional

import yaml

from exp.rl_router.batch_package import steady_state_bytes
from exp.rl_router.pilot_lambda import (
    EVAL_EPISODES,
    LAMBDA_GRID,
    PILOT_SCHEMA,
    PILOT_BATCH_SIZE,
    PILOT_BATCHES,
    _chain_is_contiguous,
    interaction_budget,
)

# G-launch-1 capacity ceiling: steady-state live feature bytes across the dump
# root. The frozen estimate is ~2 batches (~5-6 GB); 20 GB is the abort line.
CAPACITY_CEILING_BYTES = 20 * 1024**3

# M4 smoke size (plan §8).
SMOKE_EPISODES = 20


# ---------------------------------------------------------------------------
# Run matrix
# ---------------------------------------------------------------------------


def load_run_matrix(path: str | pathlib.Path) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


def resolve_run(matrix: dict, run_id: str) -> dict:
    """Return one run's frozen row, or fail loudly listing what exists."""
    for run in matrix.get("runs", []):
        if run["id"] == run_id:
            return run
    known = [r["id"] for r in matrix.get("runs", [])]
    raise SystemExit(f"run id {run_id!r} is not in the run matrix; known: {known}")


def resolve_lambda(matrix: dict, run: dict) -> float:
    """Resolve a run's λ symbol through the pilot's recorded result.

    A null λ is the pilot not having run yet. Substituting a default here would
    let a run start with an uncalibrated cost trade-off and still be reported
    against the pre-registered λ₁/λ₂ labels.
    """
    key = run["lambda"]
    value = (matrix.get("lambda") or {}).get(key)
    if value is None:
        raise SystemExit(
            f"run {run['id']}: lambda {key!r} is still null in the run matrix — "
            "run pilot_lambda.py (M5c) and record the selection before training"
        )
    return float(value)


# ---------------------------------------------------------------------------
# G-launch gates
# ---------------------------------------------------------------------------


def check_launch_gates(
    matrix: dict,
    run: dict,
    *,
    artifacts: dict,
    batch_size: int,
    seed: int,
    variant: str,
    suite: str,
    remote_live_bytes: Optional[int] = None,
    arm_yaml: Optional[str] = None,
    planned_batches: Optional[int] = None,
    bootstrap: bool = False,
) -> list[str]:
    """Return every reason this run may NOT start. Empty list = cleared.

    ``artifacts`` maps the required pre-launch products to their paths:
    ``arm_costs`` (M5a), ``warmstart_weights`` (M5b), ``pilot`` (M5c),
    ``capacity_smoke`` (M4).
    """
    problems: list[str] = []

    # --- G-launch-2: the run matches the frozen decision record ---
    if int(batch_size) != int(matrix.get("batch_size", -1)):
        problems.append(
            f"batch_size {batch_size} != run matrix {matrix.get('batch_size')} "
            "(D2 is frozen; a different N changes the estimator)"
        )
    if int(seed) != int(run["seed"]):
        problems.append(f"seed {seed} != run matrix {run['seed']} for run {run['id']}")
    if variant != run["variant"]:
        problems.append(f"variant {variant!r} != run matrix {run['variant']!r}")
    if suite != run["suite"]:
        problems.append(f"suite {suite!r} != run matrix {run['suite']!r}")
    if suite not in (matrix.get("suites") or {}):
        problems.append(f"suite {suite!r} has no entry in the run matrix")

    # --- G-launch-1: parameters and capacity are measured, not assumed ---
    try:
        resolve_lambda(matrix, run)
    except SystemExit as exc:
        problems.append(str(exc))

    costs_path = artifacts.get("arm_costs")
    if not costs_path or not pathlib.Path(costs_path).exists():
        problems.append(
            "missing M5a cost artifact (arm_costs): the reward's cost term has no "
            "measured authority (D4)"
        )
    else:
        costs = json.loads(pathlib.Path(costs_path).read_text(encoding="utf-8"))
        problems.extend(_cost_problems(costs, variant))

    warm = artifacts.get("warmstart_weights")
    if not warm or not pathlib.Path(warm).exists():
        problems.append("missing M5b warm-start weights; training would start uninitialised")

    problems.extend(_pilot_problems(artifacts.get("pilot"), matrix, run,
                                    warmstart_path=artifacts.get("warmstart_weights")))
    problems.extend(_arm_yaml_problems(arm_yaml, run, matrix))
    if planned_batches is not None:
        expected = int(matrix["episodes_per_run"]) // int(matrix["batch_size"])
        if int(planned_batches) != expected:
            problems.append(
                f"planned {planned_batches} batches but the matrix fixes "
                f"{matrix['episodes_per_run']} episodes / {matrix['batch_size']} per batch "
                f"= {expected}; a short run is not the pre-registered experiment"
            )

    if not bootstrap:
        problems.extend(_smoke_problems(artifacts.get("capacity_smoke")))

    if remote_live_bytes is not None and remote_live_bytes > CAPACITY_CEILING_BYTES:
        problems.append(
            f"the serving host already holds {remote_live_bytes / 1024**3:.1f} GB of live "
            f"features (> {CAPACITY_CEILING_BYTES / 1024**3:.0f} GB ceiling); reclaim "
            "before launch"
        )
    return problems


def _pilot_problems(pilot_path: Optional[str], matrix: dict, run: dict,
                    warmstart_path: Optional[str] = None) -> list[str]:
    """The pilot record must actually justify the λ the matrix carries.

    Checking only that a file exists lets an empty ``{}`` clear the gate, which
    is the same as having no calibration at all: the run would report results
    against the pre-registered λ₁/λ₂ labels while λ came from nowhere.
    """
    problems: list[str] = []
    if not pilot_path or not pathlib.Path(pilot_path).exists():
        return ["missing M5c pilot record; λ would be uncalibrated"]
    try:
        doc = json.loads(pathlib.Path(pilot_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [f"pilot record {pilot_path} is not valid json"]
    if not doc.get("separated"):
        problems.append(
            "pilot record does not report a separated λ grid; the supplementary "
            "candidate or an owner ruling is still outstanding (R13)"
        )
    selected = doc.get("selected") or {}
    symbol = run["lambda"]
    matrix_value = (matrix.get("lambda") or {}).get(symbol)
    if symbol not in selected:
        problems.append(f"pilot record has no selection for {symbol!r} (got {sorted(selected)})")
    elif matrix_value is None or float(selected[symbol]) != float(matrix_value):
        problems.append(
            f"run matrix {symbol}={matrix_value!r} disagrees with the pilot's "
            f"selection {selected.get(symbol)!r}; the matrix must record what the "
            "pilot actually chose"
        )
    if doc.get("schema") != PILOT_SCHEMA:
        problems.append(
            f"pilot record schema {doc.get('schema')!r} != {PILOT_SCHEMA!r}; refusing to "
            "interpret an unrecognised calibration record"
        )
    protocol = doc.get("protocol")
    if not protocol:
        # A missing protocol block used to pass silently, so a hand-written
        # record could clear the gate without describing any protocol at all.
        problems.append("pilot record carries no protocol block")
    else:
        for key, want in (("batches", PILOT_BATCHES), ("batch_size", PILOT_BATCH_SIZE),
                          ("eval_episodes", EVAL_EPISODES), ("mode", "argmax"),
                          ("eval_pool", "b_train_remainder")):
            if key not in protocol:
                problems.append(f"pilot protocol is missing {key!r}")
            elif protocol[key] != want:
                problems.append(
                    f"pilot protocol {key}={protocol[key]!r}, frozen value is {want!r}"
                )
    if doc.get("seed") is None:
        problems.append("pilot record does not say which seed every candidate shared")

    runs = doc.get("runs") or {}
    if not runs:
        # A record that selects a λ without any candidate manifests is a claim,
        # not evidence: nothing shows a candidate was ever trained.
        problems.append("pilot record has no per-candidate manifests")
    if int(doc.get("candidates_run", -1)) != len(runs):
        problems.append(
            f"pilot candidates_run={doc.get('candidates_run')} but {len(runs)} candidate "
            "manifests are present"
        )
    if len(runs) < len(LAMBDA_GRID):
        problems.append(
            f"pilot ran {len(runs)} candidates; the frozen grid has {len(LAMBDA_GRID)}"
        )
    digests = {(r.get("expected_warmstart_sha256"), r.get("expected_pilot_split_sha256"),
                r.get("expected_remainder_split_sha256")) for r in runs.values()}
    if len(digests) > 1:
        problems.append(
            "pilot candidates did not share one warm-start checkpoint and split pair; "
            "only λ may differ between them"
        )
    for candidate, record in runs.items():
        prefix = f"pilot candidate {candidate}"
        # --- the training half, from the candidate's own run manifest ---
        try:
            candidate_lambda = float(candidate)
            recorded_lambda = float(record["lambda_recorded"])
        except (KeyError, TypeError, ValueError):
            problems.append(
                f"{prefix} did not record the lambda it actually trained"
            )
        else:
            if recorded_lambda != candidate_lambda:
                problems.append(
                    f"{prefix} actually trained lambda={recorded_lambda}, not "
                    f"candidate lambda={candidate_lambda}"
                )
        if record.get("seed") is None:
            problems.append(f"{prefix} did not record the seed it trained under")
        elif int(record["seed"]) != int(doc.get("seed", -1)):
            problems.append(f"{prefix} trained under seed {record['seed']}, not {doc.get('seed')}")
        if record.get("batch_size") is not None and \
                int(record["batch_size"]) != PILOT_BATCH_SIZE:
            problems.append(
                f"{prefix} trained with batch_size {record['batch_size']}, frozen "
                f"protocol is {PILOT_BATCH_SIZE}"
            )
        if record.get("train_judge_mode") not in (None, "sample"):
            problems.append(
                f"{prefix} trained with judge mode {record['train_judge_mode']!r}; the "
                "pilot's training half must sample"
            )
        if record.get("batches_trained") is None:
            problems.append(f"{prefix} manifest is missing batches_trained")
        elif int(record["batches_trained"]) != PILOT_BATCHES:
            problems.append(
                f"{prefix} trained {record['batches_trained']} batches, frozen protocol "
                f"is {PILOT_BATCHES}"
            )
        if not _chain_is_contiguous(record.get("weights_versions") or []):
            problems.append(
                f"{prefix} weight chain {record.get('weights_versions')} is not five "
                "contiguous one-step updates; a gap, jump, or unchanged version means "
                "an update went missing or a batch trained from the wrong policy"
            )
        # --- the evaluation half ---
        if record.get("eval_mode") != "argmax":
            problems.append(
                f"{prefix} evaluated with mode {record.get('eval_mode')!r}; the frozen "
                "protocol measures the FROZEN policy (argmax)"
            )
        if record.get("eval_episodes") is None:
            problems.append(f"{prefix} manifest is missing eval_episodes")
        elif int(record["eval_episodes"]) != EVAL_EPISODES:
            problems.append(
                f"{prefix} measured {record['eval_episodes']} eval episodes, frozen "
                f"protocol is {EVAL_EPISODES}"
            )
        final = record.get("final_weights_version")
        measured = record.get("eval_weights_versions") or []
        if final and measured and measured != [final]:
            problems.append(
                f"{prefix} evaluated weights {measured}, but its training ended on "
                f"{final!r}: the realized rate must come from the batch-5 policy"
            )
        # --- inputs must be the frozen ones, not merely self-consistent ---
        if record.get("train_split_sha256") and doc.get("pilot_split_sha256") and \
                record["train_split_sha256"] != doc["pilot_split_sha256"]:
            problems.append(f"{prefix} trained on a different split than the pilot plan")
        if warmstart_path and record.get("expected_warmstart_sha256") and \
                record["expected_warmstart_sha256"] != _sha256(warmstart_path):
            problems.append(
                f"{prefix} calibrated against a different warm-start checkpoint than "
                "the one this run would start from"
            )
    return problems


def _smoke_problems(smoke_path: Optional[str]) -> list[str]:
    """Recompute the M4 claims rather than trusting a ``passed`` flag.

    A bare ``{"passed": true}`` is hand-writable, so the gate checks the report
    is the recognised schema, names the batch and run it measured, is bound to
    the package that batch consumed, and carries real before/after-reclaim
    measurements inside the capacity ceiling.
    """
    if not smoke_path or not pathlib.Path(smoke_path).exists():
        return ["missing M4 capacity smoke report (G-launch-1 capacity assertion)"]
    try:
        report = json.loads(pathlib.Path(smoke_path).read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return [f"M4 smoke report {smoke_path} is not valid json"]
    problems: list[str] = []
    if report.get("schema") != M4_SCHEMA:
        problems.append(
            f"M4 report schema {report.get('schema')!r} != {M4_SCHEMA!r}; a hand-written "
            "'passed' flag is not capacity evidence"
        )
    if not report.get("batch_id"):
        problems.append("M4 report does not name the batch it measured")
    if int(report.get("episodes", 0)) != SMOKE_EPISODES:
        problems.append(
            f"M4 report covers {report.get('episodes')} episodes, expected {SMOKE_EPISODES}"
        )
    if not report.get("bytes_per_step"):
        problems.append("M4 report has no measured per-step dump bytes")
    if not report.get("run_id"):
        problems.append("M4 report is not bound to a run id")
    if not report.get("package_sha256"):
        problems.append("M4 report is not bound to the package its batch consumed")
    if report.get("bytes_before_reclaim") is None or report.get("bytes_after_reclaim") is None:
        problems.append(
            "M4 report lacks before/after reclaim byte counts: without them the "
            "peak is an estimate, and the capacity claim is unfalsifiable"
        )
    peak = report.get("peak_bytes")
    if peak is None:
        problems.append("M4 report has no measured peak byte usage")
    elif int(peak) > CAPACITY_CEILING_BYTES:
        problems.append(
            f"M4 peak {int(peak) / 1024**3:.1f} GB exceeds the "
            f"{CAPACITY_CEILING_BYTES / 1024**3:.0f} GB ceiling"
        )
    versions = report.get("weights_versions") or []
    if len(versions) != 2 or not _advanced_by_one(*versions):
        problems.append(f"M4 report weights versions {versions} did not advance by one")
    if report.get("violations"):
        problems.append(f"M4 smoke recorded violations: {report['violations']}")
    if not report.get("passed"):
        problems.append("M4 smoke did not pass")
    return problems


def _sha256(path: str) -> str:
    import hashlib

    p = pathlib.Path(path)
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else ""


def _arm_yaml_problems(arm_yaml: Optional[str], run: dict, matrix: dict) -> list[str]:
    """The yaml that will actually be dispatched must match the frozen row.

    Comparing the matrix against values the caller derived from the matrix is a
    tautology; the yaml is the artefact the fleet consumes, so it is what has to
    agree.
    """
    if not arm_yaml:
        return []
    path = pathlib.Path(arm_yaml)
    if not path.exists():
        return [f"arm yaml {arm_yaml} does not exist"]
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    judge = ((cfg.get("checkpoints") or {}).get("cp1") or {}).get("judge") or {}
    problems: list[str] = []
    expected_arms = {"R_ts": "ts", "R_tc": "tc", "R_tsc": "tsc"}.get(run["variant"])
    if judge.get("type") != "mlp_router":
        problems.append(f"arm yaml judge.type is {judge.get('type')!r}, expected 'mlp_router'")
    if expected_arms and judge.get("arms") != expected_arms:
        problems.append(
            f"arm yaml arms={judge.get('arms')!r} does not match variant "
            f"{run['variant']} (expected {expected_arms!r})"
        )
    if judge.get("mode") != "sample":
        problems.append(
            f"arm yaml mode={judge.get('mode')!r}: a training run must sample "
            "(argmax is the frozen-evaluation setting)"
        )
    if judge.get("seed") is None or int(judge["seed"]) != int(run["seed"]):
        problems.append(
            f"arm yaml seed={judge.get('seed')!r} != run matrix seed {run['seed']}"
        )
    if not judge.get("dump_dir"):
        problems.append("arm yaml has no dump_dir; a training run must record rollouts")
    return problems


def _cost_problems(costs: dict, variant: str) -> list[str]:
    """The cost artifact must be a real per-arm GPU-time measurement."""
    problems: list[str] = []
    normalized = costs.get("normalized_costs") or {}
    if normalized.get("teacher") != 1.0:
        problems.append(f"arm_costs must be normalized to teacher=1.0 (got {normalized})")
    needed = {"R_ts": ("teacher", "student"), "R_tc": ("teacher", "cache"),
              "R_tsc": ("teacher", "student", "cache")}.get(variant, ())
    missing = [arm for arm in needed if arm not in normalized]
    if missing:
        problems.append(f"arm_costs is missing measurements for {missing} (variant {variant})")
    for arm in needed:
        prov = (costs.get("provenance") or {}).get(arm)
        if not prov or not prov.get("gpu_timed"):
            problems.append(
                f"arm_costs[{arm}] is not a server-side GPU-time measurement "
                "(D4 requires batch=1 GPU time; wall-clock is report-only)"
            )
            continue
        if not prov.get("in_process"):
            problems.append(
                f"arm_costs[{arm}] was not measured in the process that owns the "
                "model; an RPC round trip cannot be GPU-timed by its caller"
            )
        if arm == "student" and not prov.get("task_prompts"):
            problems.append(
                "arm_costs[student] records no task prompts: the ACT ensemble routes "
                "by exact prompt match, so a cost without them was not measured on "
                "the real student path"
            )
    return problems


# ---------------------------------------------------------------------------
# M4 smoke — five machine-checkable exit criteria
# ---------------------------------------------------------------------------


M4_SCHEMA = "rl_router_m4_v1"
RUN_MANIFEST_SCHEMA = "rl_router_run_v1"


def m4_smoke(
    *,
    manifest: dict,
    metrics: list[dict],
    checkpoint_versions: list[str],
    next_batch_shards: list[dict],
    dump_root: str | pathlib.Path,
    episodes: int = SMOKE_EPISODES,
    run_id: str = "",
    batch_id: str = "",
    package_sha256: str = "",
    bytes_before_reclaim: Optional[int] = None,
    bytes_after_reclaim: Optional[int] = None,
) -> dict:
    """Evaluate the 20-episode exit criteria. Returns a machine-checkable report.

    1. the three-source join is 100% complete (no missing slot, no rejection);
    2. exactly one update landed, and the version advanced exactly once;
    3. every episode of the NEXT batch ran under the new weights version;
    4. the fallback rate is recorded (a cache arm hitting an empty library must
       be visible, not inferred);
    5. per-step dump bytes and steady-state disk are measured and under budget.
    """
    violations: list[str] = []

    if not manifest.get("complete"):
        violations.append(f"join incomplete: missing {manifest.get('missing_slots')}")
    if manifest.get("rejected"):
        violations.append(f"join rejected {len(manifest['rejected'])} episode(s)")
    selected = manifest.get("training_selected", [])
    if len(selected) != episodes:
        violations.append(f"expected {episodes} selected episodes, got {len(selected)}")

    applied = [m for m in metrics if not m.get("skipped") and not m.get("admission_failed")]
    if len(applied) != 1:
        violations.append(f"expected exactly one applied update, got {len(applied)}")
    if len(checkpoint_versions) != 2 or not _advanced_by_one(*checkpoint_versions):
        # "changed" is not the contract: v0 -> v9 means eight updates went
        # missing, and that has to be visible here rather than in the results.
        violations.append(
            f"weights version did not advance by exactly one: {checkpoint_versions}"
        )

    if next_batch_shards:
        stale = sorted({
            str(s.get("weights_version")) for s in next_batch_shards
            if str(s.get("weights_version")) != checkpoint_versions[-1]
        })
        if stale:
            violations.append(
                f"next batch still ran under stale weights {stale}; expected "
                f"{checkpoint_versions[-1]}"
            )
    else:
        violations.append("no next-batch shards to check the weight rollover against")

    if applied and "arm_executed_rate" not in applied[0]:
        violations.append("metrics row does not record the executed-arm / fallback rates")

    rows = sum(int(s.get("rows", 0)) for s in selected)
    dims = {int(s.get("dim", 0)) for s in selected}
    bytes_per_step = (max(dims) * 2) if dims else 0
    live = steady_state_bytes(dump_root)
    if bytes_per_step <= 0:
        violations.append("could not measure per-step dump bytes from the manifest")
    if live > CAPACITY_CEILING_BYTES:
        violations.append(
            f"steady-state dump {live / 1024**3:.1f} GB exceeds the "
            f"{CAPACITY_CEILING_BYTES / 1024**3:.0f} GB ceiling"
        )

    peak = max(x for x in (bytes_before_reclaim, live) if x is not None)
    return {
        "schema": M4_SCHEMA,
        "passed": not violations,
        "violations": violations,
        "run_id": run_id,
        "batch_id": batch_id or manifest.get("batch_id", ""),
        "package_sha256": package_sha256,
        "episodes": len(selected),
        "steps": rows,
        "bytes_per_step": bytes_per_step,
        "peak_bytes": peak,
        "bytes_before_reclaim": bytes_before_reclaim,
        "bytes_after_reclaim": bytes_after_reclaim if bytes_after_reclaim is not None else live,
        "steady_state_bytes": live,
        "steady_state_gb": round(live / 1024**3, 3),
        "weights_versions": checkpoint_versions,
    }


# ---------------------------------------------------------------------------
# Run manifest
# ---------------------------------------------------------------------------


def _advanced_by_one(before: str, after: str) -> bool:
    """``vN`` -> ``vN+1`` numerically, not merely "different"."""
    try:
        return int(after[1:]) - int(before[1:]) == 1 and before[0] == after[0] == "v"
    except (ValueError, IndexError):
        return False


def build_run_manifest(
    *,
    matrix: dict,
    run: dict,
    lam: float,
    init_states_dir: str,
    arm_costs: dict,
    warmstart_episodes: int,
    artifacts: dict,
    pilot_candidates: Optional[int] = None,
    split_path: str = "",
    arm_yaml: str = "",
    judge_mode: str = "",
) -> dict:
    """The tracked record a launch gate can re-check and analysis can cite.

    Carries the interaction ledger's shared offset explicitly: the headline
    x-axis is not the optimizer's episode count, and the difference has to be
    reproducible from the manifest rather than from a note in a log.
    """
    # Charge the ledger by the candidates the pilot ACTUALLY ran: a
    # supplementary λ is a real interaction cost, and pricing it at the grid
    # size would under-report exactly the runs that needed extra calibration.
    ledger = interaction_budget(
        warmstart_episodes=warmstart_episodes,
        n_candidates=int(pilot_candidates) if pilot_candidates else None,
    )
    return {
        "schema": RUN_MANIFEST_SCHEMA,
        "run_id": run["id"],
        "suite": run["suite"],
        "variant": run["variant"],
        "seed": run["seed"],
        "flagship": bool(run.get("flagship")),
        "lambda_symbol": run["lambda"],
        "lambda_value": lam,
        "batch_size": matrix["batch_size"],
        "episodes_per_run": matrix["episodes_per_run"],
        "init_states_dir": init_states_dir,
        "split": split_path,
        "split_sha256": _sha256(split_path) if split_path else "",
        "arm_yaml": arm_yaml,
        "arm_yaml_sha256": _sha256(arm_yaml) if arm_yaml else "",
        "judge_mode": judge_mode,
        "training_init_domain": matrix["decisions"]["D3_training_init_domain"],
        "arm_costs": arm_costs,
        "decisions": matrix["decisions"],
        "artifacts": artifacts,
        "interaction_ledger": ledger,
        "primary_seed": (matrix.get("eval") or {}).get("primary_seed", 0),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cmd_check(args) -> None:
    matrix = load_run_matrix(args.matrix)
    run = resolve_run(matrix, args.run_id)
    artifacts = json.loads(pathlib.Path(args.artifacts).read_text(encoding="utf-8"))
    problems = check_launch_gates(
        matrix, run, artifacts=artifacts,
        batch_size=args.batch_size or matrix["batch_size"],
        seed=args.seed if args.seed is not None else run["seed"],
        variant=args.variant or run["variant"],
        suite=args.suite or run["suite"],
        remote_live_bytes=args.remote_live_bytes if args.remote_live_bytes >= 0 else None,
        arm_yaml=args.arm_yaml or None,
        planned_batches=args.planned_batches or None,
    )
    if problems:
        print("\n".join(f"- {p}" for p in problems), file=sys.stderr)
        raise SystemExit(1)
    print(f"launch gates cleared for {args.run_id}")


def _cmd_smoke(args) -> None:
    """Consume the run loop's ACTUAL layout.

    The loop fetches the remote join to ``<batch>/remote_manifest.json`` and the
    trainer keeps ``metrics.jsonl`` at the run root on the serving host, so the
    smoke reads those; the earlier fixed ``<batch>/manifest.json`` +
    ``<batch>/metrics.jsonl`` paths existed nowhere a real run writes.
    """
    from exp.rl_router.batch_package import load_shard_manifest, read_jsonl

    batch_dir = pathlib.Path(args.batch_dir)
    manifest_path = pathlib.Path(args.manifest or (batch_dir / "remote_manifest.json"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metrics = [m for m in read_jsonl(args.metrics or (batch_dir / "metrics.jsonl"))
               if not args.batch_id or m.get("batch_id") == args.batch_id]
    versions = json.loads((batch_dir / "versions.json").read_text(encoding="utf-8"))
    next_shards = (
        load_shard_manifest(args.next_shards)
        if args.next_shards and pathlib.Path(args.next_shards).exists() else []
    )
    package_sha = ""
    if args.package:
        marker = pathlib.Path(args.package) / "COMPLETE.json"
        if marker.exists():
            package_sha = json.loads(marker.read_text(encoding="utf-8"))["package_sha256"]
    report = m4_smoke(
        manifest=manifest, metrics=metrics, checkpoint_versions=versions,
        next_batch_shards=next_shards, dump_root=args.dump_root, episodes=args.episodes,
        run_id=args.run_id, batch_id=args.batch_id or manifest.get("batch_id", ""),
        package_sha256=package_sha,
        bytes_before_reclaim=args.bytes_before_reclaim or None,
    )
    out = pathlib.Path(args.out or (batch_dir / "m4_smoke.json"))
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_check = sub.add_parser("check", help="G-launch-1 / G-launch-2")
    p_check.add_argument("--matrix", required=True)
    p_check.add_argument("--run-id", required=True)
    p_check.add_argument("--artifacts", required=True)
    p_check.add_argument("--batch-size", type=int, default=0)
    p_check.add_argument("--seed", type=int, default=None)
    p_check.add_argument("--variant", default="")
    p_check.add_argument("--suite", default="")
    p_check.add_argument("--remote-live-bytes", type=int, default=-1,
                         help="live feature bytes on the SERVING host (-1 = unknown)")
    p_check.add_argument("--arm-yaml", default="")
    p_check.add_argument("--planned-batches", type=int, default=0)
    p_check.set_defaults(func=_cmd_check)

    p_smoke = sub.add_parser("smoke", help="M4 five-assertion exit criteria")
    p_smoke.add_argument("--batch-dir", required=True)
    p_smoke.add_argument("--dump-root", required=True)
    p_smoke.add_argument("--next-shards", default="")
    p_smoke.add_argument("--manifest", default="", help="default <batch-dir>/remote_manifest.json")
    p_smoke.add_argument("--metrics", default="", help="fetched metrics.jsonl")
    p_smoke.add_argument("--package", default="", help="the consumed package dir (for its sha)")
    p_smoke.add_argument("--run-id", default="")
    p_smoke.add_argument("--batch-id", default="")
    p_smoke.add_argument("--bytes-before-reclaim", type=int, default=0)
    p_smoke.add_argument("--episodes", type=int, default=SMOKE_EPISODES)
    p_smoke.add_argument("--out", default="")
    p_smoke.set_defaults(func=_cmd_smoke)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
