"""Calibrate λ by a short training pilot (§3.10 / D1, M5c).

λ sets how hard the reward penalises execution cost, which is what decides how
much of the trade-off curve a run explores. It cannot be picked by inspection:
under a *fixed* policy the arm distribution does not depend on λ at all — the
cost term shifts every episode's reward by a per-episode constant and the
batch-mean baseline subtracts most of it back. λ only becomes visible through
what it *trains*, so each candidate has to be trained, briefly, and then
measured.

Protocol (frozen):

  1. candidates λ ∈ {0.05, 0.2, 0.5};
  2. each candidate starts from the SAME warm-start checkpoint and the SAME
     seed, and trains 5 batches x 100 episodes on a pilot-only subset of
     B-train (30 inits/task, a frozen list);
  3. its batch-5 weights are frozen and evaluated in argmax mode on the
     B-train remainder that the pilot did not touch — never B-val, which stays
     held out for checkpoint selection;
  4. λ₁ = the candidate whose realized teacher rate is closest to 40%,
     λ₂ = closest to 20%.

If both targets land on one candidate, or all three sit in the same regime, one
supplementary candidate is inserted at the geometric mean of the closest pair —
at most once. Still no separation escalates to the owner rather than silently
picking a λ the protocol did not distinguish.

Usage::

    uv run exp/rl_router/pilot_lambda.py plan --split <split.yaml> --out <dir>
    uv run exp/rl_router/pilot_lambda.py select --measurements <measured.json>
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib

from typing import Optional

import numpy as np

# Bumped whenever the pilot's published contract changes; the launch gate
# refuses a record it does not recognise rather than guessing its meaning.
PILOT_SCHEMA = "rl_router_pilot_v1"

LAMBDA_GRID = (0.05, 0.2, 0.5)
TARGET_RATES = {"lambda_1": 0.40, "lambda_2": 0.20}
PILOT_INITS_PER_TASK = 30
PILOT_BATCHES = 5
PILOT_BATCH_SIZE = 100
EVAL_EPISODES = 100


def pilot_split(split_doc: dict, *, inits_per_task: int = PILOT_INITS_PER_TASK,
                seed: int = 0) -> dict:
    """Split B-train into the pilot subset and the remainder used to measure.

    Both halves are frozen lists rather than a runtime sample: the pilot must
    not train on what it is then graded on, and that separation has to be
    auditable after the fact.
    """
    rng = np.random.RandomState(seed)
    pilot, remainder = {}, {}
    for key in sorted(split_doc):
        train = list(split_doc[key]["train"])
        if inits_per_task > len(train):
            raise ValueError(
                f"{key}: pilot needs {inits_per_task} inits but B-train has {len(train)}"
            )
        order = rng.permutation(len(train))
        chosen = sorted(train[i] for i in order[:inits_per_task])
        pilot[key] = chosen
        remainder[key] = sorted(set(train) - set(chosen))
    return {"pilot": pilot, "remainder": remainder,
            "inits_per_task": inits_per_task, "seed": seed}


def select_lambdas(measurements: dict[float, float]) -> dict:
    """Pick λ₁ / λ₂ from realized teacher rates, applying the frozen tie rule.

    ``measurements`` maps λ to the teacher rate its frozen pilot weights
    produced on the held-out remainder.
    """
    if not measurements:
        raise ValueError("no measurements to select from")
    picks = {
        name: min(measurements, key=lambda lam: abs(measurements[lam] - target))
        for name, target in TARGET_RATES.items()
    }
    distinct = len(set(picks.values())) == len(picks)
    result = {
        "measurements": {str(k): v for k, v in sorted(measurements.items())},
        "targets": TARGET_RATES,
        "selected": {k: float(v) for k, v in picks.items()},
        "separated": distinct,
    }
    if not distinct:
        result["supplementary_candidate"] = supplementary_candidate(measurements)
        result["action"] = (
            "run the supplementary candidate once, then re-select; if it still does "
            "not separate, escalate to the owner (plan R13)"
        )
    return result


def supplementary_candidate(measurements: dict[float, float]) -> float:
    """Geometric mean of the two candidates whose rates bracket the targets.

    Geometric rather than arithmetic because the grid is multiplicative: the
    interesting scale between 0.05 and 0.2 is 0.1, not 0.125.
    """
    lams = sorted(measurements)
    if len(lams) < 2:
        raise ValueError("need at least two candidates to interpolate between")
    # The adjacent pair whose realized rates are furthest apart brackets the
    # regime change; splitting anywhere else cannot separate the targets.
    pairs = [(lams[i], lams[i + 1]) for i in range(len(lams) - 1)]
    lo, hi = max(pairs, key=lambda p: abs(measurements[p[1]] - measurements[p[0]]))
    return float(math.sqrt(lo * hi))


def interaction_budget(*, warmstart_episodes: int, pilot_batches: int = PILOT_BATCHES,
                       pilot_batch_size: int = PILOT_BATCH_SIZE,
                       n_candidates: Optional[int] = None,
                       eval_episodes: int = EVAL_EPISODES) -> dict:
    """Episodes charged to every variant before formal training starts.

    Charged in full to each variant of the suite, not amortised: the headline
    interaction-efficiency curve answers "what did it cost to get this router",
    and the warm-start pass and the pilot are part of that cost. Sharing them
    across variants would let the curve start from a free lunch. Variants of one
    suite therefore share a constant offset, which is recorded here so the plot
    is reproducible from the numbers rather than from a note.
    """
    # None = the frozen grid; a run whose pilot needed the supplementary
    # candidate passes the real count so the ledger is not under-charged.
    candidates = len(LAMBDA_GRID) if n_candidates is None else int(n_candidates)
    pilot_total = candidates * (pilot_batches * pilot_batch_size + eval_episodes)
    return {
        "warmstart_episodes": warmstart_episodes,
        "pilot_candidates": candidates,
        "pilot_episodes": pilot_total,
        "shared_offset": warmstart_episodes + pilot_total,
        "note": (
            "charged in full to every variant of the suite (conservative); the "
            "optimizer-only axis is reported separately as a secondary diagnostic"
        ),
    }


def write_split_yaml(part: dict[str, list[int]], source: dict, out_path: str | pathlib.Path) -> pathlib.Path:
    """Materialise one half of the pilot split as a split yaml.

    ``run_rl_router.btrain_pairs`` reads ``train`` from a split yaml, so writing
    the pilot subset and the measurement remainder as real split files is what
    makes "train here, measure there" mechanical instead of a convention someone
    has to remember.
    """
    import yaml as _yaml

    doc = {
        key: {**{k: v for k, v in source[key].items() if k != "train"},
              "train": list(part[key]), "val": []}
        for key in part
    }
    target = pathlib.Path(out_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(_yaml.safe_dump(doc, sort_keys=True), encoding="utf-8")
    return target


def candidate_manifest(
    candidate_dir: str | pathlib.Path,
    *,
    lam: float,
    seed: int,
    eval_rows: list[dict],
    teacher_rate: float,
    command: str,
    warmstart_weights: str,
    pilot_split: str,
    remainder_split: str,
) -> dict:
    """Describe what a candidate ACTUALLY ran, read from its own artifacts.

    A command string proves nothing — an arbitrary template can ignore every
    placeholder it was given — and neither does echoing back the controller's
    own inputs. Everything here is read from what the candidate produced:

      - its ``run_manifest.json`` (written by the candidate's own training run)
        supplies the seed, batch size, split identity and judge mode it really
        used;
      - the per-batch ``versions.json`` files supply the weight chain, which
        must be five contiguous updates;
      - its ``eval/arm.yaml`` supplies the evaluation mode and the weights the
        evaluation was actually pointed at;
      - the eval rows supply the measured episode count and the version those
        episodes ran under.

    Anything the candidate did not record comes back as None, which the launch
    gate treats as missing evidence rather than as agreement.
    """
    directory = pathlib.Path(candidate_dir)
    run_manifest = _read_json(directory / "run_manifest.json") or {}
    eval_yaml = _read_yaml(directory / "eval" / "arm.yaml") or {}
    eval_judge = ((eval_yaml.get("checkpoints") or {}).get("cp1") or {}).get("judge") or {}

    versions: list[list[str]] = []
    for batch in sorted(d for d in directory.glob("b[0-9]*") if d.is_dir()):
        chain = _read_json(batch / "versions.json")
        if chain:
            versions.append(chain)
    episodes = sorted({(r.get("task_uid"), r.get("attempt")) for r in eval_rows})
    eval_versions = sorted({
        str((r.get("router_outputs") or {}).get("weights_version"))
        for r in eval_rows if r.get("router_outputs")
    })
    return {
        "lambda": lam,
        "command": command,
        "dir": str(directory),
        # --- read from the candidate's own run manifest ---
        "seed": run_manifest.get("seed"),
        "batch_size": run_manifest.get("batch_size"),
        "train_split_sha256": run_manifest.get("split_sha256"),
        "train_judge_mode": run_manifest.get("judge_mode"),
        "lambda_recorded": run_manifest.get("lambda_value"),
        # --- read from the produced batches ---
        "batches_trained": len(versions),
        "weights_versions": versions,
        "version_chain_contiguous": _chain_is_contiguous(versions),
        "start_weights_version": versions[0][0] if versions else None,
        "final_weights_version": versions[-1][-1] if versions else None,
        # --- read from the evaluation ---
        "eval_mode": eval_judge.get("mode"),
        "eval_weights_path": eval_judge.get("weights_path"),
        "eval_rows": len(eval_rows),
        "eval_episodes": len(episodes),
        "eval_weights_versions": eval_versions,
        # --- the controller's inputs, recorded for cross-checking ---
        "expected_warmstart_sha256": _sha256(warmstart_weights),
        "expected_pilot_split_sha256": _sha256(pilot_split),
        "expected_remainder_split_sha256": _sha256(remainder_split),
        "seed_requested": seed,
        "teacher_rate": teacher_rate,
    }


def _chain_is_contiguous(versions: list[list[str]]) -> bool:
    """Every batch is one ``vN -> vN+1`` update, with no gap between batches.

    Merely checking adjacent endpoints is insufficient: five ``v9 -> v9``
    marker files are adjacent but contain no optimizer update at all.
    """
    if not isinstance(versions, list) or not versions:
        return False
    for chain in versions:
        if not isinstance(chain, list) or len(chain) != 2:
            return False
        before, after = chain
        if not isinstance(before, str) or not isinstance(after, str):
            return False
        if not before.startswith("v") or not after.startswith("v"):
            return False
        try:
            before_number = int(before[1:])
            after_number = int(after[1:])
        except ValueError:
            return False
        if after_number != before_number + 1:
            return False
    for earlier, later in zip(versions, versions[1:]):
        if earlier[1] != later[0]:
            return False
    return True


def _read_json(path: pathlib.Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def _read_yaml(path: pathlib.Path):
    if not path.exists():
        return None
    import yaml as _yaml

    try:
        return _yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a malformed artifact is missing evidence
        return None


def realized_teacher_rate(client_rows: list[dict]) -> float:
    """Fraction of executed steps that ran on the teacher.

    Measured on ``arm_executed``, not on what was sampled: a cache arm that hit
    an empty library ran the teacher, and the rate λ is calibrated against is
    what the fleet actually spent.
    """
    executed = [
        str(row["router_outputs"]["arm_executed"])
        for row in client_rows
        if row.get("_kind") is None and row.get("router_outputs")
    ]
    if not executed:
        raise ValueError("no executed arms in the client rows; the eval produced nothing")
    return sum(a == "teacher" for a in executed) / len(executed)


def candidate_command(
    *,
    template: str,
    lam: float,
    candidate_dir: str | pathlib.Path,
    pilot_split_path: str | pathlib.Path,
    remainder_split_path: str | pathlib.Path,
    warmstart_weights: str,
    seed: int = 0,
) -> str:
    """Render one candidate's invocation.

    Every candidate starts from the SAME warm-start checkpoint and the SAME
    seed; only λ differs. Anything else varying would make the comparison
    between candidates uninterpretable — which is the whole point of the pilot.
    """
    return template.format(
        lam=lam, candidate_dir=str(candidate_dir), seed=seed,
        pilot_split=str(pilot_split_path), remainder_split=str(remainder_split_path),
        warmstart_weights=warmstart_weights, batches=PILOT_BATCHES,
        batch_size=PILOT_BATCH_SIZE, eval_episodes=EVAL_EPISODES,
    )


def _sha256(path: str | pathlib.Path) -> str:
    import hashlib

    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _cmd_plan(args) -> None:
    import yaml

    doc = yaml.safe_load(pathlib.Path(args.split).read_text(encoding="utf-8"))
    plan = pilot_split(doc, seed=args.seed)
    plan["lambda_grid"] = list(LAMBDA_GRID)
    plan["protocol"] = {
        "batches": PILOT_BATCHES, "batch_size": PILOT_BATCH_SIZE,
        "eval_episodes": EVAL_EPISODES, "mode": "argmax", "eval_pool": "B-train remainder",
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    plan_dir = out.parent
    plan["pilot_split_yaml"] = str(write_split_yaml(plan["pilot"], doc, plan_dir / "pilot_split.yaml"))
    plan["remainder_split_yaml"] = str(
        write_split_yaml(plan["remainder"], doc, plan_dir / "remainder_split.yaml")
    )
    out.write_text(json.dumps(plan, indent=2), encoding="utf-8")
    print(f"pilot plan -> {out}")


def _cmd_run(args) -> None:
    """Train every candidate, measure it on the held-out remainder, select.

    λ cannot be read off a fixed policy: the cost term shifts every episode's
    reward by a per-episode constant and the batch-mean baseline subtracts most
    of it back, so under a frozen policy the arm distribution is independent of
    λ. Each candidate therefore has to be *trained* briefly and then measured.
    """
    import subprocess

    plan = json.loads(pathlib.Path(args.plan).read_text(encoding="utf-8"))
    template = pathlib.Path(args.command_template).read_text(encoding="utf-8").strip()
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    measurements: dict[float, float] = {}
    runs: dict[str, dict] = {}
    candidates = list(LAMBDA_GRID)
    supplementary_used = False

    while True:
        for lam in candidates:
            if lam in measurements:
                continue
            candidate_dir = out_dir / f"lam_{lam}"
            cmd = candidate_command(
                template=template, lam=lam, candidate_dir=candidate_dir,
                pilot_split_path=plan["pilot_split_yaml"],
                remainder_split_path=plan["remainder_split_yaml"],
                warmstart_weights=args.warmstart_weights, seed=args.seed,
            )
            print(f"[pilot] lambda={lam}: {cmd}")
            result = subprocess.run(cmd, shell=True, check=False)
            if result.returncode != 0:
                raise SystemExit(f"pilot candidate lambda={lam} failed (exit {result.returncode})")
            rows_path = candidate_dir / "eval" / "client_rows.jsonl"
            rows = [
                json.loads(line)
                for line in rows_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            measurements[lam] = realized_teacher_rate(rows)
            runs[str(lam)] = candidate_manifest(
                candidate_dir, lam=lam, seed=args.seed, eval_rows=rows,
                teacher_rate=measurements[lam], command=cmd,
                warmstart_weights=args.warmstart_weights,
                pilot_split=plan["pilot_split_yaml"],
                remainder_split=plan["remainder_split_yaml"],
            )
            print(f"[pilot] lambda={lam}: realized teacher rate {measurements[lam]:.3f}")

        result = select_lambdas(measurements)
        # Self-contained record: the gate must be able to check WHAT produced
        # this λ, not merely that a file exists. Protocol, seed, the warm-start
        # digest and the split identities all ride along, plus one manifest per
        # candidate actually run.
        result.update({
            "schema": PILOT_SCHEMA,
            "protocol": {
                "batches": PILOT_BATCHES, "batch_size": PILOT_BATCH_SIZE,
                "eval_episodes": EVAL_EPISODES, "mode": "argmax",
                "eval_pool": "b_train_remainder",
            },
            "seed": args.seed,
            "warmstart_weights": args.warmstart_weights,
            "warmstart_sha256": _sha256(args.warmstart_weights),
            "pilot_split_sha256": _sha256(plan["pilot_split_yaml"]),
            "remainder_split_sha256": _sha256(plan["remainder_split_yaml"]),
            "candidates_run": len(runs),
            "runs": runs,
        })
        (out_dir / "measurements.json").write_text(
            json.dumps({"teacher_rate_by_lambda": {str(k): v for k, v in measurements.items()}},
                       indent=2),
            encoding="utf-8",
        )
        (out_dir / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        if result["separated"]:
            print(json.dumps(result, indent=2))
            return
        if supplementary_used:
            raise SystemExit(
                "ALERT: the lambda grid did not separate even after the one "
                "supplementary candidate (plan R13); escalate to the owner"
            )
        supplementary_used = True
        candidates = [result["supplementary_candidate"]]
        print(f"[pilot] inserting supplementary candidate {candidates[0]}")


def build_eval_yaml(arm_yaml: str | pathlib.Path, *, weights_path: str) -> dict:
    """Turn a candidate's TRAIN arm yaml into its frozen-evaluation twin.

    Derived from the training yaml rather than written independently, so the key
    builder, the artifact and the search config are provably the ones the
    candidate trained under — the realized teacher rate has to be measured on
    the same observation space, or it is not measuring that candidate.

    The three edits are the whole protocol: ``argmax`` freezes the policy
    (sampling would measure a different distribution than the one being
    selected), no ``dump_dir`` keeps the measurement from becoming another
    training rollout, and ``weights_path`` points at the batch-5 checkpoint.
    """
    import copy

    import yaml as _yaml

    cfg = copy.deepcopy(_yaml.safe_load(
        pathlib.Path(arm_yaml).read_text(encoding="utf-8")
    ))
    judge = cfg["checkpoints"]["cp1"]["judge"]
    judge["mode"] = "argmax"
    judge["weights_path"] = str(weights_path)
    judge.pop("dump_dir", None)
    judge.pop("temperature", None)
    judge.pop("seed", None)
    return cfg


def _cmd_eval(args) -> None:
    """Measure one candidate's frozen batch-5 policy on the B-train remainder.

    This is step 3 of the frozen protocol. It is a separate invocation from the
    training half because the two differ in every way that matters — a frozen
    policy, no dump, and a pool the candidate never trained on — and because the
    controller (``run``) needs the two halves to be independently re-runnable
    when one of them dies mid-pilot.
    """
    import yaml as _yaml

    from openpi.cache.config import load_cache_config
    from openpi.conductor import ServerEndpoint, WorkerSpec

    from exp.rl_router.run_rl_router import (
        RouterBatchStrategy,
        btrain_pairs,
        make_slots,
        resolve_init_states_dir,
        run_round,
        sample_batch,
    )

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_yaml = out_dir / "arm.yaml"
    cfg = build_eval_yaml(args.arm_yaml, weights_path=args.weights)
    eval_yaml.write_text(_yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    load_cache_config(str(eval_yaml))    # allowlist + mlp_router rules, before any episode
    if cfg["checkpoints"]["cp1"]["judge"].get("dump_dir"):
        raise SystemExit("the pilot evaluation must not write rollouts")

    # Same guard as every other pass: an empty init dir makes LIBERO fall back to
    # the official pruned_init A pool, and the pilot would calibrate λ on the
    # frozen test set.
    init_dir = resolve_init_states_dir(args.init_states_dir)

    remainder = btrain_pairs(args.split)
    if len(remainder) < args.episodes:
        raise SystemExit(
            f"the remainder split holds {len(remainder)} inits but the frozen protocol "
            f"measures {args.episodes}; the pilot subset was cut too large"
        )
    # Deterministic draw, so a re-run of a candidate measures the same episodes.
    pairs = sample_batch(remainder, batch_size=args.episodes, batch_idx=0, seed=args.seed)
    yaml_id = eval_yaml.stem
    slots = make_slots(pairs, yaml_id=yaml_id)
    servers = [ServerEndpoint(*_endpoint(s)) for s in args.servers.split(",")]
    specs = [
        WorkerSpec(worker_id=f"w{i}", server_key=servers[i % len(servers)].key,
                   gpu_id=str(i % args.gpus), conda_env=args.conda_env,
                   task_suite_name=args.suite, init_states_dir=init_dir)
        for i in range(args.workers)
    ]
    strategy = RouterBatchStrategy(
        suite=args.suite, yaml_path=str(eval_yaml), run_id=args.run_id,
        batch_id="eval", weights_version=args.weights_version,
        bundle_id=f"pilot_{args.run_id}_eval_{args.weights_version}",
        slots=slots, trials_per_task=args.episodes,
    )
    _journal, client_rows = run_round(
        strategy=strategy, yaml_id=yaml_id, servers=servers, worker_specs=specs,
        journal_path=str(out_dir / "journal.jsonl"),
        rows_path=str(out_dir / "client_rows.jsonl"),
        bind_host=args.bind_host, episode_timeout_s=args.episode_timeout_s,
    )
    rate = realized_teacher_rate(client_rows)
    (out_dir / "teacher_rate.json").write_text(
        json.dumps({"teacher_rate": rate, "episodes": args.episodes,
                    "weights_version": args.weights_version,
                    "weights_path": args.weights, "split": str(args.split)}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"teacher_rate": rate, "rows": len(client_rows)}, indent=2))


def _endpoint(spec: str) -> tuple[str, int]:
    host, port = spec.rsplit(":", 1)
    return host, int(port)


def _cmd_select(args) -> None:
    raw = json.loads(pathlib.Path(args.measurements).read_text(encoding="utf-8"))
    measurements = {float(k): float(v) for k, v in raw["teacher_rate_by_lambda"].items()}
    result = select_lambdas(measurements)
    print(json.dumps(result, indent=2))
    if not result["separated"]:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_plan = sub.add_parser("plan", help="freeze the pilot / remainder init lists")
    p_plan.add_argument("--split", required=True)
    p_plan.add_argument("--seed", type=int, default=0)
    p_plan.add_argument("--out", required=True)
    p_plan.set_defaults(func=_cmd_plan)

    p_run = sub.add_parser("run", help="train + measure every candidate, then select")
    p_run.add_argument("--plan", required=True, help="output of `plan`")
    p_run.add_argument("--command-template", required=True,
                       help="file holding the per-candidate shell template; placeholders: "
                            "{lam} {candidate_dir} {pilot_split} {remainder_split} "
                            "{warmstart_weights} {batches} {batch_size} {eval_episodes}")
    p_run.add_argument("--warmstart-weights", required=True)
    p_run.add_argument("--seed", type=int, default=0,
                       help="the ONE seed every candidate shares")
    p_run.add_argument("--out-dir", required=True)
    p_run.set_defaults(func=_cmd_run)

    p_eval = sub.add_parser(
        "eval", help="freeze one candidate's batch-5 weights and measure it on the remainder")
    p_eval.add_argument("--arm-yaml", required=True,
                        help="the candidate's TRAIN arm yaml; the eval twin is derived from it")
    p_eval.add_argument("--weights", required=True,
                        help="SERVER-side path of the frozen batch-5 checkpoint")
    p_eval.add_argument("--weights-version", required=True)
    p_eval.add_argument("--split", required=True, help="the pilot's remainder split yaml")
    p_eval.add_argument("--suite", default="libero_10")
    p_eval.add_argument("--run-id", required=True)
    p_eval.add_argument("--servers", required=True)
    p_eval.add_argument("--workers", type=int, default=8)
    p_eval.add_argument("--gpus", type=int, default=1)
    p_eval.add_argument("--conda-env", default="")
    p_eval.add_argument("--init-states-dir", required=True,
                        help="B-train diff pool; empty would silently measure on the A pool")
    p_eval.add_argument("--out-dir", required=True, help="<candidate_dir>/eval")
    p_eval.add_argument("--episodes", type=int, default=EVAL_EPISODES)
    p_eval.add_argument("--seed", type=int, default=0)
    p_eval.add_argument("--bind-host", default="127.0.0.1")
    p_eval.add_argument("--episode-timeout-s", type=float, default=1800.0)
    p_eval.set_defaults(func=_cmd_eval)

    p_select = sub.add_parser("select", help="pick lambda_1 / lambda_2 from realized rates")
    p_select.add_argument("--measurements", required=True)
    p_select.set_defaults(func=_cmd_select)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
