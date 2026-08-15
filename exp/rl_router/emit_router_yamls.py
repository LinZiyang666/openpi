"""Emit the router arm yamls, with the pre-registration guards baked in (§3.3/§3.9).

Every arm reuses the *calibrated production retrieval config* (the ablation
study's suite baseline) and swaps only the verdict layer for ``mlp_router``.
That is what makes the comparison fair: the router and TIER see the same key
builder, the same artifact, and the same search — they differ in how the
verdict is decided, and nothing else.

Emitted arms:

======== ================== ======================================
variant  arms               routing
======== ================== ======================================
R_ts     teacher/student    hit_to = student sidecar
R_tc     teacher/cache      none (classic cache semantics)
R_tsc    all three          hit_to = student sidecar
======== ================== ======================================

Two yamls per arm:

  - **train**: ``mode: sample`` + a ``dump_dir`` (the RL rollouts);
  - **eval**:  ``mode: argmax`` + no dump. The frozen-evaluation guard is
    mechanical — ``--check`` re-reads every emitted eval yaml and fails if it
    could sample or could write a dump, so a copy-paste slip cannot turn the
    one-shot A-pool evaluation into another training run.

The module also holds the §3.9 pre-registration guards (init-pool exclusion,
seed roles, the Holm family). They belong beside the emitter because they guard
the same boundary from the other side: the emitter fixes what a run may *do*,
these fix what may be *claimed* from it — and both have to be settled before any
A-pool episode runs.

Usage::

    uv run exp/rl_router/emit_router_yamls.py --suite libero_10 \\
        --weights exp/rl_router/data/l10/v0.pt --student 127.0.0.1:7002 \\
        --dump-dir /data/rl_router/dump --out-dir exp/rl_router/config/libero_10
    uv run exp/rl_router/emit_router_yamls.py --check exp/rl_router/config/libero_10
"""

from __future__ import annotations

import argparse
import copy
import pathlib
import sys

import yaml

from openpi.cache.config import load_cache_config
from openpi.cache.components.mlp_router_judge import ARM_SETS

# The calibrated retrieval config each arm inherits.
BASELINE = "exp/ablation_study/config/common/{suite}_baseline.yaml"

# Frozen feature set: PROMPT_EMB is excluded because production retrieval
# abandoned it, so including it would give the router an input TIER does not use.
FEATURE_FIELDS = ["vision_0", "vision_1", "robot_state"]

VARIANTS = {"R_ts": "ts", "R_tc": "tc", "R_tsc": "tsc"}


def build_arm_yaml(
    baseline: dict,
    *,
    arms: str,
    weights_path: str,
    student_endpoint: str | None,
    mode: str,
    dump_dir: str,
    temperature: float,
    seed: int,
    hidden: int,
) -> dict:
    """Return a router arm config derived from the suite baseline."""
    cfg = copy.deepcopy(baseline)
    judge = {
        "type": "mlp_router",
        "arms": arms,
        "weights_path": weights_path,
        "feature_fields": list(FEATURE_FIELDS),
        "hidden": hidden,
        "mode": mode,
    }
    if mode == "sample":
        # Both are required by the validator under sampling: without them a run
        # cannot be replayed, and an unreplayable RL run cannot be audited.
        judge["temperature"] = temperature
        judge["seed"] = seed
    if dump_dir:
        judge["dump_dir"] = dump_dir
    cfg["checkpoints"]["cp1"]["judge"] = judge
    # CP1-only: the arms map onto executor slots that exist only there.
    cfg["checkpoints"] = {"cp1": cfg["checkpoints"]["cp1"]}
    cfg["write_policy"] = {"type": "never"}
    if "student" in ARM_SETS[arms]:
        if not student_endpoint:
            raise SystemExit(f"arms={arms} needs --student host:port for the sidecar")
        cfg["routing"] = {"hit_to": student_endpoint}
    else:
        cfg.pop("routing", None)
    return cfg


def emit(
    out_dir: pathlib.Path,
    *,
    suite: str,
    weights_path: str,
    student_endpoint: str | None,
    dump_dir: str,
    temperature: float,
    seed: int,
    hidden: int,
    variants: list[str],
) -> list[pathlib.Path]:
    baseline = yaml.safe_load(
        pathlib.Path(BASELINE.format(suite=suite)).read_text(encoding="utf-8")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[pathlib.Path] = []
    for variant in variants:
        arms = VARIANTS[variant]
        for mode, dump in (("sample", dump_dir), ("argmax", "")):
            kind = "train" if mode == "sample" else "eval"
            cfg = build_arm_yaml(
                baseline, arms=arms, weights_path=weights_path,
                student_endpoint=student_endpoint, mode=mode, dump_dir=dump,
                temperature=temperature, seed=seed, hidden=hidden,
            )
            path = out_dir / f"{variant.lower()}_{kind}.yaml"
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            # Validate through the real loader so the routing allowlist and the
            # mlp_router rules run here, not on the serving host mid-run.
            load_cache_config(str(path))
            written.append(path)
    return written


def check_eval_guards(directory: pathlib.Path) -> list[str]:
    """Machine guard for the one-shot A-pool evaluation (§3.9).

    Returns the list of violations. An eval yaml that can sample, or that
    carries a dump directory, would silently make the frozen-checkpoint
    measurement into another training run — the exact selection-bias failure
    the pre-registration is designed to prevent.
    """
    violations: list[str] = []
    for path in sorted(directory.glob("*_eval.yaml")):
        cfg = load_cache_config(str(path))
        judge = cfg.checkpoints["cp1"].judge
        if judge.mode != "argmax":
            violations.append(f"{path}: eval yaml has mode={judge.mode!r}, expected 'argmax'")
        if judge.dump_dir:
            violations.append(f"{path}: eval yaml carries dump_dir={judge.dump_dir!r}")
        if judge.seed is not None:
            violations.append(f"{path}: eval yaml carries a sampling seed")
    return violations


def check_init_pools(train_init_dir: str, eval_init_dir: str) -> list[str]:
    """A-pool / B-pool mutual exclusion, asserted in both directions.

    Training may only ever touch B-train; the official pruned_init A pool is
    measured once, at the end. One shared directory would leak the evaluation
    set into training and quietly invalidate every reported number.
    """
    train, eval_ = pathlib.Path(train_init_dir).resolve(), pathlib.Path(eval_init_dir).resolve()
    violations: list[str] = []
    if train == eval_:
        violations.append(f"training and evaluation init pools are the same path: {train}")
    if train in eval_.parents or eval_ in train.parents:
        violations.append(f"init pools are nested: train={train} eval={eval_}")
    return violations


# ---------------------------------------------------------------------------
# Pre-registration guards (§3.9)
#
# These live next to the emitter because they guard the same thing from the
# other side: the emitter fixes what a run may do, and these fix what may be
# claimed from its results. Both have to be decided before any A-pool episode
# runs, or the choice is no longer independent of the numbers.
# ---------------------------------------------------------------------------

PRIMARY_SEED = 0


def seed_roles(runs: list[dict]) -> dict:
    """Split runs into the primary family and the robustness set.

    Only seed 0 enters the multiplicity-corrected family; a second training
    seed is reported for direction agreement and its own paired test, outside
    the family. Promoting whichever seed looked better would be selection on
    the outcome, and correcting for both would spend power on a question the
    experiment is not asking.
    """
    primary = [r for r in runs if int(r["seed"]) == PRIMARY_SEED]
    robustness = [r for r in runs if int(r["seed"]) != PRIMARY_SEED]
    return {"primary": primary, "robustness": robustness}


def reject_seed_pooling(episodes: list[dict]) -> None:
    """Fail if one comparison's episodes come from more than one training seed.

    Pooling two seeds' rollouts would treat two policies as one and inflate the
    effective sample size of every paired test computed on them.
    """
    seeds = {int(e["seed"]) for e in episodes if "seed" in e}
    if len(seeds) > 1:
        raise ValueError(
            f"episodes from multiple training seeds {sorted(seeds)} in one comparison; "
            "seed-0 is primary and other seeds are reported separately (§3.9)"
        )


def holm_family(runs: list[dict]) -> list[dict]:
    """The pre-registered primary hypotheses, one statistic each.

    Per suite: the flagship variant's endpoint against TIER's two-tier arm, and
    R_tsc's endpoint against TIER's three-tier arm. Everything else in the study
    is descriptive — declaring that here, before the numbers exist, is what
    keeps the correction honest.
    """
    family = []
    for run in seed_roles(runs)["primary"]:
        if run.get("flagship"):
            family.append({
                "suite": run["suite"], "variant": run["variant"], "seed": run["seed"],
                "comparator": "tier_two_tier", "statistic": "paired_mcnemar",
            })
        if run["variant"] == "R_tsc":
            family.append({
                "suite": run["suite"], "variant": run["variant"], "seed": run["seed"],
                "comparator": "tier_three_tier", "statistic": "paired_mcnemar",
            })
    return family


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", default="", help="validate an emitted directory and exit")
    parser.add_argument("--suite", default="libero_10")
    parser.add_argument("--weights", default="")
    parser.add_argument("--student", default="", help="student sidecar 'host:port'")
    parser.add_argument("--dump-dir", default="")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--train-init-dir", default="")
    parser.add_argument("--eval-init-dir", default="")
    args = parser.parse_args()

    if args.check:
        problems = check_eval_guards(pathlib.Path(args.check))
        if args.train_init_dir and args.eval_init_dir:
            problems += check_init_pools(args.train_init_dir, args.eval_init_dir)
        if problems:
            print("\n".join(problems), file=sys.stderr)
            raise SystemExit(1)
        print(f"{args.check}: eval guards OK")
        return

    if not args.weights or not args.out_dir:
        raise SystemExit("--weights and --out-dir are required when emitting")
    variants = [v for v in args.variants.split(",") if v]
    unknown = set(variants) - set(VARIANTS)
    if unknown:
        raise SystemExit(f"unknown variants {sorted(unknown)}; valid: {sorted(VARIANTS)}")
    written = emit(
        pathlib.Path(args.out_dir), suite=args.suite, weights_path=args.weights,
        student_endpoint=args.student or None, dump_dir=args.dump_dir,
        temperature=args.temperature, seed=args.seed, hidden=args.hidden, variants=variants,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
