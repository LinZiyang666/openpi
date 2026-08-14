"""E1 dose-response driver: key quality vs history gain across key builders.

Plan section 3.2. The same rollout batch was encoded by several key builders,
which gives a key-quality spectrum on identical samples and identical targets.
The qualitative prediction is that the history/difference gain shrinks as the
key gets more sufficient.

Three things this driver does on purpose:

  * **Only the A and C groups are run.** Group B needs that builder's own
    production ``trajectory_weights``, and only the primary builder has depth
    variants in the config tree; fabricating them would score configurations
    that were never run. The plan allows the trend to be read off "B or C", so
    C carries it here and the restriction is recorded in the manifest.
  * **The modality weights are held fixed** across builders (all specs come
    from the same grid point), so the x-axis moves with the key encoding and
    nothing else.
  * **The fold split is task-stratified and reported both ways.** x and y come
    from disjoint episode halves so they cannot share an ``r_A`` term, and the
    folds are then exchanged and the pair reported together, because with
    n = 6 builders a single split is easy to over-read.

The result is exploratory: n is the number of builders, so it is a trend
corroboration, never a confirmatory claim.

Public interface: :func:`stratified_folds`, :func:`collect_rows`,
:func:`analyse`, :func:`main`.

Key dependencies: :mod:`e1_loeo_residual`, :mod:`_scoring`.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib
from typing import Any, Mapping, Optional, Sequence

import numpy as np

from exp.markov_sufficiency import _scoring
from exp.markov_sufficiency import e1_loeo_residual as e1

#: The C group carries the trend (see the module docstring for why not B).
DOSE_GROUP = "C-g1.0"


def stratified_folds(rows: Sequence[Mapping[str, Any]], seed: int = 0) -> dict[str, int]:
    """Split episodes into two folds at random, stratified by task.

    Stratifying matters because tasks differ a lot in residual scale; an
    unstratified split can put most of one task in one fold and turn a task
    effect into an apparent key-quality effect.
    """
    by_task: dict[str, list[str]] = collections.defaultdict(list)
    seen: dict[str, str] = {}
    for r in rows:
        traj = r["trajectory_id"]
        if traj not in seen:
            seen[traj] = r["task_key"]
            by_task[r["task_key"]].append(traj)

    rng = np.random.default_rng(seed)
    assignment: dict[str, int] = {}
    for task in sorted(by_task):
        eps = sorted(by_task[task])
        order = rng.permutation(len(eps))
        for rank, idx in enumerate(order):
            assignment[eps[idx]] = rank % 2
    return assignment


def collect_rows(
    specs: Mapping[str, str],
    suite: str,
    library_root: str = e1.DEFAULT_LIBRARY_ROOT,
    max_episodes: Optional[int] = None,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Run the A/C LOEO for every key builder of one suite."""
    by_builder: dict[str, list[dict[str, Any]]] = {}
    manifests: dict[str, Any] = {}
    for builder, yaml_path in sorted(specs.items()):
        scorer = _scoring.build_scorer(yaml_path)
        result = e1.run_suite(
            suite,
            builder,
            {1: scorer},
            library_root=library_root,
            gammas=(1.0,),
            ks=(e1.PRIMARY_K,),
            # No oracle pass: E1-O is a separate registered read-out on the
            # primary builder, and running it here would triple the cost.
            oracle_eps=(),
            max_episodes=max_episodes,
        )
        by_builder[builder] = result["rows"]
        manifests[builder] = {**result["manifest"], "yaml": yaml_path}
    return by_builder, manifests


def analyse(by_builder: Mapping[str, list[dict[str, Any]]], seed: int = 0) -> dict[str, Any]:
    """Both fold orientations of the cross-fitted trend, plus the raw points."""
    all_rows = [r for rows in by_builder.values() for r in rows]
    folds = stratified_folds(all_rows, seed=seed)
    forward = e1.dose_response(by_builder, DOSE_GROUP, fold_assignment=folds)
    reversed_ = e1.dose_response(by_builder, DOSE_GROUP, fold_assignment=folds, swap=True)

    points = []
    for builder, rows in sorted(by_builder.items()):
        cell = e1.aggregate(rows, DOSE_GROUP, e1.PRIMARY_K)
        points.append(
            {
                "key_builder": builder,
                "median_residual_A": cell["median_residual_A"],
                "relative_delta": cell["relative_delta"],
                "n_episodes": cell["n_episodes"],
            }
        )
    consistent = (
        forward["n"] >= 3
        and reversed_["n"] >= 3
        and np.sign(forward["spearman"]) == np.sign(reversed_["spearman"])
    )
    return {
        "group": DOSE_GROUP,
        "k": e1.PRIMARY_K,
        "fold_seed": seed,
        "forward": forward,
        "folds_reversed": reversed_,
        "sign_consistent": bool(consistent),
        "points": points,
        "role": "exploratory -- n is the number of key builders; a trend corroboration, not a verdict",
    }


def _parse_spec(raw: str) -> tuple[str, str, str]:
    suite, _, rest = raw.partition(":")
    builder, _, yaml_path = rest.partition("=")
    if not (suite and builder and yaml_path):
        raise ValueError(f"--spec must be SUITE:BUILDER=YAML, got {raw!r}")
    return suite, builder, yaml_path


def main(argv: Optional[list[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="E1 dose-response across key builders")
    ap.add_argument("--spec", action="append", required=True, metavar="SUITE:BUILDER=YAML")
    ap.add_argument("--library-root", default=e1.DEFAULT_LIBRARY_ROOT)
    ap.add_argument("--max-episodes", type=int, default=None, help="smoke runs only")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    by_suite: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for raw in args.spec:
        try:
            suite, builder, yaml_path = _parse_spec(raw)
        except ValueError as exc:
            ap.error(str(exc))
        by_suite[suite][builder] = yaml_path

    result: dict[str, Any] = {"suites": {}}
    for suite, specs in sorted(by_suite.items()):
        if len(specs) < 3:
            ap.error(f"suite {suite} has {len(specs)} key builders; the trend needs at least 3")
        rows, manifests = collect_rows(
            specs, suite, library_root=args.library_root, max_episodes=args.max_episodes
        )
        result["suites"][suite] = {"analysis": analyse(rows, seed=args.seed), "manifest": manifests}

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({s: v["analysis"] for s, v in result["suites"].items()}, indent=2))


if __name__ == "__main__":
    main()
