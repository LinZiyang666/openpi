"""Is an arm's success-rate gap real, or is it the 500-episode noise floor?

Every arm runs the same 500 ``(task_id, orig_init_state_idx)`` pairs, and the
same pair means the same seed and therefore the same initial state. The
comparison is paired, so the test can condition on the episode rather than
treating two arms as independent samples -- which matters because at 500
episodes an unpaired comparison cannot resolve anything under about 5 points.

The test is an exact-in-expectation sign-flip permutation: under the null the
two arms are exchangeable *on each episode*, so flipping the sign of each
episode's difference independently generates the null distribution. Only
episodes where the arms disagree contribute (the rest have zero difference),
which is McNemar's setting; the permutation form is used because it needs no
normal approximation at these counts.

    python paired_test.py <results_dir> --baseline <arm> [--arms a,b,c] [--iters N]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random


def load_arm(path: pathlib.Path) -> dict[tuple[int, int], bool]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return {(r["task_id"], r["orig_init_state_idx"]): bool(r["success"]) for r in rows}


def paired_sign_flip(
    base: dict[tuple[int, int], bool],
    other: dict[tuple[int, int], bool],
    *,
    iters: int,
    seed: int = 12345,
) -> dict:
    """Two-sided p for ``other - base``, over the episodes both arms ran."""
    keys = sorted(set(base) & set(other))
    if not keys:
        raise ValueError("the two arms share no episodes")
    diffs = [int(other[k]) - int(base[k]) for k in keys]
    observed = sum(diffs) / len(diffs)
    # Only disagreements carry signal; ties contribute 0 under every flip.
    nonzero = [d for d in diffs if d]
    rng = random.Random(seed)
    hits = 0
    for _ in range(iters):
        flipped = sum(d if rng.random() < 0.5 else -d for d in nonzero)
        if abs(flipped / len(diffs)) >= abs(observed) - 1e-12:
            hits += 1
    return {
        "n_paired": len(keys),
        "n_disagree": len(nonzero),
        "delta_pp": 100 * observed,
        # +1 in both terms: the observed assignment is itself one of the
        # permutations, so a p of exactly 0 is not attainable and not claimed.
        "p": (hits + 1) / (iters + 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir")
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--arms", default="", help="comma-separated; default: every other arm")
    ap.add_argument("--iters", type=int, default=2000)
    args = ap.parse_args()

    root = pathlib.Path(args.results_dir)
    base = load_arm(root / f"{args.baseline}.json")
    stems = (
        args.arms.split(",")
        if args.arms
        else sorted(p.stem for p in root.glob("*.json") if p.stem != args.baseline)
    )
    print(f"baseline {args.baseline}: {sum(base.values())}/{len(base)}")
    print(f"{'arm':16s} {'n':>5s} {'disagree':>9s} {'delta_pp':>9s} {'p':>7s}")
    for stem in stems:
        path = root / f"{stem}.json"
        if not path.is_file():
            print(f"{stem:16s} MISSING")
            continue
        out = paired_sign_flip(base, load_arm(path), iters=args.iters)
        print(f"{stem:16s} {out['n_paired']:5d} {out['n_disagree']:9d} "
              f"{out['delta_pp']:+9.1f} {out['p']:7.3f}")


if __name__ == "__main__":
    main()
