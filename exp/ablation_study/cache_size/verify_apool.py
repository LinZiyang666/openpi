"""Freeze and verify the A-pool (official pruned_init) used for evaluation.

The plan calls for materializing the A-pool rather than relying on the episode
runner's implicit fallback, for three reasons: the fallback's content depends on
the installed LIBERO version and is therefore not hashable, the anti-pollution
guard should not be loosened for evaluation, and the paired analysis needs a
verifiable join basis.

The record this writes is what **binds** a run to a specific pool: it carries
the directory alongside the digests, and ``run_size_eval`` feeds that directory
to every ``WorkerSpec.init_states_dir``. So the digest attests the inits the
workers actually loaded, not merely a copy that existed somewhere at
verification time. A record produced in one environment can no longer ride
along with another environment's default pool.

Two assertions carry the anti-pollution argument:

*   ``A ∩ B = ∅`` -- the evaluation pool shares no init with the difference pool
    the libraries are built from. This is the claim the whole leakage story
    rests on, so it must be executed and recorded, not assumed.
*   digest stability -- the same 500 inits, run after run.

Requires the LIBERO client environment (the main uv venv has no ``libero``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np
import torch


def digest_init_file(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rollup_digest(per_task_digests: dict[str, str]) -> str:
    """One digest over the whole pool, order-independent by construction."""
    return hashlib.sha256(
        "".join(f"{k}:{per_task_digests[k]}" for k in sorted(per_task_digests)).encode()
    ).hexdigest()


def states_to_rows(states) -> set[bytes]:
    """Canonical byte rows for set arithmetic across differently-ordered pools."""
    arr = np.asarray(states)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return {np.ascontiguousarray(row, dtype=np.float64).tobytes() for row in arr}


def assert_disjoint(a_dir: pathlib.Path, b_dir: pathlib.Path) -> dict:
    """A-pool vs difference-pool: zero shared init rows, per task."""
    report: dict[str, dict] = {}
    for a_file in sorted(a_dir.glob("*.init")):
        b_file = b_dir / a_file.name
        if not b_file.exists():
            raise FileNotFoundError(f"difference pool has no counterpart for {a_file.name}")
        a_rows = states_to_rows(torch.load(a_file, weights_only=False))
        b_rows = states_to_rows(torch.load(b_file, weights_only=False))
        shared = a_rows & b_rows
        report[a_file.stem] = {
            "a_count": len(a_rows),
            "b_count": len(b_rows),
            "shared": len(shared),
        }
        if shared:
            raise SystemExit(
                f"{a_file.name}: {len(shared)} init states appear in BOTH the evaluation "
                "pool and the library/difference pool -- evaluation would be contaminated"
            )
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apool-dir", required=True,
                    help="materialized A-pool, e.g. db_init/libero/<suite>_apool")
    ap.add_argument("--diff-pool-dir", required=True,
                    help="difference pool the libraries are built from")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", required=True, help="digest + disjointness record (yaml)")
    ap.add_argument("--expect-per-task", type=int, default=50)
    args = ap.parse_args()

    a_dir = pathlib.Path(args.apool_dir)
    files = sorted(a_dir.glob("*.init"))
    if len(files) != 10:
        raise SystemExit(f"expected 10 task init files in {a_dir}, found {len(files)}")

    digests = {f.stem: digest_init_file(f) for f in files}
    disjoint = assert_disjoint(a_dir, pathlib.Path(args.diff_pool_dir))

    total = 0
    for stem, info in disjoint.items():
        if info["a_count"] != args.expect_per_task:
            raise SystemExit(
                f"{stem}: A-pool has {info['a_count']} inits, expected {args.expect_per_task}"
            )
        total += info["a_count"]

    rollup = rollup_digest(digests)

    record = {
        "suite": args.suite,
        # The directory is part of the record on purpose: the runner feeds it to
        # every worker, so the digest attests the pool that was actually loaded
        # rather than one that merely existed somewhere at verification time.
        "apool_dir": str(a_dir.resolve()),
        "total_inits": total,
        "per_task_digests": digests,
        "rollup_sha256": rollup,
        "disjoint_from_difference_pool": disjoint,
    }
    import yaml

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(record, sort_keys=False))
    print(json.dumps({"total_inits": total, "rollup_sha256": rollup}, indent=2))


if __name__ == "__main__":
    main()
