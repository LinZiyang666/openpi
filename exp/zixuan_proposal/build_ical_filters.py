"""Tracked deterministic generator for the I_cal / I_val episode-filter manifests.

TRACER Phase 6, §3-B2. The `exp/zixuan_proposal/data/filters/I_cal_even.json` and
`I_val_odd.json` manifests define the held-out even/odd `orig_init_state_idx` split
that every downstream consumer (projection training rows, model-selection folds,
rebuilt D+/D- pools, normalizer/calibration inputs) must agree on. Those files exist
on disk but are **gitignored** (`.gitignore:6 exp/**/data/**`), so another host is not
guaranteed to have them. This module regenerates them deterministically from the
authoritative per-suite init maps + the held-out pool convention, so the split is
reproducible anywhere.

Split rule (shared by both suites; both are 10 tasks x 50 held-out inits):
`orig_init_state_idx` even -> I_cal, odd -> I_val. Each row is
`{task_id, orig_init_state_idx, subset_init_state_idx}` with `subset == orig`
(the manifest indexes the held-out 0..49 pool directly), sorted by (task, init).

CLI:
  python -m exp.zixuan_proposal.build_ical_filters --check   # verify existing files
  python -m exp.zixuan_proposal.build_ical_filters --write   # (re)write the manifests
"""

from __future__ import annotations

import argparse
import json
import pathlib

# ------------------------------------------------------------------
# Constants (held-out pool convention; asserted against the init maps)
# ------------------------------------------------------------------
NUM_TASKS = 10
POOL_SIZE = 50  # held-out init states per task in db_init/libero/<suite>

_REPO = pathlib.Path(__file__).resolve().parents[2]
_FILTER_DIR = _REPO / "exp/zixuan_proposal/data/filters"
_INIT_MAPS = {
    "libero_spatial": _REPO / "exp/common/data/db/libero_cache/libero_spatial_init_map.json",
    "libero_10": _REPO / "exp/common/data/db/libero_cache/libero_10_init_map.json",
}


# ------------------------------------------------------------------
# Generation
# ------------------------------------------------------------------
def generate_filters(num_tasks: int = NUM_TASKS, pool_size: int = POOL_SIZE):
    """Return (ical_even_rows, ival_odd_rows) deterministically.

    Rows are sorted by (task_id, orig_init_state_idx); `subset == orig`.
    """
    ical: list[dict] = []
    ival: list[dict] = []
    for task_id in range(num_tasks):
        for init in range(pool_size):
            row = {"task_id": task_id, "orig_init_state_idx": init, "subset_init_state_idx": init}
            (ical if init % 2 == 0 else ival).append(row)
    return ical, ival


def _assert_suite_task_count() -> None:
    """Fail loud if a suite's init map does not have exactly NUM_TASKS tasks."""
    for suite, path in _INIT_MAPS.items():
        if not path.exists():
            raise SystemExit(f"init map missing for {suite}: {path}")
        rows = json.loads(path.read_text())
        tasks = {r["task_id"] for r in rows}
        if tasks != set(range(NUM_TASKS)):
            raise SystemExit(
                f"{suite} init map has tasks {sorted(tasks)}, expected 0..{NUM_TASKS - 1}"
            )


def _validate(rows: list[dict], parity: int) -> None:
    """Fail loud on the 250-row / schema / parity contract."""
    expected_n = NUM_TASKS * (POOL_SIZE // 2)
    if len(rows) != expected_n:
        raise SystemExit(f"expected {expected_n} rows, got {len(rows)}")
    for r in rows:
        if set(r) != {"task_id", "orig_init_state_idx", "subset_init_state_idx"}:
            raise SystemExit(f"unexpected schema: {sorted(r)}")
        if r["orig_init_state_idx"] % 2 != parity:
            raise SystemExit(f"parity violation: {r}")
        if r["subset_init_state_idx"] != r["orig_init_state_idx"]:
            raise SystemExit(f"subset != orig: {r}")


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Generate/verify the I_cal/I_val filter manifests")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify existing files match the generator")
    mode.add_argument("--write", action="store_true", help="(re)write the manifests to disk")
    args = ap.parse_args()

    _assert_suite_task_count()
    ical, ival = generate_filters()
    _validate(ical, parity=0)
    _validate(ival, parity=1)

    targets = {"I_cal_even.json": ical, "I_val_odd.json": ival}
    if args.write:
        _FILTER_DIR.mkdir(parents=True, exist_ok=True)
        for name, rows in targets.items():
            (_FILTER_DIR / name).write_text(json.dumps(rows, indent=2))
            print(f"wrote {name}: {len(rows)} rows")
    else:  # --check
        for name, rows in targets.items():
            path = _FILTER_DIR / name
            if not path.exists():
                raise SystemExit(f"MISSING {name} (run --write): {path}")
            on_disk = json.loads(path.read_text())
            if on_disk != rows:
                raise SystemExit(f"MISMATCH {name}: on-disk file differs from the generator")
            print(f"OK {name}: {len(rows)} rows match")


if __name__ == "__main__":
    main()
