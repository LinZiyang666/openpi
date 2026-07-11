"""Offline solver: per-step signal table -> calibrated params JSON (TRACER Phase 5).

Loads the threshold-independent signal table produced by
``build_calibration_table.py`` and runs ``MarginGateCalibrator.calibrate`` to
minimize the Eq 24 objective ``L_cal = BadHitRate + c_miss*MissRate +
c_warm*WarmCost``. Execution is offline; the emitted params later go into a
calibrated YAML via ``emit_calibrated_yaml.py``.

Usage:
  PYTHONPATH=. uv run python exp/zixuan_proposal/solve_calibration.py \
      --rows-jsonl exp/zixuan_proposal/data/phase5_calib_rows_libero_spatial.jsonl \
      --suite libero_spatial --c-miss 1.0 --c-warm 0.75 \
      --out-json exp/zixuan_proposal/data/phase5_params_libero_spatial.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from openpi.cache.components.margin_gate_calibrator import MarginGateCalibrator  # noqa: E402


def load_rows(path: str) -> list[dict]:
    """Read the per-step signal table JSONL into a list of row dicts."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    if not rows:
        raise SystemExit(f"no signal rows in {path}")
    return rows


def solve(rows: list[dict], *, suite: str, c_miss: float, c_warm: float) -> dict:
    """Return the calibration doc {suite, n_rows, c_*, params, metrics}."""
    params = MarginGateCalibrator.calibrate(rows, c_miss=c_miss, c_warm=c_warm)
    metrics = MarginGateCalibrator.evaluate(rows, params, c_miss=c_miss, c_warm=c_warm)
    return {
        "suite": suite, "n_rows": len(rows), "c_miss": c_miss, "c_warm": c_warm,
        "params": params, "metrics": metrics,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Solve failure-gate calibration (Eq 24)")
    ap.add_argument("--rows-jsonl", required=True, help="build_calibration_table output")
    ap.add_argument("--suite", required=True)
    ap.add_argument("--c-miss", type=float, default=1.0)
    ap.add_argument("--c-warm", type=float, default=0.75, help="repo WARM_COST precedent")
    ap.add_argument("--out-json", required=True)
    args = ap.parse_args()

    doc = solve(load_rows(args.rows_jsonl), suite=args.suite,
                c_miss=args.c_miss, c_warm=args.c_warm)
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    mt = doc["metrics"]
    logger.info(
        "suite=%s n_rows=%d -> L_cal=%.4f (BHR=%.4f Miss=%.4f Warm=%.4f); params=%s",
        args.suite, doc["n_rows"], mt["L_cal"], mt["BadHitRate"], mt["MissRate"],
        mt["WarmCost"], doc["params"],
    )
    logger.info("wrote params -> %s", out)


if __name__ == "__main__":
    main()
