"""Emit a calibrated dual-retrieval YAML from a base YAML + params JSON (Phase 5).

Reads the base active YAML (illustrative, uncalibrated params) and the
``solve_calibration.py`` params JSON, writes the calibrated ``gate_betas`` /
``threshold`` / ``warm_tiers`` into the judge block and ``margin_lambda`` into
the search_strategy block, and (when b2 != 0) attaches the ``u_t_factor``. It
writes a NEW ``dual_retrieval_calibrated{,_l10}.yaml`` and never overwrites the
active YAML (which stays as the pre-calibration reference).

Usage:
  PYTHONPATH=. uv run python exp/zixuan_proposal/emit_calibrated_yaml.py \
      --base-yaml exp/zixuan_proposal/config/dual_retrieval_active.yaml \
      --params-json exp/zixuan_proposal/data/phase5_params_libero_spatial.json \
      --u-t-descriptor direction --u-t-channel state --u-t-past 2 --u-t-future 1 \
      --out-yaml exp/zixuan_proposal/config/dual_retrieval_calibrated.yaml
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def apply_params(base_cfg: dict, params: dict, u_t_factor: dict | None) -> dict:
    """Return a deep copy of base_cfg with the calibrated params written in.

    ``u_t_factor`` is attached to the judge only when the calibrated
    ``gate_betas["b2"] != 0`` (the validator ties the two together); a zero b2
    leaves the gate as a margin-only failure gate.
    """
    cfg = copy.deepcopy(base_cfg)
    cp1 = cfg["checkpoints"]["cp1"]
    judge = cp1["judge"]
    judge["gate_betas"] = dict(params["gate_betas"])
    judge["threshold"] = params["threshold"]
    judge["warm_tiers"] = params["warm_tiers"]
    if float(params["gate_betas"].get("b2", 0.0)) != 0.0:
        if u_t_factor is None:
            raise SystemExit(
                "calibrated b2 != 0 requires --u-t-descriptor/channel/past/future "
                "(the kinematic factor the gate weights)"
            )
        judge["u_t_factor"] = dict(u_t_factor)
    else:
        judge.pop("u_t_factor", None)
    cp1["search_strategy"]["margin_lambda"] = params["margin_lambda"]
    return cfg


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="Write calibrated params into a new YAML")
    ap.add_argument("--base-yaml", required=True, help="active illustrative YAML (not overwritten)")
    ap.add_argument("--params-json", required=True, help="solve_calibration.py output")
    ap.add_argument("--out-yaml", required=True, help="NEW calibrated YAML path")
    ap.add_argument("--u-t-descriptor", choices=["jerk", "direction", "dispersion", "path_length"])
    ap.add_argument("--u-t-channel", choices=["action", "state"])
    ap.add_argument("--u-t-past", type=int)
    ap.add_argument("--u-t-future", type=int)
    args = ap.parse_args()

    base_cfg = yaml.safe_load(Path(args.base_yaml).read_text(encoding="utf-8"))
    params = json.loads(Path(args.params_json).read_text(encoding="utf-8"))["params"]

    u_t_factor = None
    if args.u_t_descriptor is not None:
        if None in (args.u_t_channel, args.u_t_past, args.u_t_future):
            raise SystemExit("--u-t-descriptor requires --u-t-channel/--u-t-past/--u-t-future")
        u_t_factor = {
            "descriptor": args.u_t_descriptor, "channel": args.u_t_channel,
            "past": args.u_t_past, "future": args.u_t_future,
        }

    out_cfg = apply_params(base_cfg, params, u_t_factor)
    out = Path(args.out_yaml)
    if out.resolve() == Path(args.base_yaml).resolve():
        raise SystemExit("--out-yaml must differ from --base-yaml (do not overwrite active)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(out_cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info("wrote calibrated YAML -> %s (margin_lambda=%s, gate_betas=%s)",
                out, params["margin_lambda"], params["gate_betas"])


if __name__ == "__main__":
    main()
