"""Emit the RoboCasa365 warm-start arm over the schedule-stamped pinned libraries.

This is a NEW emitter with its own output root and its own index digest on
purpose (plan D8): ``emit_ws_search_yamls.py`` / ``emit_ws_search2_yamls.py``
are frozen by the ``source_sha256`` their runs recorded, and changing a byte of
either would make every in-flight ``run_ws_search2.py`` preflight refuse to
dispatch. The helpers that are safe to share -- the round-1 weight matrix,
the round-2 cell builder and validator, the pinned-library spec -- are
imported, never copied.

What one cell is
----------------
One retrieval weight configuration (a round-1 ``weight_matrix()`` id) served
with ``judge.type: always_warm_start`` at one resume timestep ``t``, over the
warm library of one teacher. The sweep over ``t`` is the same success-rate ~
start_t curve the Pi0.5/LIBERO line measured; the FULL_HIT and teacher arms
this is compared against come from the existing emitters.

The timestep set is the schedule's, and the schedule is an input: for GR00T
the step count is a runtime property of the served policy (baked into the
RoboCasa checkpoint, CLI-overridden on LIBERO), so ``--groot-steps`` is
required and every GR00T cell carries ``denoise_schedule: groot_n15_k<N>_v1``.
The serving guard refuses the yaml unless the live head runs exactly that
loop, and the storage binding refuses a library stamped with any other
schedule -- omission and mismatch both fail loudly.

Public interface: ``warm_teacher_spec``, ``build_warm_cell``,
``verify_warm_cell``, ``emit_warm_arms``, ``verify_index_digest``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from openpi.cache.types import PI05_V1, DenoiseSchedule, groot_n15_schedule

from exp.robocasa365 import emit_ws_search2_yamls as ws2
from exp.robocasa365 import emit_ws_search_yamls
from exp.robocasa365.emit_ws_search_yamls import weight_matrix
from exp.robocasa365.pinned_objects import canonical_json, load_pin_manifest

DEFAULT_OUT_ROOT = "exp/robocasa365/config/ws_warmstart_pnp"
DEFAULT_WARM_PRELOAD_DIR = "/data/robocasa365_cache/cache_artifacts_pnp_warm"
WARM_LIBRARY_TAG = "pnp_warm"
DIGEST_NAME = "index_digest.json"
ARM = "warm"

_TEACHER_DIGEST_DOMAIN = "robocasa365/wsw_index_teacher/v1"
_GLOBAL_DIGEST_DOMAIN = "robocasa365/wsw_index/v1"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def teacher_schedule(teacher: str, groot_steps: int | None) -> DenoiseSchedule:
    """The loop a teacher's warm library is keyed under.

    Pi0.5 is its fixed ten-step loop. GR00T's count is not a property of the
    model family, so it must be supplied -- there is deliberately no default.
    """
    if teacher == "pi05":
        return PI05_V1
    if groot_steps is None:
        raise SystemExit(
            "--groot-steps is required: the GR00T step count is a runtime property "
            "of the served policy and is not restated anywhere in this repository."
        )
    return groot_n15_schedule(groot_steps)


def warm_teacher_spec(
    teacher: str,
    pin_id: str,
    *,
    preload_dir: str = DEFAULT_WARM_PRELOAD_DIR,
    library_tag: str = WARM_LIBRARY_TAG,
) -> dict:
    """The pinned teacher spec pointed at the schedule-stamped warm library."""
    return ws2.pinned_teacher_spec(
        teacher, pin_id, preload_dir=preload_dir, library_tag=library_tag
    )


def build_warm_cell(
    weights: dict[str, float],
    calib_entry: dict,
    teacher: str,
    *,
    spec: dict,
    schedule: DenoiseSchedule,
    start_t: float,
) -> dict:
    """The round-2 main-arm cell with the verdict replaced by a forced warm start."""
    cfg = ws2.build_cell(weights, calib_entry, teacher, text_ivf=True, spec=spec)
    cfg["checkpoints"]["cp1"]["judge"] = {
        "type": "always_warm_start",
        "start_t": start_t,
    }
    if schedule is not PI05_V1:
        cfg["denoise_schedule"] = schedule.schedule_id
    return cfg


def verify_warm_cell(
    cfg: dict, cid: str, teacher: str, *, spec: dict, schedule: DenoiseSchedule
) -> None:
    """Round-2 invariants plus the warm-start shape this arm exists to test."""
    ws2.verify_cell(cfg, cid, teacher, text_ivf=True, spec=spec)
    judge = cfg["checkpoints"]["cp1"]["judge"]
    assert judge["type"] == "always_warm_start", cid
    assert round(judge["start_t"], 4) in schedule.timestep_set, (cid, judge["start_t"])
    assert not judge.get("warm_tiers"), cid
    if schedule is PI05_V1:
        assert "denoise_schedule" not in cfg, cid
    else:
        assert cfg.get("denoise_schedule") == schedule.schedule_id, cid


def emit_warm_arms(
    out_root: Path,
    calib: dict,
    pin_id: str,
    *,
    weight_cids: list[str],
    groot_steps: int | None,
    timesteps: list[float] | None = None,
    preload_dir: str = DEFAULT_WARM_PRELOAD_DIR,
    library_tag: str = WARM_LIBRARY_TAG,
) -> dict:
    """Emit both teachers' warm arms and seal the tree into one digest."""
    out_root = Path(out_root)
    configs = weight_matrix()
    unknown = sorted(set(weight_cids) - set(configs))
    if unknown:
        raise SystemExit(f"weight cells not in the round-1 matrix: {unknown}")
    per_teacher_cells: dict[str, dict[str, str]] = {}
    for teacher in sorted(ws2.TEACHERS):
        schedule = teacher_schedule(teacher, groot_steps)
        chosen = list(schedule.timesteps) if timesteps is None else timesteps
        spec = warm_teacher_spec(
            teacher, pin_id, preload_dir=preload_dir, library_tag=library_tag
        )
        calib_entry = ws2.calibration_entry(calib, spec, teacher)
        arm_dir = out_root / teacher / ARM
        arm_dir.mkdir(parents=True, exist_ok=True)
        index = {}
        for weight_cid in sorted(weight_cids):
            for t in chosen:
                cid = f"{weight_cid}__ws{t:.4f}"
                cfg = build_warm_cell(
                    configs[weight_cid],
                    calib_entry,
                    teacher,
                    spec=spec,
                    schedule=schedule,
                    start_t=round(t, 4),
                )
                verify_warm_cell(cfg, cid, teacher, spec=spec, schedule=schedule)
                path = arm_dir / f"{cid}.yaml"
                path.write_text(yaml.safe_dump(cfg, sort_keys=False))
                ws2.validate_on_disk(path)
                index[cid] = {
                    "file": path.name,
                    "weights": configs[weight_cid],
                    "start_t": round(t, 4),
                    "schedule_id": schedule.schedule_id,
                }
        (arm_dir / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
        per_teacher_cells[teacher] = {
            cid: _sha256_text((arm_dir / f"{cid}.yaml").read_text()) for cid in index
        }
    digest = build_index_digest(per_teacher_cells)
    digest_path = out_root / DIGEST_NAME
    digest_path.write_text(json.dumps(digest, indent=1, sort_keys=True))
    verify_index_digest(out_root, digest_path, expected_pin_id=pin_id)
    return digest


def source_sha256() -> dict[str, str]:
    """Hash every emitter file the cell text derives from, this one included."""
    paths = (Path(emit_ws_search_yamls.__file__), Path(ws2.__file__), Path(__file__))
    return {p.name: _sha256_text(p.read_text()) for p in paths}


def teacher_digest(teacher: str, cells: dict[str, str]) -> str:
    """Summary hash of one teacher's ``{cid: sha256(yaml_text)}`` table."""
    payload = {"domain": _TEACHER_DIGEST_DOMAIN, "teacher": teacher, "cells": cells}
    return _sha256_text(canonical_json(payload))


def build_index_digest(
    per_teacher_cells: dict[str, dict[str, str]], sources: dict[str, str] | None = None
) -> dict:
    """The frozen summary of an emitted warm tree (same schema as the ws2 digest)."""
    per_teacher = {
        teacher: {"cells": cells, "digest": teacher_digest(teacher, cells)}
        for teacher, cells in sorted(per_teacher_cells.items())
    }
    sources = source_sha256() if sources is None else sources
    payload = {
        "domain": _GLOBAL_DIGEST_DOMAIN,
        "per_teacher": per_teacher,
        "source_sha256": sources,
    }
    return {
        "per_teacher": per_teacher,
        "source_sha256": sources,
        "global_digest": _sha256_text(canonical_json(payload)),
    }


def verify_index_digest(
    config_root: Path, digest_path: Path, *, expected_pin_id: str | None = None
) -> dict:
    """Prove an emitted warm tree is still the tree its digest froze.

    Raises:
        ValueError: on the first inconsistency, naming it.
    """
    config_root = Path(config_root)
    digest_path = Path(digest_path)
    try:
        doc = json.loads(digest_path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"index digest {digest_path} does not exist") from exc
    absent = sorted({"per_teacher", "source_sha256", "global_digest"} - set(doc))
    if absent:
        raise ValueError(f"index digest {digest_path} is missing {absent}")
    live_sources = source_sha256()
    if doc["source_sha256"] != live_sources:
        raise ValueError(
            f"index digest {digest_path} was written by different emitter sources: "
            f"recorded {doc['source_sha256']}, current {live_sources}"
        )
    per_teacher = doc["per_teacher"]
    if set(per_teacher) != set(ws2.TEACHERS):
        raise ValueError(
            f"index digest {digest_path} covers teachers {sorted(per_teacher)}, "
            f"expected {sorted(ws2.TEACHERS)}"
        )
    for teacher in sorted(per_teacher):
        section = per_teacher[teacher]
        cells = section.get("cells")
        if not isinstance(cells, dict) or not cells:
            raise ValueError(f"{teacher}: index digest section has no cells table")
        arm_dir = config_root / teacher / ARM
        for cid in sorted(cells):
            path = arm_dir / f"{cid}.yaml"
            try:
                yaml_text = path.read_text()
            except FileNotFoundError as exc:
                raise ValueError(f"{teacher}: {path} is missing") from exc
            actual = _sha256_text(yaml_text)
            if actual != cells[cid]:
                raise ValueError(
                    f"{teacher}: {path} hashes to {actual}, digest says {cells[cid]}"
                )
            cfg = yaml.safe_load(yaml_text)
            if not isinstance(cfg, dict):
                raise ValueError(f"{teacher}: {path} does not contain a config mapping")
            judge = cfg.get("checkpoints", {}).get("cp1", {}).get("judge", {})
            if judge.get("type") != "always_warm_start":
                raise ValueError(f"{teacher}: {path} is not a warm-start cell")
            if expected_pin_id is not None:
                actual_pin = (
                    cfg.get("backend", {}).get("in_memory", {}).get("expected_pin_id")
                )
                if actual_pin != expected_pin_id:
                    raise ValueError(
                        f"{teacher}: {path} expects pin_id {actual_pin!r}, "
                        f"but the runtime manifest hashes to {expected_pin_id!r}"
                    )
        if section.get("digest") != teacher_digest(teacher, cells):
            raise ValueError(
                f"{teacher}: recorded digest does not match its own cells table"
            )
    rebuilt = build_index_digest(
        {t: per_teacher[t]["cells"] for t in per_teacher}, sources=doc["source_sha256"]
    )
    if rebuilt["global_digest"] != doc["global_digest"]:
        raise ValueError(
            f"index digest {digest_path} records global_digest {doc['global_digest']!r} "
            f"but its contents hash to {rebuilt['global_digest']}"
        )
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--calibration",
        required=True,
        help="calibration json of the WARM libraries (re-calibrated after the "
        "schedule-stamped re-collection; the pnp_pinned json is stale)",
    )
    ap.add_argument(
        "--pinned-objects",
        required=True,
        help="pin table; every cell carries expected_pin_id",
    )
    ap.add_argument(
        "--weight-cell",
        action="append",
        required=True,
        help="round-1 weight_matrix() cell id to serve (repeatable)",
    )
    ap.add_argument(
        "--groot-steps",
        type=int,
        help="the GR00T head's live num_inference_timesteps; required, no default",
    )
    ap.add_argument(
        "--start-t",
        type=float,
        action="append",
        help="restrict the sweep to these timesteps (default: every "
        "recoverable point of each teacher's schedule)",
    )
    ap.add_argument("--out-root", default=DEFAULT_OUT_ROOT)
    ap.add_argument("--warm-preload-dir", default=DEFAULT_WARM_PRELOAD_DIR)
    ap.add_argument("--warm-library-tag", default=WARM_LIBRARY_TAG)
    args = ap.parse_args()

    calib = json.loads(Path(args.calibration).read_text())
    pin_id, _ = load_pin_manifest(args.pinned_objects)
    digest = emit_warm_arms(
        Path(args.out_root),
        calib,
        pin_id,
        weight_cids=args.weight_cell,
        groot_steps=args.groot_steps,
        timesteps=args.start_t,
        preload_dir=args.warm_preload_dir,
        library_tag=args.warm_library_tag,
    )
    print(
        f"[emit] warm arms {sorted(digest['per_teacher'])} pin_id={pin_id} "
        f"global_digest={digest['global_digest']} -> {args.out_root}",
        flush=True,
    )


if __name__ == "__main__":
    main()
