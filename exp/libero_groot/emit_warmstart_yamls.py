"""Emit the LIBERO × GR00T warm-start sweep YAMLs.

A NEW emitter rather than a mode of ``emit_gate_yamls.py``: that emitter's
experiments never warm-start, and its writer deliberately refuses any warm tier
as a template leak. Keeping that refusal intact while producing warm-start
cells from a separate file is how plan D9's "release both guards last" is
satisfied without weakening the gate line -- warm-start yamls can only come
from here, and here they are *required*.

One cell is the suite's winning search recipe (``Binding.template_path``, the
same base the gate line inherits) with the verdict replaced by
``always_warm_start`` at one resume timestep, over a warm library built from a
schedule-stamped re-collection. The GR00T step count is a runtime property of
the served policy (``serve_groot_libero.py --denoising-steps``), so
``--denoising-steps`` is required here too and every cell carries
``denoise_schedule: groot_n15_k<N>_v1``; the serving guard refuses the yaml
unless the live head runs exactly that loop.

Public interface: ``build_warm_cell``, ``emit_warm_sweep``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pathlib

import yaml

from openpi.cache.config import load_cache_config
from openpi.cache.types import groot_n15_schedule, schedule_from_id

from exp.libero_groot import gate_pareto_bindings as gpb

ARM = "warmstart"


def _base(binding: gpb.Binding, library: str) -> dict:
    """The suite's template with this experiment's library and write policy restated."""
    if not binding.template_path.is_file():
        raise SystemExit(
            f"{binding.suite}: {binding.template_path} missing -- run "
            "`emit_gate_yamls.py --mode template` first"
        )
    cfg = copy.deepcopy(
        yaml.safe_load(binding.template_path.read_text(encoding="utf-8"))
    )
    cfg["backend"]["in_memory"]["preload_path"] = library
    cfg["write_policy"] = {"type": "never"}
    return cfg


def build_warm_cell(
    binding: gpb.Binding, *, library: str, denoising_steps: int, start_t: float
) -> dict:
    """One forced-warm-start cell at ``start_t`` under the live step count's schedule."""
    schedule = groot_n15_schedule(denoising_steps)
    if round(start_t, 4) not in schedule.timestep_set:
        raise SystemExit(
            f"start_t={start_t} is not a recoverable point of {schedule.schedule_id}: "
            f"{list(schedule.timesteps)}"
        )
    cfg = _base(binding, library)
    cp1 = cfg["checkpoints"]["cp1"]
    cp1["gate"] = {"type": "always_search"}
    cp1["judge"] = {"type": "always_warm_start", "start_t": round(start_t, 4)}
    cfg["denoise_schedule"] = schedule.schedule_id
    return cfg


def _write(cfg: dict, out_dir: pathlib.Path, yaml_id: str) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{yaml_id}.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    loaded = load_cache_config(path)  # strict schema self-check
    cp1 = loaded.checkpoints["cp1"]
    # The mirror image of emit_gate_yamls' guard: this arm exists to warm-start,
    # so a cell that lost its warm judge or its schedule would run as a plain
    # FULL_HIT/MISS sweep and be indistinguishable in the results from the
    # gate line it is compared against.
    if cp1.judge.type != "always_warm_start" or cp1.judge.start_t is None:
        raise SystemExit(f"{path}: not a warm-start cell ({cp1.judge.type})")
    if loaded.denoise_schedule is None:
        raise SystemExit(f"{path}: warm-start cell without denoise_schedule")
    return str(path)


def emit_warm_sweep(
    binding: gpb.Binding,
    *,
    library: str,
    denoising_steps: int,
    timesteps: list[float] | None = None,
    out_dir: pathlib.Path | None = None,
) -> dict[str, str]:
    """Emit one yaml per resume timestep. Returns ``{yaml_id: path}``."""
    schedule = groot_n15_schedule(denoising_steps)
    chosen = list(schedule.timesteps) if timesteps is None else timesteps
    out_dir = binding.config_root / ARM if out_dir is None else out_dir
    written: dict[str, str] = {}
    for t in chosen:
        yaml_id = f"gpws_{binding.tag}_k{schedule.num_steps}_t{t:.4f}"
        cfg = build_warm_cell(
            binding, library=library, denoising_steps=denoising_steps, start_t=t
        )
        written[yaml_id] = _write(cfg, out_dir, yaml_id)
    provenance = {
        "suite": binding.suite,
        "library": library,
        "schedule_id": schedule.schedule_id,
        "cells": {
            yaml_id: hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
            for yaml_id, p in written.items()
        },
        "emitter_sha256": hashlib.sha256(
            pathlib.Path(__file__).read_bytes()
        ).hexdigest(),
    }
    (out_dir / "index.json").write_text(
        json.dumps(provenance, indent=1, sort_keys=True) + "\n"
    )
    return written


def verify_warm_sweep(
    yaml_dir: str | pathlib.Path, *, expected_suite: str | None = None
) -> dict | None:
    """Verify a warm-start directory before an evaluator dispatches any cell.

    Non-warm directories remain valid inputs to the shared evaluators and
    return ``None``. If even one warm cell is present, however, the dedicated
    index, every cell digest, schedule, judge shape, and library binding become
    mandatory.
    """
    root = pathlib.Path(yaml_dir)
    yaml_paths = {path.stem: path for path in sorted(root.glob("*.yaml"))}
    warm_ids: set[str] = set()
    for yaml_id, path in yaml_paths.items():
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise SystemExit(f"{path}: cache recipe must be a mapping")
        cp1 = (doc.get("checkpoints") or {}).get("cp1") or {}
        judge = cp1.get("judge") or {}
        if judge.get("type") == "always_warm_start" or judge.get("warm_tiers"):
            warm_ids.add(yaml_id)
    index_path = root / "index.json"
    # An emitted warm directory keeps its identity even if every judge was
    # subsequently changed. Otherwise removing the last warm judge bypasses
    # both digest verification and the arm-shape check.
    if not warm_ids and not index_path.is_file():
        return None

    if not index_path.is_file():
        raise SystemExit(f"{root}: warm-start cells require index.json")
    try:
        provenance = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"{index_path}: unreadable warm-start index: {exc}") from exc
    if not isinstance(provenance, dict):
        raise SystemExit(f"{index_path}: warm-start index must be a mapping")
    # The original search emitter also writes index.json, as yaml_id ->
    # {file, weights}. It remains outside this warm-arm contract.
    if not warm_ids and not {"schedule_id", "cells"}.intersection(provenance):
        return None
    recorded_cells = provenance.get("cells")
    if not isinstance(recorded_cells, dict):
        raise SystemExit(f"{index_path}: 'cells' must be a digest mapping")
    if (
        not recorded_cells
        or set(recorded_cells) != set(yaml_paths)
        or warm_ids != set(yaml_paths)
    ):
        raise SystemExit(
            f"{index_path}: a warm-start directory must contain only indexed warm "
            f"cells (indexed={sorted(recorded_cells)}, warm={sorted(warm_ids)}, "
            f"all={sorted(yaml_paths)})"
        )
    if expected_suite is not None and provenance.get("suite") != expected_suite:
        raise SystemExit(
            f"{index_path}: suite {provenance.get('suite')!r} != requested "
            f"{expected_suite!r}"
        )
    schedule_id = provenance.get("schedule_id")
    try:
        schedule = schedule_from_id(schedule_id)
    except ValueError as exc:
        raise SystemExit(f"{index_path}: invalid schedule_id {schedule_id!r}") from exc
    if not schedule.schedule_id.startswith("groot_n15_k"):
        raise SystemExit(f"{index_path}: LIBERO warm sweep requires a GR00T schedule")
    library = provenance.get("library")
    if not isinstance(library, str) or not library:
        raise SystemExit(f"{index_path}: missing library identity")

    for yaml_id in sorted(warm_ids):
        path = yaml_paths[yaml_id]
        actual_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if recorded_cells[yaml_id] != actual_digest:
            raise SystemExit(
                f"{path}: sha256 {actual_digest} != index {recorded_cells[yaml_id]}"
            )
        loaded = load_cache_config(path)
        cp1 = loaded.checkpoints.get("cp1")
        if (
            not loaded.enabled
            or loaded.backend.type != "in_memory"
            or cp1 is None
            or not cp1.enabled
            or cp1.gate.type != "always_search"
            or cp1.judge.type != "always_warm_start"
            or cp1.judge.start_t is None
        ):
            raise SystemExit(f"{path}: indexed warm cell has an invalid cp1 judge")
        if loaded.denoise_schedule != schedule_id:
            raise SystemExit(
                f"{path}: denoise_schedule {loaded.denoise_schedule!r} != "
                f"index {schedule_id!r}"
            )
        schedule.snapshot_index(cp1.judge.start_t)
        preload = loaded.backend.in_memory.preload_path
        if preload != library:
            raise SystemExit(
                f"{path}: preload_path {preload!r} != indexed library {library!r}"
            )
        if loaded.write_policy.type != "never":
            raise SystemExit(
                f"{path}: warm-start eval must set write_policy.type=never"
            )
    return provenance


def main() -> None:
    """Emit the selected suite's schedule-bound warm-start cells."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument(
        "--library",
        required=True,
        help="warm library pkl built from the schedule-stamped re-collection",
    )
    ap.add_argument(
        "--denoising-steps",
        type=int,
        required=True,
        help="the served head's live num_inference_timesteps; required, no default",
    )
    ap.add_argument(
        "--start-t",
        type=float,
        action="append",
        help="restrict the sweep (default: every recoverable point)",
    )
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    binding = gpb.for_suite(args.suite)
    written = emit_warm_sweep(
        binding,
        library=args.library,
        denoising_steps=args.denoising_steps,
        timesteps=args.start_t,
        out_dir=pathlib.Path(args.out_dir) if args.out_dir else None,
    )
    for yaml_id, path in written.items():
        print(f"[emit] {yaml_id} -> {path}", flush=True)


if __name__ == "__main__":
    main()
