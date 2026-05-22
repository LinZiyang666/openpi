"""G5 historical warmup raw rebuild on libero_10.

Phase 3 / phase 4 runners cannot be reused for this job:

- ``common.run_phase._iter_eval_yamls`` only scans direct ``.yaml`` children
  and skips ``*__warmup`` suffixes, but the phase3 config directory has
  zero top-level yamls (all live under ``warmup/`` and ``eval/``). The
  command discovers nothing.
- ``phase4.runner`` requires ``--round {1,2,3,4}`` (no round semantics fit
  G5) and accepts only a single ``--recipe`` id.
- ``phase4.runner._run_warmup_for_recipe`` writes the finite factor_raw to
  a hardcoded ``data/phase4/warmup_factor_raw/<rid>.jsonl`` (no CLI
  override) and short-circuits when that file already exists, so the
  existing libero_spatial raw silently blocks any libero_10 rebuild.

This driver loads the three historical warmup yaml files that phase5 G5
cells depend on (phase3 g6 + phase4 p1/p2), patches each yaml's
``backend.in_memory.preload_path`` to point at the libero_10 pkl, runs
the warmup against ``--task-suite libero_10``, fetches the dump, and
extracts finite-only factor_raw to caller-supplied output paths.

Three files are produced:

- ``<phase3_warmup_out>/spatial16_w8_d4_phase3_g6_f1a_a_d_jerk_curv_pair__warmup.jsonl``
- ``<phase4_warmup_out>/p1_state_fut_online_act.jsonl``
- ``<phase4_warmup_out>/p2_action_fut_online_act.jsonl``

The output filename shape strictly mirrors
``phase5.spec.warmup_raw_path_for_cell`` so phase5 ``--mode run-eval``
finds the raw without further plumbing.

CLI:

    uv run python -m exp.verdict_factor_judge.phase5.g5_warmup_libero10_driver \\
      --host ...  --port ...  --warmup-trials 2  --num-workers 5 \\
      --task-suite libero_10 \\
      --preload-pkl       exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl \\
      --phase3-warmup-in  exp/verdict_factor_judge/config/spatial16/phase3/warmup \\
      --phase4-warmup-in  exp/verdict_factor_judge/config/spatial16/phase4/warmup \\
      --phase3-warmup-out exp/verdict_factor_judge/data/phase3_libero10/warmup \\
      --phase4-warmup-out exp/verdict_factor_judge/data/phase4_libero10/warmup_factor_raw

Naming note: this driver uses ``--preload-pkl`` (no fallback semantics
needed — driver always patches the path), while ``phase5/runner.py``
uses ``--preload-pkl-override`` (override of an existing default). The
two flag names are intentional and not interchangeable.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from exp.verdict_factor_judge.common.run_phase import _build_libero_argv
from exp.verdict_factor_judge.phase3.runner import _save_dump_jsonl
from exp.verdict_factor_judge.phase5.runner import Args as RunnerArgs
from exp.verdict_factor_judge.phase5.runner import _extract_finite_factor_raw
from exp.verdict_factor_judge.phase5.spec import generate_g5_cells

try:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
except ImportError:
    WebsocketClientPolicy = None  # type: ignore[assignment]


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Historical warmup yaml stem → (input subdir, output dir kind)
# ------------------------------------------------------------------

# Each entry: (yaml_stem_without_suffix, "phase3" | "phase4")
# yaml_stem matches the warmup yaml filename minus ".yaml".
# The driver fails fast if the actual on-disk stem doesn't match one of
# these — protects against silent phase3/4 renames sneaking past.
_HISTORICAL_WARMUPS: tuple[tuple[str, str], ...] = (
    ("spatial16_w8_d4_phase3_g6_f1a_a_d_jerk_curv_pair__warmup", "phase3"),
    ("spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup",   "phase4"),
    ("spatial16_w8_d4_phase4_p2_action_fut_online_act__warmup",  "phase4"),
)


# ------------------------------------------------------------------
# declared_keys resolver
# ------------------------------------------------------------------


def _build_declared_keys_map(cfg_id: str) -> dict[str, tuple[str, ...]]:
    """Map historical warmup_yaml_id → exact declared_keys for that G5 recipe.

    Source of truth: ``phase5.spec.generate_g5_cells()``. All 16 cells of a
    given recipe share the same ``declared_keys`` (only fh/ws change), so
    deduping by ``Cell.warmup_yaml_id`` yields one tuple per recipe.

    Do NOT parse from the warmup yaml's ``dump.factors`` field —
    ``v2_spec._build_dump_factor_superset()`` intentionally over-collects
    a descriptor/window superset, which would make
    ``_extract_finite_factor_raw`` keep too many or too few raw keys.
    """
    out: dict[str, tuple[str, ...]] = {}
    for cell in generate_g5_cells(cfg_id=cfg_id):
        wid = cell.warmup_yaml_id
        if wid not in out:
            out[wid] = tuple(cell.declared_keys)
        else:
            # Sanity: every cell sharing this warmup_yaml_id must have the
            # same declared_keys (only fh/ws differ across the 16 cells).
            assert out[wid] == tuple(cell.declared_keys), (
                f"G5 cells with warmup_yaml_id={wid!r} disagree on "
                "declared_keys; spec.py invariant broken."
            )
    return out


# ------------------------------------------------------------------
# Yaml preload-path patcher
# ------------------------------------------------------------------


def _patch_preload_path(src: Path, preload_pkl: str, dst: Path) -> None:
    """Load yaml, rewrite backend.in_memory.preload_path, write to dst.

    The patched copy lives at ``dst`` (typically /tmp); the original file
    on disk is not modified — phase3/phase4 warmup yamls remain canonical
    libero_spatial artifacts in the repo.
    """
    data = yaml.safe_load(src.read_text(encoding="utf-8"))
    if "backend" not in data or "in_memory" not in data["backend"]:
        raise ValueError(
            f"warmup yaml {src} missing backend.in_memory; cannot patch"
        )
    data["backend"]["in_memory"]["preload_path"] = preload_pkl
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


# ------------------------------------------------------------------
# Args construction (reuses phase5.runner.Args so _build_libero_argv sees
# every optional attribute it reads)
# ------------------------------------------------------------------


def _build_runner_args(cli) -> RunnerArgs:
    """Construct a phase5.runner.Args dataclass with safe defaults for the
    six optional attributes ``_build_libero_argv`` reads (``task_ids`` /
    ``init_states_dir`` / ``episode_filter`` / ``cuda_visible_devices`` /
    ``conda_env`` / ``per_step_log_dir``). The driver does not surface
    these as CLI flags — defaults match the empty / no-op behavior.

    ``mode='run-warmup'`` is the closest existing semantic label for what
    the driver does (a single warmup pass per yaml). ``_build_libero_argv``
    doesn't read ``args.mode``, so the choice has no runtime effect — it
    just keeps the value inside ``VALID_MODES`` so a future grep for
    ``mode == 'something'`` won't have to special-case the driver.
    """
    return RunnerArgs(
        mode="run-warmup",
        host=cli.host,
        port=cli.port,
        task_suite=cli.task_suite,
        num_workers=cli.num_workers,
        warmup_trials=cli.warmup_trials,
        eval_trials=0,
        # remaining optional fields take their dataclass defaults (empty)
    )


# ------------------------------------------------------------------
# Per-yaml driver step
# ------------------------------------------------------------------


def _run_one_historical_warmup(
    ctl,
    *,
    yaml_stem: str,
    kind: str,                 # "phase3" or "phase4"
    src_yaml: Path,
    preload_pkl: str,
    declared_keys: tuple[str, ...],
    out_path: Path,
    runner_args: RunnerArgs,
) -> None:
    """Server-load patched yaml, run LIBERO warmup, fetch dump, extract finite.

    No cache-skip on ``out_path``: per the libero_10 plan §2.7, the driver
    must not short-circuit on file existence — that's the bug that lets
    a stale libero_spatial raw silently block a libero_10 rebuild in the
    phase4 runner. If the caller wants idempotency, they should clean the
    out dir before invoking.
    """
    with tempfile.TemporaryDirectory(prefix="phase5_libero10_warmup_") as tmpdir:
        patched_yaml = Path(tmpdir) / f"{yaml_stem}__libero10.yaml"
        _patch_preload_path(src_yaml, preload_pkl, patched_yaml)

        logger.info("[%s] %s: load_cache_config", kind, yaml_stem)
        ctl.load_cache_config(
            yaml_content=patched_yaml.read_text(encoding="utf-8"),
            yaml_id=yaml_stem,
        )

        logger.info("[%s] %s: spawn LIBERO workers", kind, yaml_stem)
        cmd, env = _build_libero_argv(
            args=runner_args, yaml_id=yaml_stem, phase="warmup",
            num_trials_per_task=runner_args.warmup_trials,
        )
        subprocess.run(cmd, env=env, check=True)

        logger.info("[%s] %s: fetch_dump", kind, yaml_stem)
        raw_jsonl = Path(tmpdir) / f"{yaml_stem}.raw.jsonl"
        content = ctl.fetch_dump(yaml_stem)
        _save_dump_jsonl(content, raw_jsonl)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        _extract_finite_factor_raw(raw_jsonl, out_path, list(declared_keys))
        logger.info("[%s] %s -> %s", kind, yaml_stem, out_path)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def _parse_cli(argv: list[str] | None = None):
    p = argparse.ArgumentParser(prog="g5_warmup_libero10_driver")
    p.add_argument("--host", default="155.98.36.32")
    p.add_argument("--port", type=int, default=9000)
    p.add_argument("--task-suite", default="libero_10")
    p.add_argument("--num-workers", type=int, default=5)
    p.add_argument("--warmup-trials", type=int, default=2)
    p.add_argument("--cfg-id", default="spatial16_w8_d4")
    p.add_argument(
        "--preload-pkl", required=True,
        help="Absolute or repo-relative path to the libero_10 pkl that all "
             "three historical warmup yamls will be patched to use.",
    )
    p.add_argument(
        "--phase3-warmup-in", required=True, type=Path,
        help="Dir containing phase3 warmup yaml files "
             "(usually exp/verdict_factor_judge/config/spatial16/phase3/warmup).",
    )
    p.add_argument(
        "--phase4-warmup-in", required=True, type=Path,
        help="Dir containing phase4 warmup yaml files "
             "(usually exp/verdict_factor_judge/config/spatial16/phase4/warmup).",
    )
    p.add_argument(
        "--phase3-warmup-out", required=True, type=Path,
        help="Output dir for phase3 g6 finite factor_raw jsonl.",
    )
    p.add_argument(
        "--phase4-warmup-out", required=True, type=Path,
        help="Output dir for phase4 p1/p2 finite factor_raw jsonls.",
    )
    return p.parse_args(argv)


def _resolve_output_path(yaml_stem: str, kind: str, cli) -> Path:
    """Mirror phase5.spec.warmup_raw_path_for_cell expectations:

    - phase3 g6: ``<phase3_warmup_out>/<warmup_yaml_id>.jsonl``
    - phase4 p1/p2: ``<phase4_warmup_out>/<recipe_id>.jsonl`` (recipe_id
      derived by stripping the ``spatial16_w8_d4_phase4_`` prefix and
      ``__warmup`` suffix from the warmup_yaml_id)
    """
    if kind == "phase3":
        return cli.phase3_warmup_out / f"{yaml_stem}.jsonl"
    if kind == "phase4":
        # warmup_yaml_id = "spatial16_w8_d4_phase4_<recipe_id>__warmup"
        prefix = "spatial16_w8_d4_phase4_"
        suffix = "__warmup"
        if not (yaml_stem.startswith(prefix) and yaml_stem.endswith(suffix)):
            raise ValueError(
                f"phase4 warmup_yaml_id {yaml_stem!r} does not match "
                f"expected '{prefix}<recipe>{suffix}' shape"
            )
        recipe_id = yaml_stem[len(prefix):-len(suffix)]
        return cli.phase4_warmup_out / f"{recipe_id}.jsonl"
    raise KeyError(f"unknown warmup kind {kind!r}")


def _resolve_input_path(yaml_stem: str, kind: str, cli) -> Path:
    if kind == "phase3":
        return cli.phase3_warmup_in / f"{yaml_stem}.yaml"
    if kind == "phase4":
        return cli.phase4_warmup_in / f"{yaml_stem}.yaml"
    raise KeyError(f"unknown warmup kind {kind!r}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    cli = _parse_cli(argv)

    if WebsocketClientPolicy is None:
        raise RuntimeError(
            "openpi_client is not installed; cannot run G5 warmup driver"
        )

    declared_keys_map = _build_declared_keys_map(cli.cfg_id)

    # Sanity check: every yaml_stem we plan to drive must have a
    # declared_keys mapping; missing entry means phase3/4 spec renamed
    # something and the driver would silently fall back to the wrong key
    # set.
    for yaml_stem, _kind in _HISTORICAL_WARMUPS:
        if yaml_stem not in declared_keys_map:
            raise KeyError(
                f"warmup yaml stem {yaml_stem!r} has no declared_keys "
                "in phase5.spec.generate_g5_cells(); refusing to run "
                "(would extract the wrong factor_raw key set)."
            )

    runner_args = _build_runner_args(cli)

    for yaml_stem, kind in _HISTORICAL_WARMUPS:
        src_yaml = _resolve_input_path(yaml_stem, kind, cli)
        if not src_yaml.exists():
            raise FileNotFoundError(
                f"historical warmup yaml not found: {src_yaml}"
            )
        out_path = _resolve_output_path(yaml_stem, kind, cli)

        with WebsocketClientPolicy(host=cli.host, port=cli.port) as ctl:
            _run_one_historical_warmup(
                ctl,
                yaml_stem=yaml_stem,
                kind=kind,
                src_yaml=src_yaml,
                preload_pkl=cli.preload_pkl,
                declared_keys=declared_keys_map[yaml_stem],
                out_path=out_path,
                runner_args=runner_args,
            )

    print(
        f"[g5_warmup_libero10_driver] all {len(_HISTORICAL_WARMUPS)} "
        "historical warmups finished."
    )
    return 0


__all__ = [
    "_HISTORICAL_WARMUPS",
    "_build_declared_keys_map",
    "_build_runner_args",
    "_parse_cli",
    "_patch_preload_path",
    "_resolve_input_path",
    "_resolve_output_path",
    "_run_one_historical_warmup",
    "main",
]


if __name__ == "__main__":
    sys.exit(main())
