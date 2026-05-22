# ruff: noqa: SLF001  # white-box tests intentionally exercise _-prefixed helpers
"""Tests for the libero_10 preload-path override + G5 warmup driver.

Three layers of plumbing must thread ``preload_pkl_override`` from CLI to
emitted yaml — these tests pin every layer so a future refactor that
breaks the chain fails loudly. The G5 driver gets its own bucket
covering ``declared_keys`` resolution, output-path mirroring, and the
yaml-patch helper.

Buckets:

- ``eval`` side override pass-through (v2_spec + phase5.spec)
- ``warmup`` side override pass-through (v2_spec + phase5.spec +
  phase5.runner emit + lazy ensure)
- CLI parse (``--preload-pkl-override`` set / default empty)
- G5 driver (declared_keys map + output-path mirror + yaml patch +
  unknown stem rejection)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml

from exp.verdict_factor_judge.common import v2_spec
from exp.verdict_factor_judge.phase5 import g5_warmup_libero10_driver as drv
from exp.verdict_factor_judge.phase5 import runner as r
from exp.verdict_factor_judge.phase5 import spec as p5spec

_LIBERO10_PKL = "exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl"


def _default_libero_spatial_pkl() -> str:
    return v2_spec.CFG_SPECS["spatial16_w8_d4"]["preload_pkl"]


def _make_phase3_g6_recipe() -> tuple[list[dict], dict]:
    """Reuse the g6 recipe shape used by phase5 G5 cells. Picking phase3 g6
    rather than a 10-key phase4 recipe keeps the fixture small.
    """
    factors, composer = v2_spec.RECIPES["topk_only"]()
    return factors, composer


# ======================================================================
# 1. Eval-side override pass-through
# ======================================================================


def test_v2_spec_build_eval_yaml_override_used() -> None:
    factors, composer = _make_phase3_g6_recipe()
    out = v2_spec.build_eval_yaml(
        "spatial16_w8_d4", factors, composer,
        preload_pkl_override=_LIBERO10_PKL,
    )
    assert out["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL


def test_v2_spec_build_eval_yaml_default_unchanged() -> None:
    factors, composer = _make_phase3_g6_recipe()
    out = v2_spec.build_eval_yaml("spatial16_w8_d4", factors, composer)
    assert out["backend"]["in_memory"]["preload_path"] == _default_libero_spatial_pkl()


def test_phase5_spec_build_eval_yaml_for_cell_override() -> None:
    g5_cells = p5spec.generate_g5_cells()
    cell = g5_cells[0]
    out = p5spec.build_eval_yaml_for_cell(
        p5spec.CFG_ID_DEFAULT, cell, fh_thr=0.5, ws_thr=0.5,
        preload_pkl_override=_LIBERO10_PKL,
    )
    assert out["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL


# ======================================================================
# 2. Warmup-side override pass-through
# ======================================================================


def test_v2_spec_build_warmup_yaml_override_used() -> None:
    out = v2_spec.build_warmup_yaml(
        "spatial16_w8_d4", "test_eval_id",
        eval_factors=None,
        preload_pkl_override=_LIBERO10_PKL,
    )
    assert out["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL


def test_v2_spec_build_warmup_yaml_default_unchanged() -> None:
    out = v2_spec.build_warmup_yaml(
        "spatial16_w8_d4", "test_eval_id", eval_factors=None,
    )
    assert out["backend"]["in_memory"]["preload_path"] == _default_libero_spatial_pkl()


def test_phase5_spec_build_warmup_yaml_for_cell_override() -> None:
    g1_cells = p5spec.generate_g1_cells()
    cell = g1_cells[0]
    out = p5spec.build_warmup_yaml_for_cell(
        p5spec.CFG_ID_DEFAULT, cell,
        preload_pkl_override=_LIBERO10_PKL,
    )
    assert out["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL


def test_runner_mode_emit_warmup_yamls_threads_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """All emit'd warmup yamls under --mode emit-warmup-yamls must carry the
    libero_10 pkl path when --preload-pkl-override is set.
    """
    captured: dict[Path, dict[str, Any]] = {}

    def fake_write_yaml(path: Path, data: dict[str, Any]) -> None:
        captured[path] = data

    monkeypatch.setattr(r, "write_yaml", fake_write_yaml)

    args = r._parse_args([
        "--mode", "emit-warmup-yamls",
        "--groups", "g1",
        "--warmup-yaml-dir", str(tmp_path),
        "--preload-pkl-override", _LIBERO10_PKL,
    ])
    r._mode_emit_warmup_yamls(args)

    assert captured, "expected at least one warmup yaml to be emitted"
    for path, data in captured.items():
        assert data["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL, (
            f"emit-warmup-yamls did not thread override for {path}"
        )


def test_runner_ensure_warmup_for_cell_lazy_threads_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When a cell's raw is missing, _ensure_warmup_for_cell → _run_one_warmup
    lazy-emits the warmup yaml; that emission path must also forward the
    preload-pkl override.
    """
    captured: dict[Path, dict[str, Any]] = {}

    def fake_write_yaml(path: Path, data: dict[str, Any]) -> None:
        captured[path] = data
        # also create the file so subsequent checks see it on disk
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(data, sort_keys=False))

    # Stub out the LIBERO subprocess + dump fetch so the test is hermetic.
    monkeypatch.setattr(r, "write_yaml", fake_write_yaml)
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **kw: MagicMock(returncode=0))

    # _run_one_warmup needs ctl.load_cache_config + ctl.fetch_dump
    fake_ctl = MagicMock()
    fake_ctl.fetch_dump.return_value = b""  # empty dump → empty extract

    # _phase5_warmup_raw_dir(args) writes the dump _raw_dump under it; force
    # writes into tmp_path so we don't pollute the repo.
    warmup_yaml_dir = tmp_path / "warmup_yaml"
    warmup_jsonl_dir = tmp_path / "warmup_jsonl"

    args = r._parse_args([
        "--mode", "run-warmup",
        "--warmup-yaml-dir", str(warmup_yaml_dir),
        "--warmup-jsonl-dir", str(warmup_jsonl_dir),
        "--preload-pkl-override", _LIBERO10_PKL,
    ])

    # Pick any G1 cell (independent warmup) and run the lazy path.
    cell = p5spec.generate_g1_cells()[0]
    r._ensure_warmup_for_cell(fake_ctl, cell, args)

    # The lazy emit should have produced the warmup yaml with the override.
    emitted_yaml = warmup_yaml_dir / f"{cell.warmup_yaml_id}.yaml"
    assert emitted_yaml in captured, (
        f"lazy-emit did not produce {emitted_yaml}; captured={list(captured)}"
    )
    assert captured[emitted_yaml]["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL


# ======================================================================
# 3. CLI parse
# ======================================================================


def test_cli_preload_pkl_override_set() -> None:
    args = r._parse_args([
        "--mode", "run-eval",
        "--preload-pkl-override", _LIBERO10_PKL,
    ])
    assert args.preload_pkl_override == _LIBERO10_PKL


def test_cli_preload_pkl_override_default_empty() -> None:
    args = r._parse_args(["--mode", "run-eval"])
    assert args.preload_pkl_override == ""


# ======================================================================
# 4. G5 driver
# ======================================================================


def test_driver_declared_keys_map_phase4_p1() -> None:
    m = drv._build_declared_keys_map(p5spec.CFG_ID_DEFAULT)
    wid = "spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup"
    keys = m[wid]
    # Expected shape: 4 offline-state W-FUT keys x 2 windows = 8, plus 2
    # online-action keys at (3, 3) = 10 keys total.
    assert len(keys) == 10
    offline_state_keys = [k for k in keys if "_offline_state__" in k]
    online_action_keys = [k for k in keys if "_online_action__" in k]
    assert len(offline_state_keys) == 8
    assert len(online_action_keys) == 2
    assert "jerk_online_action__p3_f3" in keys
    assert "dispersion_online_action__p3_f3" in keys


def test_driver_declared_keys_map_phase4_p2() -> None:
    m = drv._build_declared_keys_map(p5spec.CFG_ID_DEFAULT)
    wid = "spatial16_w8_d4_phase4_p2_action_fut_online_act__warmup"
    keys = m[wid]
    assert len(keys) == 10
    offline_action_keys = [k for k in keys if "_offline_action__" in k]
    online_action_keys = [k for k in keys if "_online_action__" in k]
    # p2's base channel is action, so the 8 offline keys are action-side.
    assert len(offline_action_keys) == 8
    assert len(online_action_keys) == 2


def test_driver_declared_keys_map_phase3_g6() -> None:
    m = drv._build_declared_keys_map(p5spec.CFG_ID_DEFAULT)
    wid = "spatial16_w8_d4_phase3_g6_f1a_a_d_jerk_curv_pair__warmup"
    keys = m[wid]
    # g6 recipe: 2 online-action keys at (3, 3) — no base offline injection.
    assert set(keys) == {"jerk_online_action__p3_f3", "dispersion_online_action__p3_f3"}


def test_driver_only_stem_filter_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    """--only-stem narrows to a single warmup_yaml_id and skips the rest."""
    invoked: list[str] = []

    def fake_run(ctl, *, yaml_stem, kind, **kwargs):
        invoked.append(yaml_stem)

    monkeypatch.setattr(drv, "WebsocketClientPolicy", lambda *a, **kw: MagicMock(
        __enter__=lambda self: MagicMock(),
        __exit__=lambda self, *a: None,
    ))
    monkeypatch.setattr(drv, "_run_one_historical_warmup", fake_run)

    target = "spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup"
    real_repo = Path(__file__).resolve().parents[2]
    drv.main([
        "--preload-pkl", _LIBERO10_PKL,
        "--phase3-warmup-in", str(real_repo / "exp/verdict_factor_judge/config/spatial16/phase3/warmup"),
        "--phase4-warmup-in", str(real_repo / "exp/verdict_factor_judge/config/spatial16/phase4/warmup"),
        "--phase3-warmup-out", "/tmp",
        "--phase4-warmup-out", "/tmp",
        "--only-stem", target,
    ])
    assert invoked == [target], f"expected 1 invocation of {target!r}, got {invoked}"


def test_driver_only_stem_filter_unknown_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """--only-stem with a yaml_stem not in _HISTORICAL_WARMUPS must raise."""
    monkeypatch.setattr(drv, "WebsocketClientPolicy", object())
    with pytest.raises(KeyError, match="--only-stem"):
        drv.main([
            "--preload-pkl", _LIBERO10_PKL,
            "--phase3-warmup-in", "/tmp",
            "--phase4-warmup-in", "/tmp",
            "--phase3-warmup-out", "/tmp",
            "--phase4-warmup-out", "/tmp",
            "--only-stem", "bogus_stem__warmup",
        ])


def test_driver_unknown_stem_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """If a stem in _HISTORICAL_WARMUPS doesn't appear in
    generate_g5_cells() (e.g. phase3/4 renamed something), main() must
    refuse to run rather than silently use the wrong key set.
    """
    monkeypatch.setattr(drv, "_HISTORICAL_WARMUPS", (
        ("ghost_recipe__warmup", "phase3"),
    ))
    # We don't even reach the network step — the sanity check fires first.
    # WebsocketClientPolicy may be None in CI; patch to a sentinel so the
    # earlier installed-check passes too.
    monkeypatch.setattr(drv, "WebsocketClientPolicy", object())

    with pytest.raises(KeyError, match="ghost_recipe__warmup"):
        drv.main([
            "--preload-pkl", _LIBERO10_PKL,
            "--phase3-warmup-in", "/tmp",
            "--phase4-warmup-in", "/tmp",
            "--phase3-warmup-out", "/tmp",
            "--phase4-warmup-out", "/tmp",
        ])


def test_driver_patch_preload_path(tmp_path: Path) -> None:
    """_patch_preload_path must rewrite backend.in_memory.preload_path
    without mutating the source file or other fields.
    """
    src = tmp_path / "src.yaml"
    dst = tmp_path / "dst.yaml"
    src.write_text(yaml.safe_dump({
        "enabled": True,
        "keys": {"k1": {"weight": 1.0}},
        "backend": {
            "type": "in_memory",
            "in_memory": {
                "preload_path": "/original/path.pkl",
                "index_type": "brute_force",
            },
        },
    }))

    drv._patch_preload_path(src, "/new/path.pkl", dst)

    src_data = yaml.safe_load(src.read_text())
    dst_data = yaml.safe_load(dst.read_text())
    # source untouched
    assert src_data["backend"]["in_memory"]["preload_path"] == "/original/path.pkl"
    # destination has the new path, everything else preserved
    assert dst_data["backend"]["in_memory"]["preload_path"] == "/new/path.pkl"
    assert dst_data["backend"]["in_memory"]["index_type"] == "brute_force"
    assert dst_data["enabled"] is True
    assert dst_data["keys"] == {"k1": {"weight": 1.0}}


def test_driver_run_one_historical_warmup_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End-to-end success path for _run_one_historical_warmup.

    Stubs the LIBERO subprocess + ctl.fetch_dump so the test is hermetic,
    feeds a hand-crafted per-step dump with a mix of finite, NaN, and
    out-of-key rows, and asserts the produced out jsonl is at the
    expected path with the finite-only filtered rows.
    """
    # 1. Build a fake historical warmup yaml on disk (libero_spatial preload).
    src_yaml = tmp_path / "fake_warmup.yaml"
    src_yaml.write_text(yaml.safe_dump({
        "enabled": True,
        "backend": {
            "type": "in_memory",
            "in_memory": {
                "preload_path": "exp/common/data/cache_artifacts/libero_spatial/cp1_spatial_pool_16.pkl",
                "index_type": "brute_force",
            },
        },
    }))

    # 2. Stub LIBERO subprocess so no real worker spawns.
    fake_run = MagicMock(return_value=MagicMock(returncode=0))
    monkeypatch.setattr(drv.subprocess, "run", fake_run)

    # 3. Stub fetch_dump to return a per-step JSONL with four rows:
    #    - row 0: both declared keys finite              -> kept whole
    #    - row 1: one key NaN, one finite                -> kept whole (>=1 finite)
    #    - row 2: only out-of-declared keys              -> dropped (kept dict empty)
    #    - row 3: declared keys present but all NaN      -> dropped (no finite values)
    fake_dump = (
        b'{"factor_raw": {"k_a": 1.0, "k_b": 2.0}}\n'
        b'{"factor_raw": {"k_a": NaN, "k_b": 3.0}}\n'
        b'{"factor_raw": {"unrelated": 9.9}}\n'
        b'{"factor_raw": {"k_a": NaN, "k_b": NaN}}\n'
    )

    fake_ctl = MagicMock()
    fake_ctl.fetch_dump.return_value = fake_dump

    # 4. Wire up runner_args via _build_runner_args so we cover that path too.
    class _CLI:
        host = "h"
        port = 1
        task_suite = "libero_10"
        num_workers = 5
        warmup_trials = 2

    runner_args = drv._build_runner_args(_CLI())

    out_path = tmp_path / "out" / "fake_warmup.jsonl"
    drv._run_one_historical_warmup(
        fake_ctl,
        yaml_stem="fake_warmup",
        kind="phase3",
        src_yaml=src_yaml,
        preload_pkl=_LIBERO10_PKL,
        declared_keys=("k_a", "k_b"),
        out_path=out_path,
        runner_args=runner_args,
    )

    # ctl.load_cache_config invoked with patched yaml that points at libero_10
    fake_ctl.load_cache_config.assert_called_once()
    loaded_yaml = yaml.safe_load(
        fake_ctl.load_cache_config.call_args.kwargs["yaml_content"]
    )
    assert loaded_yaml["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL

    # subprocess.run was called once (the LIBERO worker spawn)
    assert fake_run.call_count == 1

    # Output jsonl produced at exact path with finite-only filter applied.
    # _extract_finite_factor_raw keeps rows where >=1 declared key is finite
    # (the whole declared subset is preserved verbatim — including any NaN
    # entries — so the downstream calibration sees the raw schema).
    assert out_path.exists()
    rows = [
        json.loads(line)
        for line in out_path.read_text().splitlines() if line.strip()
    ]
    # Row 0 kept whole, row 1 kept whole (k_a NaN + k_b 3.0),
    # rows 2 + 3 dropped entirely.
    assert len(rows) == 2
    assert rows[0]["factor_raw"] == {"k_a": 1.0, "k_b": 2.0}
    assert set(rows[1]["factor_raw"].keys()) == {"k_a", "k_b"}
    assert math.isnan(rows[1]["factor_raw"]["k_a"])
    assert rows[1]["factor_raw"]["k_b"] == 3.0


def test_driver_build_runner_args_threads_conda_env() -> None:
    """--conda-env on the driver CLI must land on RunnerArgs.conda_env, so
    _build_libero_argv wraps the LIBERO worker as `conda run -p ...`.
    """
    class _CLI:
        host = "h"
        port = 1
        task_suite = "libero_10"
        num_workers = 5
        warmup_trials = 2
        conda_env = "/scratch/zixuans8/libero_sim"

    args = drv._build_runner_args(_CLI())
    assert args.conda_env == "/scratch/zixuans8/libero_sim"


def test_driver_build_runner_args_has_libero_argv_fields() -> None:
    """_build_libero_argv (common.run_phase) reads six optional attrs beyond
    host/port/task_suite/num_workers/warmup_trials/eval_trials:
    task_ids, init_states_dir, episode_filter, cuda_visible_devices,
    conda_env, per_step_log_dir. Driver's _build_runner_args must construct
    an Args where every one of these is present (even if empty), so the
    helper never blows up with AttributeError during a real run.
    """
    class _CLI:
        host = "h"
        port = 1
        task_suite = "libero_10"
        num_workers = 5
        warmup_trials = 2

    args = drv._build_runner_args(_CLI())

    required_for_libero_argv = (
        "host", "port", "task_suite", "num_workers", "warmup_trials",
        "eval_trials",
        "task_ids", "init_states_dir", "episode_filter",
        "cuda_visible_devices", "conda_env", "per_step_log_dir",
    )
    for attr in required_for_libero_argv:
        assert hasattr(args, attr), f"_build_runner_args missing field {attr!r}"
    # spot-check that empty defaults are the right type (str/tuple, not None)
    assert args.task_ids == ()
    assert args.init_states_dir == ""
    assert args.per_step_log_dir == ""
    assert args.conda_env == ""


def test_driver_output_path_mirrors_phase5_lookup() -> None:
    """The output filename shape must match what
    phase5.spec.warmup_raw_path_for_cell will look up at eval time.

    - phase3 g6 → ``<phase3_warmup_out>/<warmup_yaml_id>.jsonl``
    - phase4 p1/p2 → ``<phase4_warmup_out>/<recipe_id>.jsonl``
    """
    class _CLI:
        phase3_warmup_out = Path("/p3_out")
        phase4_warmup_out = Path("/p4_out")

    cli = _CLI()
    p3 = drv._resolve_output_path(
        "spatial16_w8_d4_phase3_g6_f1a_a_d_jerk_curv_pair__warmup", "phase3", cli,
    )
    assert p3 == Path("/p3_out/spatial16_w8_d4_phase3_g6_f1a_a_d_jerk_curv_pair__warmup.jsonl")

    p4_p1 = drv._resolve_output_path(
        "spatial16_w8_d4_phase4_p1_state_fut_online_act__warmup", "phase4", cli,
    )
    assert p4_p1 == Path("/p4_out/p1_state_fut_online_act.jsonl")

    p4_p2 = drv._resolve_output_path(
        "spatial16_w8_d4_phase4_p2_action_fut_online_act__warmup", "phase4", cli,
    )
    assert p4_p2 == Path("/p4_out/p2_action_fut_online_act.jsonl")
