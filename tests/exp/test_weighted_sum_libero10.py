# ruff: noqa: SLF001  # white-box tests intentionally exercise _-prefixed helpers
"""Tests for the weighted_sum libero_10 replication code changes (plan §9).

Covers the gated code added for the libero_10 port:

- kinematic builders thread ``preload_pkl_override`` + ``cfg_id``
  (``spec.build_eval_yaml_for_cell`` / ``super_warmup.build_super_warmup_yaml``
  / ``runner._build_always_warm_yaml``), with default-unchanged backward compat;
- runner ``--cfg-id`` CLI + emit modes forward both override + cfg_id;
- new ``CFG_SPECS["spatial16_ws_d1_best_libero10"]`` entry (structure) and the
  libero_spatial entry untouched;
- ``refine_round.rank_keybuilder`` ranks a forced stem (not the global best);
- ``emit_top10`` picks exactly 10 deterministically + manifest;
- ``emit_trajectory_yamls`` base union no longer pinned to 18 / overlap 4.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from exp.verdict_factor_judge.common import v2_spec
from exp.weighted_sum import emit_top10, refine_round
from exp.weighted_sum import emit_trajectory_yamls as etj
from exp.weighted_sum.kinematic import runner as krun
from exp.weighted_sum.kinematic import spec as kspec
from exp.weighted_sum.kinematic import super_warmup as ksw

_LIBERO10_PKL = "exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl"
_LIBERO10_CFG = "spatial16_ws_d1_best_libero10"
_SPATIAL_CFG = "spatial16_ws_d1_best"


def _spatial_pkl() -> str:
    return v2_spec.CFG_SPECS[_SPATIAL_CFG]["preload_pkl"]


def _a_cell():
    return kspec.generate_all_cells()[0]


# ======================================================================
# 1. CFG_SPECS libero_10 entry + libero_spatial untouched
# ======================================================================


def test_cfg_specs_libero10_entry_structure() -> None:
    cfg = v2_spec.CFG_SPECS[_LIBERO10_CFG]
    assert cfg["key_builder_type"] == "cp1_spatial_pool_16"
    assert "libero_10" in cfg["preload_pkl"]
    ss = cfg["search_strategy"]
    assert ss["type"] == "weighted_score_sum_knn"
    assert "trajectory_depth" not in ss  # d1 single-step retrieval
    assert ss["score_normalization"]["type"] == "per_field"
    for f in ("vision_0", "vision_1", "robot_state"):
        assert cfg["keys"][f]["enabled"] is True
        assert "method" in ss["score_normalization"]["fields"][f]


def test_cfg_specs_libero_spatial_untouched() -> None:
    # The libero_10 entry must NOT repoint the libero_spatial base.
    assert "libero_spatial" in v2_spec.CFG_SPECS[_SPATIAL_CFG]["preload_pkl"]
    assert kspec.CFG_ID_DEFAULT == _SPATIAL_CFG


# ======================================================================
# 2. kinematic builders thread preload_pkl_override + cfg_id
# ======================================================================


def test_build_eval_yaml_for_cell_override_used() -> None:
    out = kspec.build_eval_yaml_for_cell(
        _a_cell(), 0.6, 0.4, preload_pkl_override=_LIBERO10_PKL
    )
    assert out["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL


def test_build_eval_yaml_for_cell_default_unchanged() -> None:
    out = kspec.build_eval_yaml_for_cell(_a_cell(), 0.6, 0.4)
    assert out["backend"]["in_memory"]["preload_path"] == _spatial_pkl()


def test_build_eval_yaml_for_cell_cfg_id_libero10() -> None:
    out = kspec.build_eval_yaml_for_cell(_a_cell(), 0.6, 0.4, cfg_id=_LIBERO10_CFG)
    assert "libero_10" in out["backend"]["in_memory"]["preload_path"]


def test_build_super_warmup_yaml_override_used() -> None:
    out = ksw.build_super_warmup_yaml(preload_pkl_override=_LIBERO10_PKL)
    assert out["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL


def test_build_super_warmup_yaml_default_unchanged() -> None:
    out = ksw.build_super_warmup_yaml()
    assert out["backend"]["in_memory"]["preload_path"] == _spatial_pkl()


def test_build_super_warmup_yaml_cfg_id_libero10() -> None:
    out = ksw.build_super_warmup_yaml(cfg_id=_LIBERO10_CFG)
    assert "libero_10" in out["backend"]["in_memory"]["preload_path"]


def test_build_always_warm_yaml_override_used() -> None:
    out = krun._build_always_warm_yaml(0.5, preload_pkl_override=_LIBERO10_PKL)
    assert out["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL


def test_build_always_warm_yaml_default_unchanged() -> None:
    out = krun._build_always_warm_yaml(0.5)
    assert out["backend"]["in_memory"]["preload_path"] == _spatial_pkl()


def test_build_always_warm_yaml_cfg_id_libero10() -> None:
    out = krun._build_always_warm_yaml(0.5, cfg_id=_LIBERO10_CFG)
    assert "libero_10" in out["backend"]["in_memory"]["preload_path"]


# ======================================================================
# 3. runner CLI + emit modes thread both flags
# ======================================================================


def test_cli_cfg_id_set_and_default() -> None:
    a = krun._parse(["--mode", "emit-warmup", "--cfg-id", _LIBERO10_CFG])
    assert a.cfg_id == _LIBERO10_CFG
    b = krun._parse(["--mode", "emit-warmup"])
    assert b.cfg_id == _SPATIAL_CFG  # default = libero_spatial


def _capture_write_yaml(monkeypatch: pytest.MonkeyPatch) -> dict[Path, dict[str, Any]]:
    captured: dict[Path, dict[str, Any]] = {}

    def fake(path: Path, data: dict[str, Any]) -> None:
        captured[Path(path)] = data

    # runner imports write_yaml locally inside each mode fn → patch the source.
    monkeypatch.setattr(
        "exp.verdict_factor_judge.common.generate_yamls.write_yaml", fake
    )
    return captured


def test_emit_warmup_mode_threads_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_write_yaml(monkeypatch)
    args = krun._parse([
        "--mode", "emit-warmup",
        "--warmup-yaml", str(tmp_path / "sw.yaml"),
        "--preload-pkl-override", _LIBERO10_PKL,
        "--cfg-id", _LIBERO10_CFG,
    ])
    krun._mode_emit_warmup(args)
    assert captured
    for data in captured.values():
        assert data["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL


def test_run_always_warm_mode_threads_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = _capture_write_yaml(monkeypatch)
    args = krun._parse([
        "--mode", "run-always-warm",
        "--always-warm-dir", str(tmp_path),
        "--preload-pkl-override", _LIBERO10_PKL,
        "--cfg-id", _LIBERO10_CFG,
    ])
    krun._mode_run_always_warm(args)
    assert len(captured) == 3  # start_t 0.3 / 0.5 / 0.7
    for data in captured.values():
        assert data["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL


def test_emit_eval_yamls_mode_threads_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The 237-cell emit-eval-yamls mode (most fragile, plan §9d) must thread
    cfg_id + preload_pkl_override into every per-cell eval yaml. Mock the
    threshold solver so no super-raw / pkl is actually loaded.
    """
    import exp.verdict_factor_judge.phase3.threshold_solver as ts

    captured = _capture_write_yaml(monkeypatch)
    # Canned solver: finite scores + fh>ws so cells emit instead of skip.
    monkeypatch.setattr(ts, "reconstruct_scores", lambda *a, **k: [0.5] * 60)
    monkeypatch.setattr(ts, "derive_thresholds", lambda *a, **k: (0.6, 0.4))
    super_raw = tmp_path / "super_raw.jsonl"
    super_raw.write_text("")  # only existence is checked before the mocked solver

    args = krun._parse([
        "--mode", "emit-eval-yamls",
        "--super-raw", str(super_raw),
        "--eval-dir", str(tmp_path / "eval"),
        "--thresholds-dir", str(tmp_path / "thr"),
        "--preload-pkl-override", _LIBERO10_PKL,
        "--cfg-id", _LIBERO10_CFG,
    ])
    krun._mode_emit_eval_yamls(args)

    assert captured, "expected at least one eval yaml emitted"
    for path, data in captured.items():
        assert data["backend"]["in_memory"]["preload_path"] == _LIBERO10_PKL, (
            f"emit-eval-yamls did not thread override for {path}"
        )


# ======================================================================
# 4. refine_round.rank_keybuilder forces a stem's own ranking
# ======================================================================


def _fake_results() -> dict:
    # kbA clearly best; kbB present with its own ordering.
    return {
        "cp1_spatial_pool_16__grid3_vision_0@6_vision_1@50_robot_state@44": {"success_rate": 0.74, "n": 100},
        "cp1_spatial_pool_16__grid3_vision_0@12_vision_1@44_robot_state@44": {"success_rate": 0.70, "n": 100},
        "cp1_max_pool__grid3_vision_0@31_vision_1@25_robot_state@44": {"success_rate": 0.66, "n": 100},
        "cp1_max_pool__grid3_vision_0@6_vision_1@25_robot_state@69": {"success_rate": 0.60, "n": 100},
    }


def test_rank_keybuilder_forced_stem() -> None:
    ranked = refine_round.rank_keybuilder(_fake_results(), "cp1_max_pool")
    assert all(yid.split("__")[0] == "cp1_max_pool" for _, yid, _ in ranked)
    assert ranked[0][0] == 0.66  # max_pool's own best, not the global 0.74


def test_rank_keybuilder_matches_pick_best_for_global() -> None:
    res = _fake_results()
    best_stem, ranked_pick = refine_round.pick_best_keybuilder(res)
    ranked_force = refine_round.rank_keybuilder(res, best_stem)
    assert best_stem == "cp1_spatial_pool_16"
    assert ranked_force == ranked_pick


def test_rank_keybuilder_unknown_stem_raises() -> None:
    with pytest.raises(ValueError):
        refine_round.rank_keybuilder(_fake_results(), "cp1_does_not_exist")


# ======================================================================
# 5. emit_top10 — exactly 10, deterministic, manifest
# ======================================================================


def _write_results_csv(path: Path, n_configs: int) -> list[str]:
    """Write a synthetic all_results.csv with n_configs distinct yaml_ids of
    descending SR. Returns the yaml_ids in descending-SR order.
    """
    header = "stage,keybuilder,yaml_id,v0,v1,rs,normalizer,n,success_rate\n"
    lines = [header]
    yids = []
    for i in range(n_configs):
        yid = f"cp1_spatial_pool_16__grid3_vision_0@{i:02d}_vision_1@50_robot_state@44"
        sr = 0.80 - 0.01 * i  # strictly descending
        yids.append(yid)
        lines.append(f"baseline,cp1_spatial_pool_16,{yid},0.0,0.5,0.44,zscore,100,{sr:.4f}\n")
    path.write_text("".join(lines))
    return yids


def test_emit_top10_picks_exactly_10(tmp_path: Path) -> None:
    csv_path = tmp_path / "all_results.csv"
    yids = _write_results_csv(csv_path, n_configs=13)
    agg = emit_top10.aggregate_mean_sr(csv_path)
    top10, runner_up = emit_top10.pick_top10(agg)
    assert len(top10) == 10
    assert [e["yaml_id"] for e in top10] == yids[:10]  # descending SR order
    assert runner_up["yaml_id"] == yids[10]


def test_emit_top10_tie_break_by_yaml_id(tmp_path: Path) -> None:
    # Two configs share the top SR; deterministic tie-break = yaml_id ascending.
    csv_path = tmp_path / "all_results.csv"
    header = "stage,keybuilder,yaml_id,v0,v1,rs,normalizer,n,success_rate\n"
    rows = [header]
    rows.append("baseline,kb,cp1__zzz,0,0,0,zscore,100,0.7000\n")
    rows.append("baseline,kb,cp1__aaa,0,0,0,zscore,100,0.7000\n")
    for i in range(10):
        rows.append(f"baseline,kb,cp1__m{i:02d},0,0,0,zscore,100,0.5000\n")
    csv_path.write_text("".join(rows))
    agg = emit_top10.aggregate_mean_sr(csv_path)
    top10, _ = emit_top10.pick_top10(agg)
    assert top10[0]["yaml_id"] == "cp1__aaa"  # ties broken yaml_id asc
    assert top10[1]["yaml_id"] == "cp1__zzz"


def test_emit_top10_main_stages_yamls_and_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "all_results.csv"
    yids = _write_results_csv(csv_path, n_configs=12)
    src = tmp_path / "src"
    src.mkdir()
    for y in yids:
        (src / f"{y}.yaml").write_text("enabled: true\n")
    out_dir = tmp_path / "top10"
    monkeypatch.setattr(sys, "argv", [
        "emit_top10.py",
        "--results-csv", str(csv_path),
        "--src-config-dirs", str(src),
        "--out-dir", str(out_dir),
    ])
    emit_top10.main()
    staged = sorted(p.name for p in out_dir.glob("*.yaml"))
    assert len(staged) == 10
    manifest = json.loads((out_dir / "top10_manifest.json").read_text())
    assert len(manifest["entries"]) == 10
    assert manifest["entries"][0]["rank"] == 1
    assert manifest["excluded_runner_up"]["yaml_id"] == yids[10]


# ======================================================================
# 6. emit_trajectory_yamls base union no longer pinned to 18 / overlap 4
# ======================================================================


def test_select_base_configs_overlap_not_four_ok(tmp_path: Path) -> None:
    """A results table whose group1/top10 overlap != 4 must still yield a clean
    union (the old _EXPECT_OVERLAP==4 / _EXPECT_BASE==18 asserts are relaxed).
    """
    csv_path = tmp_path / "all_results.csv"
    header = "stage,keybuilder,yaml_id,v0,v1,rs,normalizer,n,success_rate\n"
    rows = [header]
    kbs = ("cp1_spatial_pool_16", "cp1_max_pool", "cp1_mean_pool", "cp1_spatial_pool_64")
    for kb in kbs:
        for i in range(4):  # >=3 regular-grid configs per kb (A criterion)
            yid = f"{kb}__grid3_vision_0@{i:02d}_vision_1@50_robot_state@44"
            rows.append(f"baseline,{kb},{yid},0.0,0.5,0.44,zscore,100,{0.7 - 0.01 * i:.4f}\n")
    csv_path.write_text("".join(rows))

    top10_dir = tmp_path / "top10"
    top10_dir.mkdir()
    # 10 top10 ids that do NOT overlap group1 at all (overlap == 0, not 4).
    for i in range(10):
        (top10_dir / f"cp1_spatial_pool_16__grid3_vision_0@9{i}_vision_1@5_robot_state@5.yaml").write_text("x")

    group1, top10_ids, base_ids = etj.select_base_configs(csv_path, top10_dir)
    group1_ids = {g[2] for g in group1}
    assert base_ids == (group1_ids | top10_ids)  # exact union, no loss
    assert len(group1_ids & top10_ids) == 0  # overlap != 4 and that's fine
    assert len(group1_ids) <= 12


def _write_group1_csv(path: Path) -> None:
    header = "stage,keybuilder,yaml_id,v0,v1,rs,normalizer,n,success_rate\n"
    rows = [header]
    kbs = ("cp1_spatial_pool_16", "cp1_max_pool", "cp1_mean_pool", "cp1_spatial_pool_64")
    for kb in kbs:
        for i in range(4):
            yid = f"{kb}__grid3_vision_0@{i:02d}_vision_1@50_robot_state@44"
            rows.append(f"baseline,{kb},{yid},0.0,0.5,0.44,zscore,100,{0.7 - 0.01 * i:.4f}\n")
    path.write_text("".join(rows))


@pytest.mark.parametrize("n_top10", [9, 11])
def test_select_base_configs_top10_miscount_raises(tmp_path: Path, n_top10: int) -> None:
    # G2 R1 Item2: a stale / short / over-full top10 dir must fail loudly, not
    # silently change N_base.
    csv_path = tmp_path / "all_results.csv"
    _write_group1_csv(csv_path)
    top10_dir = tmp_path / "top10"
    top10_dir.mkdir()
    for i in range(n_top10):
        (top10_dir / f"cp1_spatial_pool_16__grid3_vision_0@9{i}_vision_1@5_robot_state@5.yaml").write_text("x")
    with pytest.raises(ValueError):
        etj.select_base_configs(csv_path, top10_dir)


# ======================================================================
# 7. run-eval single-server guard + emit_top10 idempotency (G2 R1)
# ======================================================================


def test_run_eval_rejects_dual_server() -> None:
    args = krun._parse(["--mode", "run-eval"])  # CLI default --servers is dual
    assert "," in args.servers  # precondition
    with pytest.raises(SystemExit):
        krun._mode_run_eval(args)


def test_run_eval_single_server_ok(capsys: pytest.CaptureFixture) -> None:
    args = krun._parse(["--mode", "run-eval", "--servers", "weiland.top:14000"])
    krun._mode_run_eval(args)
    out = capsys.readouterr().out
    assert "--servers weiland.top:14000" in out
    assert "weiland.top:14001" not in out  # only one endpoint
    assert "--server-workers 48" in out  # single-server worker count


def test_emit_top10_idempotent_clears_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    csv_path = tmp_path / "all_results.csv"
    yids = _write_results_csv(csv_path, n_configs=12)
    src = tmp_path / "src"
    src.mkdir()
    for y in yids:
        (src / f"{y}.yaml").write_text("enabled: true\n")
    out_dir = tmp_path / "top10"
    out_dir.mkdir()
    (out_dir / "STALE__old_config.yaml").write_text("stale\n")  # pre-existing junk
    monkeypatch.setattr(sys, "argv", [
        "emit_top10.py",
        "--results-csv", str(csv_path),
        "--src-config-dirs", str(src),
        "--out-dir", str(out_dir),
    ])
    emit_top10.main()
    staged = sorted(p.name for p in out_dir.glob("*.yaml"))
    assert len(staged) == 10
    assert "STALE__old_config.yaml" not in staged  # stale yaml cleared
