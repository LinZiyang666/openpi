"""Unit tests for exp.verdict_factor_judge.phase5.runner.

Coverage maps to plan §5.2 invariants R-1..R-5 + CLI/decision_gate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from exp.verdict_factor_judge.phase5 import runner as r
from exp.verdict_factor_judge.phase5.spec import (
    Cell,
    generate_all_cells,
    generate_g1_cells,
    generate_g5_cells,
)


# ----------------------------------------------------------------------
# CLI parsing
# ----------------------------------------------------------------------


def test_cli_mode_required() -> None:
    with pytest.raises(SystemExit):
        r._parse_args([])


def test_cli_groups_filter() -> None:
    args = r._parse_args(["--mode", "emit-warmup-yamls", "--groups", "g1,g3"])
    assert args.groups == ("g1", "g3")
    cells = r._resolve_cells(args)
    assert {c.group for c in cells} == {"g1", "g3"}
    assert len(cells) == 96


def test_cli_cell_ids_substring_filter() -> None:
    args = r._parse_args([
        "--mode", "emit-warmup-yamls",
        "--cell-ids", "win-3-3", "pat-1-1",
    ])
    cells = r._resolve_cells(args)
    for c in cells:
        assert any(s in c.yaml_id for s in ("win-3-3", "pat-1-1"))


def test_cli_unknown_group_rejected() -> None:
    with pytest.raises(SystemExit):
        r._parse_args(["--mode", "emit-warmup-yamls", "--groups", "g1,g9"])


def test_cli_default_groups_all_five() -> None:
    args = r._parse_args(["--mode", "emit-warmup-yamls"])
    assert args.groups == r.VALID_GROUPS
    assert len(r._resolve_cells(args)) == 240


# ----------------------------------------------------------------------
# Mode 1: emit-warmup-yamls — only G1-G4 emit, G5 reuses historical
# ----------------------------------------------------------------------


def test_emit_warmup_yamls_skips_g5(tmp_path: Path) -> None:
    args = r._parse_args([
        "--mode", "emit-warmup-yamls",
        "--warmup-yaml-dir", str(tmp_path),
        "--groups", "g5",
    ])
    r._mode_emit_warmup_yamls(args)
    assert list(tmp_path.glob("*.yaml")) == []


def test_emit_warmup_yamls_g3_dedup_per_base_channel(tmp_path: Path) -> None:
    """G3 12 patterns share warmup → expect 4 yamls (2 base × 2 channel)."""
    args = r._parse_args([
        "--mode", "emit-warmup-yamls",
        "--warmup-yaml-dir", str(tmp_path),
        "--groups", "g3",
    ])
    r._mode_emit_warmup_yamls(args)
    yamls = sorted(tmp_path.glob("*.yaml"))
    assert len(yamls) == 4


def test_emit_warmup_yamls_g1_one_per_cell(tmp_path: Path) -> None:
    args = r._parse_args([
        "--mode", "emit-warmup-yamls",
        "--warmup-yaml-dir", str(tmp_path),
        "--groups", "g1",
    ])
    r._mode_emit_warmup_yamls(args)
    yamls = sorted(tmp_path.glob("*.yaml"))
    assert len(yamls) == 48


# ----------------------------------------------------------------------
# Mode 3: emit-eval-yamls — solver path uses composer_weights
# ----------------------------------------------------------------------


def _write_synth_warmup_jsonl(path: Path, declared: list[str], n: int = 80) -> None:
    """Synth warmup with per-key monotone ramps so weight changes shift scores."""
    rows = []
    for i in range(n):
        raw = {k: ((i + j * 7) % n) / n for j, k in enumerate(declared)}
        rows.append({"factor_raw": raw})
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_solve_thresholds_phase5_uses_cell_weights(tmp_path: Path) -> None:
    """Two cells with the same factor list but different weight vectors
    must produce different (FH_thr, WS_thr). Catches a silent regression
    to solve_recipe / equal weights.
    """
    # Take a G3 cell — it has 10 declared keys.
    cells = [c for c in generate_all_cells() if c.group == "g3"]
    cell_a = cells[0]   # uniform pattern
    # Find a non-uniform sibling (same warmup) to share raw jsonl.
    cell_b = next(
        c for c in cells
        if c.warmup_yaml_id == cell_a.warmup_yaml_id and c.weights != cell_a.weights
    )

    raw_dir = tmp_path / "warmup_factor_raw"
    raw_dir.mkdir()
    raw_jsonl = raw_dir / f"{cell_a.warmup_yaml_id}.jsonl"
    _write_synth_warmup_jsonl(raw_jsonl, list(cell_a.declared_keys), n=80)

    args = r._parse_args([
        "--mode", "emit-eval-yamls",
        "--warmup-jsonl-dir", str(raw_dir),
    ])
    fh_a, ws_a = r._solve_thresholds_phase5(cell_a, args)
    fh_b, ws_b = r._solve_thresholds_phase5(cell_b, args)
    assert (fh_a, ws_a) != (fh_b, ws_b), (
        f"thresholds did not shift between weight vectors: "
        f"{cell_a.yaml_id}={fh_a, ws_a}  {cell_b.yaml_id}={fh_b, ws_b}"
    )


# ----------------------------------------------------------------------
# R-1..R-5 — _run_one_cell_phase5 invariants
# ----------------------------------------------------------------------


class _MockCtl:
    """Minimal ctl mock for _run_one_cell_phase5; records call order."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.preload_yaml_id: str | None = None
        self.preload_buffer: Any = None
        self.load_yaml_id: str | None = None
        self.load_yaml_content: str | None = None

    def preload_normalizer_buffer(self, yaml_id: str, buffer: Any) -> dict:
        self.calls.append(("preload", yaml_id))
        self.preload_yaml_id = yaml_id
        self.preload_buffer = buffer
        return {"n_keys": len(buffer)}

    def load_cache_config(self, *, yaml_content: str, yaml_id: str) -> None:
        self.calls.append(("load", yaml_id))
        self.load_yaml_id = yaml_id
        self.load_yaml_content = yaml_content


def _setup_run_one_cell_env(
    tmp_path: Path, cell: Cell,
    monkeypatch: pytest.MonkeyPatch,
) -> r.Args:
    """Prep eval yaml + raw jsonl + monkeypatch subprocess + libero argv."""
    # Eval yaml
    eval_dir = tmp_path / "eval_yamls"
    eval_dir.mkdir()
    yaml_dict = r.build_eval_yaml_for_cell("spatial16_w8_d4", cell, fh_thr=0.42, ws_thr=0.18)
    import yaml as _y
    (eval_dir / f"{cell.yaml_id}.yaml").write_text(_y.safe_dump(yaml_dict))

    # Raw jsonl
    raw_dir = tmp_path / "warmup_factor_raw"
    raw_dir.mkdir()
    if cell.group == "g5":
        # G5 g6 → phase3 dir; p1/p2 → phase4 dir
        if cell.base_recipe in ("p1", "p2"):
            phase4_raw_dir = tmp_path / "phase4_raw"
            phase4_raw_dir.mkdir()
            recipe_id = cell.warmup_yaml_id.replace("spatial16_w8_d4_phase4_", "").removesuffix("__warmup")
            jsonl = phase4_raw_dir / f"{recipe_id}.jsonl"
        else:
            phase3_raw_dir = tmp_path / "phase3_warmup"
            phase3_raw_dir.mkdir()
            jsonl = phase3_raw_dir / f"{cell.warmup_yaml_id}.jsonl"
    else:
        jsonl = raw_dir / f"{cell.warmup_yaml_id}.jsonl"
    _write_synth_warmup_jsonl(jsonl, list(cell.declared_keys), n=60)

    # Stub subprocess + libero argv
    monkeypatch.setattr(r.subprocess, "run", lambda *a, **kw: None)
    monkeypatch.setattr(r, "_build_libero_argv", lambda **kw: ([], {}))
    monkeypatch.setattr(r, "_summarize_per_step_log", lambda *a, **kw: {
        "n_eval_verdicts": 100, "n_full_hit": 30, "n_warm_start": 50, "n_miss": 20,
    })
    monkeypatch.setattr(r, "_aggregate_sr_from_episode_json", lambda *a, **kw: 0.92)

    args_list = [
        "--mode", "run-eval",
        "--eval-yaml-dir", str(eval_dir),
        "--warmup-jsonl-dir", str(raw_dir),
        "--summary-out", str(tmp_path / "summary.jsonl"),
        "--episode-results-dir", str(tmp_path / "episode_results"),
        "--per-step-log-dir", str(tmp_path / "per_step"),
    ]
    if cell.group == "g5" and cell.base_recipe in ("p1", "p2"):
        args_list.extend(["--phase4-warmup-raw-dir", str(tmp_path / "phase4_raw")])
    elif cell.group == "g5" and cell.base_recipe == "g6":
        args_list.extend(["--phase3-warmup-dir", str(tmp_path / "phase3_warmup")])
    return r._parse_args(args_list)


def test_R1_preload_before_load_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cell = generate_g1_cells()[0]
    args = _setup_run_one_cell_env(tmp_path, cell, monkeypatch)
    ctl = _MockCtl()
    summary_path = Path(args.summary_out)
    r._run_one_cell_phase5(ctl, cell, args, summary_path)
    # First call must be preload, second must be load.
    assert ctl.calls[0][0] == "preload"
    assert ctl.calls[1][0] == "load"
    assert ctl.calls[0][1] == cell.yaml_id
    assert ctl.calls[1][1] == cell.yaml_id


def test_R2_cell_declared_keys_drives_buffer_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Buffer passed to preload must be keyed by cell.declared_keys
    exactly — NOT by RECIPES_PHASE4 lookup.
    """
    cell = generate_g1_cells()[0]
    args = _setup_run_one_cell_env(tmp_path, cell, monkeypatch)
    ctl = _MockCtl()
    r._run_one_cell_phase5(ctl, cell, args, Path(args.summary_out))
    # preload_buffer is dict[str, list[float]]; its keys must equal declared_keys.
    assert set(ctl.preload_buffer.keys()) == set(cell.declared_keys)


def test_R3_raw_source_resolution_g5_p1_uses_phase4_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """G5 p1 cell must read from phase4_warmup_raw_dir, not phase5 own dir."""
    cell = next(c for c in generate_g5_cells() if c.base_recipe == "p1")
    args = _setup_run_one_cell_env(tmp_path, cell, monkeypatch)
    ctl = _MockCtl()
    r._run_one_cell_phase5(ctl, cell, args, Path(args.summary_out))
    # Verify buffer has keys (raw was successfully read from phase4 dir)
    assert len(ctl.preload_buffer) > 0


def test_R3_raw_source_resolution_g5_g6_uses_phase3_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = next(c for c in generate_g5_cells() if c.base_recipe == "g6")
    args = _setup_run_one_cell_env(tmp_path, cell, monkeypatch)
    ctl = _MockCtl()
    r._run_one_cell_phase5(ctl, cell, args, Path(args.summary_out))
    assert len(ctl.preload_buffer) > 0


def test_R3_raw_source_resolution_g1_uses_phase5_own_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = generate_g1_cells()[0]
    args = _setup_run_one_cell_env(tmp_path, cell, monkeypatch)
    ctl = _MockCtl()
    r._run_one_cell_phase5(ctl, cell, args, Path(args.summary_out))
    assert len(ctl.preload_buffer) > 0


def test_R4_summary_contains_phase5_specific_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    cell = generate_g1_cells()[0]
    args = _setup_run_one_cell_env(tmp_path, cell, monkeypatch)
    ctl = _MockCtl()
    r._run_one_cell_phase5(ctl, cell, args, Path(args.summary_out))
    rows = [json.loads(l) for l in Path(args.summary_out).read_text().splitlines() if l.strip()]
    assert len(rows) == 1
    row = rows[0]
    # Required phase5 fields
    for field in ("group", "base_recipe", "axis_tag"):
        assert field in row, f"missing phase5 summary field: {field}"
    # Forbidden phase4-specific fields
    for forbidden in ("round_id", "alpha", "pattern_label", "alpha_star_for_round"):
        assert forbidden not in row, f"phase5 summary contains phase4-only field: {forbidden}"


def test_R5_runs_with_RECIPES_PHASE4_emptied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hard guarantee: phase5 helper does not depend on phase4 globals.
    Empty RECIPES_PHASE4 mid-run; G1 cell must still execute fine.
    """
    monkeypatch.setattr("exp.verdict_factor_judge.phase4.spec.RECIPES_PHASE4", {})
    cell = generate_g1_cells()[0]
    args = _setup_run_one_cell_env(tmp_path, cell, monkeypatch)
    ctl = _MockCtl()
    r._run_one_cell_phase5(ctl, cell, args, Path(args.summary_out))
    assert len(ctl.calls) >= 2   # preload + load happened


# ----------------------------------------------------------------------
# Decision gate routing
# ----------------------------------------------------------------------


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows))


def test_decision_gate_g1_winner_5pp_threshold(tmp_path: Path) -> None:
    """G1 same (base, channel, desc) bucket: 5pp gap → winner."""
    summary = tmp_path / "summary.jsonl"
    _write_summary(summary, [
        # Same bucket, top1 vs top2 = 5pp
        {"yaml_id": "a", "group": "g1", "base_recipe": "p1",
         "axis_tag": "p1_state_jerk__win-3-3", "success_rate": 0.96,
         "n_eval_verdicts": 100, "n_full_hit": 30, "n_warm_start": 50, "n_miss": 20},
        {"yaml_id": "b", "group": "g1", "base_recipe": "p1",
         "axis_tag": "p1_state_jerk__win-0-3", "success_rate": 0.91,
         "n_eval_verdicts": 100, "n_full_hit": 30, "n_warm_start": 50, "n_miss": 20},
    ])
    gate = r._dump_decision_gate_table_phase5(summary)
    bucket = gate["g1"][str(("p1", "state", "jerk"))]
    assert bucket["winner"] == "a"
    assert bucket["delta"] == pytest.approx(0.05, abs=1e-9)


def test_decision_gate_g1_inconclusive_below_5pp(tmp_path: Path) -> None:
    summary = tmp_path / "summary.jsonl"
    _write_summary(summary, [
        {"yaml_id": "a", "group": "g1", "base_recipe": "p1",
         "axis_tag": "p1_state_jerk__win-3-3", "success_rate": 0.93,
         "n_eval_verdicts": 100, "n_full_hit": 30, "n_warm_start": 50, "n_miss": 20},
        {"yaml_id": "b", "group": "g1", "base_recipe": "p1",
         "axis_tag": "p1_state_jerk__win-0-3", "success_rate": 0.91,
         "n_eval_verdicts": 100, "n_full_hit": 30, "n_warm_start": 50, "n_miss": 20},
    ])
    gate = r._dump_decision_gate_table_phase5(summary)
    bucket = gate["g1"][str(("p1", "state", "jerk"))]
    assert bucket["winner"] is None
    assert "inconclusive" in bucket["reason"]


def test_decision_gate_g5_pareto_frontier(tmp_path: Path) -> None:
    """G5 outputs Pareto frontier per recipe, not winner."""
    summary = tmp_path / "summary.jsonl"
    _write_summary(summary, [
        {"yaml_id": "a", "group": "g5", "base_recipe": "p1",
         "axis_tag": "p1__fh0.5_ws0.5", "success_rate": 0.92,
         "n_eval_verdicts": 100, "n_full_hit": 30, "n_warm_start": 50, "n_miss": 20},
        {"yaml_id": "b", "group": "g5", "base_recipe": "p1",
         "axis_tag": "p1__fh0.3_ws0.5", "success_rate": 0.95,
         "n_eval_verdicts": 100, "n_full_hit": 60, "n_warm_start": 30, "n_miss": 10},
    ])
    gate = r._dump_decision_gate_table_phase5(summary)
    assert "g5" in gate
    p1_meta = gate["g5"]["p1"]
    assert p1_meta["best_sr"] == 0.95
    assert p1_meta["best_sr_yaml"] == "b"
    assert isinstance(p1_meta["frontier"], list)


def test_decision_gate_writes_5_per_group_files(tmp_path: Path) -> None:
    """Plan §4.1..§4.5: 5 separate decision files (g1..g5_decision.json).

    G2 R1 B3 regression guard: must NOT collapse to a single
    decision_gate.json aggregate file.
    """
    summary = tmp_path / "summary.jsonl"
    _write_summary(summary, [
        {"yaml_id": "a", "group": "g1", "base_recipe": "p1",
         "axis_tag": "p1_state_jerk__win-3-3", "success_rate": 0.93,
         "n_eval_verdicts": 100, "n_full_hit": 30, "n_warm_start": 50, "n_miss": 20},
        {"yaml_id": "z", "group": "g5", "base_recipe": "p1",
         "axis_tag": "p1__fh0.5_ws0.5", "success_rate": 0.94,
         "n_eval_verdicts": 100, "n_full_hit": 30, "n_warm_start": 50, "n_miss": 20},
    ])
    r._dump_decision_gate_table_phase5(summary)
    for g in ("g1", "g2", "g3", "g4", "g5"):
        out = summary.parent / f"{g}_decision.json"
        assert out.exists(), f"plan-required file missing: {out}"
        # Empty groups still produce file (with empty payload).
        json.loads(out.read_text())
