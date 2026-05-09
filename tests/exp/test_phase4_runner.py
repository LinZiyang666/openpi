"""Tests for exp/verdict_factor_judge/run_phase4.py.

Covers:
  Group A — Per-recipe map parser (CLI dispatch)
  Group B — CLI invariants per mode
  Group C — Cell list construction (round shape + per-recipe locked cells)
  Group D — preload-before-load ordering invariant
  Group E — Round-specific decision_gate selection rules
  Group F — Resume helper
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from exp.verdict_factor_judge.phase4_spec import (
    R1_ALPHAS,
    R2_OFFLINE_PATTERNS,
    R3_ONLINE_PATTERNS,
    RECIPES_PHASE4,
    cell_id_r1,
    warmup_yaml_id,
)
from exp.verdict_factor_judge.run_phase4 import (
    Args,
    Cell,
    _build_cell_list,
    _dump_decision_gate_table,
    _run_one_cell,
    _validate_cli_invariants,
    parse_per_recipe_map,
)
from exp.verdict_factor_judge.run_phase3 import _load_done_yaml_ids


# ----------------------------------------------------------------------
# Group A — per-recipe map parser
# ----------------------------------------------------------------------


def test_parse_per_recipe_alpha_map_distinct_values() -> None:
    out = parse_per_recipe_map(
        "p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6",
        valid_values=set(R1_ALPHAS), recipes=RECIPES_PHASE4,
    )
    assert out == {
        "p1_state_fut_online_act": 0.4,
        "p2_action_fut_online_act": 0.6,
    }


def test_parse_per_recipe_pattern_map_string_values() -> None:
    out = parse_per_recipe_map(
        "p1_state_fut_online_act=jerk-heavy,p2_action_fut_online_act=uniform",
        valid_values=set(R2_OFFLINE_PATTERNS), recipes=RECIPES_PHASE4,
    )
    assert out == {
        "p1_state_fut_online_act": "jerk-heavy",
        "p2_action_fut_online_act": "uniform",
    }


def test_parse_per_recipe_map_empty_returns_empty() -> None:
    assert parse_per_recipe_map(
        None, valid_values=set(R1_ALPHAS), recipes=RECIPES_PHASE4,
    ) == {}
    assert parse_per_recipe_map(
        "", valid_values=set(R1_ALPHAS), recipes=RECIPES_PHASE4,
    ) == {}


def test_parse_per_recipe_map_rejects_unknown_recipe() -> None:
    with pytest.raises(ValueError, match="unknown recipe id"):
        parse_per_recipe_map(
            "p99_bogus=0.4",
            valid_values=set(R1_ALPHAS), recipes=RECIPES_PHASE4,
        )


def test_parse_per_recipe_map_rejects_invalid_value() -> None:
    with pytest.raises(ValueError, match="not in valid set"):
        parse_per_recipe_map(
            "p1_state_fut_online_act=999.0",
            valid_values=set(R1_ALPHAS), recipes=RECIPES_PHASE4,
        )


# ----------------------------------------------------------------------
# Group B — CLI invariants per mode
# ----------------------------------------------------------------------


def test_cli_round1_emit_eval_does_not_require_alpha_star() -> None:
    """R1 emit-eval-yamls / run-eval do not need --alpha-star (R1 sweeps alpha)."""
    _validate_cli_invariants(Args(mode="emit-eval-yamls", round=1))
    _validate_cli_invariants(Args(mode="run-eval", round=1))


def test_cli_round2_requires_alpha_star() -> None:
    with pytest.raises(ValueError, match="alpha-star"):
        _validate_cli_invariants(Args(mode="emit-eval-yamls", round=2))


def test_cli_round3_requires_offline_pattern() -> None:
    with pytest.raises(ValueError, match="offline-pattern"):
        _validate_cli_invariants(Args(
            mode="emit-eval-yamls", round=3,
            alpha_star="p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6",
        ))


def test_cli_round4_requires_online_pattern() -> None:
    with pytest.raises(ValueError, match="online-pattern"):
        _validate_cli_invariants(Args(
            mode="emit-eval-yamls", round=4,
            alpha_star="p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6",
            offline_pattern="p1_state_fut_online_act=uniform,p2_action_fut_online_act=uniform",
        ))


def test_cli_emit_warmup_yaml_no_other_required_args() -> None:
    """emit-warmup-yaml + run-warmup never require alpha/pattern flags."""
    _validate_cli_invariants(Args(mode="emit-warmup-yaml", round=1))
    _validate_cli_invariants(Args(mode="run-warmup", round=1))


# ----------------------------------------------------------------------
# Group C — Cell list construction
# ----------------------------------------------------------------------


def test_round1_builds_14_cells() -> None:
    cells = _build_cell_list(Args(mode="emit-eval-yamls", round=1))
    assert len(cells) == 2 * 7   # 2 recipes x 7 alphas


def test_round2_builds_18_cells() -> None:
    cells = _build_cell_list(Args(
        mode="emit-eval-yamls", round=2,
        alpha_star="p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6",
    ))
    assert len(cells) == 2 * 9   # 2 recipes x 9 offline patterns


def test_round3_builds_10_cells() -> None:
    cells = _build_cell_list(Args(
        mode="emit-eval-yamls", round=3,
        alpha_star="p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6",
        offline_pattern="p1_state_fut_online_act=uniform,p2_action_fut_online_act=uniform",
    ))
    assert len(cells) == 2 * 5


def test_round4_builds_10_cells() -> None:
    cells = _build_cell_list(Args(
        mode="emit-eval-yamls", round=4,
        alpha_star="p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6",
        offline_pattern="p1_state_fut_online_act=uniform,p2_action_fut_online_act=uniform",
        online_pattern="p1_state_fut_online_act=uniform,p2_action_fut_online_act=uniform",
    ))
    assert len(cells) == 2 * 5


def test_round4_recipe_filter_yields_5_cells() -> None:
    """--recipe restriction (used for R4 single-recipe trigger) yields 5 cells."""
    cells = _build_cell_list(Args(
        mode="emit-eval-yamls", round=4,
        alpha_star="p1_state_fut_online_act=0.4",
        offline_pattern="p1_state_fut_online_act=uniform",
        online_pattern="p1_state_fut_online_act=uniform",
        recipe="p1_state_fut_online_act",
    ))
    assert len(cells) == 5
    assert all(c.recipe_id == "p1_state_fut_online_act" for c in cells)


def test_p1_cells_locked_at_05_05() -> None:
    """All p1 cells across all rounds are at fh_ratio=0.5, ws_ratio=0.5."""
    args = Args(mode="emit-eval-yamls", round=1)
    cells = [c for c in _build_cell_list(args) if c.recipe_id == "p1_state_fut_online_act"]
    assert all(c.fh_ratio == 0.5 and c.ws_ratio == 0.5 for c in cells)


def test_p2_cells_locked_at_05_04() -> None:
    """All p2 cells across all rounds are at fh_ratio=0.5, ws_ratio=0.4."""
    args = Args(mode="emit-eval-yamls", round=1)
    cells = [c for c in _build_cell_list(args) if c.recipe_id == "p2_action_fut_online_act"]
    assert all(c.fh_ratio == 0.5 and c.ws_ratio == 0.4 for c in cells)


def test_per_recipe_alpha_distributes_correctly() -> None:
    """R2 build with p1=0.4, p2=0.6 -> p1 cells embed alpha=0.4, p2 cells embed 0.6."""
    cells = _build_cell_list(Args(
        mode="emit-eval-yamls", round=2,
        alpha_star="p1_state_fut_online_act=0.4,p2_action_fut_online_act=0.6",
    ))
    p1 = [c for c in cells if c.recipe_id == "p1_state_fut_online_act"]
    p2 = [c for c in cells if c.recipe_id == "p2_action_fut_online_act"]
    assert all(c.alpha == 0.4 for c in p1)
    assert all(c.alpha == 0.6 for c in p2)


# ----------------------------------------------------------------------
# Group D — preload-before-load ordering (G1 R2 Blocking 4 invariant)
# ----------------------------------------------------------------------


class _RecordingCtl:
    """Mock ctl that records call order; emulates phase 3's invariant
    that preload_normalizer_buffer fires before load_cache_config."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    def preload_normalizer_buffer(self, yaml_id: str, buffer: dict) -> dict:
        self.calls.append(("preload", {"yaml_id": yaml_id, "n_keys": len(buffer)}))
        return {"ok": True, "n_keys": len(buffer)}

    def load_cache_config(self, *, yaml_content: str, yaml_id: str) -> dict:
        self.calls.append(("load", {"yaml_id": yaml_id}))
        return {"ok": True}


def test_run_one_cell_calls_preload_before_load(tmp_path: Path, monkeypatch) -> None:
    """preload_normalizer_buffer MUST run before load_cache_config — phase3
    invariant the server's WarmupPool keying depends on. Documented by
    run_phase3.py:370-385."""
    rid = "p1_state_fut_online_act"

    # Stub warmup factor_raw cache file (4 rows over 10 declared keys).
    raw_dir = tmp_path / "warmup_factor_raw"
    raw_dir.mkdir()
    raw_path = raw_dir / f"{rid}.jsonl"
    declared = list(RECIPES_PHASE4[rid]["declared_keys"])
    rows = []
    for i in range(50):
        rows.append({"factor_raw": {k: float(i) / 100.0 for k in declared}})
    raw_path.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(
        "exp.verdict_factor_judge.run_phase4._warmup_raw_dir", lambda: raw_dir,
    )

    # Stub eval yaml file
    eval_yaml = tmp_path / f"{cell_id_r1('c', rid, 0.5)}.yaml"
    eval_yaml.write_text("enabled: true\n")
    cell = Cell(
        yaml_id=cell_id_r1("c", rid, 0.5),
        yaml_path=eval_yaml,
        recipe_id=rid, round_id=1, pattern_label="a0.5",
        weights={k: 0.1 for k in declared}, fh_thr=0.5, ws_thr=0.3,
        fh_ratio=0.5, ws_ratio=0.5, alpha=0.5,
        offline_pattern_name=None, online_pattern_name=None,
        window_pattern_name=None,
    )

    # Stub libero subprocess + summary helpers
    monkeypatch.setattr(
        "exp.verdict_factor_judge.run_phase4.subprocess.run",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        "exp.verdict_factor_judge.run_phase4._build_libero_argv",
        lambda **kw: ([], {}),
    )
    monkeypatch.setattr(
        "exp.verdict_factor_judge.run_phase4._summarize_per_step_log",
        lambda *a, **kw: {"n_eval_verdicts": 0, "n_full_hit": 0, "n_warm_start": 0, "n_miss": 0},
    )
    monkeypatch.setattr(
        "exp.verdict_factor_judge.run_phase4._aggregate_sr_from_episode_json",
        lambda *a, **kw: None,
    )

    args = Args(mode="run-eval", round=1, per_step_log_dir=str(tmp_path))
    summary_path = tmp_path / "summary.jsonl"
    ctl = _RecordingCtl()
    _run_one_cell(ctl, cell, args, summary_path)

    # First call must be preload, second must be load — same yaml_id.
    assert ctl.calls[0][0] == "preload"
    assert ctl.calls[1][0] == "load"
    assert ctl.calls[0][1]["yaml_id"] == ctl.calls[1][1]["yaml_id"] == cell.yaml_id


# ----------------------------------------------------------------------
# Group E — Round-specific decision_gate
# ----------------------------------------------------------------------


def _make_summary_row(
    *, recipe_id: str, alpha: float, pattern_label: str,
    success_rate: float, n_full_hit: int = 50, n_warm_start: int = 50,
    n_miss: int = 0, offline_pattern: str | None = None,
    online_pattern: str | None = None,
) -> dict:
    n = n_full_hit + n_warm_start + n_miss
    return {
        "yaml_id": f"{recipe_id}__{pattern_label}",
        "recipe_id": recipe_id,
        "alpha": alpha,
        "pattern_label": pattern_label,
        "success_rate": success_rate,
        "n_eval_verdicts": n,
        "n_full_hit": n_full_hit,
        "n_warm_start": n_warm_start,
        "n_miss": n_miss,
        "offline_pattern": offline_pattern,
        "online_pattern": online_pattern,
    }


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_decision_gate_r1_picks_argmax_score(tmp_path: Path) -> None:
    """R1 winner per recipe = argmax (SR - 0.5*inf)."""
    summary_path = tmp_path / "per_yaml_summary.jsonl"
    rows = [
        _make_summary_row(recipe_id="p1_state_fut_online_act", alpha=a,
                          pattern_label=f"a{a:.1f}",
                          success_rate=0.85 + 0.02 * (a == 0.4))
        for a in R1_ALPHAS
    ]
    _write_summary(summary_path, rows)
    out = _dump_decision_gate_table(round_id=1, summary_path=summary_path)
    winner = out["winners"]["p1_state_fut_online_act"]
    assert winner["alpha"] == 0.4    # only alpha=0.4 has the +0.02 bump
    decision = out["trigger_decisions"]["p1_state_fut_online_act"]
    assert decision["continue"] is False    # SR 0.87 < 0.95 - 0.02 = 0.93


def test_decision_gate_r1_continue_flag_respects_anchor(tmp_path: Path) -> None:
    summary_path = tmp_path / "per_yaml_summary.jsonl"
    # All cells with SR=0.96 — well above p1 anchor 0.95 - 2pp.
    rows = [
        _make_summary_row(recipe_id="p1_state_fut_online_act", alpha=a,
                          pattern_label=f"a{a:.1f}", success_rate=0.96)
        for a in R1_ALPHAS
    ]
    _write_summary(summary_path, rows)
    out = _dump_decision_gate_table(round_id=1, summary_path=summary_path)
    assert out["trigger_decisions"]["p1_state_fut_online_act"]["continue"] is True


def test_decision_gate_r2_forces_uniform_when_delta_below_2pp(tmp_path: Path) -> None:
    """R2 with all patterns at SR ~0.93 must force pattern*=uniform."""
    summary_path = tmp_path / "per_yaml_summary.jsonl"
    rows: list[dict] = []
    for pat_name in R2_OFFLINE_PATTERNS:
        # pat-uniform = 0.93; jerk-heavy = 0.94 (delta = 1pp, below 2pp)
        sr = 0.93 + (0.01 if pat_name == "jerk-heavy" else 0.0)
        rows.append(_make_summary_row(
            recipe_id="p1_state_fut_online_act", alpha=0.5,
            pattern_label=f"a0.5__off-{pat_name}",
            success_rate=sr, offline_pattern=pat_name,
        ))
    _write_summary(summary_path, rows)
    out = _dump_decision_gate_table(round_id=2, summary_path=summary_path)
    winner = out["winners"]["p1_state_fut_online_act"]
    assert winner["offline_pattern"] == "uniform"
    assert "forcing pattern*=uniform" in out["trigger_decisions"]["p1_state_fut_online_act"]["reason"]


def test_decision_gate_r2_picks_argmax_when_delta_above_2pp(tmp_path: Path) -> None:
    summary_path = tmp_path / "per_yaml_summary.jsonl"
    rows: list[dict] = []
    for pat_name in R2_OFFLINE_PATTERNS:
        # jerk-heavy SR=0.97, others 0.92 -> delta 5pp > 2pp.
        sr = 0.97 if pat_name == "jerk-heavy" else 0.92
        rows.append(_make_summary_row(
            recipe_id="p1_state_fut_online_act", alpha=0.5,
            pattern_label=f"a0.5__off-{pat_name}",
            success_rate=sr, offline_pattern=pat_name,
        ))
    _write_summary(summary_path, rows)
    out = _dump_decision_gate_table(round_id=2, summary_path=summary_path)
    winner = out["winners"]["p1_state_fut_online_act"]
    assert winner["offline_pattern"] == "jerk-heavy"


def test_decision_gate_r3_emits_r4_trigger_flag(tmp_path: Path) -> None:
    """R3 trigger_decisions[rid].continue = (R3 winner.SR > R3 uniform.SR + 2pp)."""
    summary_path = tmp_path / "per_yaml_summary.jsonl"
    rows: list[dict] = []
    for pat_name in R3_ONLINE_PATTERNS:
        sr = 0.97 if pat_name == "jerk-heavy" else 0.93
        rows.append(_make_summary_row(
            recipe_id="p1_state_fut_online_act", alpha=0.5,
            pattern_label=f"a0.5__off-uniform__on-{pat_name}",
            success_rate=sr, offline_pattern="uniform", online_pattern=pat_name,
        ))
    _write_summary(summary_path, rows)
    out = _dump_decision_gate_table(round_id=3, summary_path=summary_path)
    decision = out["trigger_decisions"]["p1_state_fut_online_act"]
    # winner SR=0.97 vs uniform SR=0.93 -> delta=4pp > 2pp -> R4 trigger TRUE
    assert decision["continue"] is True
    assert out["next_args_suggestion"]["online-pattern"] is not None


def test_decision_gate_next_args_round_trip(tmp_path: Path) -> None:
    """The next_args_suggestion in decision_gate.json round-trips through
    parse_per_recipe_map without raising, producing a valid Cell list."""
    summary_path = tmp_path / "per_yaml_summary.jsonl"
    # R1 with p1.alpha=0.4 winner, p2.alpha=0.6 winner.
    rows = []
    for rid, winning_alpha in [
        ("p1_state_fut_online_act", 0.4),
        ("p2_action_fut_online_act", 0.6),
    ]:
        for a in R1_ALPHAS:
            sr = 0.96 if a == winning_alpha else 0.85
            rows.append(_make_summary_row(
                recipe_id=rid, alpha=a, pattern_label=f"a{a:.1f}",
                success_rate=sr,
            ))
    _write_summary(summary_path, rows)
    out = _dump_decision_gate_table(round_id=1, summary_path=summary_path)
    suggestion = out["next_args_suggestion"]["alpha-star"]
    parsed = parse_per_recipe_map(
        suggestion, valid_values=set(R1_ALPHAS), recipes=RECIPES_PHASE4,
    )
    assert parsed["p1_state_fut_online_act"] == 0.4
    assert parsed["p2_action_fut_online_act"] == 0.6
    # And the parsed map drives a valid R2 cell list.
    cells = _build_cell_list(Args(
        mode="emit-eval-yamls", round=2, alpha_star=suggestion,
    ))
    assert len(cells) == 18


# ----------------------------------------------------------------------
# Group F — Resume helper (load_done_yaml_ids reused from phase3)
# ----------------------------------------------------------------------


def test_load_done_yaml_ids_skips_processed_rows(tmp_path: Path) -> None:
    summary_path = tmp_path / "per_yaml_summary.jsonl"
    rows = [
        {"yaml_id": "cell_a"}, {"yaml_id": "cell_b"}, {"yaml_id": "cell_c"},
    ]
    _write_summary(summary_path, rows)
    done = _load_done_yaml_ids(summary_path)
    assert done == {"cell_a", "cell_b", "cell_c"}


# ----------------------------------------------------------------------
# Group G — G2 R1 fixes: per-step default + gate-safe next_args
# ----------------------------------------------------------------------


from exp.verdict_factor_judge.run_phase4 import _apply_default_data_paths


def test_run_eval_per_step_log_dir_defaults_to_round_specific_path() -> None:
    """G2 R1 Blocking 1: --mode run-eval with empty per_step_log_dir must
    auto-fill to data/phase4/r{N}_*/per_step. Without this default the
    summary rows have n_eval_verdicts=0 -> _compute_inf returns 0.0 ->
    R1 argmax(SR - 0.5*inf) silently degrades to argmax(SR)."""
    args = Args(mode="run-eval", round=1)
    assert args.per_step_log_dir == ""           # CLI default
    filled = _apply_default_data_paths(args)
    assert filled.per_step_log_dir != ""
    assert "r1_alpha/per_step" in filled.per_step_log_dir
    assert filled.episode_results_dir != ""
    assert "r1_alpha/episode_results" in filled.episode_results_dir


def test_run_eval_per_step_log_dir_respects_explicit_value() -> None:
    """When operator passes --per-step-log-dir, default fill must not overwrite."""
    args = Args(mode="run-eval", round=2, per_step_log_dir="/tmp/custom-per-step")
    filled = _apply_default_data_paths(args)
    assert filled.per_step_log_dir == "/tmp/custom-per-step"


def test_apply_default_data_paths_skips_non_run_eval_modes() -> None:
    """Default fill is gated by mode: emit-warmup-yaml / run-warmup /
    emit-eval-yamls do not need per-step paths and must not gain them."""
    for mode in ("emit-warmup-yaml", "run-warmup", "emit-eval-yamls"):
        a = Args(mode=mode, round=1)
        out = _apply_default_data_paths(a)
        assert out.per_step_log_dir == ""
        assert out.episode_results_dir == ""


def test_decision_gate_raises_when_all_rows_have_zero_verdicts(tmp_path: Path) -> None:
    """Fail-fast guard: every n_eval_verdicts=0 means per_step_log_dir was
    empty; do not produce a silently broken decision gate."""
    summary_path = tmp_path / "per_yaml_summary.jsonl"
    rows = [
        _make_summary_row(
            recipe_id="p1_state_fut_online_act", alpha=a,
            pattern_label=f"a{a:.1f}", success_rate=0.9,
            n_full_hit=0, n_warm_start=0, n_miss=0,
        )
        for a in R1_ALPHAS
    ]
    _write_summary(summary_path, rows)
    with pytest.raises(RuntimeError, match="n_eval_verdicts=0"):
        _dump_decision_gate_table(round_id=1, summary_path=summary_path)


def test_decision_gate_r1_next_args_excludes_aborted_recipe(tmp_path: Path) -> None:
    """G2 R1 Blocking 2: R1 with one recipe pass + one fail must emit
    alpha-star only for the passing recipe, set ``recipe`` to that single
    recipe id, and produce a single-recipe ``cli_command``."""
    summary_path = tmp_path / "per_yaml_summary.jsonl"
    rows = []
    # p1 winner = alpha 0.5 with SR 0.96 -> passes anchor 0.95 - 2pp.
    for a in R1_ALPHAS:
        sr = 0.96 if a == 0.5 else 0.85
        rows.append(_make_summary_row(
            recipe_id="p1_state_fut_online_act", alpha=a,
            pattern_label=f"a{a:.1f}", success_rate=sr,
        ))
    # p2 winner with SR 0.80 -> fails anchor 0.96 - 2pp = 0.94.
    for a in R1_ALPHAS:
        rows.append(_make_summary_row(
            recipe_id="p2_action_fut_online_act", alpha=a,
            pattern_label=f"a{a:.1f}", success_rate=0.80,
        ))
    _write_summary(summary_path, rows)
    out = _dump_decision_gate_table(round_id=1, summary_path=summary_path)

    # p1 passes, p2 fails.
    assert out["trigger_decisions"]["p1_state_fut_online_act"]["continue"] is True
    assert out["trigger_decisions"]["p2_action_fut_online_act"]["continue"] is False

    # alpha-star contains only p1 (p2 aborted).
    suggestion = out["next_args_suggestion"]
    assert "p1_state_fut_online_act" in (suggestion["alpha-star"] or "")
    assert "p2_action_fut_online_act" not in (suggestion["alpha-star"] or "")

    # active_recipes / recipe / cli_command reflect the single-recipe path.
    assert suggestion["active_recipes"] == ["p1_state_fut_online_act"]
    assert suggestion["recipe"] == "p1_state_fut_online_act"
    assert "--recipe p1_state_fut_online_act" in (suggestion["cli_command"] or "")


def test_decision_gate_r3_next_args_only_triggered_recipes(tmp_path: Path) -> None:
    """G2 R1 Blocking 2: R3 with one recipe triggering R4 + one not must
    have alpha-star/offline-pattern/online-pattern all restricted to the
    triggered recipe, plus an explicit ``--recipe`` CLI for single-recipe
    R4 launch."""
    summary_path = tmp_path / "per_yaml_summary.jsonl"
    rows = []
    # p1 R3: online jerk-heavy SR 0.97; uniform SR 0.92 -> delta 5pp -> trigger.
    for pat_name in R3_ONLINE_PATTERNS:
        sr = 0.97 if pat_name == "jerk-heavy" else 0.92
        rows.append(_make_summary_row(
            recipe_id="p1_state_fut_online_act", alpha=0.4,
            pattern_label=f"a0.4__off-uniform__on-{pat_name}",
            success_rate=sr, offline_pattern="uniform", online_pattern=pat_name,
        ))
    # p2 R3: all patterns at SR 0.93 -> delta < 2pp -> NO trigger.
    for pat_name in R3_ONLINE_PATTERNS:
        rows.append(_make_summary_row(
            recipe_id="p2_action_fut_online_act", alpha=0.6,
            pattern_label=f"a0.6__off-uniform__on-{pat_name}",
            success_rate=0.93, offline_pattern="uniform", online_pattern=pat_name,
        ))
    _write_summary(summary_path, rows)
    out = _dump_decision_gate_table(round_id=3, summary_path=summary_path)

    assert out["trigger_decisions"]["p1_state_fut_online_act"]["continue"] is True
    assert out["trigger_decisions"]["p2_action_fut_online_act"]["continue"] is False

    suggestion = out["next_args_suggestion"]
    # alpha-star, offline-pattern, online-pattern: only p1.
    for field in ("alpha-star", "offline-pattern", "online-pattern"):
        assert "p1_state_fut_online_act" in (suggestion[field] or "")
        assert "p2_action_fut_online_act" not in (suggestion[field] or "")
    # Recipe restriction + cli_command for single-recipe R4.
    assert suggestion["active_recipes"] == ["p1_state_fut_online_act"]
    assert suggestion["recipe"] == "p1_state_fut_online_act"
    assert "--recipe p1_state_fut_online_act" in (suggestion["cli_command"] or "")
    assert "--round 4" in (suggestion["cli_command"] or "")


def test_serve_host_serve_port_aliases_accepted() -> None:
    """G2 R1 NB1: argparse accepts both --host/--port and the plan-doc
    spelling --serve-host/--serve-port."""
    from exp.verdict_factor_judge.run_phase4 import _parse_args

    parsed = _parse_args([
        "--mode", "run-warmup", "--round", "1",
        "--serve-host", "10.0.0.1", "--serve-port", "12345",
    ])
    assert parsed.host == "10.0.0.1"
    assert parsed.port == 12345

    parsed2 = _parse_args([
        "--mode", "run-warmup", "--round", "1",
        "--host", "10.0.0.2", "--port", "23456",
    ])
    assert parsed2.host == "10.0.0.2"
    assert parsed2.port == 23456
