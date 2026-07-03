"""Tests for ``exp.verdict_factor_judge.run_phase`` orchestration helpers.

The shim is hard to E2E without a real openpi server + LIBERO sim, so we
unit-test the deterministic pieces:

  1. JSONL aggregation (``_parse_dump_to_buffer``) — NaN skipping, cap.
  2. Per-yaml protocol order — patched ``WebsocketClientPolicy`` + patched
     ``subprocess.run`` record the call sequence; we assert the exact
     ws/subprocess interleaving.
  3. Warmup subprocess failure surfaces to the caller (``_run_one_yaml``
     does NOT swallow ``CalledProcessError``).
  4. ``--skip-warmup`` short-circuit (no warmup ws calls / no warmup
     subprocess / no fetch / no preload / no unload).
  5. Phase loop skips ``*__warmup.yaml`` files when iterating the dir.

For the ws monkeypatch we substitute a recorder class onto
``run_phase.WebsocketClientPolicy``; for subprocess we monkeypatch
``run_phase.subprocess.run``.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from exp.verdict_factor_judge import run_phase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _default_args(tmp_path: Path) -> run_phase.Args:
    """Build a baseline Args for tests; per-test edits via dataclasses.replace."""
    phase_dir = tmp_path / "phase1"
    phase_dir.mkdir(parents=True, exist_ok=True)
    return run_phase.Args(
        phase_dir=str(phase_dir),
        host="localhost",
        port=8000,
        task_suite="libero_spatial",
        num_workers=1,
        warmup_trials=1,
        eval_trials=1,
        per_step_log_dir="",
        summary_out="",
    )


def _write_yaml(path: Path, body: str = "stub: true\n") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")


class FakeClient:
    """Recording stand-in for ``WebsocketClientPolicy``.

    Each instance shares the class-level ``calls`` list so the test
    can assert ordering across multiple ``with`` blocks.
    """

    calls: list[tuple[str, dict[str, Any]]] = []
    fetch_dump_payload: bytes = b""
    preload_ack: dict = {"__ack__": "preload_normalizer_buffer", "n_keys": 0}

    def __init__(self, host: str = "", port: int | None = None, api_key: str | None = None) -> None:
        # Args are accepted but irrelevant for the recorder.
        type(self).calls.append(("__init__", {"host": host, "port": port}))

    def __enter__(self) -> "FakeClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def load_cache_config(self, *, yaml_content: str | None = None,
                          yaml_path: str | None = None,
                          yaml_id: str | None = None) -> dict:
        type(self).calls.append((
            "load_cache_config",
            {"yaml_id": yaml_id, "has_content": yaml_content is not None},
        ))
        return {"__ack__": "load_cache_config", "version": 1}

    def fetch_dump(self, warmup_yaml_id: str) -> bytes:
        type(self).calls.append((
            "fetch_dump", {"warmup_yaml_id": warmup_yaml_id},
        ))
        return type(self).fetch_dump_payload

    def preload_normalizer_buffer(self, eval_yaml_id: str, buffer: dict) -> dict:
        type(self).calls.append((
            "preload_normalizer_buffer",
            {"eval_yaml_id": eval_yaml_id, "n_keys": len(buffer)},
        ))
        return type(self).preload_ack

    def unload_warmup_buffer(self, eval_yaml_id: str) -> dict:
        type(self).calls.append((
            "unload_warmup_buffer", {"eval_yaml_id": eval_yaml_id},
        ))
        return {"__ack__": "unload_warmup_buffer"}


@pytest.fixture(autouse=True)
def _reset_fake_client() -> None:
    """Wipe the class-level recorder between tests so ordering is per-test."""
    FakeClient.calls = []
    FakeClient.fetch_dump_payload = b""
    FakeClient.preload_ack = {
        "__ack__": "preload_normalizer_buffer",
        "n_keys": 0,
    }


# ---------------------------------------------------------------------------
# 1. _parse_dump_to_buffer
# ---------------------------------------------------------------------------


def test_aggregate_warmup_jsonl_skips_nan_and_caps_buffer() -> None:
    """NaN values are dropped + each list capped at ``max_per_key``."""
    rows = []
    # Key A: 250 valid floats + 50 NaNs → after parse must be capped to 200.
    for i in range(250):
        rows.append({"factor_raw": {"A": float(i)}})
    for _ in range(50):
        rows.append({"factor_raw": {"A": float("nan")}})
    # Key B: 5 floats, all valid.
    for i in range(5):
        rows.append({"factor_raw": {"B": 0.1 * i}})
    content = ("\n".join(json.dumps(r) for r in rows)).encode("utf-8")

    buffer = run_phase._parse_dump_to_buffer(content, max_per_key=200)

    assert set(buffer) == {"A", "B"}
    assert len(buffer["A"]) == 200
    # All NaNs filtered + cap respected (FIFO retention).
    assert all(not math.isnan(v) for v in buffer["A"])
    assert buffer["A"] == [float(i) for i in range(200)]
    assert len(buffer["B"]) == 5


def test_aggregate_warmup_jsonl_tolerates_blank_lines_and_garbage() -> None:
    """Robustness: blank lines + a torn last line don't corrupt the result."""
    good = json.dumps({"factor_raw": {"K": 1.0}})
    payload = b"\n".join([good.encode(), b"", b"not-json", good.encode(), b""])
    buffer = run_phase._parse_dump_to_buffer(payload)
    assert buffer == {"K": [1.0, 1.0]}


# ---------------------------------------------------------------------------
# 2. _run_one_yaml protocol order
# ---------------------------------------------------------------------------


def test_run_one_yaml_calls_protocol_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _default_args(tmp_path)
    eval_path = Path(args.phase_dir) / "x_phase1_y.yaml"
    warmup_path = Path(args.phase_dir) / "x_phase1_y__warmup.yaml"
    _write_yaml(eval_path)
    _write_yaml(warmup_path)

    # Provide a JSONL payload so step 4 produces a non-empty buffer (and
    # thus step 6 actually runs preload_normalizer_buffer).
    FakeClient.fetch_dump_payload = json.dumps(
        {"factor_raw": {"k1": 1.0, "k2": 2.0}}
    ).encode() + b"\n"
    FakeClient.preload_ack = {
        "__ack__": "preload_normalizer_buffer", "n_keys": 2,
    }

    subprocess_calls: list[list[str]] = []

    def fake_run(cmd, *args_, **kwargs):
        subprocess_calls.append(list(cmd))

        class _Done:
            returncode = 0
        return _Done()

    monkeypatch.setattr(run_phase, "WebsocketClientPolicy", FakeClient)
    monkeypatch.setattr(run_phase.subprocess, "run", fake_run)

    summary = run_phase._run_one_yaml(args, eval_path)

    # Expected ordering (interleaved across ws + subprocess):
    #   load_cache_config(warmup) → subprocess(warmup) → fetch_dump
    #   → load_cache_config(eval) → preload_normalizer_buffer
    #   → subprocess(eval) → unload_warmup_buffer
    op_order = [name for (name, _) in FakeClient.calls if name != "__init__"]
    assert op_order == [
        "load_cache_config",
        "fetch_dump",
        "load_cache_config",
        "preload_normalizer_buffer",
        "unload_warmup_buffer",
    ], FakeClient.calls

    # Verify the yaml_id payloads match the strict warmup vs eval contract.
    by_name = [(n, p) for (n, p) in FakeClient.calls if n != "__init__"]
    assert by_name[0][1]["yaml_id"] == "x_phase1_y__warmup"
    assert by_name[1][1]["warmup_yaml_id"] == "x_phase1_y__warmup"
    assert by_name[2][1]["yaml_id"] == "x_phase1_y"
    assert by_name[3][1]["eval_yaml_id"] == "x_phase1_y"
    assert by_name[4][1]["eval_yaml_id"] == "x_phase1_y"

    # Two subprocesses: warmup then eval, both with --yaml-id matching.
    assert len(subprocess_calls) == 2
    warmup_cmd, eval_cmd = subprocess_calls
    assert "--phase" in warmup_cmd
    assert warmup_cmd[warmup_cmd.index("--phase") + 1] == "warmup"
    assert warmup_cmd[warmup_cmd.index("--yaml-id") + 1] == "x_phase1_y__warmup"
    assert eval_cmd[eval_cmd.index("--phase") + 1] == "eval"
    assert eval_cmd[eval_cmd.index("--yaml-id") + 1] == "x_phase1_y"

    # Summary echoes the count from the preload ack.
    assert summary["n_warmup_buffer_keys"] == 2
    assert summary["yaml_id"] == "x_phase1_y"
    assert summary["warmup_yaml_id"] == "x_phase1_y__warmup"


# ---------------------------------------------------------------------------
# 3. Failure propagation
# ---------------------------------------------------------------------------


def test_run_one_yaml_continues_on_warmup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warmup subprocess failure must surface to the outer loop, not be eaten.

    The outer ``main()`` then catches + skips this yaml, but
    ``_run_one_yaml`` itself MUST raise so the partial state is visible."""
    args = _default_args(tmp_path)
    eval_path = Path(args.phase_dir) / "x_phase1_y.yaml"
    warmup_path = Path(args.phase_dir) / "x_phase1_y__warmup.yaml"
    _write_yaml(eval_path)
    _write_yaml(warmup_path)

    def fake_run(cmd, *args_, **kwargs):
        # Identify warmup vs eval by --phase value in the cmd.
        phase_idx = cmd.index("--phase")
        if cmd[phase_idx + 1] == "warmup":
            raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

        class _Done:
            returncode = 0
        return _Done()

    monkeypatch.setattr(run_phase, "WebsocketClientPolicy", FakeClient)
    monkeypatch.setattr(run_phase.subprocess, "run", fake_run)

    with pytest.raises(subprocess.CalledProcessError):
        run_phase._run_one_yaml(args, eval_path)

    # No fetch_dump / preload / unload should have happened — the warmup
    # subprocess crash short-circuited the protocol.
    op_order = [name for (name, _) in FakeClient.calls if name != "__init__"]
    assert op_order == ["load_cache_config"], FakeClient.calls


def test_main_skips_failing_yaml_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``main`` records a fail count + keeps going when one yaml raises."""
    args = _default_args(tmp_path)
    # Two eval yamls; first will fail, second will succeed.
    for stem in ("a_phase1", "b_phase1"):
        _write_yaml(Path(args.phase_dir) / f"{stem}.yaml")
        _write_yaml(Path(args.phase_dir) / f"{stem}__warmup.yaml")

    def fake_run(cmd, *args_, **kwargs):
        phase_idx = cmd.index("--phase")
        yaml_id_idx = cmd.index("--yaml-id")
        if "a_phase1" in cmd[yaml_id_idx + 1] and cmd[phase_idx + 1] == "warmup":
            raise subprocess.CalledProcessError(returncode=2, cmd=cmd)

        class _Done:
            returncode = 0
        return _Done()

    monkeypatch.setattr(run_phase, "WebsocketClientPolicy", FakeClient)
    monkeypatch.setattr(run_phase.subprocess, "run", fake_run)

    rc = run_phase.main(args)
    # Non-zero exit because at least one yaml failed.
    assert rc == 1
    # We expect the second yaml's full protocol to have run after the first
    # failed — i.e. at least one ``unload_warmup_buffer`` call.
    op_names = [name for (name, _) in FakeClient.calls if name != "__init__"]
    assert "unload_warmup_buffer" in op_names


# ---------------------------------------------------------------------------
# 4. --skip-warmup
# ---------------------------------------------------------------------------


def test_skip_warmup_flag_short_circuits_to_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``skip_warmup=True``, only the eval yaml is loaded + run; no
    warmup subprocess, no fetch_dump, no preload, no unload."""
    args = replace(_default_args(tmp_path), skip_warmup=True)
    eval_path = Path(args.phase_dir) / "x_phase1_y.yaml"
    _write_yaml(eval_path)
    # Note: deliberately do NOT create the sibling warmup yaml — the skip
    # path must not even check for it.

    subprocess_calls: list[list[str]] = []

    def fake_run(cmd, *args_, **kwargs):
        subprocess_calls.append(list(cmd))

        class _Done:
            returncode = 0
        return _Done()

    monkeypatch.setattr(run_phase, "WebsocketClientPolicy", FakeClient)
    monkeypatch.setattr(run_phase.subprocess, "run", fake_run)

    run_phase._run_one_yaml(args, eval_path)

    op_order = [name for (name, _) in FakeClient.calls if name != "__init__"]
    # Only the eval load_cache_config — no fetch/preload/unload.
    assert op_order == ["load_cache_config"], FakeClient.calls
    # And only the eval subprocess fired.
    assert len(subprocess_calls) == 1
    cmd = subprocess_calls[0]
    assert cmd[cmd.index("--phase") + 1] == "eval"
    assert cmd[cmd.index("--yaml-id") + 1] == "x_phase1_y"


# ---------------------------------------------------------------------------
# 5. Phase loop skip pattern
# ---------------------------------------------------------------------------


def test_phase_loop_skips_warmup_yaml_files(tmp_path: Path) -> None:
    """``_iter_eval_yamls`` must NEVER return ``*__warmup.yaml`` files."""
    phase_dir = tmp_path / "phase1"
    phase_dir.mkdir()
    eval_files = ["a.yaml", "b.yaml", "c.yaml"]
    warmup_files = ["a__warmup.yaml", "b__warmup.yaml", "c__warmup.yaml"]
    for f in eval_files + warmup_files:
        (phase_dir / f).write_text("stub: true\n")
    # Throw in a non-yaml red herring to verify the suffix filter.
    (phase_dir / "README.md").write_text("ignored")

    found = run_phase._iter_eval_yamls(phase_dir)
    assert sorted(p.name for p in found) == sorted(eval_files)
    # Sanity: all returned paths are real files in the phase_dir.
    for p in found:
        assert p.parent == phase_dir
        assert not p.stem.endswith("__warmup")


# ---------------------------------------------------------------------------
# SR forwarding + resume helpers (added with --resume / --episode-results-dir)
# ---------------------------------------------------------------------------


def test_aggregate_sr_from_episode_json_counts_success_true(tmp_path: Path) -> None:
    """``_aggregate_sr_from_episode_json`` must count ``success: True`` rows."""
    p = tmp_path / "episode_results.json"
    p.write_text(json.dumps([
        {"task_id": 0, "episode_id": 0, "success": True},
        {"task_id": 0, "episode_id": 1, "success": False},
        {"task_id": 1, "episode_id": 0, "success": True},
        {"task_id": 1, "episode_id": 1, "success": True},
    ]))
    assert run_phase._aggregate_sr_from_episode_json(p) == 0.75


def test_aggregate_sr_from_episode_json_handles_missing_or_empty(tmp_path: Path) -> None:
    """Missing / empty / malformed inputs must return ``None`` (no crash)."""
    missing = tmp_path / "nope.json"
    assert run_phase._aggregate_sr_from_episode_json(missing) is None

    empty = tmp_path / "empty.json"
    empty.write_text("[]")
    assert run_phase._aggregate_sr_from_episode_json(empty) is None

    bad = tmp_path / "bad.json"
    bad.write_text("not json {")
    assert run_phase._aggregate_sr_from_episode_json(bad) is None


def test_load_done_yaml_ids_collects_yaml_id_field(tmp_path: Path) -> None:
    """``_load_done_yaml_ids`` returns the set of ``yaml_id`` values from
    a JSONL summary, tolerating blank / malformed lines."""
    p = tmp_path / "summary.jsonl"
    p.write_text(
        json.dumps({"yaml_id": "alpha", "success_rate": 0.9}) + "\n"
        + "\n"  # blank line
        + "garbage{\n"
        + json.dumps({"yaml_id": "beta", "success_rate": None}) + "\n"
        + json.dumps({"no_yaml": "xx"}) + "\n"  # missing yaml_id
    )
    assert run_phase._load_done_yaml_ids(p) == {"alpha", "beta"}

    # Missing file → empty set, not error.
    assert run_phase._load_done_yaml_ids(tmp_path / "nope.jsonl") == set()


def test_resolve_episode_results_dir_explicit_wins(tmp_path: Path) -> None:
    """Explicit ``--episode-results-dir`` overrides the per_step_log_dir fallback."""
    args = _default_args(tmp_path)
    args = replace(
        args,
        per_step_log_dir=str(tmp_path / "per_step"),
        episode_results_dir=str(tmp_path / "explicit"),
    )
    assert run_phase._resolve_episode_results_dir(args) == tmp_path / "explicit"


def test_resolve_episode_results_dir_fallback_to_per_step_sibling(tmp_path: Path) -> None:
    """When only ``--per-step-log-dir`` is set we put SR JSONs at its sibling."""
    args = _default_args(tmp_path)
    args = replace(args, per_step_log_dir=str(tmp_path / "data" / "cfg" / "per_step"))
    assert run_phase._resolve_episode_results_dir(args) == tmp_path / "data" / "cfg" / "episode_results"


def test_resolve_episode_results_dir_disabled_when_neither_set(tmp_path: Path) -> None:
    """No per_step_log_dir + no episode_results_dir → SR forwarding off."""
    args = _default_args(tmp_path)
    assert run_phase._resolve_episode_results_dir(args) is None


def test_build_libero_argv_forwards_save_episode_results_only_on_eval(tmp_path: Path) -> None:
    """``--save-episode-results`` MUST be forwarded only when ``phase=='eval'``;
    warmup phase uses AlwaysWarmStart so its SR is not the metric."""
    args = _default_args(tmp_path)
    target = tmp_path / "episode_results" / "yaml_alpha.json"

    # eval phase → flag present
    eval_cmd, _ = run_phase._build_libero_argv(
        args=args, yaml_id="yaml_alpha", phase="eval",
        num_trials_per_task=1, episode_results_path=target,
    )
    assert "--save-episode-results" in eval_cmd
    assert str(target) in eval_cmd

    # warmup phase → flag absent even when path is given
    warmup_cmd, _ = run_phase._build_libero_argv(
        args=args, yaml_id="yaml_alpha", phase="warmup",
        num_trials_per_task=1, episode_results_path=target,
    )
    assert "--save-episode-results" not in warmup_cmd

    # No path → no flag regardless of phase
    no_path_cmd, _ = run_phase._build_libero_argv(
        args=args, yaml_id="yaml_alpha", phase="eval",
        num_trials_per_task=1, episode_results_path=None,
    )
    assert "--save-episode-results" not in no_path_cmd


def test_build_libero_argv_emits_canonical_collect_gate_dir_flag(tmp_path: Path) -> None:
    """Verdict-factor runners emit the CANONICAL ``--collect-gate-dir`` (plan
    §2.9.3 / T14d); ``--per-step-log-dir`` survives only as main.py's deprecated
    alias and must NOT be the flag the orchestrator spawns with."""
    args = replace(_default_args(tmp_path), per_step_log_dir=str(tmp_path / "per_step"))
    cmd, _ = run_phase._build_libero_argv(
        args=args, yaml_id="yaml_alpha", phase="eval", num_trials_per_task=1,
    )
    assert "--collect-gate-dir" in cmd
    assert "--per-step-log-dir" not in cmd
    assert str(tmp_path / "per_step") in cmd


def test_summarize_gate_log_is_general_reexport() -> None:
    """The de-specialized summary (plan §2.11 R1) is the general
    ``openpi.serving.per_step_recorder.summarize_gate_log``, re-exported from
    run_phase; the verdict-specific ``_summarize_per_step_log`` is gone."""
    from openpi.serving.per_step_recorder import summarize_gate_log as general

    assert run_phase.summarize_gate_log is general
    assert not hasattr(run_phase, "_summarize_per_step_log")


def test_main_resume_skips_already_done_yamls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``main(--resume)`` with an existing summary must skip done yamls and
    run only the missing ones, appending (not truncating)."""
    phase_dir = tmp_path / "phase1"
    phase_dir.mkdir()
    # Three eval yamls + their warmup siblings (loop expects siblings).
    for name in ("a", "b", "c"):
        _write_yaml(phase_dir / f"{name}.yaml")
        _write_yaml(phase_dir / f"{name}__warmup.yaml")

    summary_path = tmp_path / "summary.jsonl"
    # Pre-seed: yaml "a" is already done.
    summary_path.write_text(
        json.dumps({"yaml_id": "a", "success_rate": 0.5, "n_full_hit": 1,
                    "n_warm_start": 0, "n_miss": 0, "n_eval_verdicts": 1,
                    "n_warmup_buffer_keys": 0,
                    "warmup_yaml_id": "a__warmup"}) + "\n"
    )

    # Stub _run_one_yaml so we don't touch ws/subprocess; record which ran.
    ran: list[str] = []

    def _fake_run_one_yaml(_args: run_phase.Args, eval_path: Path) -> dict:
        ran.append(eval_path.stem)
        return {
            "yaml_id": eval_path.stem,
            "warmup_yaml_id": f"{eval_path.stem}__warmup",
            "n_warmup_buffer_keys": 0,
            "success_rate": 0.42,
            "n_eval_verdicts": 0,
            "n_full_hit": 0,
            "n_warm_start": 0,
            "n_miss": 0,
        }

    monkeypatch.setattr(run_phase, "_run_one_yaml", _fake_run_one_yaml)

    args = replace(
        _default_args(tmp_path),
        phase_dir=str(phase_dir),
        summary_out=str(summary_path),
        resume=True,
    )
    rc = run_phase.main(args)
    assert rc == 0
    # Only b and c should have run — a was already in summary.
    assert ran == ["b", "c"]

    # Summary file should now contain a (preserved) + b + c (appended).
    rows = [json.loads(line) for line in summary_path.read_text().splitlines() if line.strip()]
    assert [r["yaml_id"] for r in rows] == ["a", "b", "c"]
    # Pre-existing row's SR is preserved unchanged.
    assert rows[0]["success_rate"] == 0.5


def test_main_without_resume_truncates_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Legacy behaviour: no ``--resume`` truncates summary on start so
    re-runs don't double rows."""
    phase_dir = tmp_path / "phase1"
    phase_dir.mkdir()
    _write_yaml(phase_dir / "x.yaml")
    _write_yaml(phase_dir / "x__warmup.yaml")

    summary_path = tmp_path / "summary.jsonl"
    summary_path.write_text(json.dumps({"yaml_id": "x", "stale": True}) + "\n")

    def _fake_run_one_yaml(_args: run_phase.Args, eval_path: Path) -> dict:
        return {"yaml_id": eval_path.stem, "warmup_yaml_id": f"{eval_path.stem}__warmup",
                "n_warmup_buffer_keys": 0, "success_rate": 1.0,
                "n_eval_verdicts": 0, "n_full_hit": 0, "n_warm_start": 0, "n_miss": 0}

    monkeypatch.setattr(run_phase, "_run_one_yaml", _fake_run_one_yaml)
    args = replace(
        _default_args(tmp_path),
        phase_dir=str(phase_dir),
        summary_out=str(summary_path),
        resume=False,
    )
    rc = run_phase.main(args)
    assert rc == 0
    rows = [json.loads(line) for line in summary_path.read_text().splitlines() if line.strip()]
    # Stale row gone; only the freshly produced row remains.
    assert len(rows) == 1
    assert rows[0]["yaml_id"] == "x"
    assert rows[0].get("stale") is None
