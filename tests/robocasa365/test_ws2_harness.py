"""Tests for the ws2 driver harness: two-view resume, real scheduler order, finalizer.

Nothing here starts a server, a worker or a driver thread; the conductor's own
``EpisodeScheduler`` is driven directly so the ORDER assertions are about what
the scheduler actually does, not about graph insertion order.
"""

from __future__ import annotations

import json
import os

import pytest

from openpi.conductor.scheduler import EpisodeScheduler
from openpi.conductor.task import ServerEndpoint, TaskGraph

from exp.robocasa365.run_ws_search2 import (
    Ws2ArmStrategy,
    batched,
    build_cell_specs,
    family_of,
    finalize,
    hold_workers_between_batches,
    interleave_cells,
    pin_manifest_sha,
    resolve_cells,
)
from exp.robocasa365.run_collect import build_run_plan, write_run_plan
from exp.robocasa365.run_ws_search import EVAL_NO_COLLECT_ROOT, WsSearchStrategy

TASKS = [("OpenDrawer", 2), ("CloseFridge", 2)]
CIDS = ["iso_a", "grid_b", "grid3_c", "grid3v_d", "grid4_e", "iso_f"]


def _strategy(cid: str, prefix: str = "ws2") -> WsSearchStrategy:
    return WsSearchStrategy(
        cid=cid, run_prefix=prefix, teacher="groot_tp", layout=1, style=1,
        base_seed=1_000_000, replan_steps=5, tasks=TASKS,
    )


def _servers(n: int = 2) -> list[ServerEndpoint]:
    return [ServerEndpoint("h", 23160 + i) for i in range(n)]


def _full_views(cids, config_dir, servers):
    """(strategies, full graphs, assignment, ranks) — the immutable view."""
    from openpi.conductor.driver import assign_servers

    ordered = interleave_cells(cids)
    ranks = {cid: i for i, cid in enumerate(ordered)}
    strategies, weights = {}, {}
    for cid in ordered:
        (config_dir / f"{cid}.yaml").write_text(f"# {cid}\n")
        s = _strategy(cid)
        strategies[s.run_id] = s
        for yaml_id, (_, n) in zip(s.yaml_ids, TASKS):
            weights[yaml_id] = n
    assignment = assign_servers(weights, servers, None, {s.key: 4 for s in servers})
    graphs = {rid: s.plan(sorted(s.yaml_ids), assignment) for rid, s in strategies.items()}
    return strategies, graphs, assignment, ranks, weights


# ------------------------------------------------------------------
# family interleaving
# ------------------------------------------------------------------


def test_family_of_reads_whole_segments():
    assert family_of("grid3v_vision_0@50") == "grid3v"
    assert family_of("grid_vision_0@50_robot_state@50") == "grid"
    assert family_of("iso_robot_state") == "iso"
    assert family_of("weird") == "other"


def test_interleave_covers_every_family_in_any_prefix():
    cids = [f"{fam}_{i}" for fam in ("iso", "grid", "grid3", "grid3v", "grid4") for i in range(4)]
    order = interleave_cells(cids)
    assert sorted(order) == sorted(cids)
    assert {family_of(c) for c in order[:5]} == {"iso", "grid", "grid3", "grid3v", "grid4"}
    assert interleave_cells(cids) == order  # deterministic


def test_interleave_drains_exhausted_families_without_loss():
    cids = ["iso_a", "grid_a", "grid_b", "grid_c"]
    order = interleave_cells(cids)
    assert sorted(order) == sorted(cids)
    assert order[0] == "iso_a"


# ------------------------------------------------------------------
# scheduler-visible order (real EpisodeScheduler)
# ------------------------------------------------------------------


def test_scheduler_setup_order_is_family_interleaved(tmp_path):
    servers = _servers(1)
    strategies, graphs, assignment, ranks, _ = _full_views(CIDS, tmp_path, servers)
    specs = build_cell_specs(strategies, graphs, set(), ranks, tmp_path)
    graph = Ws2ArmStrategy(specs).plan([], assignment)

    sched = EpisodeScheduler(graph, eval_concurrency=99)
    # Interleaving is per CELL (each cell contributes one stage per task), so
    # read the cell sequence the scheduler actually walks.
    cells_in_order: list[str] = []
    for stage in sched.pending_setups():
        cid = stage.yaml_id.split("__")[0].split("-", 1)[1]
        if cid not in cells_in_order:
            cells_in_order.append(cid)
    # The five distinct families appear before any family repeats; a cid-sorted
    # stage_id would have given grid,grid3,grid3v,grid4,iso in blocks instead.
    assert [family_of(c) for c in cells_in_order[:5]] == list(
        dict.fromkeys(family_of(c) for c in cells_in_order[:5])
    )
    assert len({family_of(c) for c in cells_in_order[:5]}) == 5
    assert cells_in_order == interleave_cells(CIDS)


def test_dispatch_order_needs_the_rank_prefix(tmp_path):
    """The frozen mechanism is ``next_task``'s ``sorted(stage_id)`` walk.

    ``pending_setups`` returns insertion order, so an order test built on it
    passes even with the rank prefix stripped. This one dispatches for real and
    compares against the same graph re-keyed by bare yaml_id — which is what
    the code would degrade to without the prefix.
    """
    import dataclasses

    # Four cells per family: with only one or two the cid-sorted order still
    # touches every family early by accident and the counter-example is vacuous.
    cids = [f"{fam}_{i}" for fam in ("iso", "grid", "grid3", "grid3v", "grid4")
            for i in range(4)]
    servers = _servers(1)
    strategies, graphs, assignment, ranks, _ = _full_views(cids, tmp_path, servers)
    graph = Ws2ArmStrategy(build_cell_specs(strategies, graphs, set(), ranks, tmp_path)).plan(
        [], assignment)

    def dispatch_cell_sequence(g):
        sched = EpisodeScheduler(g, eval_concurrency=99)
        for stage in sched.pending_setups():
            sched.mark_setup_running(stage.stage_id)
            sched.mark_setup_done(stage.stage_id)
        seen: list[str] = []
        while len(seen) < len(cids):
            task = sched.next_task(servers[0].key)
            if task is None:
                break
            cid = task.yaml_id.split("__")[0].split("-", 1)[1]
            if cid not in seen:
                seen.append(cid)
            sched.mark_result(task.task_uid, success=True, retriable=False, attempt=task.attempt)
        return seen

    ranked = dispatch_cell_sequence(graph)
    assert [family_of(c) for c in ranked[:5]] == list(
        dict.fromkeys(family_of(c) for c in ranked[:5])
    )
    assert len({family_of(c) for c in ranked[:5]}) == 5

    # Same graph without the rank prefix: cid-sorted stage ids walk the
    # families in blocks, which is exactly the round-1 sampling bias.
    bare = TaskGraph()
    for stage in graph.stages.values():
        bare.add_stage(dataclasses.replace(stage, stage_id=stage.yaml_id))
    assert len({family_of(c) for c in dispatch_cell_sequence(bare)[:5]}) < 5


def test_scheduler_order_survives_partial_progress(tmp_path):
    servers = _servers(1)
    strategies, graphs, assignment, ranks, _ = _full_views(CIDS, tmp_path, servers)

    first = [s.stage_id for s in EpisodeScheduler(
        Ws2ArmStrategy(build_cell_specs(strategies, graphs, set(), ranks, tmp_path)).plan([], assignment),
        eval_concurrency=99).pending_setups()]

    # Drop one whole cell (resume) — the remaining ranks keep their order.
    done_cell = next(iter(strategies))
    done = {ep.task_uid for yid in strategies[done_cell].yaml_ids
            for ep in graphs[done_cell].stages[yid].episodes}
    second = [s.stage_id for s in EpisodeScheduler(
        Ws2ArmStrategy(build_cell_specs(strategies, graphs, done, ranks, tmp_path)).plan([], assignment),
        eval_concurrency=99).pending_setups()]
    # The finished cell's stages are gone; every survivor keeps its rank and
    # hence its relative position.
    assert second == [sid for sid in first if done_cell not in sid]


# ------------------------------------------------------------------
# two-view resume contract
# ------------------------------------------------------------------


def test_run_plan_hash_is_invariant_across_resume(tmp_path):
    servers = _servers(2)
    strategies, graphs, _, _, _ = _full_views(CIDS[:2], tmp_path, servers)
    run_id, strategy = next(iter(strategies.items()))
    plan_path = tmp_path / f"run_plan_{run_id}.json"

    payload = build_run_plan(strategy, graphs[run_id], EVAL_NO_COLLECT_ROOT)
    write_run_plan(plan_path, payload)
    first_hash = payload["plan_hash"]

    # A resume rebuilds the FULL graph (never the pruned one) — same hash, and
    # write_run_plan accepts it instead of aborting on parameter drift.
    again = build_run_plan(strategy, graphs[run_id], EVAL_NO_COLLECT_ROOT)
    assert again["plan_hash"] == first_hash
    write_run_plan(plan_path, again)


def test_pruned_graph_would_break_the_run_plan_hash(tmp_path):
    """Why the two views exist: a pruned graph yields a different plan hash."""
    import dataclasses

    servers = _servers(1)
    strategies, graphs, _, _, _ = _full_views(CIDS[:1], tmp_path, servers)
    run_id, strategy = next(iter(strategies.items()))
    full = build_run_plan(strategy, graphs[run_id], EVAL_NO_COLLECT_ROOT)

    graph = graphs[run_id]
    first_yaml = strategy.yaml_ids[0]
    stage = graph.stages[first_yaml]
    graph.stages[first_yaml] = dataclasses.replace(stage, episodes=stage.episodes[1:])
    pruned = build_run_plan(strategy, graph, EVAL_NO_COLLECT_ROOT)
    assert pruned["plan_hash"] != full["plan_hash"]
    assert len(pruned["uids"]) < len(full["uids"])


def test_completed_cell_creates_no_stage_and_no_bundle_load(tmp_path):
    servers = _servers(1)
    strategies, graphs, assignment, ranks, _ = _full_views(CIDS[:2], tmp_path, servers)
    done_cell = next(iter(strategies))
    done = {ep.task_uid for yid in strategies[done_cell].yaml_ids
            for ep in graphs[done_cell].stages[yid].episodes}

    specs = build_cell_specs(strategies, graphs, done, ranks, tmp_path)
    assert done_cell not in specs
    graph = Ws2ArmStrategy(specs).plan([], assignment)
    assert all(done_cell not in stage.yaml_id for stage in graph.stages.values())


def test_active_episodes_keep_identity_and_gain_bundle(tmp_path):
    servers = _servers(1)
    strategies, graphs, _, ranks, _ = _full_views(CIDS[:1], tmp_path, servers)
    run_id = next(iter(strategies))
    first_uid = graphs[run_id].stages[strategies[run_id].yaml_ids[0]].episodes[0].task_uid

    specs = build_cell_specs(strategies, graphs, {first_uid}, ranks, tmp_path)
    episodes = [ep for eps in specs[run_id]["episodes_by_yaml"].values() for ep in eps]
    assert first_uid not in {ep.task_uid for ep in episodes}
    assert all(ep.bundle_id == run_id for ep in episodes)
    # Identity fields untouched by the active view.
    full = [ep for yid in strategies[run_id].yaml_ids for ep in graphs[run_id].stages[yid].episodes]
    by_uid = {ep.task_uid: ep for ep in full}
    for ep in episodes:
        original = by_uid[ep.task_uid]
        assert (ep.yaml_id, ep.task_id, ep.episode_idx, ep.extra) == (
            original.yaml_id, original.task_id, original.episode_idx, original.extra)


def test_extra_contract_is_stamped_for_every_episode(tmp_path):
    servers = _servers(1)
    strategies, graphs, _, ranks, _ = _full_views(CIDS[:1], tmp_path, servers)
    from exp.robocasa365.episode_runner import REQUIRED_EXTRA_KEYS

    specs = build_cell_specs(strategies, graphs, set(), ranks, tmp_path)
    for eps in specs[next(iter(specs))]["episodes_by_yaml"].values():
        for ep in eps:
            assert all(k in ep.extra for k in REQUIRED_EXTRA_KEYS)
            assert ep.extra["base_seed"] == 1_000_000
            assert ep.experiment == "groot_tp" and "/" not in ep.experiment


# ------------------------------------------------------------------
# bundle load memoisation
# ------------------------------------------------------------------


class _RecordingCtl:
    def __init__(self) -> None:
        self.loads: list[tuple[str, str]] = []

    def load_cache_config(self, *, yaml_content, yaml_id, bundle_id):
        del yaml_content
        self.loads.append((yaml_id, bundle_id))
        return {"version": len(self.loads)}


def test_bundle_loaded_once_per_server_and_bundle(tmp_path):
    servers = _servers(1)
    strategies, graphs, assignment, ranks, _ = _full_views(CIDS[:2], tmp_path, servers)
    arm = Ws2ArmStrategy(build_cell_specs(strategies, graphs, set(), ranks, tmp_path))
    graph = arm.plan([], assignment)
    ctl = _RecordingCtl()
    for stage in graph.stages.values():
        arm.on_stage_begin(stage, ctl, None)
    # Two cells x 2 task-stages each, but only two loads (one per bundle).
    assert len(graph.stages) == 4
    assert len(ctl.loads) == 2
    assert {bundle for _, bundle in ctl.loads} == set(strategies)
    assert all(yaml_id == bundle for yaml_id, bundle in ctl.loads)
    assert all(bundle != "default" for _, bundle in ctl.loads)


class _FlakyCtl(_RecordingCtl):
    """Fails the first load, then behaves."""

    def __init__(self) -> None:
        super().__init__()
        self.failures = 0

    def load_cache_config(self, *, yaml_content, yaml_id, bundle_id):
        if not self.failures:
            self.failures += 1
            raise ConnectionError("transient ctl error")
        return super().load_cache_config(
            yaml_content=yaml_content, yaml_id=yaml_id, bundle_id=bundle_id)


def test_failed_bundle_load_is_retried_not_memoised(tmp_path):
    """A memo written before the call would leave the cell bound to nothing."""
    servers = _servers(1)
    strategies, graphs, assignment, ranks, _ = _full_views(CIDS[:1], tmp_path, servers)
    arm = Ws2ArmStrategy(build_cell_specs(strategies, graphs, set(), ranks, tmp_path))
    stage = next(iter(arm.plan([], assignment).stages.values()))
    ctl = _FlakyCtl()

    with pytest.raises(ConnectionError):
        arm.on_stage_begin(stage, ctl, None)
    assert ctl.loads == []

    # The driver's setup retry must reach the server this time.
    arm.on_stage_begin(stage, ctl, None)
    assert len(ctl.loads) == 1
    # ... and only once thereafter.
    arm.on_stage_begin(stage, ctl, None)
    assert len(ctl.loads) == 1


def test_resume_hook_reloads_bundle_in_a_fresh_process(tmp_path):
    servers = _servers(1)
    strategies, graphs, assignment, ranks, _ = _full_views(CIDS[:1], tmp_path, servers)
    arm = Ws2ArmStrategy(build_cell_specs(strategies, graphs, set(), ranks, tmp_path))
    graph = arm.plan([], assignment)
    stage = next(iter(graph.stages.values()))
    ctl = _RecordingCtl()
    arm.on_resume(stage, ctl, None)
    arm.on_resume(stage, ctl, None)
    assert len(ctl.loads) == 1


# ------------------------------------------------------------------
# finalizer: round-1-shaped products the existing tools read
# ------------------------------------------------------------------


def _journal_line(uid: str, yaml_id: str, *, success: bool, accepted=True, status="done"):
    return json.dumps({
        "task_uid": uid, "yaml_id": yaml_id, "success": success,
        "accepted": accepted, "status": status, "error": None,
    })


def test_finalize_splits_per_cell_and_existing_tools_read_it(tmp_path):
    servers = _servers(1)
    strategies, graphs, _, ranks, _ = _full_views(CIDS[:2], tmp_path, servers)
    expected = {}
    lines = []
    for run_id, strategy in strategies.items():
        uids = []
        for yaml_id in strategy.yaml_ids:
            for ep in graphs[run_id].stages[yaml_id].episodes:
                uids.append(ep.task_uid)
                lines.append(_journal_line(ep.task_uid, yaml_id, success=ep.episode_idx == 0))
        expected[run_id] = {"cid": strategy._cid, "uids": uids}  # noqa: SLF001
    central = tmp_path / "journal_central_ws2.jsonl"
    central.write_text("\n".join(lines) + "\n")

    complete = finalize(central, tmp_path, teacher="groot_tp", expected_by_run=expected)
    assert all(complete.values())

    for run_id, spec in expected.items():
        summary = json.loads((tmp_path / f"summary_{run_id}.json").read_text())
        assert summary["cid"] == spec["cid"]
        assert set(summary["tasks"]) == {name for name, _ in TASKS}
        assert summary["macro_sr"] == pytest.approx(0.5)
        assert summary["n_missing"] == 0 and summary["n_err"] == 0

    # The round-1 analyzer reads the split journals with cid/task/idx intact.
    from exp.robocasa365.analyze_ws_search_stats import load_journals

    cells, grid = load_journals(tmp_path, 2, "ws2")
    assert set(cells) == {s._cid for s in strategies.values()}  # noqa: SLF001
    assert sorted({t for t, _ in grid}) == sorted(name for name, _ in TASKS)
    for outcomes in cells.values():
        assert set(outcomes) == set(grid)


def test_finalize_reports_missing_uids_as_incomplete(tmp_path):
    servers = _servers(1)
    strategies, graphs, _, _, _ = _full_views(CIDS[:1], tmp_path, servers)
    run_id, strategy = next(iter(strategies.items()))
    uids, lines = [], []
    for yaml_id in strategy.yaml_ids:
        for ep in graphs[run_id].stages[yaml_id].episodes:
            uids.append(ep.task_uid)
    # Journal only the first episode: the rest are missing.
    lines.append(_journal_line(uids[0], strategy.yaml_ids[0], success=True))
    central = tmp_path / "journal_central_ws2.jsonl"
    central.write_text("\n".join(lines) + "\n")

    complete = finalize(central, tmp_path, teacher="groot_tp",
                        expected_by_run={run_id: {"cid": strategy._cid, "uids": uids}})  # noqa: SLF001
    assert complete[run_id] is False
    summary = json.loads((tmp_path / f"summary_{run_id}.json").read_text())
    assert summary["n_missing"] == len(uids) - 1


def test_finalize_is_idempotent_and_ignores_other_cells(tmp_path):
    servers = _servers(1)
    strategies, graphs, _, _, _ = _full_views(CIDS[:2], tmp_path, servers)
    run_id, strategy = next(iter(strategies.items()))
    uids, lines = [], []
    for yaml_id in strategy.yaml_ids:
        for ep in graphs[run_id].stages[yaml_id].episodes:
            uids.append(ep.task_uid)
            lines.append(_journal_line(ep.task_uid, yaml_id, success=True))
    lines.append(_journal_line("some-other-run__X:eval:0:0", "some-other-run__X", success=True))
    central = tmp_path / "journal_central_ws2.jsonl"
    central.write_text("\n".join(lines) + "\n")

    expected = {run_id: {"cid": strategy._cid, "uids": uids}}  # noqa: SLF001
    finalize(central, tmp_path, teacher="groot_tp", expected_by_run=expected)
    once = (tmp_path / f"journal_{run_id}.jsonl").read_text()
    finalize(central, tmp_path, teacher="groot_tp", expected_by_run=expected)
    assert (tmp_path / f"journal_{run_id}.jsonl").read_text() == once
    assert "some-other-run" not in once


def test_finalize_passes_rejected_rows_through_untouched(tmp_path):
    servers = _servers(1)
    strategies, graphs, _, _, _ = _full_views(CIDS[:1], tmp_path, servers)
    run_id, strategy = next(iter(strategies.items()))
    first = graphs[run_id].stages[strategy.yaml_ids[0]].episodes[0]
    uids = [ep.task_uid for yid in strategy.yaml_ids for ep in graphs[run_id].stages[yid].episodes]
    central = tmp_path / "journal_central_ws2.jsonl"
    central.write_text(
        _journal_line(first.task_uid, first.yaml_id, success=True, accepted=False) + "\n"
        + _journal_line(first.task_uid, first.yaml_id, success=False) + "\n"
    )
    finalize(central, tmp_path, teacher="groot_tp",
             expected_by_run={run_id: {"cid": strategy._cid, "uids": uids}})  # noqa: SLF001
    split = (tmp_path / f"journal_{run_id}.jsonl").read_text().splitlines()
    assert len(split) == 2  # the stale row is preserved, not dropped
    summary = json.loads((tmp_path / f"summary_{run_id}.json").read_text())
    # ... but only the accepted terminal record scores.
    assert summary["tasks"][first.yaml_id.rsplit("__", 1)[-1]]["succ"] == 0


# ------------------------------------------------------------------
# manifest gating
# ------------------------------------------------------------------


def test_ws2_phase_reads_the_full_index(tmp_path):
    (tmp_path / "index.json").write_text(json.dumps({c: {} for c in CIDS}))
    cells, sha = resolve_cells("ws2", tmp_path, "")
    assert cells == sorted(CIDS) and sha == ""


def test_control_phase_requires_a_manifest(tmp_path):
    with pytest.raises(SystemExit, match="requires --manifest"):
        resolve_cells("ws2c", tmp_path, "")


def test_control_phase_takes_cells_only_from_the_manifest(tmp_path):
    manifest = tmp_path / "selection_manifest.json"
    manifest.write_text(json.dumps({"segments": {"ws2c": {"cells": ["iso_a", "grid_b"]}}}))
    cells, sha = resolve_cells("ws2c", tmp_path, str(manifest))
    assert cells == ["iso_a", "grid_b"] and len(sha) == 64


def test_missing_segment_is_refused(tmp_path):
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"segments": {"ws2c": {"cells": []}}}))
    with pytest.raises(SystemExit, match="no segment 'ws2e'"):
        resolve_cells("ws2e", tmp_path, str(manifest))


def test_resume_refuses_a_changed_manifest(tmp_path):
    pin_manifest_sha(tmp_path, "ws2c", "a" * 64)
    pin_manifest_sha(tmp_path, "ws2c", "a" * 64)  # same sha: fine
    with pytest.raises(SystemExit, match="never re-select"):
        pin_manifest_sha(tmp_path, "ws2c", "b" * 64)


# ------------------------------------------------------------------
# batching (scheduler cost mitigation)
# ------------------------------------------------------------------


def test_batches_preserve_order_and_are_lossless():
    cells = [f"c{i:03d}" for i in range(132)]
    batches = batched(cells, 12)
    assert len(batches) == 11
    assert {len(b) for b in batches} == {12}
    assert [c for b in batches for c in b] == cells
    # 0 disables batching entirely (one graph, round-1-like).
    assert batched(cells, 0) == [cells]
    # A final short batch is kept, not dropped.
    assert [len(b) for b in batched(cells, 50)] == [50, 50, 32]


class _FakeDriver:
    """Minimal stand-in exposing the handle_pull the wrapper decorates."""

    def __init__(self, msg, payload):
        self._msg, self._payload = msg, payload
        self.calls = 0

    def handle_pull(self, server_key):
        del server_key
        self.calls += 1
        return self._msg, self._payload


def test_non_final_batch_holds_workers_instead_of_shutting_them_down():
    from openpi.conductor import protocol as _proto

    driver = _FakeDriver(_proto.MSG_SHUTDOWN, {})
    hold_workers_between_batches(driver)
    msg, payload = driver.handle_pull("h:23160")
    # A worker that receives MSG_SHUTDOWN exits; between batches it must idle.
    assert msg == _proto.MSG_ASSIGN
    assert payload["none"] is True and payload["backoff_ms"] > 0


def test_holding_does_not_disturb_real_assignments():
    from openpi.conductor import protocol as _proto

    driver = _FakeDriver(_proto.MSG_ASSIGN, {"task": {"task_uid": "x"}})
    hold_workers_between_batches(driver)
    msg, payload = driver.handle_pull("h:23160")
    assert msg == _proto.MSG_ASSIGN and payload == {"task": {"task_uid": "x"}}


def test_finalize_tolerates_a_torn_last_line(tmp_path):
    """A hard crash can tear the journal; products must still materialise."""
    servers = _servers(1)
    strategies, graphs, _, _, _ = _full_views(CIDS[:1], tmp_path, servers)
    run_id, strategy = next(iter(strategies.items()))
    uids, lines = [], []
    for yaml_id in strategy.yaml_ids:
        for ep in graphs[run_id].stages[yaml_id].episodes:
            uids.append(ep.task_uid)
            lines.append(_journal_line(ep.task_uid, yaml_id, success=True))
    central = tmp_path / "journal_central_ws2.jsonl"
    central.write_text("\n".join(lines) + '\n{"task_uid": "ws2-iso_a__l1s1_gro')

    complete = finalize(central, tmp_path, teacher="groot_tp",
                        expected_by_run={run_id: {"cid": strategy._cid, "uids": uids}})  # noqa: SLF001
    assert complete[run_id] is True


@pytest.mark.parametrize(
    ("teacher", "rejected"),
    [("groot_tp", False), ("pi05", False), ("not_a_teacher", True)],
)
def test_driver_accepts_both_teachers(teacher, rejected):
    """The pi0.5 arm reuses this driver; a narrowed choice list blocks the phase.

    Drives the REAL CLI: argparse rejects an out-of-choices value with
    "invalid choice" before any other validation, so a run that gets past that
    message proves the teacher is accepted (it still fails on the missing
    required args, which is fine). Endpoint homogeneity is enforced by
    ``validate_teacher_endpoints``, not by this list, so narrowing it buys no
    safety -- it only blocks a phase, as it did on 2026-08-27.
    """
    import subprocess
    import sys

    proc = subprocess.run(
        [sys.executable, "-m", "exp.robocasa365.run_ws_search2", "--teacher", teacher],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": "src:."},
    )
    assert ("invalid choice" in proc.stderr) is rejected, proc.stderr[-400:]


def test_driver_teachers_match_the_adapters():
    """An argparse list narrower than ADAPTERS is a silent phase blocker."""
    import subprocess
    import sys

    from exp.robocasa365.episode_runner import ADAPTERS

    for teacher in ADAPTERS:
        proc = subprocess.run(
            [sys.executable, "-m", "exp.robocasa365.run_ws_search2", "--teacher", teacher],
            capture_output=True, text=True, timeout=120,
            env={**os.environ, "PYTHONPATH": "src:."},
        )
        assert "invalid choice" not in proc.stderr, f"{teacher} has an adapter but the CLI rejects it"
