"""D0 checker: sampling determinism and every census rejection class."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from exp.dispatch_surface.d0_check import (
    CONTROL_ROWS_PER_TASK,
    CONTROL_SEED,
    EXPECTED_NUM_STEPS,
    START_T_WS,
    census_identity,
    select_gpu_sample,
)


@dataclasses.dataclass
class _Payload:
    action_chunk: torch.Tensor
    intermediates: dict
    denoising_num_steps: int = EXPECTED_NUM_STEPS


@dataclasses.dataclass
class _Entry:
    id: str
    trajectory_id: str
    step_idx: int
    payload: _Payload


def _entry(eid: str, traj: str = "t0", step: int = 0) -> _Entry:
    chunk = torch.zeros(10, 32, dtype=torch.float32)
    return _Entry(eid, traj, step, _Payload(chunk, {START_T_WS: chunk.clone()}))


def _rows(n_tasks: int = 2, per_task: int = 20) -> list[dict]:
    out = []
    for t in range(n_tasks):
        for i in range(per_task):
            out.append({
                "episode_id": f"ep_{t}_{i}", "step_idx": i, "task_id": t,
                "winner_id": "e0", "s": 0.9, "v": 1.0,
                # one clear extreme per task, everything else flat
                "y_tau7": 100.0 if i == 0 else 1.0, "y_tau10": 2.0,
            })
    return out


def _sidecar(ids) -> dict:
    return {i: np.zeros((10, 32), dtype=np.float32) for i in ids}


# ---------------- sampling ----------------

def test_extremes_are_taken_whole_not_sampled():
    rows = _rows()
    extreme, _ = select_gpu_sample(rows, CONTROL_SEED)
    thr = float(np.percentile([r["y_tau7"] for r in rows], 99))
    assert {(r["episode_id"], r["step_idx"]) for r in extreme} == {
        (r["episode_id"], r["step_idx"]) for r in rows if r["y_tau7"] > thr
    }


def test_controls_are_per_task_and_exclude_extremes():
    rows = _rows()
    extreme, controls = select_gpu_sample(rows, CONTROL_SEED)
    ex_keys = {(r["episode_id"], r["step_idx"]) for r in extreme}
    assert not any((c["episode_id"], c["step_idx"]) in ex_keys for c in controls)
    by_task = {}
    for c in controls:
        by_task[c["task_id"]] = by_task.get(c["task_id"], 0) + 1
    assert set(by_task) == {r["task_id"] for r in rows}
    assert all(v == CONTROL_ROWS_PER_TASK for v in by_task.values())


def test_control_draw_is_deterministic_under_the_frozen_seed():
    rows = _rows()
    a = select_gpu_sample(rows, CONTROL_SEED)[1]
    b = select_gpu_sample(rows, CONTROL_SEED)[1]
    assert [(r["episode_id"], r["step_idx"]) for r in a] == [
        (r["episode_id"], r["step_idx"]) for r in b
    ]


def test_a_different_seed_moves_the_controls():
    rows = _rows()
    a = select_gpu_sample(rows, CONTROL_SEED)[1]
    b = select_gpu_sample(rows, CONTROL_SEED + 1)[1]
    assert [(r["episode_id"], r["step_idx"]) for r in a] != [
        (r["episode_id"], r["step_idx"]) for r in b
    ]


def test_formal_sample_refuses_a_missing_task_pool():
    with pytest.raises(ValueError, match="expected"):
        select_gpu_sample(_rows(n_tasks=9), CONTROL_SEED, expected_tasks=set(range(10)))


def test_formal_sample_refuses_fewer_than_two_controls_per_task():
    rows = _rows(n_tasks=1, per_task=2)
    with pytest.raises(ValueError, match="requires exactly"):
        select_gpu_sample(rows, CONTROL_SEED, expected_tasks={0})


# ---------------- census ----------------

def test_clean_census_passes():
    rows = [dict(r, winner_id="e0") for r in _rows(n_tasks=1, per_task=3)]
    entries = [_entry("e0")]
    assert census_identity(rows, entries, _sidecar(["e0"]))["passed"]


@pytest.mark.parametrize(
    "mutate,needle",
    [
        (lambda e, s: setattr(e[0].payload, "denoising_num_steps", 8), "denoising_num_steps"),
        (lambda e, s: e[0].payload.intermediates.clear(), "no intermediates"),
        (lambda e, s: s.pop("e0"), "missing from noise sidecar"),
        (lambda e, s: e[0].payload.intermediates.__setitem__(START_T_WS, torch.zeros(4, 4)),
         "intermediate shape"),
        (lambda e, s: e[0].payload.intermediates.__setitem__(
            START_T_WS, torch.zeros(10, 32, dtype=torch.float64)), "dtype"),
        (lambda e, s: s.__setitem__("e0", np.zeros((3, 3), dtype=np.float32)), "sidecar shape"),
    ],
)
def test_census_rejects_each_corruption(mutate, needle):
    rows = [dict(r, winner_id="e0") for r in _rows(n_tasks=1, per_task=3)]
    entries = [_entry("e0")]
    side = _sidecar(["e0"])
    mutate(entries, side)
    res = census_identity(rows, entries, side)
    assert not res["passed"]
    assert any(needle in p for p in res["problems"]), res["problems"]


def test_census_rejects_duplicate_table_rows():
    rows = [dict(r, winner_id="e0") for r in _rows(n_tasks=1, per_task=3)]
    rows.append(dict(rows[0]))
    res = census_identity(rows, [_entry("e0")], _sidecar(["e0"]))
    assert not res["passed"]
    assert any("duplicate table row" in p for p in res["problems"])


def test_census_rejects_entry_join_collision():
    rows = [dict(r, winner_id="e0") for r in _rows(n_tasks=1, per_task=3)]
    entries = [_entry("e0", "t0", 0), _entry("e1", "t0", 0)]
    res = census_identity(rows, entries, _sidecar(["e0", "e1"]))
    assert not res["passed"]
    assert any("join collision" in p for p in res["problems"])


def test_census_rejects_duplicate_entry_ids_and_non_float32_actions():
    rows = [dict(r, winner_id="e0") for r in _rows(n_tasks=1, per_task=3)]
    duplicate = _entry("e0", "t1", 1)
    duplicate.payload.action_chunk = torch.zeros(10, 32, dtype=torch.float64)
    res = census_identity(rows, [_entry("e0"), duplicate], _sidecar(["e0"]))
    assert not res["passed"]
    assert any("duplicate library entry id" in p for p in res["problems"])
    assert any("action dtype" in p for p in res["problems"])


def test_census_rejects_winner_not_in_library():
    rows = [dict(r, winner_id="ghost") for r in _rows(n_tasks=1, per_task=3)]
    res = census_identity(rows, [_entry("e0")], _sidecar(["e0"]))
    assert not res["passed"]
    assert any("not in the library" in p for p in res["problems"])


# ---------------- step-name parsing (regression) ----------------

class _FakeH5(dict):
    """Minimal stand-in: the real files use zero-padded group names."""


def test_step_index_parses_zero_padded_names():
    from exp.dispatch_surface.d0_check import step_group_index

    h5 = _FakeH5({"step_0000": 1, "step_0007": 1, "step_0103": 1, "attrs_blob": 1})
    assert step_group_index(h5) == {0: "step_0000", 7: "step_0007", 103: "step_0103"}


def test_step_index_also_handles_unpadded_names():
    from exp.dispatch_surface.d0_check import step_group_index

    assert step_group_index(_FakeH5({"step_0": 1, "step_12": 1})) == {0: "step_0", 12: "step_12"}


def test_step_index_rejects_two_names_for_one_parsed_step():
    from exp.dispatch_surface.d0_check import step_group_index

    with pytest.raises(ValueError, match="both parse"):
        step_group_index(_FakeH5({"step_0": 1, "step_0000": 1}))


# ---------------- resume-parity timestep (regression) ----------------

def test_accumulated_start_t_differs_from_the_literal_tier():
    from exp.dispatch_surface.d0_check import EXPECTED_NUM_STEPS, START_T_WS, accumulated_start_t

    acc = accumulated_start_t(START_T_WS, EXPECTED_NUM_STEPS)
    assert acc != START_T_WS
    # float32 accumulation drift, not a step-count error
    assert abs(acc - START_T_WS) < 1e-6


def test_accumulated_start_t_matches_the_loops_own_arithmetic():
    from exp.dispatch_surface.d0_check import accumulated_start_t

    for start_t, n in ((0.3, 10), (0.5, 10), (0.7, 10), (0.9, 10)):
        t = torch.tensor(1.0, dtype=torch.float32)
        dt = torch.tensor(-1.0 / n, dtype=torch.float32)
        for _ in range(round((1.0 - start_t) * n)):
            t = t + dt
        assert accumulated_start_t(start_t, n) == float(t)


# ---------------- D0 -> fit binding (G2R1-B2) ----------------

def test_controls_must_be_exactly_two_per_task_never_fewer():
    """A task with a thin pool used to silently contribute fewer controls."""
    from exp.dispatch_surface.d0_check import select_gpu_sample

    thin = [{"episode_id": "e0", "step_idx": 0, "task_id": 0, "winner_id": "w",
             "s": 0.9, "v": 1.0, "y_tau7": 100.0, "y_tau10": 2.0},
            {"episode_id": "e1", "step_idx": 1, "task_id": 0, "winner_id": "w",
             "s": 0.9, "v": 1.0, "y_tau7": 1.0, "y_tau10": 2.0}]
    with pytest.raises(ValueError, match="exactly 2 controls"):
        select_gpu_sample(thin, CONTROL_SEED, expected_tasks={0})


def test_census_rejects_a_duplicate_library_entry_id():
    """A duplicate id is collapsed by the id->entry map, so one payload would
    stand in for another during replay."""
    rows = [dict(r, winner_id="e0") for r in _rows(n_tasks=1, per_task=3)]
    entries = [_entry("e0", "t0", 0), _entry("e0", "t1", 5)]
    res = census_identity(rows, entries, _sidecar(["e0"]))
    assert not res["passed"]
    assert any("duplicate library entry id" in p for p in res["problems"])


def test_census_rejects_a_non_float32_action_chunk():
    rows = [dict(r, winner_id="e0") for r in _rows(n_tasks=1, per_task=3)]
    e = _entry("e0")
    e.payload.action_chunk = torch.zeros(10, 32, dtype=torch.float64)
    res = census_identity(rows, entries := [e], _sidecar(["e0"]))
    assert not res["passed"]
    assert any("action_chunk dtype" in p for p in res["problems"])
    assert entries


def test_h5_tree_attestation_changes_when_one_file_changes(tmp_path):
    """The tree digest must be content-based, not a file count."""
    from exp.dispatch_surface.d0_check import _h5_tree_attestation

    a, b = tmp_path / "a.h5", tmp_path / "b.h5"
    a.write_bytes(b"one")
    b.write_bytes(b"two")
    before = _h5_tree_attestation(tmp_path, [a, b])
    b.write_bytes(b"two!")
    after = _h5_tree_attestation(tmp_path, [a, b])
    assert before != after


# ---------------- D0 -> fit bindings that the D0 validator cannot reach ----------------

def _d0_and_fit(tmp_path, *, lib_sha, sidecar_sha="S" * 64, split_suite="libero_10"):
    """Record/args pair for the bindings fit_surface adds on top of D0's own.

    D0's validator re-attests the paths IT recorded, so a record that cleared a
    different library at a different path passes it. Only the fit can catch
    that, by cross-checking against the rebuild record.
    """
    import json

    from exp.dispatch_surface.d0_check import D0_PROTOCOL
    from exp.dispatch_surface.fit_surface import _file_sha256

    table = tmp_path / "t.jsonl"; table.write_text("{}\n")
    weights = tmp_path / "w.npz"; weights.write_bytes(b"w")
    yaml_f = tmp_path / "c.yaml"; yaml_f.write_text("k: v\n")
    (tmp_path / "r.json").write_text(json.dumps(
        {"library_sha256": lib_sha, "noise_sidecar_sha256": sidecar_sha}))
    (tmp_path / "s.json").write_text(json.dumps({"suite": split_suite}))
    (tmp_path / "d0.json").write_text(json.dumps({
        "D0": "PASS", "protocol": D0_PROTOCOL, "suite": "libero_10",
        "inputs": {"files": {
            "table": {"sha256": _file_sha256(table)},
            "weights_npz": {"sha256": _file_sha256(weights)},
            "cache_yaml": {"sha256": _file_sha256(yaml_f)},
            "library_pkl": {"sha256": "L" * 64},
            "noise_sidecar": {"sha256": "S" * 64},
        }},
    }))
    args = type("A", (), {})()
    args.d0_record = str(tmp_path / "d0.json")
    args.table, args.weights_npz, args.cache_yaml = str(table), str(weights), str(yaml_f)
    args.rebuild_record = str(tmp_path / "r.json")
    args.split_manifest = str(tmp_path / "s.json")
    return args


@pytest.fixture
def _skip_attestation(monkeypatch):
    """Isolate the fit-side bindings from D0's own path re-attestation."""
    monkeypatch.setattr(
        "exp.dispatch_surface.d0_check.validate_input_attestation", lambda a: None)


def test_fit_rejects_a_d0_record_that_cleared_a_different_library(tmp_path, _skip_attestation):
    from exp.dispatch_surface.fit_surface import _validate_d0_record

    args = _d0_and_fit(tmp_path, lib_sha="X" * 64)
    with pytest.raises(SystemExit, match="not the same artifact"):
        _validate_d0_record(args)


def test_fit_rejects_a_d0_record_that_cleared_a_different_sidecar(tmp_path, _skip_attestation):
    from exp.dispatch_surface.fit_surface import _validate_d0_record

    args = _d0_and_fit(tmp_path, lib_sha="L" * 64, sidecar_sha="Z" * 64)
    with pytest.raises(SystemExit, match="not the same artifact"):
        _validate_d0_record(args)


def test_fit_rejects_a_d0_record_from_the_other_suite(tmp_path, _skip_attestation):
    from exp.dispatch_surface.fit_surface import _validate_d0_record

    args = _d0_and_fit(tmp_path, lib_sha="L" * 64, split_suite="libero_spatial")
    with pytest.raises(SystemExit, match="split manifest"):
        _validate_d0_record(args)


def test_fit_accepts_a_matching_d0_record(tmp_path, _skip_attestation):
    from exp.dispatch_surface.fit_surface import _validate_d0_record

    args = _d0_and_fit(tmp_path, lib_sha="L" * 64)
    assert _validate_d0_record(args)["D0"] == "PASS"
