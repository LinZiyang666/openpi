"""Unit tests for ``exp/trajectory_deviation/compute_deviate_scores.py`` (plan §10 + §18.B3).

We cannot hit a real policy server in CI, so these tests cover:

- The pure deviate-score math (``_pairwise_l2_mean`` + ``compute_deviate_score``)
  with hand-computed reference values so any drift in the formula (e.g.
  accidental switch to RMS or wrong flatten axis) is caught.
- The §18.B3 pdist equivalence: mean of pairwise L2s matches the naive
  ``v[:, None, :] - v[None, :, :]`` broadcast.
- The aggregate pipeline end-to-end against a stubbed ``load_gt_episode``
  so the jsonl parsing + per-episode dispatch stays locked.
- Phase1Runner / Phase2Runner against a fake client, to lock the roll-out
  contract (``client.infer(obs)["actions"]``) and the jsonl record shape.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import numpy as np
import pytest

import exp.trajectory_deviation.compute_deviate_scores as cds


# ---------------------------------------------------------------------------
# _pairwise_l2_mean: §18.B3 pdist-vs-naive equivalence
# ---------------------------------------------------------------------------


def test_pairwise_l2_mean_matches_naive_broadcast() -> None:
    """Plan §18.B3: the pdist-backed implementation must match the naive
    ``v[:, None, :] - v[None, :, :]`` formulation to avoid numerical drift
    when swapping implementations."""
    rng = np.random.default_rng(0)
    v = rng.standard_normal((7, 11)).astype(np.float64)
    # Naive O(M^2) with temporary (M, M, D) tensor.
    diff = v[:, None, :] - v[None, :, :]
    naive = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(v.shape[0], k=1)
    naive_mean = naive[iu].mean()
    assert cds._pairwise_l2_mean(v) == pytest.approx(naive_mean, rel=1e-12)


def test_pairwise_l2_mean_single_sample_returns_zero() -> None:
    """With M<2, pairwise mean is undefined; the function returns 0.0 so
    the downstream ``max(bg, floor)`` keeps deviate_score finite."""
    assert cds._pairwise_l2_mean(np.zeros((1, 5))) == 0.0
    assert cds._pairwise_l2_mean(np.zeros((0, 5))) == 0.0


# ---------------------------------------------------------------------------
# compute_deviate_score: full pipeline on hand-built inputs
# ---------------------------------------------------------------------------


def test_compute_deviate_score_matches_reference_values() -> None:
    """Hand-constructed case: 2 cycles, 2 samples, trivial action-chunks.

    G2 §26 MF-1 moved this metric back to "first-action L2 only" — so only
    ``chunk[:, 0, :]`` of the bg/cache stacks matters; downstream horizon
    entries (here a single dummy row at ``H=2``) must not bleed into the
    result. Cycle 0 locks ``bg_l2=0 → deviate_score uses floor`` and cycle
    1 locks the ``cache_first_action vs gt_first_action`` path.
    """
    # (M=2, T=2, H=2, Ad=2) — the second horizon row is *noise* we want to
    # make sure the metric ignores.
    bg = np.array([
        [[[0.0, 0.0], [99.0, 99.0]], [[0.0, 0.0], [99.0, 99.0]]],
        [[[0.0, 0.0], [99.0, 99.0]], [[1.0, 1.0], [99.0, 99.0]]],
    ], dtype=np.float32)
    cache = np.array([
        [[0.5, 0.5], [42.0, 42.0]],
        [[2.0, 2.0], [42.0, 42.0]],
    ], dtype=np.float32)
    # GT is first-action only: (T=2, Ad=2).
    gt_first = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
    ], dtype=np.float32)

    out = cds.compute_deviate_score(bg, cache, gt_first, floor=0.1)

    # Cycle 0: all bg[:,0,0,:] identical → 0 (the H=1 row of 99s is ignored).
    assert out["background_l2"][0] == pytest.approx(0.0)
    # Cycle 1: two samples (0,0) and (1,1) at horizon 0 → L2 = sqrt(2), single pair → mean = sqrt(2).
    assert out["background_l2"][1] == pytest.approx(np.sqrt(2.0), rel=1e-6)
    # Cache[0] vs GT[0] per cycle (Ad=2 distance).
    assert out["cache_l2"][0] == pytest.approx(np.sqrt(0.5), rel=1e-6)
    assert out["cache_l2"][1] == pytest.approx(np.sqrt(8.0), rel=1e-6)
    # deviate_score[0] uses floor (bg == 0 → max(0, 0.1) = 0.1).
    assert out["deviate_score"][0] == pytest.approx(np.sqrt(0.5) / 0.1, rel=1e-6)
    # deviate_score[1] uses bg_l2 (√2 > 0.1).
    assert out["deviate_score"][1] == pytest.approx(np.sqrt(8.0) / np.sqrt(2.0), rel=1e-6)


def test_compute_deviate_score_rejects_mismatched_shapes() -> None:
    """Shape guards must fire before any nan-propagation surprise."""
    # bg matches cache (M, T, H, Ad); gt has wrong action_dim.
    bg = np.zeros((2, 3, 5, 2), dtype=np.float32)
    cache = np.zeros((3, 5, 2), dtype=np.float32)
    gt = np.zeros((3, 3), dtype=np.float32)  # wrong action_dim (3 vs 2)
    with pytest.raises(ValueError):
        cds.compute_deviate_score(bg, cache, gt)
    # bg must be 4-D (M, T, H, Ad).
    good_gt = np.zeros((3, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        cds.compute_deviate_score(np.zeros((3, 5, 2)), cache, good_gt)
    # cache must be 3-D.
    with pytest.raises(ValueError):
        cds.compute_deviate_score(bg, np.zeros((3, 2)), good_gt)


# ---------------------------------------------------------------------------
# aggregate: jsonl parsing + per-episode dispatch
# ---------------------------------------------------------------------------


def test_aggregate_reads_jsonl_and_writes_deviate_score_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end Phase 3 without touching h5py: monkeypatch
    ``load_gt_episode`` so this test runs in the plain venv."""
    bg_path = tmp_path / "bg_cfgA.jsonl"
    cache_path = tmp_path / "cache_cfgA.jsonl"

    # 1 episode, 2 samples, 2 cycles, 1 horizon step, 2 action dims.
    bg_records = [
        {"config": "cfgA", "episode": "task_0/episode_0", "sample_idx": 0,
         "chunks": [[[0.0, 0.0]], [[0.0, 0.0]]]},
        {"config": "cfgA", "episode": "task_0/episode_0", "sample_idx": 1,
         "chunks": [[[0.0, 0.0]], [[1.0, 1.0]]]},
    ]
    bg_path.write_text("\n".join(json.dumps(r) for r in bg_records) + "\n")

    cache_record = {
        "config": "cfgA", "episode": "task_0/episode_0",
        "chunks": [[[0.5, 0.5]], [[2.0, 2.0]]],
    }
    cache_path.write_text(json.dumps(cache_record) + "\n")

    # G2 §26 MF-1: gt return shape is now (T, Ad) — per-cycle first action.
    fake_gt_first = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

    def fake_load(gt_dir, ep_name):  # noqa: ARG001
        assert ep_name == "task_0/episode_0"
        return [], fake_gt_first

    monkeypatch.setattr(cds, "load_gt_episode", fake_load)

    out_path = tmp_path / "deviate_score_cfgA.json"
    out = cds.aggregate(bg_path, cache_path, gt_dir=tmp_path, out_path=out_path, floor=0.1)

    assert "task_0/episode_0" in out
    scores = out["task_0/episode_0"]
    assert scores["background_l2"][0] == pytest.approx(0.0)
    assert scores["background_l2"][1] == pytest.approx(np.sqrt(2.0), rel=1e-6)
    assert scores["cache_l2"][1] == pytest.approx(np.sqrt(8.0), rel=1e-6)
    # JSON file must match the returned dict exactly.
    assert json.loads(out_path.read_text()) == {
        k: v for k, v in out.items()
    } or json.loads(out_path.read_text()) == out


def test_aggregate_filters_stale_jsonl_rows_when_episode_list_is_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Per-config filtering must also constrain Phase 3 aggregation.

    This prevents stale rows from an earlier wider run in the same out-dir
    from leaking back into ``deviate_score_*.json``.
    """
    bg_path = tmp_path / "bg_cfgA.jsonl"
    cache_path = tmp_path / "cache_cfgA.jsonl"
    bg_records = [
        {"config": "cfgA", "episode": "task_0/episode_0", "sample_idx": 0,
         "chunks": [[[0.0, 0.0]]]},
        {"config": "cfgA", "episode": "task_0/episode_1", "sample_idx": 0,
         "chunks": [[[9.0, 9.0]]]},
    ]
    cache_records = [
        {"config": "cfgA", "episode": "task_0/episode_0", "chunks": [[[0.0, 0.0]]]},
        {"config": "cfgA", "episode": "task_0/episode_1", "chunks": [[[9.0, 9.0]]]},
    ]
    bg_path.write_text("\n".join(json.dumps(r) for r in bg_records) + "\n")
    cache_path.write_text("\n".join(json.dumps(r) for r in cache_records) + "\n")

    monkeypatch.setattr(
        cds,
        "load_gt_episode",
        lambda *a, **k: ([], np.zeros((1, 2), dtype=np.float32)),
    )

    out = cds.aggregate(
        bg_path,
        cache_path,
        gt_dir=tmp_path,
        out_path=tmp_path / "out.json",
        episodes=["task_0/episode_0"],
    )

    assert list(out) == ["task_0/episode_0"]


def test_aggregate_skips_episodes_without_cache_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """If Phase 1 finished for an episode but Phase 2 crashed mid-way, we
    must warn and skip — not blow up with KeyError. The state JSON carries
    that episode as failed; the operator can re-run Phase 2 with --resume.
    """
    bg_path = tmp_path / "bg_cfgA.jsonl"
    cache_path = tmp_path / "cache_cfgA.jsonl"
    bg_path.write_text(json.dumps({
        "config": "cfgA", "episode": "lonely", "sample_idx": 0,
        "chunks": [[[0.0, 0.0]]],
    }) + "\n")
    cache_path.write_text("")  # empty

    # G2 §26 MF-1: gt shape is (T, Ad).
    monkeypatch.setattr(cds, "load_gt_episode", lambda *a, **k: ([], np.zeros((1, 2), np.float32)))

    with caplog.at_level("WARNING"):
        out = cds.aggregate(bg_path, cache_path, tmp_path, tmp_path / "out.json")
    assert out == {}
    assert "lonely" in caplog.text


# ---------------------------------------------------------------------------
# Phase 1 / Phase 2 runners against a fake client
# ---------------------------------------------------------------------------


class _FakeClient:
    """Minimal stand-in for ``WebsocketClientPolicy``.

    ``infer`` returns a canned per-cycle chunk plus a counter we read in
    tests to make sure the runner called infer once per obs step.
    """

    def __init__(self, chunk: np.ndarray) -> None:
        self._chunk = chunk
        self.infer_calls = 0
        self.episode_start_kwargs: Optional[Dict[str, Any]] = None
        self.episode_end_kwargs: Optional[Dict[str, Any]] = None
        self.closed = False

    def episode_start(self, *, experiment, task, episode_id, episode_name) -> None:
        self.episode_start_kwargs = dict(
            experiment=experiment, task=task,
            episode_id=episode_id, episode_name=episode_name,
        )

    def episode_end(self, *, success: bool) -> None:
        self.episode_end_kwargs = {"success": success}

    def infer(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:  # noqa: ARG002
        self.infer_calls += 1
        return {"actions": self._chunk.copy()}

    def close(self) -> None:
        self.closed = True


def _fake_common(tmp_path: Path, client: _FakeClient, episodes: List[str]) -> cds._PhaseCommon:
    return cds._PhaseCommon(
        config_id="cfgA",
        gt_dir=tmp_path / "gt",
        episodes=episodes,
        out_dir=tmp_path / "out",
        host="127.0.0.1",
        port=9000,
        client_factory=lambda host, port: client,
    )


def test_phase1_runner_writes_per_sample_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lock the Phase-1 jsonl record schema: aggregate() reads these keys
    directly and any rename would break Phase 3 silently."""
    chunk = np.full((5, 7), 0.5, dtype=np.float32)
    client = _FakeClient(chunk)
    common = _fake_common(tmp_path, client, episodes=["task_0/episode_0"])

    # Stub load_gt_episode to return 3 cycles of identical obs.
    fake_obs = [{"observation/image": np.zeros((4, 4, 3), np.uint8)} for _ in range(3)]
    monkeypatch.setattr(cds, "load_gt_episode", lambda *a, **k: (fake_obs, np.zeros((3, 5, 7))))

    runner = cds.Phase1Runner(
        state_path=tmp_path / "phase1_state.json",
        common=common,
        M=2,
        max_retries=0,
    )
    runner.units = {u.unit_key: u for u in runner.build_units()}
    assert set(runner.units.keys()) == {
        "cfgA:task_0/episode_0:0", "cfgA:task_0/episode_0:1",
    }

    result = runner.execute_unit(runner.units["cfgA:task_0/episode_0:1"])
    assert result == {"T": 3}
    assert client.infer_calls == 3
    assert client.episode_start_kwargs["experiment"] == "deviate_score_phase1"
    # episode_end must fire even though we return successfully — the try/finally
    # ensures Phase 1 cleans up the connection regardless.
    assert client.episode_end_kwargs == {"success": True}
    assert client.closed is True

    rows = [
        json.loads(line)
        for line in (common.out_dir / "bg_cfgA.jsonl").read_text().splitlines()
    ]
    assert rows == [{
        "config": "cfgA", "episode": "task_0/episode_0", "sample_idx": 1,
        "chunks": chunk[None, :, :].repeat(3, axis=0).tolist(),
    }]


def test_phase_runners_share_execute_unit_call_sequence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cleanup/07 lock: after merging Phase1Runner and Phase2Runner into a
    shared ``_PhaseRunner`` body, the call sequence in ``execute_unit`` must
    stay ``episode_start → infer×T → episode_end(success=True)`` for BOTH
    phases. This is the structural invariant the plan asked to preserve
    while deliberately NOT touching lifecycle (F2 follow-up)."""

    class _SeqClient(_FakeClient):
        def __init__(self, chunk: np.ndarray) -> None:
            super().__init__(chunk)
            self.seq: List[str] = []

        def episode_start(self, *, experiment, task, episode_id, episode_name) -> None:
            self.seq.append("start")
            super().episode_start(
                experiment=experiment, task=task,
                episode_id=episode_id, episode_name=episode_name,
            )

        def episode_end(self, *, success: bool) -> None:
            self.seq.append(f"end:{success}")
            super().episode_end(success=success)

        def infer(self, obs):
            self.seq.append("infer")
            return super().infer(obs)

    chunk = np.full((3, 7), 0.5, dtype=np.float32)
    fake_obs = [{"x": 0}, {"x": 1}]
    monkeypatch.setattr(
        cds, "load_gt_episode", lambda *a, **k: (fake_obs, np.zeros((2, 5, 7)))
    )

    # Phase 1 — single sample.
    client1 = _SeqClient(chunk)
    common1 = _fake_common(tmp_path / "p1", client1, episodes=["task_0/episode_0"])
    r1 = cds.Phase1Runner(
        state_path=tmp_path / "p1" / "state.json", common=common1, M=1, max_retries=0,
    )
    r1.units = {u.unit_key: u for u in r1.build_units()}
    r1.execute_unit(next(iter(r1.units.values())))

    assert client1.seq == ["start", "infer", "infer", "end:True"]
    assert client1.closed is True
    assert client1.episode_start_kwargs is not None
    assert client1.episode_start_kwargs["experiment"] == "deviate_score_phase1"

    # Phase 2 — same sequence, different experiment tag, no sample_idx.
    client2 = _SeqClient(chunk)
    common2 = _fake_common(tmp_path / "p2", client2, episodes=["task_0/episode_0"])
    r2 = cds.Phase2Runner(
        state_path=tmp_path / "p2" / "state.json", common=common2, max_retries=0,
    )
    r2.units = {u.unit_key: u for u in r2.build_units()}
    r2.execute_unit(next(iter(r2.units.values())))

    assert client2.seq == ["start", "infer", "infer", "end:True"]
    assert client2.closed is True
    assert client2.episode_start_kwargs is not None
    assert client2.episode_start_kwargs["experiment"] == "deviate_score_phase2"

    # Structural: both subclasses must route through the same `_PhaseRunner`
    # body so a future change to execute_unit touches both phases at once.
    assert isinstance(r1, cds._PhaseRunner)
    assert isinstance(r2, cds._PhaseRunner)


def test_phase2_runner_writes_cache_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk = np.full((5, 7), 0.25, dtype=np.float32)
    client = _FakeClient(chunk)
    common = _fake_common(tmp_path, client, episodes=["task_3/episode_1"])

    fake_obs = [{"observation/image": np.zeros((4, 4, 3), np.uint8)}]
    monkeypatch.setattr(cds, "load_gt_episode", lambda *a, **k: (fake_obs, np.zeros((1, 5, 7))))

    runner = cds.Phase2Runner(
        state_path=tmp_path / "phase2_state.json",
        common=common,
        max_retries=0,
    )
    runner.units = {u.unit_key: u for u in runner.build_units()}
    result = runner.execute_unit(runner.units["cfgA:task_3/episode_1"])
    assert result == {"T": 1}
    assert client.closed is True
    rows = [
        json.loads(line)
        for line in (common.out_dir / "cache_cfgA.jsonl").read_text().splitlines()
    ]
    assert rows == [{
        "config": "cfgA", "episode": "task_3/episode_1",
        "chunks": [chunk.tolist()],
    }]


def test_phase_runner_closes_client_when_episode_start_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed episode_start still owns an open websocket connection.

    Step 2 creates one short-lived client per unit, so this edge case must
    close the client or a flaky server can leak file descriptors until the
    client process hits ``EMFILE``.
    """

    class _StartFailClient(_FakeClient):
        def episode_start(self, *, experiment, task, episode_id, episode_name) -> None:
            super().episode_start(
                experiment=experiment,
                task=task,
                episode_id=episode_id,
                episode_name=episode_name,
            )
            raise RuntimeError("start failed")

    chunk = np.full((5, 7), 0.5, dtype=np.float32)
    client = _StartFailClient(chunk)
    common = _fake_common(tmp_path, client, episodes=["task_0/episode_0"])
    monkeypatch.setattr(
        cds,
        "load_gt_episode",
        lambda *a, **k: ([{"x": 0}], np.zeros((1, 5, 7))),
    )

    runner = cds.Phase1Runner(
        state_path=tmp_path / "phase1_state.json",
        common=common,
        M=1,
        max_retries=0,
    )
    runner.units = {u.unit_key: u for u in runner.build_units()}

    with pytest.raises(RuntimeError, match="start failed"):
        runner.execute_unit(next(iter(runner.units.values())))

    assert client.closed is True
    assert client.episode_end_kwargs is None
    assert client.infer_calls == 0


def test_phase_runner_uses_gt_prompt_as_cache_task_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The websocket episode_start task becomes the server-side task_key.

    Cache artifacts are keyed by the LIBERO task description, so Step 2 must
    pass the GT prompt rather than a synthetic cfg/episode/sample label or the
    in-memory backend will filter every candidate out.
    """
    chunk = np.full((5, 7), 0.5, dtype=np.float32)
    client = _FakeClient(chunk)
    common = _fake_common(tmp_path, client, episodes=["task_0/episode_0"])
    prompt = "pick up the black bowl next to the plate and place it on the plate"
    monkeypatch.setattr(
        cds,
        "load_gt_episode",
        lambda *a, **k: ([{"prompt": prompt}], np.zeros((1, 5, 7))),
    )

    runner = cds.Phase1Runner(
        state_path=tmp_path / "phase1_state.json",
        common=common,
        M=1,
        max_retries=0,
    )
    runner.units = {u.unit_key: u for u in runner.build_units()}

    runner.execute_unit(next(iter(runner.units.values())))

    assert client.episode_start_kwargs is not None
    assert client.episode_start_kwargs["task"] == prompt


# ---------------------------------------------------------------------------
# parallel_run smoke: BaseRunState must have it (locked by Layer A1 + §10.1)
# ---------------------------------------------------------------------------


def test_base_run_state_supports_parallel_run(tmp_path: Path) -> None:
    """§10.1 explicitly adds ``parallel_run`` to BaseRunState so Phase 1/2
    can use a thread pool. Lock that the method still exists (the Phase
    runners would silently fall back to serial if a refactor removed it)."""
    import exp.common._run_state_base as base

    assert hasattr(base.BaseRunState, "parallel_run")


# ---------------------------------------------------------------------------
# discover_episodes
# ---------------------------------------------------------------------------


def _write_gt_stub(
    path: Path,
    *,
    success: bool | None,
    task_id: int | None = None,
    init_state_idx: int | None = None,
    orig_init_state_idx: int | None = None,
) -> None:
    """Minimal HDF5 with just the attrs discover_episodes cares about."""
    import h5py  # type: ignore[import]

    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        if success is not None:
            f.attrs["success"] = bool(success)
        if task_id is not None:
            f.attrs["task_id"] = int(task_id)
        if init_state_idx is not None:
            f.attrs["init_state_idx"] = int(init_state_idx)
        if orig_init_state_idx is not None:
            f.attrs["orig_init_state_idx"] = int(orig_init_state_idx)


def test_discover_episodes_skips_failed_gt_by_default(tmp_path: Path) -> None:
    """G2 §26 MF-4: failed GT rollouts never reach the goal; their deviate
    scores are noise for Step 3. Default discover must drop them."""
    _write_gt_stub(tmp_path / "task_0" / "episode_0.h5", success=True)
    _write_gt_stub(tmp_path / "task_0" / "episode_1.h5", success=False)
    _write_gt_stub(tmp_path / "task_3" / "episode_0.h5", success=True)

    names = cds.discover_episodes(tmp_path)
    assert names == ["task_0/episode_0", "task_3/episode_0"]


def test_discover_episodes_include_failed_flag_keeps_them(tmp_path: Path) -> None:
    """Escape hatch: ``include_failed=True`` keeps success=False episodes
    so a researcher can deliberately study the off-manifold regime."""
    _write_gt_stub(tmp_path / "task_0" / "episode_0.h5", success=True)
    _write_gt_stub(tmp_path / "task_0" / "episode_1.h5", success=False)

    names = cds.discover_episodes(tmp_path, include_failed=True)
    assert names == ["task_0/episode_0", "task_0/episode_1"]


def test_discover_episodes_warns_and_skips_unknown_by_default(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    """Legacy HDF5 missing the ``success`` attr (pre-Layer-C-MF-3 data)
    must warn AND skip by default — silently including them would let
    unknown-quality rollouts leak into Step 3 aggregation."""
    _write_gt_stub(tmp_path / "task_0" / "episode_0.h5", success=True)
    _write_gt_stub(tmp_path / "task_0" / "episode_legacy.h5", success=None)

    with caplog.at_level("WARNING"):
        names = cds.discover_episodes(tmp_path)
    assert names == ["task_0/episode_0"]
    assert any("episode_legacy" in rec.message for rec in caplog.records)


def test_discover_episodes_include_unknown_flag_keeps_legacy(
    tmp_path: Path, caplog: pytest.LogCaptureFixture,
) -> None:
    _write_gt_stub(tmp_path / "task_0" / "episode_legacy.h5", success=None)

    with caplog.at_level("WARNING"):
        names = cds.discover_episodes(tmp_path, include_unknown=True)
    assert names == ["task_0/episode_legacy"]
    # Even with include_unknown the warning still fires so the operator
    # notices the stale data.
    assert any("missing 'success'" in rec.message for rec in caplog.records)


def test_load_failed_units_by_config_uses_only_failed_original_inits(tmp_path: Path) -> None:
    path = tmp_path / "cache_eval_results.json"
    path.write_text(json.dumps([
        {"config_id": "cfgA", "task_id": 3, "init_state_idx": 0,
         "orig_init_state_idx": 15, "success": False},
        {"config_id": "cfgA", "task_id": 3, "init_state_idx": 1,
         "orig_init_state_idx": 28, "success": True},
        {"config_id": "cfgB", "task_id": 4, "init_state_idx": 9,
         "success": False},
    ]))

    failed = cds.load_failed_units_by_config(path)

    assert failed == {
        "cfgA": {(3, 15)},
        "cfgB": {(4, 9)},
    }


def test_filter_episodes_by_failed_units_matches_gt_original_init_attrs(tmp_path: Path) -> None:
    _write_gt_stub(
        tmp_path / "task_0" / "episode_0.h5",
        success=True,
        task_id=3,
        init_state_idx=0,
        orig_init_state_idx=15,
    )
    _write_gt_stub(
        tmp_path / "task_0" / "episode_1.h5",
        success=True,
        task_id=3,
        init_state_idx=1,
        orig_init_state_idx=28,
    )
    _write_gt_stub(
        tmp_path / "task_1" / "episode_0.h5",
        success=True,
        task_id=4,
        init_state_idx=0,
        orig_init_state_idx=2,
    )

    episodes = cds.discover_episodes(tmp_path)
    filtered = cds.filter_episodes_by_failed_units(tmp_path, episodes, {(3, 15), (4, 9)})

    assert filtered == ["task_0/episode_0"]
