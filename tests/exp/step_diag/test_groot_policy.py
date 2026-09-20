"""CPU tests for exp/step_diag/groot.py with fake policy / runner objects: the shadow policy executes
the upstream full loop unchanged, draws its samples through ``staged.denoise_loop`` with a fresh
head-input mapping per call and the live schedule untouched, records warm rows from the winner's
snapshots, and the evidence wrapper reports steps_run / hit meta of the production object."""

from __future__ import annotations

import contextlib
import json
from types import SimpleNamespace

import numpy as np
import torch

from exp.step_diag import groot as G
from exp.step_diag import recorder as R
from openpi.cache.types import groot_n15_schedule

H, D = 16, 32


class _FakeRunner:
    """Staged runner double: run_stage3 = upstream loop (steps_run = live K), resume counts remaining steps."""

    def __init__(self, k=4):
        self._k = k
        self._model = SimpleNamespace(action_head=SimpleNamespace(num_inference_timesteps=k))
        self.head_input_calls = 0
        self.stage3_calls = 0
        self._timer = None

    @contextlib.contextmanager
    def session(self):
        yield

    def live_schedule(self):
        return groot_n15_schedule(self._k)

    def run_stage1(self, normalized):
        return SimpleNamespace(state=torch.zeros(1, D))

    def run_stage2_llm(self, stage1):
        return SimpleNamespace(action_inputs={"x": 1}, backbone_outputs={"b": 2})

    def _head_inputs(self, stage2):
        self.head_input_calls += 1
        return {"mapped": self.head_input_calls}

    def run_stage3(self, stage2, noise=None):
        self.stage3_calls += 1
        return SimpleNamespace(action_pred=torch.full((1, H, D), 7.0), steps_run=self._k)

    def run_stage3_from(self, stage2, start_x, start_t, *, schedule=None):
        remaining = schedule.remaining_steps(float(start_t))
        return SimpleNamespace(action_pred=(start_x[None] if start_x.dim() == 2 else start_x) + remaining, steps_run=remaining)


class _FakePolicy:
    def apply_transforms(self, obs):
        return obs

    def unapply_transforms(self, d):
        return {"action": np.asarray(d["action"])}


def _spec(**over):
    base = dict(experiment_id="e", env_id="groot_rc", arm_id="shadow", mode="shadow", k_full=4, k_set=(1, 2, 3),
                warm_ts=(0.75, 0.5), n_primary=4, n_dense_extra=2, action_shape=(H, D), config_sha="c")
    base.update(over)
    return R.DiagSpec(**base)


def _rows(rec):
    return [json.loads(line) for line in rec.rows_path.read_text().splitlines()]


def _start(policy, uid="u1"):
    policy.on_episode_start(experiment="robocasa365", task="CloseFridge", episode_id=0,
                            extra_metadata={"task_uid": uid, "attempt": 1, "task_id": 1, "orig_init_state_idx": 0, "seed": 2_000_000})


def test_shadow_policy_executes_upstream_loop_and_samples_with_fresh_head_inputs(tmp_path, monkeypatch):
    calls = []

    def fake_denoise_loop(head, backbone_outputs, action_inputs, *, noise, num_steps, **kw):
        calls.append((backbone_outputs["mapped"], int(num_steps), noise.clone()))
        return noise * 0.5 + num_steps  # deterministic in (noise, k)

    monkeypatch.setattr(G._staged, "denoise_loop", fake_denoise_loop)
    runner = _FakeRunner(k=4)
    rec = R.DiagRecorder(_spec(), tmp_path)
    policy = G.GrootDiagPolicy(_FakePolicy(), runner, orchestrator=None, diag=rec, schedule=groot_n15_schedule(4))
    _start(policy)
    out = policy.get_action({"state": np.zeros(D, dtype=np.float32)})
    assert np.array_equal(out["action"], np.full((H, D), 7.0))  # the upstream loop's action, executed as is
    assert runner.stage3_calls == 1 and runner._model.action_head.num_inference_timesteps == 4  # live head untouched
    # 4 full + 4 x 3 reduced samples, each with a fresh head-input mapping
    assert len(calls) == 4 * (1 + 3) and runner.head_input_calls == len(calls)
    assert sorted({m for m, _, _ in calls}) == list(range(1, len(calls) + 1))
    # every k of sample n shares the sample's noise
    by_n = {}
    for m, k, z in calls:
        by_n.setdefault((m - 1) // 4, []).append((k, z))
    for n, ks in by_n.items():
        assert [k for k, _ in ks] == [4, 1, 2, 3] and all(torch.equal(ks[0][1], z) for _, z in ks)
    policy.on_episode_end(True)
    rows = _rows(rec)
    ok = rows[0]
    assert ok["status"] == "ok" and ok["executed_steps"] == 4 and ok["n_stage3_calls"] == 1
    assert ok["warm_status"] == {"0.7500": "no_candidate", "0.5000": "no_candidate"}
    z = np.load(tmp_path / rows[-1]["arrays"])
    assert z["a_full_0000"].shape == (4, H, D) and z["a_k1_0000"].shape == (4, H, D) and np.all(z["a_exec_0000"] == 7.0)
    assert np.allclose(z["a_k2_0000"][0], (calls[0][2] * 0.5 + 2).numpy())


def test_shadow_policy_warm_rows_from_winner_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(G._staged, "denoise_loop", lambda head, b, a, *, noise, num_steps, **kw: noise + num_steps)
    runner = _FakeRunner(k=4)
    rec = R.DiagRecorder(_spec(), tmp_path)
    winner = SimpleNamespace(score=0.9, entry_id="e1", payload=SimpleNamespace(
        intermediates={0.75: torch.ones(H, D), 0.5: torch.ones(H, D) * 2}))
    orch = SimpleNamespace(check=lambda cp, **kw: winner, broadcast_action=lambda a: None, clear=lambda: None,
                           on_task_begin=lambda k: None, on_task_end=lambda: None,
                           on_episode_start=lambda **kw: None, on_episode_end=lambda: None)
    policy = G.GrootDiagPolicy(_FakePolicy(), runner, orchestrator=orch, diag=rec, schedule=groot_n15_schedule(4))
    _start(policy)
    policy.get_action({"state": np.zeros(D, dtype=np.float32)})
    policy.on_episode_end(False)
    rows = _rows(rec)
    ok = rows[0]
    assert ok["warm_status"] == {"0.7500": "ok", "0.5000": "ok"} and ok["top1_score"] == 0.9 and ok["top1_entry_id"] == "e1"
    assert ok["shadow_nfe"] == 4 * 4 + 4 * (1 + 2 + 3) + (1 + 2)
    z = np.load(tmp_path / rows[-1]["arrays"])
    assert np.all(z["a_warm_0.7500_0000"] == 1 + 1) and np.all(z["a_warm_0.5000_0000"] == 2 + 2)  # snapshot + remaining steps


def test_evidence_wrapper_records_steps_and_hit_meta(tmp_path):
    runner = _FakeRunner(k=4)

    class _Inner:
        def __init__(self):
            self.mode = "warm"

        def get_action(self, obs):
            if self.mode == "warm":
                out = runner.run_stage3_from(None, torch.zeros(H, D), 0.75, schedule=groot_n15_schedule(4))
                return {"action": out.action_pred.numpy(), "__hit_meta__": {"hit_type": "WARM_START", "start_t": 0.75}}
            if self.mode == "hit":
                return {"action": np.zeros((1, H, D)), "__hit_meta__": {"hit_type": "FULL_HIT"}}
            out = runner.run_stage3(None)
            return {"action": out.action_pred.numpy(), "__hit_meta__": {"hit_type": "MISS"}}

    inner = _Inner()
    rec = R.DiagRecorder(_spec(arm_id="warm_t0.75", mode="warm", k_set=(), warm_ts=()), tmp_path)
    wrapper = G.GrootEvidencePolicy(inner, runner, rec, schedule_id="groot_n15_k4_v1")
    _start(wrapper)
    wrapper.get_action({})
    inner.mode = "miss"
    wrapper.get_action({})
    inner.mode = "hit"
    wrapper.get_action({})
    wrapper.on_episode_end(True)
    rows = _rows(rec)
    ok = [r for r in rows if r["status"] == "ok"]
    assert [(r["hit_type"], r["start_t"], r["executed_steps"], r["n_stage3_calls"]) for r in ok] == [
        ("WARM_START", 0.75, 1, 1), ("MISS", None, 4, 1), ("FULL_HIT", None, 0, 0)]
    assert ok[2]["a_exec_missing"] is True and rows[-1]["n_decisions"] == 3
