"""CPU tests for exp/step_diag/recorder.py with stub sampling callables: the executed action and the
global torch RNG are untouched, every k of a sample shares its initial noise, the dense rule is
deterministic and attempt-free, shadow failures degrade to error rows, finalize rows close the
episode, and the arrays file is bound by sha256."""

import hashlib
import json
import pathlib

import numpy as np
import pytest
import torch

from exp.step_diag import recorder as R

H, D = 6, 4


def _spec(**over):
    base = dict(experiment_id="exp", env_id="pi05_rc", arm_id="shadow", mode="shadow", k_full=10, k_set=(1, 3),
                warm_ts=(0.3,), n_primary=4, n_dense_extra=2, action_shape=(H, D), config_sha="abc")
    base.update(over)
    return R.DiagSpec(**base)


class _Stub:
    """Records the noises it was given per k; returns noise-dependent chunks."""

    def __init__(self):
        self.calls = []

    def sample(self, z, k):
        self.calls.append((k, z.clone()))
        return z * 0.5 + k

    def resume(self, x_t, t):
        return x_t + t

    def top1(self):
        return (0.9, "entry-1", {0.3: torch.ones(H, D)})


def _identity(uid="u1", attempt=1, init_idx=3):
    return R.EpisodeIdentity.from_episode_start(
        experiment="robocasa365", task="CloseFridge", episode_id=init_idx,
        extra_metadata={"task_uid": uid, "attempt": attempt, "task_id": 1, "orig_init_state_idx": init_idx, "seed": 2000003})


def _rows(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def test_rows_arrays_pairing_and_rng_isolation(tmp_path):
    rec = R.DiagRecorder(_spec(), tmp_path, launch_id="L1")
    stub = _Stub()
    rec.begin_episode(_identity())
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    a_exec = torch.arange(H * D, dtype=torch.float32).reshape(H, D)
    row = rec.record(a_exec=a_exec, executed_steps=10, n_stage3_calls=10, hit_type="MISS", start_t=None,
                     schedule_id="pi05_v1", sample=stub.sample, resume=stub.resume, top1=stub.top1)
    assert torch.equal(torch.random.get_rng_state(), before)  # private generator only
    assert row["status"] == "ok" and row["n_full"] == 4 and len(row["noise_ids"]) == 4
    # every k of sample n used the same initial noise as the full run of sample n
    by_k = {}
    for k, z in stub.calls:
        by_k.setdefault(k, []).append(z)
    assert len(by_k[10]) == 4 and len(by_k[1]) == 4 and len(by_k[3]) == 4
    for n in range(4):
        assert torch.equal(by_k[10][n], by_k[1][n]) and torch.equal(by_k[10][n], by_k[3][n])
    assert row["warm_status"] == {"0.3000": "ok"} and row["top1_entry_id"] == "entry-1"
    assert row["shadow_nfe"] == 4 * 10 + 4 * (1 + 3) + 3
    rec.finalize_episode(True)
    rows = _rows(rec.rows_path)
    assert [r["status"] for r in rows] == ["ok", "finalize"]
    fin = rows[-1]
    assert fin["terminal"] is True and fin["outcome"] is True and fin["n_decisions"] == 1
    arrays = tmp_path / rows[0]["arrays"]
    assert hashlib.sha256(arrays.read_bytes()).hexdigest() == rows[0]["arrays_sha256"]
    z = np.load(arrays)
    assert np.array_equal(z["a_exec_0000"], a_exec.numpy())  # executed action stored verbatim
    assert z["a_full_0000"].shape == (4, H, D) and z["a_k1_0000"].shape == (4, H, D) and z["a_warm_0.3000_0000"].shape == (H, D)
    assert z["a_full_0000"].dtype == np.float32


def test_dense_rule_is_deterministic_and_attempt_free():
    hits = [R.is_dense_decision("pi05_rc", "CloseFridge", 3, i) for i in range(200)]
    assert any(hits) and not all(hits)
    assert hits == [R.is_dense_decision("pi05_rc", "CloseFridge", 3, i) for i in range(200)]
    # noise identity changes with attempt (a retry is a new draw) but the dense selection does not
    assert R.noise_seed("e", "pi05_rc", "T", 5, 1, 0, 0) != R.noise_seed("e", "pi05_rc", "T", 5, 2, 0, 0)


def test_dense_decision_adds_full_samples_only(tmp_path):
    rec = R.DiagRecorder(_spec(), tmp_path)
    stub = _Stub()
    rec.begin_episode(_identity())
    dense_idx = next(i for i in range(500) if R.is_dense_decision("pi05_rc", "CloseFridge", 3, i))
    for i in range(dense_idx + 1):
        stub.calls.clear()
        row = rec.record(a_exec=torch.zeros(H, D), executed_steps=10, n_stage3_calls=10, hit_type="MISS", start_t=None,
                         schedule_id="pi05_v1", sample=stub.sample, resume=stub.resume, top1=stub.top1)
    assert row["dense"] is True and row["n_full"] == 6
    ks = [k for k, _ in stub.calls]
    assert ks.count(10) == 6 and ks.count(1) == 4 and ks.count(3) == 4  # extras are full-only


def test_shadow_exception_becomes_error_row_and_no_candidate_is_recorded(tmp_path):
    rec = R.DiagRecorder(_spec(), tmp_path)
    rec.begin_episode(_identity())

    def bad_sample(z, k):
        raise RuntimeError("boom")

    row = rec.record(a_exec=torch.zeros(H, D), executed_steps=10, n_stage3_calls=10, hit_type="MISS", start_t=None,
                     schedule_id="pi05_v1", sample=bad_sample, resume=None, top1=None)
    assert row["status"] == "error" and "boom" in row["error_reason"]
    stub = _Stub()
    row2 = rec.record(a_exec=torch.zeros(H, D), executed_steps=10, n_stage3_calls=10, hit_type="MISS", start_t=None,
                      schedule_id="pi05_v1", sample=stub.sample, resume=stub.resume, top1=lambda: (None, None, None))
    assert row2["status"] == "ok" and row2["warm_status"] == {"0.3000": "no_candidate"}
    rec.finalize_episode(False)
    rows = _rows(rec.rows_path)
    assert [r["status"] for r in rows] == ["error", "ok", "finalize"] and rows[-1]["n_decisions"] == 2


def test_evidence_only_rows_and_unbalanced_start(tmp_path):
    rec = R.DiagRecorder(_spec(mode="plain", k_set=(), warm_ts=()), tmp_path)
    rec.begin_episode(_identity("u1"))
    rec.record(a_exec=torch.zeros(H, D), executed_steps=2, n_stage3_calls=2, hit_type="MISS", start_t=None, schedule_id="pi05_v1")
    rec.begin_episode(_identity("u2"))  # unbalanced: closes u1 as non-terminal
    rec.finalize_episode(True)
    rows = _rows(rec.rows_path)
    fins = [r for r in rows if r["status"] == "finalize"]
    assert [f["task_uid"] for f in fins] == ["u1", "u2"]
    assert fins[0]["terminal"] is False and fins[0]["reason"] == "unbalanced_episode_start" and fins[1]["terminal"] is True
    ok = [r for r in rows if r["status"] == "ok"][0]
    assert ok["executed_steps"] == 2 and ok["n_full"] == 0 and ok["arrays"] is not None


def test_record_outside_episode_is_dropped(tmp_path):
    rec = R.DiagRecorder(_spec(), tmp_path)
    assert rec.record(a_exec=torch.zeros(H, D), executed_steps=1, n_stage3_calls=1, hit_type="MISS", start_t=None,
                      schedule_id="pi05_v1") is None


def test_remaining_steps_by_schedule_direction():
    assert R._remaining_steps(10, 0.3, "pi05_rc") == 3 and R._remaining_steps(4, 0.75, "groot_rc") == 1
    assert R._remaining_steps(8, 0.5, "groot_libero_10") == 4


@pytest.mark.parametrize("t,exp", [(0.1, 1), (0.2, 2), (0.3, 3)])
def test_pi05_remaining(t, exp):
    assert R._remaining_steps(10, t, "pi05_libero_spatial") == exp


def test_finalize_records_client_stamp_and_mismatch(tmp_path):
    rec = R.DiagRecorder(_spec(), tmp_path, launch_id="SRV")
    ident = R.EpisodeIdentity.from_episode_start(
        experiment="robocasa365", task="CloseFridge", episode_id=1,
        extra_metadata={"task_uid": "u", "attempt": 1, "launch_id": "L-drv", "arm_id": "shadow", "experiment_id": "exp",
                        "config_sha": "abc", "lane": "main"})
    rec.begin_episode(ident)
    rec.finalize_episode(True)
    fin = _rows(rec.rows_path)[-1]
    assert fin["launch_id"] == "SRV"  # the server's own launch id
    assert fin["client_stamp"] == {"launch_id": "L-drv", "arm_id": "shadow", "experiment_id": "exp", "config_sha": "abc"}
    assert fin["stamp_mismatch"] == [] and fin["lane"] == "main"
    rec.begin_episode(R.EpisodeIdentity.from_episode_start(
        experiment="robocasa365", task="CloseFridge", episode_id=2,
        extra_metadata={"task_uid": "v", "attempt": 1, "arm_id": "plain_k2", "config_sha": ""}))
    rec.finalize_episode(False)
    fin2 = _rows(rec.rows_path)[-1]
    assert fin2["stamp_mismatch"] == ["arm_id"] and fin2["client_stamp"]["launch_id"] is None  # empty sha is not a mismatch


def test_shape_mismatch_is_an_error_row_and_sessions_are_isolated(tmp_path):
    rec = R.DiagRecorder(_spec(), tmp_path)
    stub = _Stub()
    rec.begin_episode(_identity())
    row = rec.record(a_exec=torch.zeros(H + 1, D), executed_steps=10, n_stage3_calls=10, hit_type="MISS", start_t=None,
                     schedule_id="pi05_v1", sample=stub.sample, resume=stub.resume, top1=stub.top1)
    assert row["status"] == "error" and "action_shape" in row["error_reason"]
    rec.finalize_episode(False)
    # two connections interleave: each session keeps its own decision counter and arrays
    s1, s2 = rec.session(), rec.session()
    s1.begin_episode(_identity("c1", init_idx=1))
    s2.begin_episode(_identity("c2", init_idx=2))
    s1.record(a_exec=torch.zeros(H, D), executed_steps=10, n_stage3_calls=10, hit_type="MISS", start_t=None,
              schedule_id="pi05_v1", sample=stub.sample, resume=stub.resume, top1=stub.top1)
    s2.record(a_exec=torch.zeros(H, D), executed_steps=10, n_stage3_calls=10, hit_type="MISS", start_t=None,
              schedule_id="pi05_v1", sample=stub.sample, resume=stub.resume, top1=stub.top1)
    s1.record(a_exec=torch.zeros(H, D), executed_steps=10, n_stage3_calls=10, hit_type="MISS", start_t=None,
              schedule_id="pi05_v1", sample=stub.sample, resume=stub.resume, top1=stub.top1)
    s2.finalize_episode(True)
    s1.finalize_episode(False)
    rows = _rows(rec.rows_path)
    fins = {r["task_uid"]: r for r in rows if r["status"] == "finalize"}
    assert fins["c1"]["n_decisions"] == 2 and fins["c2"]["n_decisions"] == 1
    assert fins["c1"]["session_id"] != fins["c2"]["session_id"] and fins["c1"]["terminal"] and fins["c2"]["terminal"]
    assert [r["decision_idx"] for r in rows if r["task_uid"] == "c1" and r["status"] == "ok"] == [0, 1]
    z1, z2 = np.load(tmp_path / fins["c1"]["arrays"]), np.load(tmp_path / fins["c2"]["arrays"])
    assert "a_full_0001" in z1.files and "a_full_0001" not in z2.files
    rec.close()  # idempotent on closed sessions
    assert len([r for r in _rows(rec.rows_path) if r["status"] == "finalize"]) == 3
