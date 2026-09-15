"""CurveRegistry: identity, sharing, isolation, atomicity, file snapshots."""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from openpi.cache.components.online_rit import ContinuationFeedback, OnlineRiskCurves
from openpi.cache.online_state import CurveRegistry, RegistryKey

TIERS = [7, 6, 4]


def test_serving_registry_requires_a_persistent_root(tmp_path):
    registry = CurveRegistry(require_persistence=True)
    with pytest.raises(ValueError, match="requires judge.state_log_dir"):
        registry.attach(yaml_id="a", library_sha256="l", fingerprint="f", factory=_factory())
    key = registry.attach(yaml_id="a", library_sha256="l", fingerprint="f", factory=_factory(), log_dir=str(tmp_path))
    assert registry.log_dir(key).is_dir()


@pytest.mark.parametrize("operation", ["feedback", "snapshot"])
def test_persistence_failure_stays_invalid_after_storage_recovers(tmp_path, monkeypatch, operation):
    from exp.online_rit.pick_terminal_state import pick_terminal

    registry = CurveRegistry(state_log_root=str(tmp_path))
    key = registry.attach(yaml_id="io", library_sha256="a" * 64, fingerprint="f", factory=_factory())
    def broken(*args, **kwargs):
        raise OSError("simulated full disk")
    with monkeypatch.context() as patch:
        if operation == "feedback":
            patch.setattr(registry, "_append_feedback_locked", broken)
            snapshot = registry.decision_snapshot(key, ("io", "e", 0, 0), .5, .3)
            with pytest.raises(OSError):
                registry.record_batch(key, snapshot, [])  # zero learned batches must still invalidate
        else:
            patch.setattr(type(tmp_path), "write_text", broken)
            with pytest.raises(OSError):
                registry.flush(key)
    assert registry.is_invalid(key)
    registry.flush(key, "recovered")
    with pytest.raises(SystemExit, match="invalid"):
        pick_terminal(registry.log_dir(key))


def _factory():
    return lambda: OnlineRiskCurves(
        knots=[0.0, 0.5, 1.0], tier_indices=TIERS, alpha=0.05, window=64, n_min=5
    )


def test_same_identity_shares_and_different_identity_isolates():
    reg = CurveRegistry()
    k1 = reg.attach(yaml_id="a", library_sha256="1" * 64, fingerprint="fp", factory=_factory())
    k2 = reg.attach(yaml_id="a", library_sha256="1" * 64, fingerprint="fp", factory=_factory())
    k3 = reg.attach(yaml_id="b", library_sha256="1" * 64, fingerprint="fp", factory=_factory())
    k4 = reg.attach(yaml_id="a", library_sha256="2" * 64, fingerprint="fp", factory=_factory())
    assert k1 == k2 and reg.curves(k1) is reg.curves(k2)
    assert reg.curves(k1) is not reg.curves(k3) and reg.curves(k1) is not reg.curves(k4)
    assert reg.describe(k1)["n_attached"] == 2


def test_fingerprint_conflict_and_missing_identity_are_refused():
    reg = CurveRegistry()
    reg.attach(yaml_id="a", library_sha256="1" * 64, fingerprint="fp", factory=_factory())
    with pytest.raises(ValueError):
        reg.attach(yaml_id="a", library_sha256="1" * 64, fingerprint="other", factory=_factory())
    with pytest.raises(ValueError):
        reg.attach(yaml_id="", library_sha256="1" * 64, fingerprint="fp", factory=_factory())
    with pytest.raises(ValueError):
        reg.attach(yaml_id="a", library_sha256="", fingerprint="fp", factory=_factory())


def test_concurrent_commits_lose_nothing_and_never_tear():
    reg = CurveRegistry()
    key = reg.attach(yaml_id="a", library_sha256="1" * 64, fingerprint="fp", factory=_factory())
    n_threads, per_thread = 64, 25
    errors: list[Exception] = []

    def worker(tid: int) -> None:
        rng = np.random.default_rng(tid)
        try:
            for n in range(per_thread):
                s = float(rng.uniform(0, 1))
                snap = reg.decision_snapshot(key, ("a", f"u{tid}", 0, n), s, 0.5)
                # every tier's pre-decision value must come from one revision
                assert len(snap.q_pre) == len(TIERS)
                fb = [ContinuationFeedback(t, float(rng.uniform(0, 1)), "shadow") for t in TIERS]
                reg.record_batch(key, snap, fb)
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    c = reg.curves(key)
    assert c.n_updates == n_threads * per_thread
    assert c.n_feedback == n_threads * per_thread * len(TIERS)


def test_second_connection_sees_first_connections_learning():
    reg = CurveRegistry()
    key = reg.attach(yaml_id="a", library_sha256="1" * 64, fingerprint="fp", factory=_factory())
    reg.attach(yaml_id="a", library_sha256="1" * 64, fingerprint="fp", factory=_factory())
    before = reg.decision_snapshot(key, ("a", "b", 0, 0), 0.5, 0.5)
    assert before.cuts[7] is None
    for n in range(10):
        snap = reg.decision_snapshot(key, ("a", "x", 0, n), 0.5, 0.5)
        reg.record_batch(key, snap, [ContinuationFeedback(7, 0.1, "executed")])
    after = reg.decision_snapshot(key, ("a", "b", 0, 1), 0.5, 0.5)
    assert after.cuts[7] is not None and after.decision_revision == 10


def test_snapshots_and_feedback_rows_are_written(tmp_path):
    reg = CurveRegistry(state_log_root=str(tmp_path))
    key = reg.attach(
        yaml_id="arm1", library_sha256="1" * 64, fingerprint="fp", factory=_factory(), snapshot_every=3
    )
    d = tmp_path / RegistryKey("arm1", "1" * 64).dirname / reg.server_instance_id
    assert reg.log_dir(key) == d
    assert (d / "state_00000000_attach.json").exists()
    for n in range(7):
        snap = reg.decision_snapshot(key, ("arm1", "u", 0, n), 0.5, 0.5)
        reg.record_batch(key, snap, [ContinuationFeedback(7, 0.1, "executed")], episode={"task_uid": "u"})
    names = sorted(p.name for p in d.glob("state_*_periodic.json"))
    assert names == ["state_00000003_periodic.json", "state_00000006_periodic.json"]
    reg.flush(key, "task_end")
    latest = json.loads((d / "state_latest.json").read_text())
    assert latest["n_updates"] == 7 and latest["reason"] == "task_end"
    rows = [json.loads(line) for line in (d / "feedback.jsonl").read_text().splitlines()]
    assert len(rows) == 7
    assert rows[0]["decision"]["decision_id"] == ["arm1", "u", 0, 0]
    assert rows[-1]["diag"]["update_revision_after"] == 7
    assert rows[0]["episode"]["task_uid"] == "u"


def test_judge_state_log_dir_wins_and_processes_are_isolated(tmp_path):
    reg_a = CurveRegistry(state_log_root=str(tmp_path / "root"), server_instance_id="procA")
    reg_b = CurveRegistry(state_log_root=str(tmp_path / "root"), server_instance_id="procB")
    ka = reg_a.attach(yaml_id="arm", library_sha256="1" * 64, fingerprint="fp", factory=_factory(), log_dir=str(tmp_path / "cfg"))
    kb = reg_b.attach(yaml_id="arm", library_sha256="1" * 64, fingerprint="fp", factory=_factory(), log_dir=str(tmp_path / "cfg"))
    assert reg_a.log_dir(ka) == tmp_path / "cfg" / RegistryKey("arm", "1" * 64).dirname / "procA"
    assert reg_b.log_dir(kb) == tmp_path / "cfg" / RegistryKey("arm", "1" * 64).dirname / "procB"
    assert reg_a.log_dir(ka) != reg_b.log_dir(kb)


def test_invalid_observation_marks_the_stream_and_its_snapshots(tmp_path):
    reg = CurveRegistry(state_log_root=str(tmp_path))
    key = reg.attach(yaml_id="arm", library_sha256="1" * 64, fingerprint="fp", factory=_factory())
    assert reg.is_invalid(key) is False
    reg.mark_invalid(key, ["executed:warm875:not finite"], decision={"decision_idx": 4})
    assert reg.is_invalid(key) and reg.describe(key)["invalid_reasons"] == ["4:executed:warm875:not finite"]
    latest = json.loads((reg.log_dir(key) / "state_latest.json").read_text())
    assert latest["flow_invalid"] is True and latest["reason"] == "invalid"
    from openpi.cache.components.online_rit import verify_state_sha

    verify_state_sha(latest)  # sealed over the final document, extra fields included
    latest["flow_invalid"] = False
    with pytest.raises(ValueError):
        verify_state_sha(latest)


def test_frozen_entry_counts_observations_only(tmp_path):
    reg = CurveRegistry(state_log_root=str(tmp_path))
    def factory():
        return OnlineRiskCurves(
            knots=[0.0, 0.5, 1.0], tier_indices=TIERS, alpha=0.05, window=64, n_min=5, update_enabled=False
        )

    key = reg.attach(yaml_id="f", library_sha256="1" * 64, fingerprint="fp", factory=factory)
    snap = reg.decision_snapshot(key, ("f", "u", 0, 0), 0.5, 0.5)
    diag = reg.record_batch(key, snap, [ContinuationFeedback(7, 0.1, "executed")], n_rejected=1)
    assert diag["learned"] is False and diag["update_revision_after"] == 0
    assert reg.curves(key).n_observed == 1 and reg.curves(key).n_rejected == 1
