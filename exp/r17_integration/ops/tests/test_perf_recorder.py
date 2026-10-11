"""CPU-only recorder correctness: rates, missing timings, and safe log tails."""
import json
from pathlib import Path

from exp.r17_integration.ops import perf_recorder as perf


def test_partial_line_and_rotation(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_bytes(b'{"time": 1}\n{"time":')
    rows, cursor, cov = perf.read_increment(path, None)
    assert rows == [{"time": 1}]
    assert cov["initial"]
    with path.open("ab") as f:
        f.write(b' 2}\n')
    rows, cursor, _ = perf.read_increment(path, cursor)
    assert rows == [{"time": 2}]
    assert perf.read_increment(path, cursor)[0] == []
    path.write_text('{"time": 3}\n')
    assert perf.read_increment(path, cursor)[0] == [{"time": 3}]


def test_driver_and_server_merge_and_missing_queue(tmp_path):
    (tmp_path / "driver").mkdir()
    (tmp_path / "replica_0").mkdir()
    (tmp_path / "driver/attempts.jsonl").write_text(json.dumps(dict(
        task_uid="one", accepted=True, error=None, duration_s=3, time=100)) + "\n")
    (tmp_path / "replica_0/serving.jsonl").write_text(json.dumps(dict(
        elapsed_seconds=.5, service_seconds=.5, queue_seconds=None, time=100,
        timing_scope="end_to_end_including_stage_queue_wait; service_seconds_is_not_kernel_time")) + "\n")
    (tmp_path / "replica_0/batching.jsonl").write_text(json.dumps(dict(
        schema="r17.fastwam.stage_batch.v1", stage="stage3", size=2, time=100)) + "\n")
    specs = [dict(name="a", path=str(tmp_path), cap=8, scope=scope) for scope in ("driver", "server")]
    result = perf.collect_files(specs, {}, 99)
    m = result["campaigns"]["a"]
    assert m["accepted_total"] == 1
    report = perf.metrics_report(m, 30, 8)
    assert report["episodes_per_min"] == 2
    assert report["policy_calls_per_s"] == 1/30
    assert report["queue_timing_status"] == "unsupported"
    assert "service_ms" not in report["values"]
    assert report["batches"]["stage3"]["p50"] == 2
    assert report["counts"]["explicit_window_expiries"] == 0
    assert perf.collect_files(specs, result["cursors"], 99)["campaigns"]["a"]["counts"]["server_calls"] == 0


def test_initial_partial_is_live_final_partial_is_terminal(tmp_path):
    for name, receipt in (("hs3_live", dict(status="PARTIAL", cleanup=[])),
                          ("hs3_closed", dict(status="PARTIAL", all_owned_closed=True)),
                          ("hs3_failed", dict(status="FAIL"))):
        p = tmp_path / name
        p.mkdir()
        (p / "campaign.json").write_text(json.dumps(dict(model="dp", workers=24)))
        (p / "RECEIPT.json").write_text(json.dumps(receipt))
    assert [c["name"] for c in perf.discover(tmp_path)] == ["hs3_live"]


def test_tether_slot_budget(tmp_path, monkeypatch):
    args = type("Args", (), dict(eval_root=tmp_path, interval=30))()
    recorder = perf.Recorder(args)
    class Worker:
        def poll(self):
            return None
    def start():
        recorder.worker = Worker()
    calls = []
    def remote(node, specs):
        calls.append(node)
        return dict(error="node_offline")
    monkeypatch.setattr(recorder, "start_worker", start)
    monkeypatch.setattr(recorder, "remote", remote)
    monkeypatch.setattr(perf, "node_snapshot", lambda previous: {})
    first = recorder.sample()
    assert first["tether_calls"] == 2
    assert calls == ["h100"]  # Other slot belongs to the persistent 107 collector.
    calls.clear()
    second = recorder.sample()
    assert second["tether_calls"] == 2
    assert sorted(calls) == ["h100", "timan108"]
    calls.clear()
    assert recorder.sample()["tether_calls"] == 1
    assert calls == ["h100"]


def test_analysis_ignores_restart_bootstrap_and_pairs_episode_timing(tmp_path):
    (tmp_path/"driver").mkdir()
    attempt = dict(task_uid="one", attempt=1, accepted=True, error=None, duration_s=1, time=30)
    (tmp_path/"driver/attempts.jsonl").write_text(json.dumps(attempt) + "\n")
    rows = [dict(task_uid="one", attempt=1, accepted=True, _kind="client_timing",
                 infers=2, infer_ms=240, env_ms=200, img_ms=0),
            *[dict(task_uid="one", attempt=1, accepted=True, server_timing=dict(infer_ms=100)) for _ in range(2)]]
    (tmp_path/"driver/per_step.jsonl").write_text("".join(json.dumps(r)+"\n" for r in rows))
    config = dict(name="a", path=str(tmp_path), cap=8, workers=1, server_node="local")
    samples = []
    for stamp, pid, accepted, calls in ((0, 1, 100, 100), (30, 1, 1, 2), (60, 2, 100, 100), (90, 2, 1, 4)):
        metrics = perf.empty_metrics()
        metrics["counts"].update(accepted=accepted, server_calls=calls)
        samples.append(dict(time=stamp, utc=str(stamp), pid=pid, configs=[config],
                            campaigns=dict(a=metrics), nodes={}, tether_calls=1, collection_seconds=1))
    path = tmp_path/"perf.jsonl"
    path.write_text("".join(json.dumps(r)+"\n" for r in samples))
    report = perf.analyze_samples(path)
    assert report["measured_seconds"] == 60
    campaign = report["campaigns"]["a"]
    assert campaign["episodes_per_min"] == 2
    assert campaign["policy_calls_per_s"] == .1
    assert campaign["completed_cohort"]["matched_episodes"] == 1
    assert campaign["completed_cohort"]["client"]["transport_plus_protocol_residual_ms"] == 20
    assert campaign["completed_cohort"]["client"]["residual_idle_setup_ms"] == 560
    assert "2.00 | 0.10" in perf.summary_text(samples)


def test_exclusive_dp_mode_never_contacts_h100_or_108(tmp_path, monkeypatch):
    args = type("Args", (), dict(eval_root=tmp_path, interval=30, no_h100=True, no_timan108=True))()
    recorder = perf.Recorder(args)
    class Worker:
        def poll(self):
            return None
    monkeypatch.setattr(recorder, "start_worker", lambda: setattr(recorder, "worker", Worker()))
    monkeypatch.setattr(recorder, "remote", lambda *_: (_ for _ in ()).throw(AssertionError("remote probe")))
    monkeypatch.setattr(perf, "node_snapshot", lambda previous: {})
    first = recorder.sample()
    assert first["tether_calls"] == 1
    assert "h100" not in first["nodes"]
    assert recorder.sample()["tether_calls"] == 0
