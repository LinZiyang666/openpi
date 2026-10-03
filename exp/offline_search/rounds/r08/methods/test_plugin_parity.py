"""R8 methods through S1's CPU fake-policy request-tape parity harness.

No socket/model/server is created. Reuse S1's real _ConnPolicy/session/observer
driver and extend its state digest with every R8 controller variable.
"""
import hashlib
import json
import pickle
from pathlib import Path
from unittest.mock import patch

import pytest

from exp.offline_search.closed_loop import plugin
from exp.offline_search.debug.tests import test_server_parity as harness
from exp.offline_search.debug.server.diagnostics import method_chain


def read_debug_output(directory):
    """Read both legacy and per-process journals, reporting killed final lines."""
    records, skipped = [], []
    for path in sorted(directory.glob("decisions*.jsonl")):
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.endswith(b"\n"):
                    skipped.append(dict(path=str(path), line=line_number, reason="unterminated final line"))
                    continue
                records.append(json.loads(line))
    stats = [json.loads(path.read_text()) for path in sorted(directory.glob("writer_stats*.json"))]
    return records, stats, skipped


@pytest.mark.parametrize("per_process", [False, True])
@pytest.mark.parametrize("final_line", ['{"decision_id":"killed"}', '{"decision_id":'])
def test_debug_output_layout_and_unterminated_line(tmp_path, per_process, final_line):
    suffix = "_123" if per_process else ""
    path = tmp_path / ("decisions" + suffix + ".jsonl")
    path.write_text('{"decision_id":"ok"}\n' + final_line)
    (tmp_path / ("writer_stats" + suffix + ".json")).write_text('{"written":1}')
    if per_process:
        (tmp_path / "decisions_456.jsonl").write_text('{"decision_id":"second"}\n')
        (tmp_path / "writer_stats_456.json").write_text('{"written":1}')
    records, stats, skipped = read_debug_output(tmp_path)
    assert [r["decision_id"] for r in records] == (["ok", "second"] if per_process else ["ok"])
    assert sum(s["written"] for s in stats) == len(records)
    assert skipped == [dict(path=str(path), line=2, reason="unterminated final line")]


@pytest.mark.parametrize("variant", ["FL", "W10", "W5", "IP", "P10", "O5b"])
def test_new_method_request_tape_debug_parity(tmp_path, variant):
    name = "r8_pi05_spatial_P10" if variant == "P10" else "r8_pi05_spatial_50_" + variant
    path = Path("/tmp/r8_S5/fits") / (name + ".pkl")
    if not path.is_file():
        pytest.skip("run rounds.r08.ops.prefit before the artifact parity suite")
    original_runtime, original_digest, original_infer = harness.runtime, harness.digest, plugin._ConnPolicy.infer

    def fitted(_family):
        with path.open("rb") as stream:
            blob = pickle.load(stream)
        return blob["method"], blob

    def runtime(method, blob, directory, family):
        rt = original_runtime(method, blob, directory, family)
        rt.policy_tail = variant in ("IP", "P10", "O5b")
        rt.request_cameras = variant in ("W10", "W5")
        rt.oracle = variant == "O5b"
        rt.judge = plugin.JudgeSpec.parse("guard_only") if rt.policy_tail else None
        rt.opts.os_fit_artifact = str(path)
        return rt

    def digest(session, inner):
        state = []
        for method in method_chain(session.method):
            state.append({k: v for k, v in vars(method).items() if k.startswith(("_r8", "_oracle", "lottery_"))})
        return hashlib.sha256(original_digest(session, inner).encode() + pickle.dumps(state, protocol=4)).hexdigest()

    def infer(connection, obs, *args, **kwargs):
        if variant == "O5b":
            obs = dict(obs, __oracle__=dict(status="partial", privileged=True, objects=[
                dict(object_id="goal0", status="available", distance=.04, lifted=False, satisfied=False,
                     predicate_known=True, in_window=True),
                dict(object_id="unresolved", status="unsupported", reason="body unresolved")]))
        return original_infer(connection, obs, *args, **kwargs)

    with patch.object(harness, "fitted", fitted), patch.object(harness, "runtime", runtime), \
            patch.object(harness, "digest", digest), patch.object(plugin._ConnPolicy, "infer", infer):
        off = harness.replay(variant, tmp_path / "off", False)
        on = harness.replay(variant, tmp_path / "on", True)
    assert len(off[0]) >= 200
    assert off[:4] == on[:4]  # response bytes, full method state, RNG, legacy log records
    records, stats, skipped = read_debug_output(tmp_path / "on/debug")
    assert stats and sum(s["written"] for s in stats) == len(on[0])
    assert all(s["drained"] and not s["errors"] for s in stats)
    assert not skipped, skipped
    assert len(records) == len(on[0])
    assert all(r["diag_status"]["status"] == "available" for r in records)
    if variant == "W5":
        assert all(r["vision"] for r in records)
    if variant in ("W5", "W10"):
        assert any(r["camera_mode"] == "wrist_only" for r in records)
    if variant in ("IP", "P10", "O5b"):
        assert any(r["src"] == "policy_tail" for r in records)
        assert any(r["policy_calls"] for r in records)
    if variant == "O5b":
        assert all(r["oracle_status"]["status"] == "available" for r in records)
        anchors = [r["diag"] for r in records if r["diag"].get("fresh")]
        assert anchors and all(d["oracle_status"] == "partial" for d in anchors)
        assert all([o["status"] for o in d["oracle_objects"]] == ["available", "unsupported"] for d in anchors)
    report = dict(variant=variant, requests=len(on[0]), response_bytes_identical=True,
                  method_state_identical=True, rng_states_identical=True, legacy_records_identical=True,
                  writer_drained=True, skipped_final_lines=skipped)
    (tmp_path / "parity_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
