import json
from pathlib import Path
import shutil

import numpy as np
import pytest

from exp.offline_search.debug import reader, schema
from exp.offline_search.debug.fixtures import make_synthetic_arm
from exp.offline_search.debug.capacity import capacity, budget, forecast_campaign


@pytest.fixture
def arm(tmp_path):
    return reader.open_arm(make_synthetic_arm(tmp_path / "run"), "synthetic")


def test_arrays_preserve_requested_order_duplicates_and_prompts(arm):
    ids = arm.decisions().decision_id.tolist()
    chosen = [ids[20], ids[0], ids[20], ids[-1]]
    data = arm.decision_arrays(["state_wire", "served_chunk", "prompt"], chosen)
    assert data["state_wire"].dtype == np.float64
    assert data["served_chunk"].shape == (4, 10, 32)
    assert np.array_equal(data["state_wire"][0], data["state_wire"][2])
    assert data["prompt"].tolist() == ["move the object"] * 4
    assert arm.images(chosen)["wrist"].shape == (4, 16, 16, 3)
    with pytest.raises(KeyError, match="missing decision_ids"):
        arm.decision_arrays(["state_wire"], ["absent"])
    with pytest.raises(KeyError, match="missing field"):
        arm.decision_arrays(["absent"], chosen)


def test_roundtrip_cache_is_disposable_and_invalidates(arm):
    before = arm.decisions()
    after = reader.open_arm(arm.run_root, "synthetic").decisions()
    assert isinstance(after.diag.iloc[0], dict)
    assert isinstance(after.rows.iloc[0], list)
    assert before.decision_id.tolist() == after.decision_id.tolist()
    assert (arm.debug_dir / "derived/decisions.parquet").exists()
    path = arm.server_dirs[0] / "decisions.jsonl"
    rows = reader.read_jsonl(path)
    rows[0]["look_reason"] = "new_reason"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert "new_reason" in arm.decisions().look_reason.values


def test_accepted_filter_retry_retention_and_pairing(arm):
    path = arm.journal_path
    journal = reader.read_jsonl(path)
    journal[0]["accepted"] = False
    path.write_text("".join(json.dumps(r) + "\n" for r in journal))
    assert len(arm.episodes()) == 5
    assert len(arm.episodes(False)) == 6
    assert len(arm.decisions(accepted_only=False)) == 54
    assert len(arm.decisions()) == 46
    assert len(reader.pair(arm, arm)) == 5
    assert reader.pair(arm, arm).journal_success_a.tolist() == reader.pair(arm, arm).journal_success_b.tolist()


def test_conflicting_accepted_attempts_rejected(arm):
    rows = reader.read_jsonl(arm.journal_path)
    rows.append(dict(rows[0], attempt=2))
    arm.journal_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    with pytest.raises(ValueError, match="duplicate accepted"):
        arm.journal()


def test_orphan_event_visible_in_outer_join(arm):
    ep = arm.episodes().iloc[0]
    path = arm.debug_dir / "client" / ep.episode_key / "events.jsonl"
    events = reader.read_jsonl(path)
    events.append(dict(ev="decision", decision_id=ep.episode_key + ":1:999", decision_seq=999))
    path.write_text("".join(json.dumps(r) + "\n" for r in events))
    ds = arm.decisions()
    assert ds.server_join.eq("missing_server").sum() == 1
    assert ds.loc[ds.decision_seq.eq(999), "episode_key"].iloc[0] == ep.episode_key


def test_duplicate_server_ids_are_not_silently_deduplicated(arm):
    path = arm.server_dirs[0] / "decisions.jsonl"
    with path.open("a") as f:
        f.write(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate server decision"):
        arm.decisions()


def test_ragged_contacts_merge_across_blocks(arm):
    ep = arm.episodes().iloc[0]
    path = arm.debug_dir / "client" / ep.episode_key / "controls_0000.npz"
    data = reader.read_npz(path)
    n = len(data["control_idx"])
    for j, (lo, hi) in enumerate(((0, 20), (20, n))):
        arrays = {k: v[lo:hi] for k, v in data.items() if k != "contact_off"}
        arrays["contact_off"] = np.arange(hi - lo + 1, dtype=np.int32)
        schema.write_npz_block(path.parent / ("controls_%04d.npz" % j), arrays)
    merged = arm.controls(ep.episode_key)
    assert np.array_equal(merged["contact_off"], np.arange(n + 1))
    assert merged["contact_geom"].shape == (n, 2)
    assert np.array_equal(merged["qpos"], data["qpos"])


def test_variable_actuator_substeps_are_padded_losslessly(arm):
    ep = arm.episodes().iloc[0]
    directory = arm.debug_dir / "client" / ep.episode_key
    original = reader.read_npz(directory / "controls_0000.npz")
    n = len(original["control_idx"])
    for j, (lo, hi, steps) in enumerate(((0, 20, 1), (20, n, 3))):
        data = {k: v[lo:hi] for k, v in original.items() if k != "contact_off"}
        data["contact_off"] = np.arange(hi - lo + 1, dtype=np.int32)
        data["act_substep"] = np.full((hi - lo, steps, 7), j, np.float32)
        data["act_substep_count"] = np.full(hi - lo, steps, np.int32)
        schema.write_npz_block(directory / ("controls_%04d.npz" % j), data)
    data = arm.controls(ep.episode_key)
    assert data["act_substep"].shape == (n, 3, 7)
    assert np.isnan(data["act_substep"][:20, 1:]).all()
    assert data["act_substep_count"].tolist() == [1] * 20 + [3] * (n - 20)


def test_aug_catalog_snapshot_alignment(arm):
    ids = arm.decisions().decision_id.tolist()[::11]
    assert arm.aug("policy_shadow", ids)["decision_id"].tolist() == ids
    assert arm.aug("policy_shadow", ids)["chunk"].shape == (len(ids), 10, 32)
    assert len(arm.catalog()) == 16
    catalog_dir = arm.run_root / arm.manifest.pop("catalog")
    catalog_dir.rename(catalog_dir.with_name("explicit_cell"))
    arm.manifest["arm_spec"] = dict(cell="explicit_cell")
    assert arm.catalog() is None  # A serving cell cannot distinguish current from the big library.
    assert {"reset", "final", "000000"} <= set(arm.snapshots(arm.episodes().episode_key.iloc[0]))


def test_process_logs_metas_stats_and_truncated_tail(arm):
    directory = arm.server_dirs[0]
    original = directory / "decisions.jsonl"
    rows = reader.read_jsonl(original)
    original.write_text("".join(json.dumps(r) + "\n" for r in rows[:20]))
    process = directory / "decisions_23.jsonl"
    process.write_text("".join(json.dumps(dict(r, pid=23)) + "\n" for r in rows[20:]) + '{"killed":')
    meta = reader.read_json(directory / "meta.json")
    (directory / "meta_23.json").write_text(json.dumps(dict(meta, pid=23)))
    (directory / "writer_stats_23.json").write_text(json.dumps(dict(pid=23, bytes=123)))
    arm = reader.open_arm(arm.run_root, "synthetic")
    ds = arm.decisions()
    assert len(ds) == 54 and ds.server_join.eq("verified").all()
    assert len(arm.server_metas) == 2
    assert arm.record_meta(arm.server_records().iloc[-1].to_dict())["pid"] == 23
    assert arm.read_issues[0]["code"] == "truncated_jsonl_tail"
    cached = reader.open_arm(arm.run_root, "synthetic")
    assert len(cached.decisions()) == 54
    assert cached.read_issues == arm.read_issues
    report = capacity(cached)
    assert directory.name + "/writer_stats_23.json" in report["io_stats"]
    total = report["source_bytes"]
    assert np.isclose(report["episode_bytes"]["total"] + report["shared_metadata_bytes"] + report["unaccepted_bytes"], total)
    assert report["read_issues"] == arm.read_issues


def test_unterminated_jsonl_is_skipped_and_interior_corruption_is_fatal(tmp_path):
    path = tmp_path / "log.jsonl"
    path.write_bytes(b'{"ok": 1}\n{"unterminated": 2}')
    with pytest.warns(RuntimeWarning, match="final line without newline skipped"):
        assert reader.read_jsonl(path) == [dict(ok=1)]
    path.write_bytes(b'{"broken":\n{"ok": 1}\n')
    with pytest.raises(ValueError, match="invalid JSON"):
        reader.read_jsonl(path)


def test_subset_init_comes_from_uid_and_original_init_is_preserved(arm):
    path = arm.server_dirs[0] / "decisions.jsonl"
    rows = reader.read_jsonl(path)
    for r in rows:
        r.pop("orig_init_state_idx", None)
        r["init"] = 100 + int(r["task_uid"].rsplit(":", 1)[1])
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ds = arm.decisions()
    assert ds.init.tolist() == [int(uid.rsplit(":", 1)[1]) for uid in ds.task_uid]
    assert ds.orig_init_state_idx.eq(ds.init + 100).all()
    assert reader.uid_identity(dict(task_uid="arm:eval:3:21", init=88, orig_init_state_idx=77))["orig_init_state_idx"] == 77


def test_new_process_metadata_is_discovered_by_existing_reader(arm):
    directory = arm.server_dirs[0]
    meta = dict(arm.server_meta, pid=99, H=16)
    (directory / "meta_99.json").write_text(json.dumps(meta))
    assert arm.record_meta(dict(_server_dir=str(directory), pid=99))["H"] == 16


def test_only_requested_npz_members_are_decompressed(arm, monkeypatch):
    ids = arm.decisions().decision_id.tolist()[:3]
    original = np.lib.npyio.NpzFile.__getitem__
    loaded = []
    def track(archive, key):
        loaded.append(key)
        return original(archive, key)
    monkeypatch.setattr(np.lib.npyio.NpzFile, "__getitem__", track)
    arm.decision_arrays(["state_wire", "prompt"], ids)
    assert set(loaded) <= {"decision_id", "state_wire", "prompts", "prompt_idx"}
    loaded.clear()
    arm.controls(arm.episodes().episode_key.iloc[0], ["is_settle"])
    assert set(loaded) == {"is_settle"}


def test_capacity_reconciles_sources_and_forecast(arm):
    data = capacity(arm, remaining_arms=70)
    total = sum(p.stat().st_size for p in arm.debug_dir.rglob("*") if p.is_file() and "derived" not in p.relative_to(arm.debug_dir).parts)
    assert data["source_bytes"] == total
    assert np.isclose(data["episode_bytes"]["total"] + data["shared_metadata_bytes"] + data["unaccepted_bytes"], total)
    assert data["forecast"]["estimated_remaining_bytes"] == data["episode_bytes"]["mean"] * 35000
    assert data["fields"]["server/img_wrist"]["compressed"] > 0
    forecast = forecast_campaign([arm], [dict(name="synthetic", model="pi05", suite="synthetic", expected_episodes=500),
                                        dict(name="future", model="pi05", suite="synthetic")])
    assert forecast["status"] == "available"
    assert forecast["remaining_episodes"] == 994
    arm.manifest["suite"] = arm.server_meta.pop("suite")
    assert forecast_campaign([arm], [dict(name="future", model="pi05", suite="synthetic")])["status"] == "available"
    arm.server_meta["suite"] = None
    assert capacity(arm)["suite"] == "synthetic"
    unavailable = forecast_campaign([arm], [dict(name="unknown", model="groot", suite="synthetic")])
    assert unavailable["status"] == "unavailable"
    assert unavailable["estimated_remaining_bytes"] is None


def test_budget_recomputed_counts_and_measured_vs_assumed(arm):
    data = budget(arm)
    assert data["counts"] == dict(N=54, V=28., M=0., full_looks=28., wrist_looks=0., third_looks=0., camera_completions=0., stage23_calls=0.)
    assert np.isclose(data["ledgers"]["declared"]["owner_IR"], .152 * 28 / 54)
    path = arm.server_dirs[0] / "decisions.jsonl"
    rows = reader.read_jsonl(path)
    rows[0].update(camera_mode="wrist_only", camera_completions=1, policy_calls=1, stage23_calls=1, owner_cost=.0646 + .0502 + .848)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    data = budget(arm)
    assert data["ledgers"]["measured"]["work"] > data["ledgers"]["r4_assumed"]["work"]
    assert not data["warnings"]


def test_missing_dispatch_counts_report_unavailable(arm):
    path = arm.server_dirs[0] / "decisions.jsonl"
    rows = reader.read_jsonl(path)
    rows[0].pop("policy_calls")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert budget(arm)["status"] == "unavailable"


def test_p3_adapter_minimal_legacy_capture(tmp_path):
    root = tmp_path / "legacy"
    arm_dir = root / "runs/old"
    uid = "old:eval:0:1"
    ek = schema.episode_key(uid, 1)
    server = arm_dir / "server_old"
    client = arm_dir / "client_telemetry" / ek
    server.mkdir(parents=True)
    client.mkdir(parents=True)
    (arm_dir / "client").mkdir()
    (arm_dir / "client/journal.jsonl").write_text(json.dumps(dict(task_uid=uid, attempt=1, task_id=0, init=1, accepted=True, success=True, status="done")) + "\n")
    records = [dict(ev="p3_v2_startup", catalog=dict(model="groot", interface={})),
               dict(ev="dec", uid=uid, attempt=1, step=0, vision=True, hit=True),
               dict(ev="p3_decision", uid=uid, attempt=1, step=0, input_archive="input.npz")]
    (server / "decisions_old.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    schema.write_npz_block(server / "input.npz", dict(raw_state=np.zeros(8), executed_chunk=np.zeros((16, 32)),
                                                        wire_chunk=np.zeros((16, 7)), policy_chunk=np.ones((16, 32))))
    records = [dict(ev="decision", decision_step=0), dict(ev="control", control=0, decision_step=0, chunk_offset=0,
                action_issued=[0.] * 7, reward=0., done=True,
                after=dict(qpos=[0.] * 12, qvel=[0.] * 12, observation_numeric=dict(robot0_eef_pos=[0., 0., 0.]))),
               dict(ev="rollout_end", controls=1, decisions=1, success=True, environment_seed=7)]
    (client / "controls.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    arm = reader.p3v2_adapter(root, "old")
    assert arm.server_meta["model"] == "groot"
    ids = arm.decisions().decision_id.tolist()
    assert arm.decision_arrays(["served_wire"], ids)["served_wire"].shape == (1, 16, 7)
    assert arm.controls(ek)["eef_pos"].shape == (1, 3)
    assert arm.aug("policy_shadow", ids)["chunk"].shape == (1, 16, 32)
    with pytest.raises(KeyError, match="unsupported"):
        arm.images(ids)
    assert not (arm.debug_dir / "derived").exists()


def test_p3_tables_without_server_logs_remain_read_only(tmp_path):
    import pandas as pd
    root = tmp_path / "legacy"
    directory = root / "runs/old/client"
    directory.mkdir(parents=True)
    uid = "old:eval:0:1"
    (directory / "journal.jsonl").write_text(json.dumps(dict(task_uid=uid, attempt=1, accepted=True, success=False, status="failed")) + "\n")
    table_dir = root / "tables/cell"
    table_dir.mkdir(parents=True)
    pd.DataFrame([dict(arm="old", uid=uid, attempt=1, task_id=0, init=1, model="groot", step=0, vision=True,
                       hit=True, source="cache", actual_controls=5)]).to_csv(table_dir / "decisions.csv", index=False)
    pd.DataFrame([dict(arm="old", uid=uid, attempt=1, task_id=0, init=1, model="groot", suite="spatial", lib="current")]).to_csv(table_dir / "episodes.csv", index=False)
    before = {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    arm = reader.p3v2_adapter(root, "old")
    assert arm.server_meta["model"] == "groot"
    assert arm.decisions().actual_controls.iloc[0] == 5
    assert arm.episodes().suite.iloc[0] == "spatial"
    assert before == {str(p): p.read_bytes() for p in root.rglob("*") if p.is_file()}
