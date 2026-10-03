"""R8 review regressions: support chains, terminal outcomes, process shards."""

import json
from copy import deepcopy

import numpy as np
import pytest

from exp.offline_search.debug.tools.physical import (
    adapters,
    audits,
    blind_drift,
    forensics,
    pairs,
    segmentation,
    selfcheck,
)
from exp.offline_search.debug.tools.physical.common import (
    Unavailable,
    input_summary,
    load_episodes,
    report,
)


def support_tape(ep, contacts):
    ep = deepcopy(ep)
    ep.meta["entities"] = dict(
        body_names=[
            "world",
            "bottom",
            "middle",
            "top",
            "bottom_collision_child",
            "drawer",
        ],
        body_parentid=[0, 0, 0, 0, 1, 0],
        geom_names=[
            "floor",
            "bottom_collision",
            "middle_collision",
            "top_collision",
            "drawer_collision",
        ],
        geom_bodyid=[0, 4, 2, 3, 5],
        joint_bodyid=[1, 2, 3, 5],
        joint_type=[0, 0, 0, 2],
        movable=[
            dict(name=name, body_id=i)
            for i, name in enumerate(["bottom", "middle", "top", "drawer"], 1)
        ],
    )
    # Drawer is an articulation, while all three object roots are free.
    ep.meta["entities"]["movable"][-1]["body_id"] = 5
    ep.controls["obj_pos"] = np.zeros((ep.n, 4, 3))
    ep.controls["contact_geom"] = np.tile(
        np.asarray(contacts, dtype=np.int32).reshape(-1, 2), (ep.n, 1)
    )
    ep.controls["contact_off"] = np.arange(ep.n + 1, dtype=np.int32) * len(contacts)
    return ep


@pytest.mark.parametrize("ground", [0, 4])
def test_stack_and_in_drawer_support_propagates_through_child_bodies(
    physical_episode, ground
):
    ep = support_tape(physical_episode, [[3, 2], [2, 1], [ground, 1]])
    row = next(
        r for r in selfcheck.check(ep) if r["check"] == "resting_objects_contact_table"
    )
    assert row["passed"] and row["denominator"] == 3
    assert row["supported_free_body_ids"] == [1, 2, 3]
    assert row["support_rule"] == "contact chain to any non-free body"


def test_floating_contact_cycle_and_internal_contact_are_not_support(physical_episode):
    ep = support_tape(physical_episode, [[1, 2], [2, 3], [3, 1], [1, 1]])
    row = next(
        r for r in selfcheck.check(ep) if r["check"] == "resting_objects_contact_table"
    )
    assert not row["passed"]
    assert row["missing_objects"] == ["bottom", "middle", "top"]


def test_step_cap_is_failure_with_physical_label_and_censored_late_grasp(
    physical_episode,
):
    ep = physical_episode
    ep.meta.update(termination_reason="step_cap", success=True)
    ep.controls["action"][-5:, 6] = -1
    ep.controls["action"][-2:, 6] = 1
    row, _ = forensics.analyse(ep)
    assert row["status"] == "available" and row["success"] is False
    assert row["outcome"] == "timeout" and row["termination_reason"] == "step_cap"
    assert row["label"] != "success"
    assert audits.grasps(ep)[-1]["lift_within_window"] is None
    assert not adapters.calibrate([ep])["near_radius_by_task"]
    with pytest.raises(Unavailable, match="no successful discovery"):
        segmentation.fit([ep])


@pytest.mark.parametrize(
    "metadata",
    [
        dict(termination_reason="exception"),
        dict(journal_status="error"),
        dict(journal_error="physics error"),
    ],
)
def test_exception_is_invalid_and_excluded_from_physics_and_fits(
    physical_episode, metadata
):
    physical_episode.meta.update(metadata, success=True)
    assert physical_episode.identity["success"] is None
    assert physical_episode.identity["outcome_status"] == "invalid"
    with pytest.raises(Unavailable, match="invalid episode outcome"):
        forensics.analyse(physical_episode)
    with pytest.raises(Unavailable, match="invalid episode outcome"):
        audits.grasps(physical_episode)
    with pytest.raises(Unavailable, match="invalid episode outcome"):
        audits.drops(physical_episode)
    with pytest.raises(Unavailable, match="invalid episode outcome"):
        pairs.validate_pair(physical_episode, physical_episode)
    assert not adapters.calibrate([physical_episode])["near_radius_by_task"]
    with pytest.raises(Unavailable, match="no successful discovery"):
        segmentation.fit([physical_episode])
    tables = forensics.build([physical_episode])
    assert tables["episodes_forensics"][0]["status"] == "unavailable"
    assert not tables["controls_truth"]


def test_journal_success_is_authoritative(physical_episode):
    physical_episode.meta.update(success=True, journal_success=False)
    assert forensics.analyse(physical_episode)[0]["success"] is False


def test_single_arm_pairs_report_missing_comparison(physical_episode):
    tables, summary = pairs.build([physical_episode], "A")
    assert summary["admitted_pairs"] == 0
    assert summary["matched_pairs"] == 0
    assert tables["pairs"][0]["status"] == "unavailable"
    assert "no comparison arm" in tables["pairs"][0]["reason"]


@pytest.mark.parametrize("process_layout", [False, True])
@pytest.mark.parametrize("tail", ['{"decision_id":', '{"ignored": true}'])
def test_reader_layout_subset_init_tail_reporting_and_readonly(
    tmp_path, process_layout, tail
):
    from exp.offline_search.debug.fixtures import make_synthetic_arm

    make_synthetic_arm(tmp_path, n_episodes=2, model="groot")
    server = next((tmp_path / "runs/synthetic/debug").glob("server_*"))
    records = [
        json.loads(s) for s in (server / "decisions.jsonl").read_text().splitlines()
    ]
    for i, r in enumerate(records):
        if i % 2:
            r["init"] = 42  # old source index; task_uid still subset init 0
            r.pop("orig_init_state_idx", None)
        else:
            r.update(init=0, orig_init_state_idx=42)  # new producer semantics
    (server / "decisions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records) + tail
    )
    (server / "writer_stats.json").write_text(json.dumps(dict(pid=1, dropped=0)))
    if process_layout:
        for base, suffix in (
            ("decisions", ".jsonl"),
            ("meta", ".json"),
            ("writer_stats", ".json"),
        ):
            (server / (base + suffix)).rename(server / (base + "_1" + suffix))
        # Unrelated earlier process must not supply metadata to these episodes.
        (server / "meta_0.json").write_text(json.dumps(dict(pid=0, model="pi05")))
    before = {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    episodes = load_episodes(tmp_path, ["synthetic"])
    assert len(episodes) == 2
    assert all(
        ep.decisions
        and all(d["init"] == 0 and d["orig_init_state_idx"] == 42 for d in ep.decisions)
        for ep in episodes
    )
    assert all(ep.manifest["model"] == "groot" for ep in episodes)
    assert episodes[0].manifest["state_dim"] == 8
    capture = input_summary(episodes)
    assert (
        capture["arms"]["synthetic"]["read_issues"][0]["code"] == "truncated_jsonl_tail"
    )
    assert len(capture["arms"]["synthetic"]["writer_stats"]) == 1
    after = {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in tmp_path.rglob("*")
        if p.is_file()
    }
    assert before == after
    out = tmp_path / "output"
    payload = report(
        out,
        "regression",
        {"episodes": [ep.identity for ep in episodes]},
        episodes=episodes,
    )
    assert payload["summary"]["input_capture"] == capture


def test_missing_catalog_is_reported_as_missing_evidence(physical_episode):
    rows = blind_drift.drift(physical_episode, blind_drift.Library())
    assert len(rows) == 8
    assert all(
        r["status"] == "unavailable" and "catalog unavailable" in r["reason"]
        for r in rows
    )


def test_exception_backfill_is_rejected_without_aborting_other_analysis(
    physical_episode, tmp_path
):
    physical_episode.meta.update(
        termination_reason="exception",
        backfill=dict(admission="PASS", library_rows=[0]),
    )
    (tmp_path / "episode.json").write_text(json.dumps(physical_episode.meta))
    np.savez(tmp_path / "controls_0000.npz", **physical_episode.controls)
    library = blind_drift.Library(backfill_root=tmp_path)
    assert not library.physics
    assert "invalid episode outcome" in library.rejected_backfills[0]


def test_scratch_catalog_option_reads_catalog_without_touching_capture(tmp_path):
    from exp.offline_search.debug.fixtures import make_synthetic_arm
    import pandas as pd

    make_synthetic_arm(tmp_path, n_episodes=1)
    catalog = tmp_path / "scratch_rows.parquet"
    pd.DataFrame(
        [dict(row=i, task_id=0, episode=0, step=i, next=i + 1) for i in range(100)]
    ).to_parquet(catalog)
    out = tmp_path / "output"
    blind_drift.main(
        [
            "--run-root",
            str(tmp_path),
            "--arms",
            "synthetic",
            "--out",
            str(out),
            "--catalog-file",
            str(catalog),
        ]
    )
    result = json.loads((out / "blind_drift.json").read_text())
    assert result["tables"]["blind_controls"]
    assert all(
        "catalog unavailable" not in r.get("reason", "")
        for r in result["tables"]["blind_controls"]
    )
    assert not (tmp_path / "runs/synthetic/debug/derived").exists()
