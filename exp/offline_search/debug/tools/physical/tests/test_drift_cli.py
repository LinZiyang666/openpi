from copy import deepcopy
import importlib
import json

import numpy as np
import pytest

from exp.offline_search.debug.tools.physical.blind_drift import Library, drift
from exp.offline_search.debug.tools.physical.common import (
    load_episodes,
    normalize_legacy,
    Episode,
)


def catalog():
    return [
        dict(row=i, task_id=0, episode=0, step=i, next=i + 1 if i < 15 else -1)
        for i in range(16)
    ]


def test_successor_chains_never_terminal_clamp():
    lib = Library(catalog=catalog())
    assert lib.advance([0, 15, -1], 1).tolist() == [1, -1, -1]
    lib.catalog[1]["task_id"] = 1
    assert lib.advance([0], 1).tolist() == [-1]


def test_robot_fallback_is_only_a_measured_decision_check(physical_episode, tmp_path):
    np.save(tmp_path / "rs.npy", np.arange(16 * 8, dtype=float).reshape(16, 8))
    np.save(tmp_path / "success.npy", np.ones(16, bool))
    lib = Library(tmp_path, catalog())

    class Reader:
        def decision_arrays(self, keys, ids):
            return {"state_norm": np.array([lib.rs[1] + 0.1])}

    physical_episode.reader_arm = Reader()
    rows = drift(physical_episode, lib)
    checks = [r for r in rows if r.get("robot_status") == "available"]
    assert len(checks) == 8
    assert all(r["object_relative_status"] == "unavailable" for r in rows)
    assert rows[0]["evidence_control_idx"] == 6
    assert rows[1]["status"] == "unavailable"


def test_certified_control_rate_backfill_and_object_drift(physical_episode, tmp_path):
    demo = deepcopy(physical_episode)
    demo.meta.update(
        episode_key="library_a1",
        backfill=dict(admission="PASS", library_rows=list(range(16))),
        capture_errors=[],
    )
    directory = tmp_path / "client" / demo.key
    directory.mkdir(parents=True)
    (directory / "episode.json").write_text(json.dumps(demo.meta))
    np.savez_compressed(
        directory / "controls_0000.npz",
        **{k: v for k, v in demo.controls.items() if not k.startswith("contact_")},
    )
    lib = Library(catalog=catalog(), backfill_root=tmp_path)
    assert len(lib.physics) == 16
    live = physical_episode
    live.controls["eef_pos"][7:, 0] += 0.02
    live.controls["qpos"][7:, 0] += 0.02
    rows = drift(live, lib)
    assert all(r["object_relative_status"] == "available" for r in rows)
    assert np.isclose(rows[0]["robot_qpos_rms"], 0.02 / 3.0)
    assert rows[0]["object_relative_rms"] > 0
    demo.meta["backfill"]["admission"] = "REJECTED"
    (directory / "episode.json").write_text(json.dumps(demo.meta))
    assert not Library(catalog=catalog(), backfill_root=tmp_path).physics


def test_legacy_adapter_extracts_physics_and_preserves_unknown_predicate():
    before = dict(
        observation_numeric={
            "robot0_eef_pos": [0, 0, 0.1],
            "object_1_pos": [0, 0, 0.1],
            "object_1_quat": [0, 0, 0, 1],
        },
        predicates={'["in","object_1","target_1"]': None},
        entity_ids={"body_names": ["target_1"]},
        sim_state=[0.0, 1.0],
        qpos=[0.0, 1.0],
    )
    after = dict(before, body_xpos=[[1.0, 0.0, 0.1]])
    raw = [dict(before=before, after=after, wait_phase=True, environment_seed=7)]
    ep = Episode(dict(episode_key="p3", suite="libero_10"), dict(legacy_records=raw))
    ep = normalize_legacy(ep)
    assert ep.controls["obj_pos"].shape == (1, 1, 3)
    assert np.isnan(ep.controls["predicates"][0, 0])
    assert ep.meta["p3_contact_names_invalid"]
    assert ep.meta["env_seed"] == 7


@pytest.mark.parametrize(
    "tool",
    [
        "selfcheck",
        "forensics",
        "grasp_audit",
        "drop_audit",
        "paired_diverge",
        "twin_divergence",
        "blind_drift",
        "segmentation_bench",
        "episode_card",
        "arm_rollup",
    ],
)
def test_each_cli_on_shared_schema_fixture(tool, tmp_path):
    from exp.offline_search.debug.fixtures import make_synthetic_arm

    root = tmp_path / "run"
    make_synthetic_arm(root, n_episodes=3, model="groot")
    config = tmp_path / "adapter.json"
    config.write_text(
        json.dumps(
            dict(
                environment="libero",
                obj_quat_order="wxyz",
                goal_objects=[dict(predicate=0, object="object", destination="target")],
                destinations=dict(target=[0.5, 0.0, 0.0]),
            )
        )
    )
    out = tmp_path / "out"
    argv = [
        "--run-root",
        str(root),
        "--arms",
        "synthetic",
        "--out",
        str(out),
        "--procs",
        "1",
        "--adapter-config",
        str(config),
    ]
    if tool in ("paired_diverge", "twin_divergence"):
        argv += ["--reference-arm", "synthetic"]
    importlib.import_module("exp.offline_search.debug.tools.physical." + tool).main(
        argv
    )
    result = json.loads((out / (tool + ".json")).read_text())
    assert result["tool"] == tool
    assert (out / (tool + ".md")).exists()
    assert list(out.glob("*.csv"))
    if tool in ("episode_card", "arm_rollup"):
        assert list(out.glob("*.png"))
        assert list(out.glob("*.html"))
    if tool == "forensics":
        assert result["denominators"]["episodes_forensics"]["available"] == 3
    if tool == "segmentation_bench":
        assert result["tables"]["fits"]
        assert all(r["passed"] for r in result["tables"]["prefix_replay"])


def test_shared_reader_joins_real_decision_ids(tmp_path):
    from exp.offline_search.debug.fixtures import make_synthetic_arm

    make_synthetic_arm(tmp_path, n_episodes=2)
    episodes = load_episodes(tmp_path, ["synthetic"])
    assert len(episodes) == 2
    assert all(
        ep.decision_at(2)["decision_id"].startswith(ep.key + ":") for ep in episodes
    )
    assert all(ep.controls["control_idx"][0] == 0 for ep in episodes)
