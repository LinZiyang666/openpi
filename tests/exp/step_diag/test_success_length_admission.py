"""Length admission rejects unbound launches and preserves per-arm statistics during partial runs."""

import json

import numpy as np
import pytest

from exp.step_diag.analysis import success_length as S
from tests.exp.step_diag.test_libero_selfstart import ENV_ID, TASKS, TEACHER, _write_libero_arm


def _rewrite(path, edit):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        edit(row)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("fault", ["environment", "arm", "manifest_conflict", "launch_membership"])
def test_length_excludes_unbound_launch_evidence(tmp_path, fault):
    root = tmp_path / "arms"
    _write_libero_arm(root, tmp_path / "srv", "full", {0: [1], 1: [1]})
    directory = root / TEACHER / ENV_ID / "full"
    path = directory / "launch_L0.json"
    launch = json.loads(path.read_text())
    if fault == "environment":
        launch["env_id"] = "groot_libero_spatial"
    elif fault == "arm":
        launch["arm_id"] = "plain_k1"
        _rewrite(directory / "per_step_x.jsonl", lambda row: row.update(arm_id="plain_k1"))
    elif fault == "manifest_conflict":
        (directory / "launch_duplicate.json").write_text(json.dumps(dict(launch, started="other")))
    else:
        other = dict(launch, launch_id="L1", driver_run_id="other", expected=launch["expected"][1:])
        (directory / "launch_L1.json").write_text(json.dumps(other))
        launch["expected"] = launch["expected"][:1]
        _rewrite(directory / "journal_x.jsonl", lambda row: row.update(run_id="other"))
        _rewrite(directory / "per_step_x.jsonl", lambda row: row.update(run_id="other", launch_id="L1"))
    path.write_text(json.dumps(launch))
    data, problems = S.arm_episodes(TEACHER, "full", (root,), env_dir=ENV_ID)
    assert problems
    if fault == "launch_membership":
        assert {key[0] for key in data} == {TASKS[1]}
    else:
        assert not data


@pytest.mark.parametrize("field,value", [
    ("init_idx", None), ("task_id", None), ("seed", None), ("init_pool_sha256", None),
    ("num_steps_wait", None), ("num_steps_wait", -1), ("num_steps_wait", "10"),
    ("n_decisions", 0), ("n_decisions", -1), ("n_env_steps", 0),
])
def test_libero_summary_requires_identity_and_valid_counts(tmp_path, field, value):
    root = tmp_path / "arms"
    _write_libero_arm(root, tmp_path / "srv", "full", {0: [1]}, summary_edit=lambda uid, rows: [
        dict(row, **{field: value}) for row in rows
    ])
    data, problems = S.arm_episodes(TEACHER, "full", (root,), env_dir=ENV_ID)
    assert not data and problems


@pytest.mark.parametrize("with_reference", [False, True])
def test_unpaired_lengths_include_tasks_not_yet_run_by_reference(with_reference):
    def key(task):
        return task, 0, 7, "pool", ENV_ID

    data = {"full": {key("a"): [(True, 10, 2)]} if with_reference else {},
            "plain_k1": {key("a"): [(True, 20, 4)], key("b"): [(True, 40, 8)]}}
    row = S.analyse(data, [], list(data), 2, np.random.default_rng(0))["arms"]["plain_k1"]
    assert row["n_success"] == 2 and row["unpaired_macro_length"] == 6
    assert row.get("paired", {}).get("n_pairs", 0) == int(with_reference)


def test_complete_arm_has_partial_pairs_against_incomplete_reference(tmp_path, monkeypatch, capsys):
    root = tmp_path / "arms"
    _write_libero_arm(root, tmp_path / "srv", "full", {0: [1]})
    _write_libero_arm(root, tmp_path / "srv", "plain_k1", {i: [1] * 50 for i in range(10)})
    output = tmp_path / "lengths.json"
    monkeypatch.setattr("sys.argv", ["success_length", "--benchmark", "libero", "--roots", str(root),
                                    "--env-ids", ENV_ID, "--out", str(output)])
    S.main()
    row = json.loads(output.read_text())["decisions"][f"{ENV_ID}_m1"]["arms"]["plain_k1"]
    assert row["status"] == "formal" and row["n_success"] == 500
    assert row["paired"]["status"] == "partial" and row["paired"]["n_pairs"] == 1
    assert "[PARTIAL]" in capsys.readouterr().out
