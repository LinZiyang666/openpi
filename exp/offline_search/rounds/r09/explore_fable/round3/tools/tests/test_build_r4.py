"""Spec-level tests for the round-4 six-cell batch builder (no closed loop, no artifacts written outside tmp)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from exp.offline_search.rounds.r09.explore_fable.round3.tools import build_r4 as b

FROZEN = (b.R8_MAIN, b.R8_ABL, b.MANIFEST_SRC)


def _ready():
    return all(p.exists() for p in FROZEN) and all((b.HEADS / f"head_{m}_{s}_{l}_motion_pertask.npz").exists() for m, s, l in b.CELLS)


pytestmark = pytest.mark.skipif(not _ready(), reason="frozen R8 rows / corrector heads not available on this machine")


def test_batch_shape_and_names(tmp_path):
    arms, prefit, prov = b.build(tmp_path)
    assert len(arms) == 21 and len(prefit) == 15 and len(prov) == 6
    names = [a["name"] for a in arms]
    assert len(set(names)) == 21
    for m, s, l in b.CELLS:
        cell = [a for a in arms if a["name"].startswith(f"r9f4_{m}_{s}_{l}_")]
        kinds = sorted(a["name"].split(f"r9f4_{m}_{s}_{l}_")[1] for a in cell)
        assert kinds == (["A", "np_corr05", "np_corr05_esc", "onlynp"] if m == "pi05" else ["A", "np_corr05", "onlynp"])


def test_controls_are_verbatim_copies_of_frozen_rows(tmp_path):
    arms, _, prov = b.build(tmp_path)
    r8, abl = b._rows(b.R8_MAIN), b._rows(b.R8_ABL)
    for a in arms:
        if a["name"].endswith("_A"):
            src = r8[prov[a["name"][5:-2]]["cache_control_source"]]
            assert (a["method"], a["kwargs"], a["plugin_args"], a["client_overrides"]) == (
                src["method"], src["kwargs"], src["plugin_args"], src["client_overrides"])
            assert "full_model" not in a and "--os-judge" not in a["plugin_args"]
        if a["name"].endswith("_onlynp"):
            src = abl[prov[a["name"][5:-7]]["onlynp_control_source"]]
            assert (a["method"], a["kwargs"], a["plugin_args"]) == (src["method"], src["kwargs"], src["plugin_args"])
            assert a["full_model"] is True and src["kwargs"]["disabled_guards"] == ["stuck", "terminal", "overtime"]


def test_stack_arms_use_cell_frozen_judge_and_half_corrector(tmp_path):
    arms, prefit, _ = b.build(tmp_path)
    for a in arms:
        if "np_corr05" not in a["name"]:
            continue
        m = a["model"]
        kw = a["kwargs"]
        assert kw["max_calls"] == 0 and kw["force_trigger_at"] == [] and kw["burst"] == 2 and kw["hold_decisions"] == 2
        assert kw["closed_sign"] == b.CLOSED_SIGN[m] and kw["empty_aperture"] == b.APERTURE[m]
        abbr = f"{b.ABBR[m]}_{b.SUITE_ABBR[a['suite']]}_"
        assert Path(kw["onlynp_fit"]).name.startswith("r8abl_onlynp_" + abbr)
        corr = prefit[Path(kw["corrected_fit"]).stem]
        assert corr["method"].endswith(":CorrectedCache") and corr["kwargs"]["blend"] == 0.5 and corr["kwargs"]["correct_gripper"] is False
        assert corr["kwargs"]["lib"] == ("big" if a["name"].split("_")[3] == "500" else "current")
        assert Path(corr["kwargs"]["head_path"]).name == f"head_{m}_{a['suite']}_{a['name'].split('_')[3]}_motion_pertask.npz"
        cls = a["method"].rsplit(":", 1)[1]
        if a["name"].endswith("_esc"):
            assert m == "pi05" and cls == "NpGraspEsc3" and kw["lag_threshold"] == 12 and kw["deadline"] == 80
        else:
            assert cls == b.STACK_CLASS[m] and "lag_threshold" not in kw
        assert a["plugin_args"][-1] == str(Path(tmp_path) / "fits" / f"{a['name']}.pkl")
        assert "--os-judge" in a["plugin_args"] and "--os-policy-tail" in a["plugin_args"]


def test_nothing_is_task_indexed_and_manifest_is_eval_split(tmp_path):
    arms, prefit, prov = b.build(tmp_path)
    text = json.dumps(arms) + json.dumps(prefit)
    for forbidden in ("task_budget", "per_task", "task_ids", "task_switch", "hard_tasks"):
        assert forbidden not in text
    manifest = json.loads(b.MANIFEST_SRC.read_text())
    assert {(int(t), int(i)) for t, i in manifest} == {(t, i) for t in range(10) for i in range(20, 30)}
    for p in prov.values():
        assert p["head_train_inits"] == [0, 19] and p["eval_inits"] == [20, 29]


def _fake_arm(root, arm, outcomes, decisions):
    """outcomes: {(task, init): success}; decisions: list of (init, vision, hit)."""
    run = root / "runs" / arm
    (run / "client").mkdir(parents=True)
    (run / "server_1").mkdir()
    with (run / "client" / "journal.jsonl").open("w") as f:
        for (t, i), s in outcomes.items():
            f.write(json.dumps(dict(task_uid=f"{arm}:eval:{t}:{i}", accepted=True, status="done", success=s)) + "\n")
        f.write(json.dumps(dict(task_uid=f"{arm}:eval:0:20", accepted=False, status="failed", success=False)) + "\n")
    with (run / "server_1" / f"decisions_{arm}_1.jsonl").open("w") as f:
        for i, v, h in decisions:
            f.write(json.dumps(dict(ev="dec", uid=f"{arm}:eval:0:{i}", vision=v, hit=h)) + "\n")
        f.write(json.dumps(dict(ev="meta", uid=f"{arm}:eval:0:20")) + "\n")


def test_arm_outcomes_uses_only_the_eval_split(tmp_path):
    arm = "r9f4_pi05_l10_500_np_corr05"
    _fake_arm(tmp_path, arm, {(0, 20): True, (0, 21): False, (1, 22): True, (0, 5): False},
              [(20, True, True), (20, False, True), (21, True, False), (22, False, True), (5, True, False)])
    out = b.arm_outcomes(tmp_path, arm)
    assert out["n"] == 3 and abs(out["sr"] - 2 / 3) < 1e-9          # the train-split episode (init 5) is excluded
    assert out["n_dec"] == 4 and abs(out["look"] - 0.5) < 1e-9 and abs(out["call"] - 0.25) < 1e-9
    assert abs(out["ir"] - (0.152 * 0.5 + 0.848 * 0.25)) < 1e-9
    assert b.arm_outcomes(tmp_path, "r9f4_pi05_l10_500_A") is None    # missing arm -> None, never a crash


def test_score_marks_incomplete_batches(tmp_path, capsys):
    _fake_arm(tmp_path, "r9f4_groot_spatial_50_np_corr05", {(0, 20): True}, [(20, True, True)])
    _fake_arm(tmp_path, "r9f4_groot_spatial_50_onlynp", {(0, 20): False}, [(20, True, False)])
    table, verdict = b.score(tmp_path)
    assert table["groot_spatial_50"]["np_corr05"]["sr"] == 1.0 and table["groot_spatial_50"]["A"] is None
    assert not verdict["H1"]["complete"] and not verdict["H1"]["passed"] and not verdict["H3"]["complete"]
    assert (tmp_path / "score.json").exists()
