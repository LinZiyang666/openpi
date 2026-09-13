"""``exp/libero_groot/verify_shadow_h5``: the cohort acceptance gate on synthetic attempts.

150 episodes (10 tasks x 15) built from a synthetic shadow manifest, tiny H5
files stamped the way the collector stamps them, and client rows carrying the
termination evidence. Every rejection class the plan names is exercised: a
truncated-but-self-consistent failure, an exception exit, a wrong orig index,
a missing H5, a missing client row, two H5 files for one episode, an episode
outside the manifest, a stale schedule stamp, and a duplicate acceptance across
attempts. The retry filter names exactly the missing episodes.
"""

from __future__ import annotations

import json
import math
import pathlib
import subprocess
import sys

import h5py
import numpy as np
import pytest

from exp.libero_groot import emit_task_map as tm
from exp.libero_groot import verify_shadow_h5 as v

SUITE = "libero_spatial"
CAP = v.MAX_STEPS[SUITE]
# Canonical ``task.name`` (locates the .init pool) and the instruction ``task.language``
# (what the client sends and the collector stores) are two strings per task.
TASKS = [f"pick_up_the_black_bowl_number_{t}_and_place_it_on_the_plate" for t in range(10)]
LANGUAGES = [f"pick up the black bowl number {t} and place it on the plate" for t in range(10)]


def _task_map(tmp_path, *, names=None, languages=None, suite=SUITE) -> pathlib.Path:
    """A benchmark task map as ``emit_task_map.py`` writes it."""
    names = TASKS if names is None else names
    languages = LANGUAGES if languages is None else languages
    doc = {"protocol": tm.PROTOCOL, "suite": suite,
           "tasks": [{"task_id": t, "name": names[t], "language": languages[t], "bddl_file": f"{names[t]}.bddl",
                      "problem_folder": suite} for t in range(10)]}
    p = tmp_path / "task_map.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _manifest(tmp_path) -> pathlib.Path:
    rng = np.random.default_rng(0)
    assignment, pools = {}, {}
    for t in range(10):
        indices = [int(x) for x in rng.choice(50, size=v.TRIALS, replace=False)]
        assignment[str(t)] = {"task_name": TASKS[t], "fit": indices[:10], "cal": indices[10:]}
        pools[TASKS[t]] = {"indices": indices}
    doc = {"suite": SUITE, "assignment": assignment, "pool_digests": {"shadow": pools}}
    p = tmp_path / "shadow_manifest.json"
    p.write_text(json.dumps(doc), encoding="utf-8")
    return p


def _write_h5(path: pathlib.Path, *, episode_id: int, task: str, success: bool, n_steps: int,
              schedule=v.SCHEDULE_ID, k=v.NUM_STEPS, num_steps=None, drop_snapshot=False,
              task_id=None, orig=None, first_step=0) -> None:
    """A collector-shaped file: identity attrs as the client's ``episode_start`` metadata
    stamps them (``task_id`` defaults to ``episode_id // 15``; ``orig`` is written only
    when given), the schedule stamp, and ``n_steps`` groups starting at ``first_step``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.attrs["episode_id"] = episode_id
        f.attrs["task"] = task
        f.attrs["success"] = success
        f.attrs["num_steps"] = n_steps if num_steps is None else num_steps
        f.attrs["denoise_schedule_id"] = schedule
        f.attrs["denoising_num_steps"] = k
        f.attrs["task_id"] = int(episode_id // v.TRIALS if task_id is None else task_id)
        if orig is not None:
            f.attrs["orig_init_state_idx"] = int(orig)
        for i in range(first_step, first_step + n_steps):
            g = f.create_group(f"step_{i:04d}")
            for ds in v.REQUIRED_STEP_DATASETS:
                g.create_dataset(ds, data=np.zeros(2, dtype=np.float16))
            for ds in v.REQUIRED_SNAPSHOTS:
                if drop_snapshot and ds == v.REQUIRED_SNAPSHOTS[-1]:
                    continue
                g.create_dataset(ds, data=np.zeros(2, dtype=np.float16))


def _client_row(t, subset, orig, *, success, steps=None, reason=None, **over):
    if success:
        steps = 60 if steps is None else steps
        reason = "success" if reason is None else reason
    else:
        steps = CAP + v.NUM_STEPS_WAIT if steps is None else steps
        reason = "step_cap" if reason is None else reason
    infers = math.ceil((steps - v.NUM_STEPS_WAIT) / v.REPLAN_STEPS)
    row = {"task_id": t, "init_state_idx": subset, "orig_init_state_idx": orig, "episode_id": t * v.TRIALS + subset,
           "seed": v.SEED, "success": success, "task_suite_name": SUITE, "termination_reason": reason,
           "client_timing": {"steps": steps, "infers": infers}, "max_steps": CAP,
           "num_steps_wait": v.NUM_STEPS_WAIT, "replan_steps": v.REPLAN_STEPS}
    row.update(over)
    return row


@pytest.mark.parametrize("steps,accepted", [(CAP + v.NUM_STEPS_WAIT, True), (CAP + v.NUM_STEPS_WAIT + 1, False)])
def test_success_is_also_bounded_by_the_frozen_episode_cap(tmp_path, steps, accepted):
    row = _client_row(0, 0, 17, success=True, steps=steps)
    path = tmp_path / "episode.h5"
    _write_h5(path, episode_id=0, task=LANGUAGES[0], success=True,
              n_steps=row["client_timing"]["infers"], orig=17)
    problems = v.judge_episode(SUITE, (0, 0), {"task_name": TASKS[0], "task_language": LANGUAGES[0], "orig": 17},
                               row, v._h5_identity(path))
    assert (not problems) == accepted
    if not accepted:
        assert any("exceeds frozen cap" in problem for problem in problems)


def _build_attempt(root: pathlib.Path, name: str, expected: dict, keys, *, mutate=None):
    """Write client rows (per task) and H5 files (srv<i>/<exp>/) for ``keys``."""
    adir = root / name
    rows_by_task: dict[int, list] = {}
    for t, s in keys:
        exp = expected[(t, s)]
        success = (t + s) % 3 != 0
        row = _client_row(t, s, exp["orig"], success=success)
        h5_kw = {"episode_id": t * v.TRIALS + s, "task": exp.get("task_language", LANGUAGES[t]), "success": success,
                 "n_steps": row["client_timing"]["infers"], "task_id": t, "orig": exp["orig"]}
        h5_path = adir / f"srv{t % 5}" / f"acb_shadow_{SUITE}" / f"episode_{t * v.TRIALS + s:04d}.h5"
        if mutate is not None:
            row, h5_kw, h5_path = mutate((t, s), row, h5_kw, h5_path)
        if row is not None:
            rows_by_task.setdefault(t, []).append(row)
        if h5_kw is not None:
            _write_h5(h5_path, **h5_kw)
    adir.mkdir(parents=True, exist_ok=True)
    for t, rows in rows_by_task.items():
        (adir / f"client_task_{t}.json").write_text(json.dumps(rows), encoding="utf-8")
    return adir


@pytest.fixture
def cohort(tmp_path):
    manifest = _manifest(tmp_path)
    expected = v.expected_episodes(json.loads(manifest.read_text()))
    return manifest, expected, tmp_path / "attempts"


def test_expected_episodes_is_the_manifest_in_pool_order(cohort):
    manifest, expected, _ = cohort
    assert len(expected) == 150 and set(expected) == {(t, s) for t in range(10) for s in range(15)}
    doc = json.loads(manifest.read_text())
    for t in range(10):
        pool = doc["pool_digests"]["shadow"][TASKS[t]]["indices"]
        assert [expected[(t, s)]["orig"] for s in range(15)] == pool
    flt = v.episode_filter(expected, [(3, 0), (3, 14)])
    assert flt == [{"task_id": 3, "subset_init_state_idx": 0, "orig_init_state_idx": expected[(3, 0)]["orig"]},
                   {"task_id": 3, "subset_init_state_idx": 14, "orig_init_state_idx": expected[(3, 14)]["orig"]}]


def test_manifest_inconsistencies_fail(cohort):
    manifest, _, _ = cohort
    doc = json.loads(manifest.read_text())
    doc["assignment"]["0"]["cal"][0] = 999
    with pytest.raises(SystemExit, match="pool indices"):
        v.expected_episodes(doc)
    doc = json.loads(manifest.read_text())
    doc["pool_digests"]["shadow"][TASKS[1]]["indices"][0] = doc["pool_digests"]["shadow"][TASKS[1]]["indices"][1]
    doc["assignment"]["1"]["fit"] = doc["pool_digests"]["shadow"][TASKS[1]]["indices"][:10]
    doc["assignment"]["1"]["cal"] = doc["pool_digests"]["shadow"][TASKS[1]]["indices"][10:]
    with pytest.raises(SystemExit, match="distinct"):
        v.expected_episodes(doc)


def test_a_clean_attempt_is_fully_accepted(cohort):
    manifest, expected, root = cohort
    _build_attempt(root, "attempt_0", expected, sorted(expected))
    res = v.verify(SUITE, manifest, root)
    assert res["ok"] and res["n_accepted"] == 150 and res["rejected"] == [] and res["missing"] == []
    assert res["retry_filter"] == [] and res["duplicates"] == []
    rec = res["accepted"][0]
    assert set(rec) >= {"task_id", "task_name", "subset_init_state_idx", "orig_init_state_idx", "attempt", "h5",
                        "h5_sha256", "client_json", "success", "steps", "infers", "termination_reason"}
    assert len({(r["task_id"], r["subset_init_state_idx"]) for r in res["accepted"]}) == 150
    assert 0.0 < res["success_rate"] < 1.0
    assert all(r["termination_reason"] in ("success", "step_cap") for r in res["accepted"])
    assert all(r["steps"] == CAP + v.NUM_STEPS_WAIT for r in res["accepted"] if not r["success"])


def _mutations():
    def truncated(key, row, h5, path):  # failed early, counts self-consistent -> still rejected
        if key == (0, 0):
            row = _client_row(0, 0, row["orig_init_state_idx"], success=False, steps=100)
            h5["n_steps"] = row["client_timing"]["infers"]
        return row, h5, path

    def exception(key, row, h5, path):
        if key == (0, 1):
            row = _client_row(0, 1, row["orig_init_state_idx"], success=False, steps=40, reason="exception")
            h5["n_steps"] = row["client_timing"]["infers"]
        return row, h5, path

    def wrong_orig(key, row, h5, path):
        if key == (0, 2):
            row = dict(row, orig_init_state_idx=row["orig_init_state_idx"] + 1)
        return row, h5, path

    def no_h5(key, row, h5, path):
        return (row, None, path) if key == (1, 0) else (row, h5, path)

    def no_client(key, row, h5, path):
        return (None, h5, path) if key == (1, 1) else (row, h5, path)

    def stale_schedule(key, row, h5, path):
        if key == (2, 0):
            h5 = dict(h5, schedule="groot_n15_k4_v1", k=4)
        return row, h5, path

    def count_mismatch(key, row, h5, path):
        if key == (2, 1):
            h5 = dict(h5, n_steps=h5["n_steps"] + 1)
        return row, h5, path

    def num_steps_attr(key, row, h5, path):
        if key == (2, 2):
            h5 = dict(h5, num_steps=h5["n_steps"] + 1)
        return row, h5, path

    def missing_snapshot(key, row, h5, path):
        if key == (2, 3):
            h5 = dict(h5, drop_snapshot=True)
        return row, h5, path

    def h5_success_disagrees(key, row, h5, path):
        if key == (2, 4):
            h5 = dict(h5, success=not h5["success"])
        return row, h5, path

    def wrong_seed(key, row, h5, path):
        return (dict(row, seed=8), h5, path) if key == (3, 0) else (row, h5, path)

    def short_cap(key, row, h5, path):  # client claims a shorter cap and stops there
        if key == (3, 1):
            row = _client_row(3, 1, row["orig_init_state_idx"], success=False, steps=100 + v.NUM_STEPS_WAIT, max_steps=100)
            h5["n_steps"] = row["client_timing"]["infers"]
        return row, h5, path

    def infers_off(key, row, h5, path):
        if key == (3, 2):
            row = dict(row, client_timing={"steps": row["client_timing"]["steps"], "infers": row["client_timing"]["infers"] + 1})
        return row, h5, path

    def all_of_them(key, row, h5, path):
        for f in (truncated, exception, wrong_orig, no_h5, no_client, stale_schedule, count_mismatch, num_steps_attr,
                  missing_snapshot, h5_success_disagrees, wrong_seed, short_cap, infers_off):
            row, h5, path = f(key, row, h5, path)
        return row, h5, path

    return all_of_them


REJECTED_KEYS = {(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 0), (3, 1), (3, 2)}


def test_every_rejection_class_is_reported_and_only_those_are_retried(cohort):
    manifest, expected, root = cohort
    _build_attempt(root, "attempt_0", expected, sorted(expected), mutate=_mutations())
    res = v.verify(SUITE, manifest, root)
    assert not res["ok"]
    assert res["n_accepted"] == 150 - len(REJECTED_KEYS)
    rejected = {(r["task_id"], r["subset"]): r["problems"] for r in res["rejected"]}
    assert set(rejected) == REJECTED_KEYS
    assert any("step_cap termination with steps=100" in p for p in rejected[(0, 0)])
    assert any("not success/step_cap" in p for p in rejected[(0, 1)])
    assert any("orig_init_state_idx" in p for p in rejected[(0, 2)])
    assert rejected[(1, 0)] == ["no H5 file"]
    assert rejected[(1, 1)] == ["no client terminal row"]
    assert any("stamped 'groot_n15_k4_v1'/4" in p for p in rejected[(2, 0)])
    assert any("step groups but the client made" in p for p in rejected[(2, 1)])
    assert any("H5 num_steps" in p for p in rejected[(2, 2)])
    assert any("missing datasets" in p for p in rejected[(2, 3)])
    assert any("H5 success" in p for p in rejected[(2, 4)])
    assert any("seed=8" in p for p in rejected[(3, 0)])
    assert any("max_steps=100" in p for p in rejected[(3, 1)])
    assert any("infers=" in p and "ceil" in p for p in rejected[(3, 2)])
    assert {(m["task_id"], m["subset_init_state_idx"]) for m in res["missing"]} == REJECTED_KEYS
    assert {(r["task_id"], r["subset_init_state_idx"]) for r in res["retry_filter"]} == REJECTED_KEYS
    assert all(r["orig_init_state_idx"] == expected[(r["task_id"], r["subset_init_state_idx"])]["orig"]
               for r in res["retry_filter"])


def test_a_retry_attempt_completes_the_cohort_without_touching_the_first(cohort):
    manifest, expected, root = cohort
    _build_attempt(root, "attempt_0", expected, sorted(expected), mutate=_mutations())
    first = v.verify(SUITE, manifest, root)
    retry_keys = [(r["task_id"], r["subset_init_state_idx"]) for r in first["retry_filter"]]
    _build_attempt(root, "attempt_1", expected, retry_keys)
    res = v.verify(SUITE, manifest, root)
    assert res["ok"] and res["n_accepted"] == 150 and res["duplicates"] == []
    by_key = {(r["task_id"], r["subset_init_state_idx"]): r for r in res["accepted"]}
    assert all(by_key[k]["attempt"] == "attempt_1" for k in retry_keys)
    assert all(by_key[k]["attempt"] == "attempt_0" for k in set(expected) - set(retry_keys))
    # The first attempt's rejected material is still there, still rejected, never re-read as accepted.
    assert {(r["task_id"], r["subset"]) for r in res["rejected"]} == REJECTED_KEYS


def test_two_valid_terminals_for_one_episode_are_a_duplicate_not_last_wins(cohort):
    manifest, expected, root = cohort
    _build_attempt(root, "attempt_0", expected, sorted(expected))
    _build_attempt(root, "attempt_1", expected, [(4, 4)])
    res = v.verify(SUITE, manifest, root)
    assert not res["ok"]
    assert res["duplicates"] == [{"task_id": 4, "subset": 4, "attempts": ["attempt_0", "attempt_1"]}]
    assert res["n_accepted"] == 150


def test_same_attempt_conflicts_invalidate_the_episode_and_foreign_episodes_are_rejected(cohort):
    """Two H5 files or two client rows for one episode in one attempt: the attempt
    cannot vouch for it (no first/last wins), so the episode goes missing -> ok=False."""
    manifest, expected, root = cohort
    adir = _build_attempt(root, "attempt_0", expected, sorted(expected))
    _write_h5(adir / "srv0" / "x" / "episode_dup.h5", episode_id=5, task=LANGUAGES[0], success=True, n_steps=3,
              orig=expected[(0, 5)]["orig"])
    _write_h5(adir / "srv1" / "x" / "episode_foreign.h5", episode_id=10 * v.TRIALS + 3, task="other", success=True, n_steps=3)
    rows = json.loads((adir / "client_task_3.json").read_text())
    (adir / "client_task_3.json").write_text(json.dumps([*rows, dict(rows[0])]))
    res = v.verify(SUITE, manifest, root)
    assert not res["ok"] and res["n_accepted"] == 148
    problems = {(r["task_id"], r["subset"]): r["problems"][0] for r in res["rejected"]}
    assert "2 H5 files for one episode in this attempt" in problems[(0, 5)]
    assert "2 client rows for one episode in this attempt" in problems[(3, 0)]
    assert "not in the shadow manifest" in problems[(10, 3)]
    assert {(m["task_id"], m["subset_init_state_idx"]) for m in res["missing"]} == {(0, 5), (3, 0)}
    # A clean retry attempt of exactly those two completes the cohort.
    _build_attempt(root, "attempt_1", expected, [(0, 5), (3, 0)])
    res = v.verify(SUITE, manifest, root)
    assert res["ok"] and res["n_accepted"] == 150 and res["duplicates"] == []


def test_identity_attrs_are_bound_to_client_and_manifest(cohort):
    manifest, expected, root = cohort

    def mutate(key, row, h5, path):
        if key == (1, 2):
            h5 = dict(h5, orig=h5["orig"] + 1)  # right episode_id, wrong initial state
        if key == (1, 3):
            h5 = dict(h5, task_id=2)
        if key == (1, 4):
            h5 = dict(h5, orig=None)  # collector attr missing altogether
        if key == (1, 5):
            h5 = dict(h5, first_step=1)  # step_0001..: the first decision is absent
        return row, h5, path

    _build_attempt(root, "attempt_0", expected, sorted(expected), mutate=mutate)
    res = v.verify(SUITE, manifest, root)
    problems = {(r["task_id"], r["subset"]): r["problems"] for r in res["rejected"]}
    assert set(problems) == {(1, 2), (1, 3), (1, 4), (1, 5)}
    assert any("H5 orig_init_state_idx attr" in p for p in problems[(1, 2)])
    assert any("H5 task_id attr 2 != 1" in p for p in problems[(1, 3)])
    assert any("lacks attrs ['orig_init_state_idx']" in p for p in problems[(1, 4)])
    assert any("not step_0000" in p for p in problems[(1, 5)])
    rec = res["accepted"][0]
    assert rec["task_name"] == TASKS[0] and rec["task_language"] == LANGUAGES[0]
    assert rec["client_json_sha256"] and rec["client_row"]["termination_reason"] in ("success", "step_cap")


def test_task_map_binds_canonical_names_and_instructions(cohort, tmp_path):
    manifest, expected, root = cohort
    task_map = tm.load_task_map(_task_map(tmp_path), SUITE)
    _build_attempt(root, "attempt_0", expected, sorted(expected))
    res = v.verify(SUITE, manifest, root, task_map)
    assert res["ok"] and res["task_map_bound"]
    assert res["tasks"]["3"] == {"task_name": TASKS[3], "task_language": LANGUAGES[3]}
    # The manifest's canonical name must be the benchmark's task.name for that id.
    other = tm.load_task_map(_task_map(tmp_path, names=[*TASKS[:9], "KITCHEN_SCENE3_turn_on_the_stove"]), SUITE)
    with pytest.raises(SystemExit, match="manifest task_name"):
        v.verify(SUITE, manifest, root, other)
    # An H5 whose instruction is not the benchmark's task.language is rejected.
    drift = tm.load_task_map(_task_map(tmp_path, languages=[*LANGUAGES[:9], "turn on the stove"]), SUITE)
    res = v.verify(SUITE, manifest, root, drift)
    assert not res["ok"] and all(r["task_id"] == 9 for r in res["rejected"])
    assert any("benchmark task.language" in p for p in res["rejected"][0]["problems"])
    # Without a map the canonical name is never compared to the instruction (l10-style names).
    exp = {"task_name": "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket", "orig": 17}
    row = _client_row(0, 0, 17, success=True, steps=15)
    h5 = tmp_path / "e.h5"
    _write_h5(h5, episode_id=0, task="put both the alphabet soup and the tomato sauce in the basket", success=True,
              n_steps=1, orig=17)
    assert v.judge_episode(SUITE, (0, 0), exp, row, v._h5_identity(h5)) == []


def test_task_map_validation():
    base = {"protocol": tm.PROTOCOL, "suite": SUITE,
            "tasks": [{"task_id": t, "name": TASKS[t], "language": LANGUAGES[t]} for t in range(10)]}
    assert set(tm.validate_task_map(base, SUITE)) == set(range(10))
    for bad, fragment in [
        ({**base, "suite": "libero_10"}, "not a"),
        ({**base, "tasks": base["tasks"][:9]}, "0..9"),
        ({**base, "tasks": [*base["tasks"][:9], {"task_id": 9, "name": TASKS[0], "language": LANGUAGES[9]}]}, "not unique"),
        ({**base, "tasks": [*base["tasks"][:9], {"task_id": 9, "name": TASKS[9], "language": ""}]}, "bad or duplicate"),
    ]:
        with pytest.raises(SystemExit, match=fragment):
            tm.validate_task_map(bad, SUITE)


def test_inconsistent_instructions_within_a_task_fail_the_cohort(cohort):
    manifest, expected, root = cohort

    def mutate(key, row, h5, path):
        if key == (4, 4):
            h5 = dict(h5, task=LANGUAGES[4] + " please")
        return row, h5, path

    _build_attempt(root, "attempt_0", expected, sorted(expected), mutate=mutate)
    res = v.verify(SUITE, manifest, root)
    assert not res["ok"] and res["n_accepted"] == 150
    assert res["inconsistent_languages"] == {4: sorted([LANGUAGES[4], LANGUAGES[4] + " please"])}


def test_cli_writes_filters_manifests_and_per_task_retry_files(cohort, tmp_path):
    manifest, expected, root = cohort
    out = tmp_path / "out"
    cmd = [sys.executable, "-m", "exp.libero_groot.verify_shadow_h5", "--suite", SUITE, "--shadow-manifest", str(manifest),
           "--task-map", str(_task_map(tmp_path)), "--attempts-root", str(root), "--out-dir", str(out), "--emit-full-filter"]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=pathlib.Path(v.__file__).resolve().parents[2])
    assert r.returncode == 0, r.stderr
    assert "no attempts yet" in r.stdout
    for t in range(10):
        rows = json.loads((out / f"filter_task_{t}.json").read_text())
        assert [r["subset_init_state_idx"] for r in rows] == list(range(15))
        assert all(r["task_id"] == t for r in rows)
    _build_attempt(root, "attempt_0", expected, sorted(expected), mutate=_mutations())
    r = subprocess.run(cmd[:-1], capture_output=True, text=True, cwd=pathlib.Path(v.__file__).resolve().parents[2])
    assert r.returncode == 1
    acc = json.loads((out / "accepted_shadow_manifest.json").read_text())
    assert acc["ok"] is False and acc["n_accepted"] == 150 - len(REJECTED_KEYS)
    assert acc["task_map_bound"] and acc["task_map_sha256"] and acc["tasks"]["0"]["task_language"] == LANGUAGES[0]
    rej = json.loads((out / "rejected.json").read_text())
    assert len(rej["missing"]) == len(REJECTED_KEYS)
    retry_files = sorted(p.name for p in (out / "retry").glob("filter_task_*.json"))
    assert retry_files == ["filter_task_0.json", "filter_task_1.json", "filter_task_2.json", "filter_task_3.json"]
    assert {r["subset_init_state_idx"] for r in json.loads((out / "retry" / "filter_task_2.json").read_text())} == {0, 1, 2, 3, 4}
