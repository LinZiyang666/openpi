"""The conductor runner's evidence path and its resume filter.

Everything here is pure: no driver, no server, no sim. What is pinned is the
set of failures that only show up after a run has already finished -- a retried
episode duplicating its per-step rows, a results file missing the identity
fields the analysis joins on, and a merge sidecar that agrees with itself.
"""

from __future__ import annotations

import json


from openpi.conductor import ServerEndpoint

from exp.libero_groot.run_conductor import GrootSweepStrategy
from exp.libero_groot.run_conductor import arms_with_work_left
from exp.libero_groot.run_conductor import keep_accepted_rows
from exp.libero_groot.run_conductor import read_journal_outcomes
from exp.libero_groot.run_conductor import write_arm_artifacts

SERVERS = [ServerEndpoint("h", 23160 + i) for i in range(6)]


def _strategy(arms=("fh05", "fh20"), trials=2, num_tasks=3, all_arms=None, done_uids=None):
    return GrootSweepStrategy(
        task_suite="libero_spatial",
        yaml_paths={a: f"/tmp/{a}.yaml" for a in arms},
        all_arms=list(all_arms if all_arms is not None else arms),
        servers=SERVERS,
        trials=trials,
        num_tasks=num_tasks,
        done_uids=done_uids,
    )


# -- the plan ------------------------------------------------------------


def test_each_arm_is_spread_over_every_server():
    strategy = _strategy()
    graph = strategy.plan(["fh05", "fh20"], {})
    assert len(graph.stages) == 2 * len(SERVERS)
    for arm in ("fh05", "fh20"):
        siblings = [s for s in graph.stages.values() if s.yaml_id == arm]
        assert {s.server for s in siblings} == set(SERVERS)
        assert sum(len(s.episodes) for s in siblings) == 3 * 2


def test_plan_validates_so_a_partition_bug_cannot_hang_the_run():
    """``plan`` calls validate(); a duplicated uid would otherwise deadlock."""
    strategy = _strategy()
    graph = strategy.plan(["fh05", "fh20"], {})
    graph.validate()  # idempotent, and already run inside plan


def test_episode_index_carries_the_join_fields_the_wire_drops():
    strategy = _strategy(arms=("fh05",), trials=50, num_tasks=10)
    strategy.plan(["fh05"], {})
    uid = "fh05:eval:3:7"
    # episode_id must match the shared helper, or the arm will not join against
    # the ones the cell scheduler already produced.
    assert strategy.episode_index[uid] == {
        "task_id": 3,
        "init_state_idx": 7,
        "orig_init_state_idx": 7,
        "episode_id": 3 * 50 + 7,
    }


def test_setup_is_skipped_for_an_empty_sibling():
    """A bundle load for zero episodes is the swap burst that faulted the GPU."""
    strategy = _strategy(arms=("fh05",), trials=1, num_tasks=1)
    graph = strategy.plan(["fh05"], {})
    empty = [s for s in graph.stages.values() if not s.episodes]
    assert empty, "1 episode over 6 servers must leave empty siblings"

    calls = []

    class _Ctl:
        def load_cache_config(self, **kw):
            calls.append(kw)

    for stage in empty:
        strategy.on_stage_begin(stage, _Ctl(), None)
    assert calls == []


# -- per-step dedup ------------------------------------------------------


def _outcome(uid, attempt):
    return {uid: {"task_uid": uid, "status": "done", "success": True, "attempt": attempt}}


def test_a_retried_episode_contributes_one_set_of_rows():
    rows = [
        {"task_uid": "a", "step_idx": 0, "attempt": 1},
        {"task_uid": "a", "step_idx": 1, "attempt": 1},
        {"task_uid": "a", "step_idx": 0, "attempt": 2},
        {"task_uid": "a", "step_idx": 1, "attempt": 2},
        {"task_uid": "b", "step_idx": 0, "attempt": 1},
    ]
    outcomes = {**_outcome("a", 2), **_outcome("b", 1)}
    kept = keep_accepted_rows(rows, outcomes)
    assert [r["attempt"] for r in kept if r["task_uid"] == "a"] == [2, 2]
    assert len([r for r in kept if r["task_uid"] == "b"]) == 1


def test_a_fenced_duplicate_of_the_same_attempt_is_dropped():
    """Max-attempt cannot separate these: the generation number is identical.

    A superseded dispatch reports at the same generation as the accepted one, so
    keeping both puts two rows on one (episode_id, step_idx) and the integrity
    gate rejects the arm after it has already run.
    """
    rows = [
        {"task_uid": "a", "step_idx": 0, "attempt": 1, "accepted": True},
        {"task_uid": "a", "step_idx": 0, "attempt": 1, "accepted": False},
    ]
    kept = keep_accepted_rows(rows, _outcome("a", 1))
    assert len(kept) == 1
    assert kept[0]["accepted"] is True


def test_rows_for_an_episode_with_no_accepted_outcome_are_dropped():
    """Such an episode is absent from the results side; keeping rows fails I3."""
    rows = [{"task_uid": "ghost", "step_idx": 0, "attempt": 1}]
    assert keep_accepted_rows(rows, {}) == []


def test_rows_without_a_uid_are_left_alone():
    rows = [{"step_idx": 0}, {"task_uid": "a", "step_idx": 0, "attempt": 1}]
    assert len(keep_accepted_rows(rows, _outcome("a", 1))) == 2


# -- resume filter -------------------------------------------------------


def test_a_finished_arm_is_not_walked_again(tmp_path):
    journal = tmp_path / "j.jsonl"
    lines = [
        {"task_uid": f"fh05:eval:0:{i}", "yaml_id": "fh05", "status": "done"}
        for i in range(4)
    ]
    lines.append({"task_uid": "fh20:eval:0:0", "yaml_id": "fh20", "status": "done"})
    journal.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    remaining, counts = arms_with_work_left(journal, ["fh05", "fh20"], expected=4)
    assert remaining == ["fh20"]
    assert counts == {"fh05": 4, "fh20": 1}


def test_a_retried_uid_does_not_inflate_the_count(tmp_path):
    journal = tmp_path / "j.jsonl"
    dup = {"task_uid": "fh05:eval:0:0", "yaml_id": "fh05", "status": "done"}
    journal.write_text("\n".join(json.dumps(dup) for _ in range(4)), encoding="utf-8")
    remaining, counts = arms_with_work_left(journal, ["fh05"], expected=4)
    assert remaining == ["fh05"]
    assert counts == {"fh05": 1}


def test_missing_journal_means_everything_still_has_work(tmp_path):
    remaining, counts = arms_with_work_left(tmp_path / "nope.jsonl", ["a", "b"], expected=1)
    assert remaining == ["a", "b"]
    assert counts == {}


# -- artifacts -----------------------------------------------------------


def _outcomes(uids, *, success=True):
    return {u: {"task_uid": u, "status": "done", "success": success} for u in uids}


def test_results_rows_carry_the_analysis_join_fields(tmp_path):
    strategy = _strategy(arms=("fh05",), trials=2, num_tasks=2)
    strategy.plan(["fh05"], {})
    uids = sorted(strategy.episode_index)
    results_path, _ = write_arm_artifacts(
        results_dir=tmp_path / "results",
        per_step_dir=tmp_path / "per_step",
        arm="fh05",
        planned_uids=uids,
        outcomes=_outcomes(uids),
        episode_index=strategy.episode_index,
    )
    rows = json.loads(results_path.read_text())
    assert len(rows) == 4
    for row in rows:
        assert {"task_id", "orig_init_state_idx", "episode_id", "success"} <= set(row)


def test_the_sidecar_disagrees_when_an_episode_never_reported(tmp_path):
    """Its two counts come from different places, which is the whole point.

    An episode whose retries are exhausted never gets a terminal journal
    record, so a sidecar derived from the results file would agree with the
    results file and report nothing.
    """
    strategy = _strategy(arms=("fh05",), trials=2, num_tasks=2)
    strategy.plan(["fh05"], {})
    uids = sorted(strategy.episode_index)
    _, sidecar = write_arm_artifacts(
        results_dir=tmp_path / "results",
        per_step_dir=tmp_path / "per_step",
        arm="fh05",
        planned_uids=uids,
        outcomes=_outcomes(uids[:3]),  # one episode never came back
        episode_index=strategy.episode_index,
    )
    meta = json.loads(sidecar.read_text())
    assert meta["transport"] == "tcp"
    assert meta["episodes_expected"] == 4
    assert meta["episodes_reported"] == 3


def test_journal_outcomes_keep_only_terminal_records(tmp_path):
    journal = tmp_path / "j.jsonl"
    journal.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {"task_uid": "a", "status": "dispatched"},
                {"task_uid": "a", "status": "done", "success": True},
                {"task_uid": "b", "status": "failed", "success": False},
            ]
        ),
        encoding="utf-8",
    )
    out = read_journal_outcomes(journal)
    assert set(out) == {"a", "b"}
    assert out["a"]["success"] is True


# -- what the audit found ------------------------------------------------


def test_episode_index_covers_arms_with_no_work_left():
    """A resume's finished arms are exactly the ones missing their artifacts.

    They are filtered out of the stage graph, so if the index dropped them too
    there would be no way to ever write what a crashed predecessor owed.
    """
    strategy = _strategy(arms=("fh20",), all_arms=("fh05", "fh20"), trials=2, num_tasks=1)
    assert {uid.split(":", 1)[0] for uid in strategy.episode_index} == {"fh05", "fh20"}


def test_journalled_episodes_are_dropped_from_the_plan():
    """Not just skipped: a sibling with nothing left must look empty.

    The scheduler's resume empties ``pending`` but leaves ``Stage.episodes``
    full, and the strategy only sees the latter -- so a shard whose work is done
    would still pay a full bundle reload at on_stage_begin.
    """
    strategy = _strategy(arms=("fh05",), trials=2, num_tasks=3)
    all_uids = sorted(strategy.episode_index)
    done = set(all_uids[:4])
    strategy_resume = _strategy(arms=("fh05",), trials=2, num_tasks=3, done_uids=done)
    graph = strategy_resume.plan(["fh05"], {})
    planned = {ep.task_uid for s in graph.stages.values() for ep in s.episodes}
    assert planned == set(all_uids) - done


def test_the_sidecar_lands_beside_the_per_step_file_not_the_results(tmp_path):
    """aggregate globs the results dir for *.json and rejects unknown stems.

    A sidecar left there fails the phase under the name ``<arm>.merge``.
    """
    strategy = _strategy(arms=("fh05",), trials=1, num_tasks=1)
    uids = sorted(strategy.episode_index)
    results_path, sidecar = write_arm_artifacts(
        results_dir=tmp_path / "results",
        per_step_dir=tmp_path / "per_step",
        arm="fh05",
        planned_uids=uids,
        outcomes=_outcomes(uids),
        episode_index=strategy.episode_index,
    )
    assert results_path.parent.name == "results"
    assert sidecar.parent.name == "per_step"
    assert [p.name for p in (tmp_path / "results").glob("*.json")] == ["fh05.json"]


def test_a_fenced_stale_result_does_not_overwrite_the_accepted_one(tmp_path):
    """The driver journals rejected results too, marked accepted=false.

    Taking the last line would let a stale fatal error bury the retry that
    actually succeeded, and the arm would under-report its success rate.
    """
    journal = tmp_path / "j.jsonl"
    journal.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {"task_uid": "a", "status": "done", "success": True,
                 "attempt": 2, "accepted": True},
                {"task_uid": "a", "status": "failed", "success": False,
                 "attempt": 1, "accepted": False},
            ]
        ),
        encoding="utf-8",
    )
    out = read_journal_outcomes(journal)
    assert out["a"]["success"] is True
    assert out["a"]["attempt"] == 2


def test_records_predating_the_accepted_field_fall_back_to_attempt(tmp_path):
    journal = tmp_path / "j.jsonl"
    journal.write_text(
        "\n".join(
            json.dumps(x)
            for x in [
                {"task_uid": "a", "status": "failed", "success": False, "attempt": 1},
                {"task_uid": "a", "status": "done", "success": True, "attempt": 2},
            ]
        ),
        encoding="utf-8",
    )
    assert read_journal_outcomes(journal)["a"]["success"] is True


def test_the_artifacts_pass_the_real_integrity_gate(tmp_path):
    """End to end against the analyzer, not against a description of it.

    The layout bug this pins was not "the paths look wrong" -- each path looked
    reasonable on its own. It was that ``aggregate`` globs the results directory
    for ``*.json`` and rejects any stem outside the arm set, so a sidecar
    written beside the results failed the phase under the name ``<arm>.merge``.
    Only running the gate shows that.
    """
    from exp.libero_groot.analysis.gate_pareto.analyze_gate_pareto import (
        aggregate_arm,
        check_arm_integrity,
    )

    results_dir = tmp_path / "eval_results"
    per_step_dir = tmp_path / "eval_per_step"
    strategy = _strategy(arms=("fh05",), trials=2, num_tasks=2)
    strategy.plan(["fh05"], {})
    uids = sorted(strategy.episode_index)

    # One decision row per episode, keyed the way the recorder writes them.
    rows = [
        {
            "yaml_id": "fh05",
            "task_uid": uid,
            "episode_id": strategy.episode_index[uid]["episode_id"],
            "step_idx": 0,
            "hit_type": "MISS",
            "searched": True,
            "attempt": 1,
        }
        for uid in uids
    ]
    per_step_dir.mkdir(parents=True)
    (per_step_dir / "fh05.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )

    write_arm_artifacts(
        results_dir=results_dir,
        per_step_dir=per_step_dir,
        arm="fh05",
        planned_uids=uids,
        outcomes=_outcomes(uids),
        episode_index=strategy.episode_index,
    )

    # What aggregate() does per arm, with the same file discovery.
    present = {p.stem for p in results_dir.glob("*.json") if not p.name.endswith(".partial.json")}
    assert present == {"fh05"}, "a stray artifact in the results dir fails the phase"

    results = json.loads((results_dir / "fh05.json").read_text())
    per_step = [
        json.loads(line)
        for line in (per_step_dir / "fh05.jsonl").read_text().splitlines()
        if line.strip()
    ]
    merge = json.loads((per_step_dir / "fh05.merge.json").read_text())
    check_arm_integrity(results, per_step, expect_ep=len(uids), merge=merge, arm="fh05")

    point = aggregate_arm(results, per_step, expect_ep=len(uids), arm="fh05")
    assert point["n_ep"] == len(uids)
    assert point["teacher_ratio"] == 1.0
