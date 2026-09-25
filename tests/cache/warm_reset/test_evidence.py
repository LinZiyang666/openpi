"""Server-side evidence rows and the trusted-input admission checker (plan §4.7, §9.4)."""

from __future__ import annotations

import copy
import hashlib
import re
from types import SimpleNamespace

import pytest

from openpi.cache.warm_reset.evidence import (
    ExpectedEpisode,
    GrootWarmResetEvidencePolicy,
    WarmResetEvidencePolicy,
    decision_problems,
    episode_problems,
)
from openpi.cache.warm_reset.runtime import WarmResetSession
from openpi.conductor.task import EpisodeTask
from tests.cache.warm_reset._support import (
    EPISODE,
    EXTRA,
    expected_for,
    obs_for,
    pi05_stack,
    read_rows,
    run_episode,
    spec_for,
)

# ------------------------------------------------------------------
# Wrapper: rows, closure, lifecycle
# ------------------------------------------------------------------


class _Inner:
    """Interceptor stand-in: records lifecycle, answers with a canned hit meta."""

    def __init__(self, meta=None, fail_at=None):
        self.calls = []
        self.meta = meta or {"hit_type": "WARM_START", "start_t": 0.2, "winner_id": "e1", "warm_reset": {"k": 10}}
        self.fail_at = fail_at
        self.n = 0

    def infer(self, obs, **kw):
        self.n += 1
        if self.fail_at == self.n:
            raise RuntimeError("boom")
        return {"actions": obs, "__hit_meta__": dict(self.meta)}

    def on_task_begin(self):
        self.calls.append("task_begin")

    def on_episode_start(self, **kw):
        self.calls.append(("episode_start", kw["task"]))

    def on_episode_end(self, success):
        self.calls.append(("episode_end", success))

    def on_task_end(self):
        self.calls.append("task_end")

    def prefill_trajectory(self, *a, **kw):
        return None


def _wrapper(tmp_path, inner, arm="warmreset_t0.2", cls=WarmResetEvidencePolicy):
    spec = spec_for("pi05", arm, evidence_dir=str(tmp_path / "ev"))
    session = WarmResetSession(spec)
    return cls(inner, session=session, family="pi05", bundle_id="b", yaml_id="y", yaml_sha256="s",
               schedule_id="pi05_v1", k=10), session


def test_one_row_per_request_and_one_finalize_per_episode(tmp_path):
    inner = _Inner()
    served, session = _wrapper(tmp_path, inner)
    served.on_task_begin()
    for success in (True, False):
        served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
        for _ in range(3):
            served.infer({"x": 1})
        served.on_episode_end(success)
    rows = read_rows(served.evidence_path)
    assert re.fullmatch(rf"warm_reset_y_.+_\d+_{session.conn_id}\.jsonl", served.evidence_path.name)
    assert [r["row_kind"] for r in rows] == ["decision"] * 3 + ["finalize"] + ["decision"] * 3 + ["finalize"]
    assert [r["decision_idx"] for r in rows if r["row_kind"] == "decision"] == [0, 1, 2, 0, 1, 2]
    assert [r["episode_seq"] for r in rows] == [0] * 4 + [1] * 4
    assert [r["success"] for r in rows] == [True] * 4 + [False] * 4
    fin = rows[3]
    assert (fin["terminal"], fin["outcome"], fin["n_decisions"]) == (True, True, 3)
    first = rows[0]
    assert first["identity"] == {**EPISODE, **EXTRA}
    assert (first["task_uid"], first["attempt"], first["yaml_id"], first["bundle_id"]) == ("y:eval:4:3", 1, "y", "b")
    assert first["warm_reset"] == {"k": 10} and first["status"] == "ok" and first["error"] is None
    assert inner.calls == ["task_begin", ("episode_start", EPISODE["task"]), ("episode_end", True),
                           ("episode_start", EPISODE["task"]), ("episode_end", False)]


def test_connection_drop_closes_the_episode_non_terminal(tmp_path):
    inner = _Inner()
    served, _ = _wrapper(tmp_path, inner)
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    served.infer({})
    served.on_task_end()
    rows = read_rows(served.evidence_path)
    assert [r["row_kind"] for r in rows] == ["decision", "finalize"]
    assert rows[1]["terminal"] is False and rows[1]["outcome"] is None
    assert all(r["success"] is None for r in rows)
    assert inner.calls[-1] == "task_end"


def test_error_rows_keep_the_index_and_reraise(tmp_path):
    served, _ = _wrapper(tmp_path, _Inner(fail_at=2))
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    served.infer({})
    with pytest.raises(RuntimeError, match="boom"):
        served.infer({})
    served.infer({})
    served.on_episode_end(True)
    rows = read_rows(served.evidence_path)
    assert [(r["decision_idx"], r["status"]) for r in rows[:3]] == [(0, "ok"), (1, "error"), (2, "ok")]
    assert rows[1]["error"] == "RuntimeError: boom" and rows[1]["warm_reset"] is None
    assert rows[3]["n_decisions"] == 3


def test_request_outside_an_episode_is_refused_without_a_row(tmp_path):
    served, _ = _wrapper(tmp_path, _Inner())
    with pytest.raises(RuntimeError, match="outside an episode"):
        served.infer({})
    assert not served.evidence_path.exists()


def test_unterminated_episode_is_closed_when_the_next_one_starts(tmp_path):
    served, _ = _wrapper(tmp_path, _Inner())
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    served.infer({})
    served.on_episode_start(**EPISODE, extra_metadata=dict(EXTRA))
    served.infer({})
    served.on_episode_end(True)
    rows = read_rows(served.evidence_path)
    assert [(r["row_kind"], r["episode_seq"], r["success"]) for r in rows] == [
        ("decision", 0, None), ("finalize", 0, None), ("decision", 1, True), ("finalize", 1, True)]
    assert rows[1]["terminal"] is False


def test_identity_conflicts_and_missing_seed_keys_fail_before_any_decision(tmp_path):
    served, _ = _wrapper(tmp_path, _Inner())
    with pytest.raises(ValueError, match="conflicts"):
        served.on_episode_start(**EPISODE, extra_metadata={**EXTRA, "task": "other"})
    self_served, _ = _wrapper(tmp_path, _Inner(), arm="selfwarmreset_t0.2")
    with pytest.raises(ValueError, match="seed keys"):
        self_served.on_episode_start(**EPISODE, extra_metadata={"task_uid": "u"})


def test_hasattr_surface_is_the_inner_one(tmp_path):
    served, _ = _wrapper(tmp_path, _Inner())
    assert hasattr(served, "prefill_trajectory") and not hasattr(served, "get_action")
    groot, _ = _wrapper(tmp_path, SimpleNamespace(get_action=lambda o: {}), cls=GrootWarmResetEvidencePolicy)
    assert not hasattr(groot, "prefill_trajectory") and not hasattr(groot, "infer")


# ------------------------------------------------------------------
# Admission: positive controls
# ------------------------------------------------------------------


@pytest.fixture(params=[("warmreset_t0.2", True), ("warmreset_t0.2", False),
                        ("selfresetfinal_t0.2", True), ("selfresetfinal_t0.2", False)],
                ids=["cache-success", "cache-failure", "self-success", "self-failure"])
def episode(request, tmp_path):
    arm, success = request.param
    stack = pi05_stack(tmp_path, arm)
    run_episode(stack, 10, success=success)
    return SimpleNamespace(stack=stack, rows=read_rows(stack.served.evidence_path),
                           expected=expected_for(stack, n=10, outcome=success))


def test_valid_episodes_are_admitted_and_priced(episode):
    result = episode_problems(episode.rows, expected=episode.expected)
    assert not +result["problems"], result["problems"]
    self_arm = episode.stack.spec.self_start
    assert result["continuation_nfe"] == 20  # N = 2 per decision
    assert result["self_start_nfe"] == (100 if self_arm else 0)  # K = 10 per decision
    assert result["total_nfe"] == (120 if self_arm else 20)
    finalize = episode.rows[-1]
    assert finalize["row_kind"] == "finalize" and "warm_reset" not in finalize
    inner = episode.rows[0]["warm_reset"]
    assert ("self_seed" in inner) == self_arm and inner["self_start_calls"] == (1 if self_arm else 0)
    assert decision_problems(episode.rows[0], expected=episode.expected) == []


# ------------------------------------------------------------------
# Admission: every negative is rejected, and never priced
# ------------------------------------------------------------------


def _decisions(rows):
    return [r for r in rows if r["row_kind"] == "decision"]


def _mutations():
    def inner(key, value, i=0):
        def f(rows):
            _decisions(rows)[i]["warm_reset"][key] = value
        return f

    def outer(key, value, i=0):
        def f(rows):
            _decisions(rows)[i][key] = value
        return f

    def finalize(key, value):
        def f(rows):
            rows[-1][key] = value
        return f

    def drop_tail(rows):
        del rows[-3:-1]  # the last two decisions; the finalize (and success) stay

    def drop_middle(rows):
        del rows[4]

    def duplicate(rows):
        rows.insert(3, copy.deepcopy(rows[3]))

    def out_of_range(rows):
        _decisions(rows)[-1]["decision_idx"] = 10

    def no_finalize(rows):
        del rows[-1]

    def two_finalize(rows):
        rows.append(copy.deepcopy(rows[-1]))

    def other_session(rows):
        rows[0]["conn_id"] = "another-connection"

    def success_stamp(rows):
        rows[2]["success"] = not rows[2]["success"]

    def drop_inner(rows):
        _decisions(rows)[0]["warm_reset"] = None

    return {
        "decision_gap": drop_tail,
        "decision_gap_middle": drop_middle,
        "duplicate_decision": duplicate,
        "decision_gap_range": out_of_range,
        "finalize_missing": no_finalize,
        "duplicate_finalize": two_finalize,
        "non_terminal": finalize("terminal", False),
        "decision_count_mismatch": finalize("n_decisions", 9),
        "outcome_mismatch": finalize("outcome", None),
        "outcome_mismatch_stamp": success_stamp,
        "duplicate_session": other_session,
        "identity_mismatch": outer("task_uid", "other:eval:4:3"),
        "identity_mismatch_attempt": outer("attempt", 2),
        "identity_mismatch_yaml": outer("yaml_id", "z"),
        "identity_mismatch_bundle": outer("bundle_id", "c"),
        "identity_mismatch_sha": outer("yaml_sha256", "0" * 64),
        "identity_mismatch_wire": lambda rows: rows[1]["identity"].update(orig_init_state_idx=4),
        "spec_mismatch": inner("spec_digest", "0" * 64),
        "spec_mismatch_outer": outer("spec_digest", "0" * 64),
        "spec_mismatch_source": inner("source", "not-an-arm-source"),
        "spec_mismatch_point": inner("point", "not-a-start-point"),
        "spec_mismatch_kind": inner("kind", "exact"),
        "spec_mismatch_level": inner("level", True),
        "schedule_mismatch": inner("start_t", 0.3),
        "schedule_mismatch_outer": outer("schedule_id", "groot_n15_k8_v1"),
        "schedule_mismatch_times": inner("t", [9.0, 8.0]),
        "schedule_mismatch_dt": inner("dt", 0.0),
        "schedule_mismatch_family": outer("family", "unknown-model"),
        "warm_reset_missing": drop_inner,
        "hit_type_mismatch": outer("hit_type", "MISS"),
        "steps_mismatch": inner("continuation_nfe", 3),
        "extra_stage3_calls": inner("n_stage3_calls", 2),
        "decision_error": outer("status", "error"),
        "schema_mismatch": outer("schema", "warm_reset_evidence_v0"),
        "invalid_row": outer("row_kind", "summary"),
        "invalid_field": inner("n_stage3_calls", True),
        "decision_nfe_mismatch": inner("decision_nfe", 99),
    }


MUTATIONS = _mutations()


@pytest.mark.parametrize("name", sorted(MUTATIONS))
def test_negative_is_rejected_with_its_code(episode, name):
    rows = copy.deepcopy(episode.rows)
    MUTATIONS[name](rows)
    result = episode_problems(rows, expected=episode.expected)
    code = re.sub(r"_(middle|range|stamp|attempt|yaml|bundle|sha|wire|outer|source|point|kind|level|times|dt|family)$", "", name)
    assert result["problems"][code] > 0, (name, dict(result["problems"]))
    assert result["total_nfe"] is None and result["continuation_nfe"] is None


@pytest.mark.parametrize("name,mutate", [
    ("self_start_missing", lambda w: w.update(self_start=False)),
    ("extra_self_start_calls", lambda w: w.update(self_start_calls=2)),
    ("self_seed_mismatch", lambda w: w.update(self_seed=w["self_seed"] + 1)),
    ("self_direct_nfe_mismatch", lambda w: w.update(self_direct_nfe=9, decision_nfe=w["continuation_nfe"] + 9)),
])
def test_self_arm_proof_is_required(tmp_path, name, mutate):
    stack = pi05_stack(tmp_path, "selfwarmreset_t0.2")
    run_episode(stack, 4)
    rows = read_rows(stack.served.evidence_path)
    mutate(rows[1]["warm_reset"])
    assert episode_problems(rows, expected=expected_for(stack, n=4))["problems"][name] > 0


def test_cache_arm_must_not_carry_self_fields(tmp_path):
    stack = pi05_stack(tmp_path, "warmreset_t0.2")
    run_episode(stack, 4)
    for key, value in (("self_start", True), ("self_seed", 1), ("self_direct_nfe", 10), ("self_start_calls", 1)):
        rows = read_rows(stack.served.evidence_path)
        rows[0]["warm_reset"][key] = value
        assert episode_problems(rows, expected=expected_for(stack, n=4))["problems"]["self_start_on_cache_arm"] > 0


def test_empty_evidence_and_bad_expectations(tmp_path):
    stack = pi05_stack(tmp_path, "warmreset_t0.2")
    run_episode(stack, 3)
    rows = read_rows(stack.served.evidence_path)
    assert episode_problems([], expected=expected_for(stack, n=3))["problems"]["server_evidence_missing"] == 1
    assert episode_problems(rows, expected=expected_for(stack, n=0))["problems"]["no_decisions"] == 1
    for bad in ({"spec_digest": "0" * 64}, {"outcome": 1}, {"n_decisions": True}, {"k": 8}, {"start_t": 0.25}):
        result = episode_problems(rows, expected=expected_for(stack, n=3, **bad))
        assert result["problems"]["invalid_expected"] == 1 and result["total_nfe"] is None, bad


@pytest.mark.parametrize("field", ["conn_id", "episode_seq"])
def test_invalid_session_fields_report_a_problem(tmp_path, field):
    stack = pi05_stack(tmp_path, "warmreset_t0.2")
    run_episode(stack, 1)
    for value in ([], {}, None, True):
        rows = read_rows(stack.served.evidence_path)
        rows[0][field] = value
        result = episode_problems(rows, expected=expected_for(stack, n=1))
        assert result["problems"]["invalid_field"] and result["total_nfe"] is None


def test_expected_identity_must_be_a_consistent_mapping(tmp_path):
    stack = pi05_stack(tmp_path, "warmreset_t0.2")
    run_episode(stack, 1)
    rows = read_rows(stack.served.evidence_path)
    for identity in (None, [], {**EPISODE, **EXTRA, "attempt": 9}):
        result = episode_problems(rows, expected=expected_for(stack, n=1, identity=identity))
        assert result["problems"] == {"invalid_expected": 1}


def test_old_server_rows_without_evidence_are_rejected(tmp_path):
    """A server that silently ignored the block (plan F3) writes no rows at all."""
    stack = pi05_stack(tmp_path, "warmreset_t0.2")
    assert episode_problems([], expected=expected_for(stack, n=5))["problems"] == {"server_evidence_missing": 1}


# ------------------------------------------------------------------
# Trusted-input adaptation sample (test layer only; plan §4.7.3)
# ------------------------------------------------------------------


def expected_from_trusted(*, journal, per_step, task: EpisodeTask, run_id: str, task_name: str,
                          yaml_text: str, spec, schedule_id: str, k: int, start_t: float) -> ExpectedEpisode:
    """What an exp analyser does: expectations from the accepted terminal and driver-stamped rows only."""
    terminals = [r for r in journal if r["task_uid"] == task.task_uid and r["run_id"] == run_id
                 and r.get("accepted") is True and r.get("status") in ("done", "failed")]
    if len(terminals) != 1:
        raise ValueError(f"{len(terminals)} accepted terminals")
    term = terminals[0]
    if term.get("error") or type(term.get("success")) is not bool:
        raise ValueError("accepted terminal is not a clean outcome")
    if term["yaml_id"] != task.yaml_id:
        raise ValueError("terminal and dispatched task disagree on yaml_id")
    rows = [r for r in per_step
            if r.get("run_id") == run_id and r.get("yaml_id") == task.yaml_id and r.get("accepted") is True
            and r.get("task_uid") == task.task_uid and r.get("attempt") == term["attempt"] and "hit_type" in r]
    if any(r.get("success") is not term["success"] for r in rows):
        raise ValueError("stale success on a per-step row")
    steps = [r["step_idx"] for r in rows]
    if len(set(steps)) != len(steps) or any(type(s) is not int or s < 0 for s in steps):
        raise ValueError("duplicate or invalid step_idx")
    identity = {"experiment": task.experiment, "task": task_name, "episode_id": task.episode_idx,
                "task_id": task.task_id, "orig_init_state_idx": task.orig_init_state_idx,
                "task_uid": task.task_uid, "attempt": term["attempt"]}
    return ExpectedEpisode(
        task_uid=task.task_uid, attempt=term["attempt"], outcome=term["success"], n_decisions=len(rows),
        yaml_id=task.yaml_id, bundle_id=task.bundle_id, spec=spec, spec_digest=spec.digest(),
        yaml_sha256=hashlib.sha256(yaml_text.encode("utf-8")).hexdigest(), schedule_id=schedule_id, k=k,
        start_t=start_t, identity=identity,
    )


def _libero_task(attempt=1):
    return EpisodeTask(task_uid="arm_a:eval:4:3", yaml_id="arm_a", phase="eval", experiment="libero_10",
                       task_id=4, episode_idx=3, orig_init_state_idx=3, server_host="h", server_port=1,
                       bundle_id="bundle_7", attempt=attempt, extra={"num_trials_per_task": 50})


def _libero_rows(task, run_id, n, *, success, attempt):
    from examples.libero.episode_runner import _hit_row

    rows = []
    for i in range(n):
        row = _hit_row(task, 5 * i, {"hit_type": "WARM_START", "start_t": 0.2}, 50)
        row.update(success=success, attempt=attempt, accepted=True, yaml_id=task.yaml_id, run_id=run_id)
        rows.append(row)
    return rows


def test_trusted_inputs_admit_a_libero_episode(tmp_path):
    from examples.libero.episode_runner import _episode_extra_metadata

    task = _libero_task()
    stack = pi05_stack(tmp_path, "selfwarmreset_t0.2", yaml_id="arm_a", bundle_id="bundle_7")
    language = "put both the alphabet soup and the cream cheese box in the basket"
    stack.served.on_task_begin()
    stack.served.on_episode_start(experiment=task.experiment, task=language, episode_id=task.episode_idx,
                                  extra_metadata=_episode_extra_metadata(task))
    for _ in range(4):
        stack.served.infer(obs_for(stack))
    stack.served.on_episode_end(True)
    per_step = _libero_rows(task, "run1", 4, success=True, attempt=1)
    assert per_step[0]["episode_id"] != task.episode_idx  # the global id is not the wire episode id
    # another run of the same uid / attempt and a stale attempt never join
    per_step += _libero_rows(task, "run0", 6, success=False, attempt=1)
    per_step += [dict(r, attempt=0, success=False) for r in _libero_rows(task, "run1", 2, success=False, attempt=0)]
    journal = [{"task_uid": task.task_uid, "attempt": 1, "status": "done", "accepted": True, "success": True,
                "error": None, "run_id": "run1", "yaml_id": "arm_a"}]
    expected = expected_from_trusted(journal=journal, per_step=per_step, task=task, run_id="run1",
                                     task_name=language, yaml_text=stack.yaml_path.read_text(), spec=stack.spec,
                                     schedule_id="pi05_v1", k=10, start_t=0.2)
    assert expected.n_decisions == 4 and expected.yaml_id != expected.bundle_id
    result = episode_problems(read_rows(stack.served.evidence_path), expected=expected)
    assert not +result["problems"] and result["total_nfe"] == 4 * (10 + 2)


def test_trusted_inputs_for_a_robocasa_episode_use_the_canonical_name(tmp_path):
    task = EpisodeTask(task_uid="arm_b:eval:1:7", yaml_id="arm_b", phase="eval", experiment="robocasa365",
                       task_id=1, episode_idx=7, orig_init_state_idx=7, server_host="h", server_port=1,
                       bundle_id="bundle_b", attempt=1)
    stack = pi05_stack(tmp_path, "warmreset_t0.2", yaml_id="arm_b", bundle_id="bundle_b")
    stack.served.on_task_begin()
    meta = {"task_uid": task.task_uid, "attempt": 1, "task_id": 1, "orig_init_state_idx": 7, "seed": 2_000_007}
    stack.served.on_episode_start(experiment="robocasa365", task="CloseFridge", episode_id=7, extra_metadata=meta)
    stack.served.infer(obs_for(stack))
    stack.served.on_episode_end(False)
    per_step = [{"task_uid": task.task_uid, "yaml_id": "arm_b", "step_idx": -1, "prompt": "p"}]  # header row
    per_step += [{"task_uid": task.task_uid, "yaml_id": "arm_b", "step_idx": 0, "hit_type": "WARM_START",
                  "success": False, "attempt": 1, "accepted": True, "run_id": "r"}]
    for row in per_step:
        row.update(success=False, attempt=1, accepted=True, run_id="r")
    journal = [{"task_uid": task.task_uid, "attempt": 1, "status": "failed", "accepted": True, "success": False,
                "error": None, "run_id": "r", "yaml_id": "arm_b"}]
    expected = expected_from_trusted(journal=journal, per_step=per_step, task=task, run_id="r",
                                     task_name="CloseFridge", yaml_text=stack.yaml_path.read_text(),
                                     spec=stack.spec, schedule_id="pi05_v1", k=10, start_t=0.2)
    assert expected.n_decisions == 1  # the header row has no hit_type
    assert not +episode_problems(read_rows(stack.served.evidence_path), expected=expected)["problems"]
    wrong = ExpectedEpisode(**{**expected.__dict__, "identity": {**expected.identity, "task": "close the fridge"}})
    assert episode_problems(read_rows(stack.served.evidence_path), expected=wrong)["problems"]["identity_mismatch"]


@pytest.mark.parametrize("defect", ["no_terminal", "terminal_error", "two_terminals", "stale_success", "dup_step"])
def test_untrustworthy_inputs_are_refused_by_the_adapter(defect):
    task = _libero_task()
    journal = [{"task_uid": task.task_uid, "attempt": 1, "status": "done", "accepted": True, "success": True,
                "error": None, "run_id": "run1", "yaml_id": "arm_a"}]
    per_step = _libero_rows(task, "run1", 3, success=True, attempt=1)
    if defect == "no_terminal":
        journal = []
    elif defect == "terminal_error":
        journal[0]["error"] = "worker crashed"
    elif defect == "two_terminals":
        journal.append(dict(journal[0], attempt=2, success=False))
    elif defect == "stale_success":
        per_step[1]["success"] = False
    else:
        per_step[2]["step_idx"] = per_step[1]["step_idx"]
    with pytest.raises(ValueError):
        expected_from_trusted(journal=journal, per_step=per_step, task=task, run_id="run1", task_name="t",
                              yaml_text="", spec=spec_for("pi05", "warmreset_t0.2"), schedule_id="pi05_v1", k=10,
                              start_t=0.2)
