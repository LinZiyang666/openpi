"""Tests for the ws2 evidence seam: base-class hook, opt-in wiring, join dedup.

The header hook is the only new surface on ``RobocasaEpisodeRunner``; these
tests pin (a) the base class stays byte-identical in what it reports, (b) the
ws2 subclass emits exactly one header carrying the values the environment
actually produced at its single reset, and (c) a retried episode's evidence
resolves to the last attempt only.
"""

from __future__ import annotations

import pytest

from exp.robocasa365.episode_runner import RobocasaEpisodeRunner
from exp.robocasa365.ws2_episode_runner import HEADER_STEP_IDX, Ws2EpisodeRunner

from tests.robocasa365.test_robocasa_episode_runner import (
    PROMPT,
    _FakeAdapter,
    _FakeClient,
    _FakeEnv,
    _make_task,
    _report,
)


def _runner(cls, *, horizon=3, env=None, client=None):
    client = client or _FakeClient()
    env = env or _FakeEnv()
    runner = cls(
        _FakeAdapter(),
        client_factory=lambda server: client,
        gym_make=lambda task_name, layout, style, **kw: env,
        horizon_fn=lambda name: horizon,
        handshake_probe=lambda server, timeout_s: None,
        connect_deadline_s=0.05,
        connect_retries=2,
    )
    return runner, client, env


def test_base_runner_emits_no_header_row():
    runner, _, env = _runner(RobocasaEpisodeRunner)
    result = runner.run(_make_task(), _report)
    assert result.per_step_rows, "the hit-meta rows must still be reported"
    assert all(row["step_idx"] >= 0 for row in result.per_step_rows)
    assert all("prompt" not in row for row in result.per_step_rows)
    assert env.reset_seeds == [107]  # base_seed 100 + orig_init_state_idx 7


def test_base_hook_returns_empty_list():
    runner, _, _ = _runner(RobocasaEpisodeRunner)
    assert runner._episode_header_rows(_make_task(), prompt="x", seed=1) == []  # noqa: SLF001


def test_ws2_runner_emits_exactly_one_header_with_reset_truth():
    runner, _, env = _runner(Ws2EpisodeRunner)
    task = _make_task()
    result = runner.run(task, _report)

    headers = [row for row in result.per_step_rows if row["step_idx"] == HEADER_STEP_IDX]
    assert len(headers) == 1
    header = headers[0]
    assert header["task_uid"] == task.task_uid
    assert header["yaml_id"] == task.yaml_id
    assert header["attempt"] == task.attempt
    # The values the env actually produced at its single reset — no replay.
    assert header["prompt"] == PROMPT
    assert header["seed"] == env.reset_seeds[0] == 107
    # The header leads; decision rows follow in step order.
    assert result.per_step_rows[0] is header
    assert [row["step_idx"] for row in result.per_step_rows[1:]] == sorted(
        row["step_idx"] for row in result.per_step_rows[1:]
    )


def test_ws2_runner_resets_once_per_episode():
    runner, _, env = _runner(Ws2EpisodeRunner)
    runner.run(_make_task(), _report)
    assert len(env.reset_seeds) == 1


def test_worker_entry_defaults_to_base_runner():
    from exp.robocasa365 import worker_entry

    args = worker_entry.parse_args([
        "--worker-id", "w0", "--server-key", "h:1", "--driver-host", "h",
        "--driver-port", "1", "--teacher", "groot_tp", "--connect-deadline-s", "1",
        "--episode-deadline-s", "1", "--terminate-grace-s", "1",
    ])
    assert args.episode_header_rows is False
    runner = worker_entry.build_runner(args)
    inner = runner._inner  # noqa: SLF001 - WatchdogRunner wraps the real runner
    assert type(inner) is RobocasaEpisodeRunner


def test_worker_entry_opt_in_selects_evidence_runner():
    from exp.robocasa365 import worker_entry

    args = worker_entry.parse_args([
        "--worker-id", "w0", "--server-key", "h:1", "--driver-host", "h",
        "--driver-port", "1", "--teacher", "groot_tp", "--connect-deadline-s", "1",
        "--episode-deadline-s", "1", "--terminate-grace-s", "1",
        "--episode-header-rows",
    ])
    assert args.episode_header_rows is True
    runner = worker_entry.build_runner(args)
    assert isinstance(runner._inner, Ws2EpisodeRunner)  # noqa: SLF001


def test_default_path_never_imports_the_evidence_module():
    """Opt-in must be structural: a plain worker must not load ws2 code at all.

    Runs in a subprocess because this test module imports the ws2 runner at
    top level, which would pre-populate ``sys.modules`` and make the check
    vacuous in-process.
    """
    import subprocess
    import sys

    script = (
        "import sys; from exp.robocasa365 import worker_entry; "
        "args = worker_entry.parse_args(["
        "'--worker-id','w0','--server-key','h:1','--driver-host','h',"
        "'--driver-port','1','--teacher','groot_tp','--connect-deadline-s','1',"
        "'--episode-deadline-s','1','--terminate-grace-s','1']); "
        "worker_entry.build_runner(args); "
        "print('ws2' if 'exp.robocasa365.ws2_episode_runner' in sys.modules else 'clean')"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "clean"


def test_ws2_spawn_passes_the_flag_and_new_session(monkeypatch):
    from exp.robocasa365 import run_ws_search2
    from openpi.conductor import WorkerSpec

    captured = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(run_ws_search2.subprocess, "Popen", fake_popen)
    run_ws_search2.ws2_spawn_fn(
        WorkerSpec(worker_id="w0", server_key="h:1", gpu_id="3"),
        "dh", 42,
        worker_python="/py", robocasa_cwd="/cwd", repo_root="/repo",
        egl_lib_dir="/egl", egl_vendor_dir="/vendor", teacher="groot_tp",
        connect_deadline_s=1.0, episode_deadline_s=2.0, terminate_grace_s=3.0,
        max_cached_envs=1,
    )
    assert "--episode-header-rows" in captured["cmd"]
    assert captured["cmd"][:3] == ["/py", "-m", "exp.robocasa365.worker_entry"]
    assert captured["kwargs"]["start_new_session"] is True


# ------------------------------------------------------------------
# retry dedup in the attribution join
# ------------------------------------------------------------------


def test_join_uses_last_attempt_block_only():
    from exp.robocasa365.analyze_ws2_vs_ws1 import attempt_rows, join_buckets

    uid = "ws2-iso_x__l1s1_groot_tp__OpenCabinet:eval:0:0"
    rows = [
        # attempt 1: a winner from bucket 0, then the episode died and retried
        {"task_uid": uid, "step_idx": -1, "attempt": 1, "prompt": "stale", "seed": 1},
        {"task_uid": uid, "step_idx": 0, "attempt": 1,
         "winner_id": "OpenCabinet/episode_0000:3"},
        # attempt 2: the accepted run, winners from bucket 1
        {"task_uid": uid, "step_idx": -1, "attempt": 2, "prompt": "fresh", "seed": 1},
        {"task_uid": uid, "step_idx": 0, "attempt": 2,
         "winner_id": "OpenCabinet/episode_0007:4"},
    ]
    assert len(attempt_rows(rows, {"attempt": 2})) == 2

    variants = {
        "trajectory_to_bucket": {
            "OpenCabinet/episode_0000": 0,
            "OpenCabinet/episode_0007": 1,
        },
        "buckets": [
            {"bucket_index": 0, "representative": {"prompt": "stale", "status": "resolved"}},
            {"bucket_index": 1, "representative": {"prompt": "fresh", "status": "resolved"}},
        ],
    }
    joined = join_buckets(rows, {uid: {"attempt": 2}}, variants)
    assert len(joined) == 1
    assert joined[0]["eval_prompt"] == "fresh"
    assert joined[0]["top_bucket"] == 1
    assert joined[0]["matched"] is True
    assert joined[0]["n_searches"] == 1


def test_join_ignores_a_stale_block_that_landed_last():
    """A fenced attempt can arrive AFTER its replacement (driver appends on arrival)."""
    from exp.robocasa365.analyze_ws2_vs_ws1 import attempt_rows, join_buckets

    uid = "ws2-iso_x__l1s1_groot_tp__OpenCabinet:eval:0:0"
    rows = [
        # the accepted attempt lands first ...
        {"task_uid": uid, "step_idx": -1, "attempt": 2, "prompt": "fresh",
         "seed": 1, "accepted": True},
        {"task_uid": uid, "step_idx": 0, "attempt": 2,
         "winner_id": "OpenCabinet/episode_0007:4", "accepted": True},
        # ... then the slow, fenced attempt-1 result arrives and is appended.
        {"task_uid": uid, "step_idx": -1, "attempt": 1, "prompt": "stale",
         "seed": 1, "accepted": False},
        {"task_uid": uid, "step_idx": 0, "attempt": 1,
         "winner_id": "OpenCabinet/episode_0000:3", "accepted": False},
    ]
    assert [r["attempt"] for r in attempt_rows(rows, {"attempt": 2}) if r["step_idx"] == -1] == [2]

    variants = {
        "trajectory_to_bucket": {
            "OpenCabinet/episode_0000": 0,
            "OpenCabinet/episode_0007": 1,
        },
        "buckets": [
            {"bucket_index": 0, "representative": {"prompt": "stale", "status": "resolved"}},
            {"bucket_index": 1, "representative": {"prompt": "fresh", "status": "resolved"}},
        ],
    }
    joined = join_buckets(rows, {uid: {"attempt": 2}}, variants)
    assert joined[0]["eval_prompt"] == "fresh"
    assert joined[0]["top_bucket"] == 1


def test_join_flags_unmapped_winner_ids():
    from exp.robocasa365.analyze_ws2_vs_ws1 import join_buckets

    uid = "ws2-iso_x__l1s1_groot_tp__OpenDrawer:eval:1:0"
    rows = [
        {"task_uid": uid, "step_idx": -1, "attempt": 1, "prompt": "p", "seed": 5},
        {"task_uid": uid, "step_idx": 0, "attempt": 1,
         "winner_id": "Unknown/episode_0001:2"},
    ]
    variants = {"trajectory_to_bucket": {}, "buckets": []}
    joined = join_buckets(rows, {uid: {"attempt": 1}}, variants)
    assert joined[0]["n_unmapped_winners"] == 1
    assert joined[0]["top_bucket"] is None
    assert joined[0]["matched"] is None


def test_join_skips_episodes_absent_from_the_accepted_set():
    from exp.robocasa365.analyze_ws2_vs_ws1 import join_buckets

    rows = [{"task_uid": "not-accepted", "step_idx": -1, "attempt": 1, "prompt": "p", "seed": 0}]
    assert join_buckets(rows, {}, {"trajectory_to_bucket": {}, "buckets": []}) == []


@pytest.mark.parametrize("cls", [RobocasaEpisodeRunner, Ws2EpisodeRunner])
def test_hit_meta_rows_identical_across_runners(cls):
    """The evidence runner adds a header; it must not alter decision rows."""
    runner, _, _ = _runner(cls)
    rows = [r for r in runner.run(_make_task(), _report).per_step_rows if r["step_idx"] >= 0]
    assert rows
    assert all(set(r) == {"task_uid", "yaml_id", "step_idx", "hit_type",
                          "winner_id", "cp1_score", "searched"} for r in rows)


def test_header_row_shape_is_frozen():
    """The header's identity fields come from the dispatched task, unchanged."""
    task = _make_task()
    runner, _, _ = _runner(Ws2EpisodeRunner)
    header = runner._episode_header_rows(task, prompt="p", seed=9)[0]  # noqa: SLF001
    assert header == {
        "task_uid": task.task_uid,
        "yaml_id": task.yaml_id,
        "step_idx": HEADER_STEP_IDX,
        "attempt": task.attempt,
        "prompt": "p",
        "seed": 9,
    }


def test_join_binds_the_attempt_and_run_id_the_journal_accepted():
    """Dispatch generations restart at 1 in a fresh process — run_id separates them."""
    from exp.robocasa365.analyze_ws2_vs_ws1 import attempt_rows, join_buckets

    uid = "ws2-iso_x__l1s1_groot_tp__OpenCabinet:eval:0:0"
    rows = [
        # pre-crash run: attempt 1 under run A
        {"task_uid": uid, "step_idx": -1, "attempt": 1, "run_id": "runA",
         "prompt": "pre-crash", "seed": 1},
        {"task_uid": uid, "step_idx": 0, "attempt": 1, "run_id": "runA",
         "winner_id": "OpenCabinet/episode_0000:3"},
        # the re-run after resume: attempt is 1 AGAIN, only run_id differs
        {"task_uid": uid, "step_idx": -1, "attempt": 1, "run_id": "runB",
         "prompt": "after-resume", "seed": 1},
        {"task_uid": uid, "step_idx": 0, "attempt": 1, "run_id": "runB",
         "winner_id": "OpenCabinet/episode_0007:4"},
    ]
    accepted = {uid: {"attempt": 1, "run_id": "runB"}}
    picked = attempt_rows(rows, accepted[uid])
    assert [r.get("run_id") for r in picked] == ["runB", "runB"]

    variants = {
        "trajectory_to_bucket": {"OpenCabinet/episode_0000": 0, "OpenCabinet/episode_0007": 1},
        "buckets": [
            {"bucket_index": 0, "representative": {"prompt": "pre-crash", "status": "resolved"}},
            {"bucket_index": 1, "representative": {"prompt": "after-resume", "status": "resolved"}},
        ],
    }
    joined = join_buckets(rows, accepted, variants)
    assert joined[0]["eval_prompt"] == "after-resume"
    assert joined[0]["top_bucket"] == 1
    assert joined[0]["run_id"] == "runB"


def test_join_refuses_a_verdict_on_an_unresolved_bucket():
    """A representative whose replay failed cannot support a match verdict."""
    from exp.robocasa365.analyze_ws2_vs_ws1 import join_buckets

    uid = "ws2-iso_x__l1s1_groot_tp__OpenDrawer:eval:1:0"
    rows = [
        {"task_uid": uid, "step_idx": -1, "attempt": 1, "prompt": "open the drawer", "seed": 5},
        {"task_uid": uid, "step_idx": 0, "attempt": 1, "winner_id": "OpenDrawer/episode_0002_a01:7"},
    ]
    variants = {
        "trajectory_to_bucket": {"OpenDrawer/episode_0002_a01": 0},
        "buckets": [{"bucket_index": 0, "ambiguous": False,
                     "representative": {"prompt": None, "status": "unresolved"}}],
    }
    joined = join_buckets(rows, {uid: {"attempt": 1}}, variants)
    assert joined[0]["top_bucket"] == 0
    assert joined[0]["bucket_variant_status"] == "unresolved"
    assert joined[0]["matched"] is None


def test_join_refuses_a_verdict_on_an_ambiguous_bucket():
    """A bucket spanning >1 task has no single variant to compare against."""
    from exp.robocasa365.analyze_ws2_vs_ws1 import join_buckets

    uid = "ws2-iso_x__l1s1_groot_tp__OpenDrawer:eval:1:0"
    rows = [
        {"task_uid": uid, "step_idx": -1, "attempt": 1, "prompt": "open the drawer", "seed": 5},
        {"task_uid": uid, "step_idx": 0, "attempt": 1, "winner_id": "OpenDrawer/episode_0002_a01:7"},
    ]
    variants = {
        "trajectory_to_bucket": {"OpenDrawer/episode_0002_a01": 0},
        "buckets": [{"bucket_index": 0, "ambiguous": True,
                     "representative": {"prompt": "open the drawer", "status": "resolved"}}],
    }
    joined = join_buckets(rows, {uid: {"attempt": 1}}, variants)
    assert joined[0]["bucket_ambiguous"] is True
    assert joined[0]["matched"] is None


def test_join_reports_object_class_from_the_bucket_map():
    from exp.robocasa365.analyze_ws2_vs_ws1 import join_buckets

    uid = "ws2-iso_x__l1s1_groot_tp__PickPlaceCounterToCabinet:eval:6:0"
    rows = [
        {"task_uid": uid, "step_idx": -1, "attempt": 1, "prompt": "pick the apple", "seed": 9},
        {"task_uid": uid, "step_idx": 0, "attempt": 1,
         "winner_id": "PickPlaceCounterToCabinet/episode_0011_a01:3"},
    ]
    variants = {
        "trajectory_to_bucket": {"PickPlaceCounterToCabinet/episode_0011_a01": 4},
        "buckets": [{"bucket_index": 4, "ambiguous": False, "representative": {
            "prompt": "pick the apple", "status": "resolved",
            "object_class": [{"name": "obj", "cat": "apple"}]}}],
    }
    joined = join_buckets(rows, {uid: {"attempt": 1}}, variants)
    assert joined[0]["matched"] is True
    assert joined[0]["bucket_object_class"] == [{"name": "obj", "cat": "apple"}]
