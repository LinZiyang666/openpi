"""Contract tests for the ws2 artefacts and gates raised at G2.

Four surfaces that synthetic component tests could not cover: the REAL
selection manifest that decides the control arm's sample, the driver's failure
propagation, the orchestrator's readiness/resource admission, and the formal
analysis completeness gate.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = REPO / "exp" / "robocasa365" / "config" / "ws_search2" / "selection_manifest.json"
WS1_INDEX = REPO / "exp" / "robocasa365" / "config" / "ws_search" / "groot_tp" / "index.json"


# ------------------------------------------------------------------
# the real, committed selection manifest
# ------------------------------------------------------------------


def _manifest() -> dict:
    if not MANIFEST.exists():
        pytest.fail(f"the audited selection manifest is missing: {MANIFEST}")
    return json.loads(MANIFEST.read_text())


def test_manifest_ws2c_is_the_frozen_twelve_cell_sample():
    segment = _manifest()["segments"]["ws2c"]
    index = json.loads(WS1_INDEX.read_text())
    cells = segment["cells"]
    assert len(cells) == len(set(cells)) == 12
    assert set(cells) <= set(index), "every control cell must exist in the round-1 matrix"
    iso = [c for c in cells if c.split("_", 1)[0] == "iso"]
    assert sorted(iso) == sorted(segment["iso_cids"]) and len(iso) == 4
    assert len(segment["top8_cids"]) == 8
    assert not set(segment["top8_cids"]) & set(iso), "top-8 excludes the iso anchors"


def test_manifest_records_its_statistical_parameters():
    segment = _manifest()["segments"]["ws2c"]
    # Frozen protocol: the round-1 analyzer's own defaults, so the tied set
    # this selection saw is the published one.
    assert segment["params"] == {
        "alpha": 0.05, "episodes": 8, "resamples": 20000, "run_prefix": "ws1", "seed": 12345,
    }
    assert _manifest()["algorithm"] == "ws2_selection_v1"


def test_manifest_pins_every_source_journal_by_hash():
    segment = _manifest()["segments"]["ws2c"]
    sources = segment["source_journals"]
    assert len(sources) == 132, "the selection must be traceable to the full round-1 matrix"
    assert all(len(sha) == 64 and set(sha) <= set("0123456789abcdef") for sha in sources.values())
    assert all(name.startswith("journal_ws1-") and name.endswith(".jsonl") for name in sources)


def test_manifest_leader_matches_the_published_round1_result():
    """Independent confirmation the selection reproduces the reported readout.

    ``ws_search_round1.md`` §2 names ``grid2 v2@87.5/rs@12.5`` as the leader;
    a selection that ranked differently would mean the protocol drifted.
    """
    segment = _manifest()["segments"]["ws2c"]
    assert segment["leader"] == "grid_vision_2@87_robot_state@12"
    assert segment["leader"] in segment["cells"]
    assert segment["padding_used"] is False


# ------------------------------------------------------------------
# driver failure propagation (subprocess: exit code is the contract)
# ------------------------------------------------------------------


def _phase_fixture(tmp_path, cids, *, episodes=2, tasks=("OpenDrawer", "CloseFridge")):
    """A runnable ws2 phase on disk: config dir, index, env config."""
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    for cid in cids:
        (config_dir / f"{cid}.yaml").write_text(f"# {cid}\n")
    (config_dir / "index.json").write_text(json.dumps({c: {} for c in cids}))
    env_config = tmp_path / "env.env"
    env_config.write_text(
        "WORKER_PYTHON=/bin/true\nROBOCASA_CWD=/tmp\nREPO_ROOT=/tmp\n"
        "EGL_LIB_DIR=/tmp\nEGL_VENDOR_DIR=/tmp\nGROOT_SERVERS=h:23160\n"
    )
    return config_dir, env_config, [(name, episodes) for name in tasks]


def _main_argv(config_dir, env_config, data_dir, **extra):
    argv = [
        "run_ws_search2", "--teacher", "groot_tp", "--servers", "h:23160",
        "--config-dir", str(config_dir), "--env-config", str(env_config),
        "--data-dir", str(data_dir), "--role", "driver", "--bind-port", "23999",
        "--episodes", "2", "--tasks", "OpenDrawer,CloseFridge",
    ]
    for key, value in extra.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return argv


@pytest.mark.parametrize("fail_on_batch", [1, 2])
def test_a_failing_batch_stops_the_phase_through_real_main(monkeypatch, tmp_path, fail_on_batch):
    """First AND middle batch failures must halt the phase, unfinalised.

    Drives the production ``main()`` — argument plumbing, batch loop, thread
    handoff and the finalize guard all included — with only ``ConductorDriver``
    faked, so a regression in the (unapproved, race-prone) batching deviation
    cannot slip through.
    """
    import sys

    from exp.robocasa365 import run_ws_search2 as mod

    cids = ["iso_a", "grid_b", "grid3_c", "grid3v_d"]
    config_dir, env_config, _ = _phase_fixture(tmp_path, cids)
    started: list[int] = []

    class _FakeDriver:
        def __init__(self, *a, **k):
            del a, k
            self.port = 24000 + len(started)
            self._n = len(started) + 1
            started.append(self._n)

        def run(self):
            if self._n == fail_on_batch:
                raise RuntimeError(f"simulated batch {self._n} failure")

        def handle_pull(self, server_key):
            del server_key
            return ("assign", {"none": True, "backoff_ms": 1})

    finalized: list[int] = []
    monkeypatch.setattr(mod, "ConductorDriver", _FakeDriver)
    monkeypatch.setattr(mod, "finalize", lambda *a, **k: finalized.append(1) or {})
    monkeypatch.setattr(mod, "validate_teacher_endpoints", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", _main_argv(config_dir, env_config, tmp_path,
                                                cells_per_batch=2))

    with pytest.raises(SystemExit) as excinfo:
        mod.main()

    assert "ws2 driving failed" in str(excinfo.value)
    assert not finalized, "a failed phase must not write products"
    # The failing batch is the last one attempted: nothing after it ran.
    assert started == list(range(1, fail_on_batch + 1))


def test_all_batches_run_and_finalize_once_on_success(monkeypatch, tmp_path):
    import sys

    from exp.robocasa365 import run_ws_search2 as mod

    cids = ["iso_a", "grid_b", "grid3_c", "grid3v_d"]
    config_dir, env_config, _ = _phase_fixture(tmp_path, cids)
    started: list[int] = []

    class _FakeDriver:
        def __init__(self, *a, **k):
            del a, k
            self.port = 24100 + len(started)
            started.append(len(started) + 1)

        def run(self):
            return None

        def handle_pull(self, server_key):
            del server_key
            return ("shutdown", {})

    finalized: list[int] = []

    def fake_finalize(central, data_dir, *, teacher, expected_by_run):
        del central, data_dir, teacher
        finalized.append(1)
        return {run_id: True for run_id in expected_by_run}

    monkeypatch.setattr(mod, "ConductorDriver", _FakeDriver)
    monkeypatch.setattr(mod, "finalize", fake_finalize)
    monkeypatch.setattr(mod, "validate_teacher_endpoints", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", _main_argv(config_dir, env_config, tmp_path,
                                                cells_per_batch=2))
    mod.main()
    assert started == [1, 2], "4 cells / 2 per batch = 2 batches"
    assert len(finalized) == 1, "products are written exactly once, after the last batch"


def test_batching_requires_a_fixed_bind_port(monkeypatch, tmp_path):
    """Workers reconnect by address; an ephemeral port would strand them."""
    import sys

    from exp.robocasa365 import run_ws_search2 as mod

    cids = ["iso_a", "grid_b", "grid3_c"]
    config_dir, env_config, _ = _phase_fixture(tmp_path, cids)
    argv = _main_argv(config_dir, env_config, tmp_path, cells_per_batch=1)
    argv[argv.index("--bind-port") + 1] = "0"
    monkeypatch.setattr(mod, "validate_teacher_endpoints", lambda *a, **k: None)
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit, match="fixed --bind-port"):
        mod.main()


def test_worker_runs_two_non_empty_batches_over_one_port(tmp_path):
    """Real driver + real WorkerLoop: batch 1 holds, batch 2 rebinds, worker follows.

    The whole point of the batching deviation is that the fleet SURVIVES a
    batch boundary. So both batches carry real episodes, the second driver
    binds the SAME port the first one used, and the worker must execute batch
    1's episodes, stay alive through the hold, execute batch 2's, and only
    exit when the final batch answers shutdown.
    """
    import socket
    import threading
    import time

    from openpi.conductor import task as _task
    from openpi.conductor.driver import ConductorDriver
    from openpi.conductor.strategy import ExperimentStrategy
    from openpi.conductor.task import EpisodeTask, ServerEndpoint, Stage, TaskGraph

    from exp.robocasa365.run_ws_search2 import hold_workers_between_batches

    server = ServerEndpoint("127.0.0.1", 23160)

    def free_port() -> int:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            return sock.getsockname()[1]

    port = free_port()

    class _Batch(ExperimentStrategy):
        """One stage of real episodes; no bundle load (ctl is unused here)."""

        def __init__(self, cell: str, n: int) -> None:
            self._cell, self._n = cell, n

        def plan(self, yamls, assignment):
            del yamls, assignment
            graph = TaskGraph()
            yaml_id = f"ws2-{self._cell}__l1s1_groot_tp__OpenDrawer"
            graph.add_stage(Stage(
                stage_id=f"0000__{yaml_id}", yaml_id=yaml_id, phase="eval", server=server,
                episodes=[
                    EpisodeTask(
                        task_uid=_task.make_task_uid(yaml_id, "eval", 0, i),
                        yaml_id=yaml_id, phase="eval", experiment="groot_tp",
                        task_id=0, episode_idx=i, orig_init_state_idx=i,
                        server_host=server.host, server_port=server.port,
                        bundle_id=f"ws2-{self._cell}", extra={},
                    )
                    for i in range(self._n)
                ],
            ))
            return graph

    class _Runner:
        def __init__(self) -> None:
            self.ran: list[str] = []

        def run(self, task, report):
            del report
            self.ran.append(task.task_uid)
            return _task.EpisodeResult(task_uid=task.task_uid, success=True, n_steps=1)

        def close(self):
            return None

    def start_batch(cell: str, n: int, *, final: bool):
        driver = ConductorDriver(
            _Batch(cell, n), yaml_weights={f"ws2-{cell}__l1s1_groot_tp__OpenDrawer": n},
            servers=[server], journal_path=str(tmp_path / f"journal_{cell}.jsonl"),
            ctl_factory=lambda s: None, bind_host="127.0.0.1", bind_port=port,
        )
        if not final:
            hold_workers_between_batches(driver)
        thread = threading.Thread(target=driver.run, daemon=True)
        thread.start()
        deadline = time.time() + 10
        while driver.port is None and time.time() < deadline:
            time.sleep(0.01)
        assert driver.port == port, "each batch must bind the same address the fleet holds"
        return driver, thread

    runner = _Runner()
    from openpi.conductor.worker import WorkerLoop

    loop = WorkerLoop("w0", server.key, runner,
                      connect=lambda: socket.create_connection(("127.0.0.1", port), timeout=5))
    worker = threading.Thread(target=loop.run_forever, daemon=True)

    driver1, thread1 = start_batch("iso_a", 2, final=False)
    worker.start()
    # Batch 1: its episodes must actually run.
    deadline = time.time() + 30
    while len(runner.ran) < 2 and time.time() < deadline:
        time.sleep(0.05)
    assert len(runner.ran) == 2, f"batch 1 episodes did not run: {runner.ran}"

    # The held batch answers "no task, back off" instead of dismissing the
    # fleet; the worker is still alive when this batch's driver stops.
    time.sleep(0.5)
    assert worker.is_alive(), "the hold must keep the fleet for the next batch"
    driver1.stop() if hasattr(driver1, "stop") else None
    thread1.join(timeout=5.0)

    # Batch 2 rebinds the same port; the worker reconnects on its own backoff.
    driver2, thread2 = start_batch("grid_b", 2, final=True)
    deadline = time.time() + 60
    while len(runner.ran) < 4 and time.time() < deadline:
        time.sleep(0.05)
    assert len(runner.ran) == 4, f"batch 2 episodes did not run after reconnect: {runner.ran}"
    assert {uid.split("__")[0] for uid in runner.ran} == {"ws2-iso_a", "ws2-grid_b"}

    # The FINAL batch is not held, so its shutdown dismisses the fleet.
    worker.join(timeout=30.0)
    assert not worker.is_alive(), "the final batch must dismiss the fleet"
    thread2.join(timeout=5.0)


# ------------------------------------------------------------------
# orchestrator readiness + resource gates
# ------------------------------------------------------------------


class _Recorder:
    """Captures remote scripts and replays scripted outputs."""

    def __init__(self, outputs: list[str]) -> None:
        self.scripts: list[str] = []
        self.calls: list[tuple[str, str]] = []
        self._outputs = list(outputs)

    def __call__(self, node: str, script: str, *, echo_only: bool,
                 home: str = "/home/weiland") -> str:
        del echo_only
        self.calls.append((node, home))
        self.scripts.append(script)
        return self._outputs.pop(0) if self._outputs else ""


def _gate_args(**over):
    import argparse

    base = dict(node="weilandserver", min_free_ram_gb=40, min_free_vram_mb=8500,
                min_gpu_temp_c=44, tmux_prefix="ws2s")
    base.update(over)
    return argparse.Namespace(**base)


def _probe_output(*, port_used="0", session="FREE", ram="120",
                  vram=("20000", "19800", "19900"), temp="55"):
    return "\n".join([port_used, session, ram, f"{vram[0]}, {temp}", vram[1], vram[2]])


def test_preflight_passes_on_a_healthy_host(monkeypatch, capsys):
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    rec = _Recorder([_probe_output()])
    monkeypatch.setattr(orch, "texec", rec)
    orch.preflight_gate(_gate_args(), 23160, "ws2s23160")
    assert "preflight ok" in capsys.readouterr().out
    script = rec.scripts[0]
    for needle in ("ss -tln", "tmux has-session", "free -g", "nvidia-smi"):
        assert needle in script
    assert script.count("memory.free") == 3, "VRAM must be read three times"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"port_used": "1"}, "already listening"),
        ({"session": "TAKEN"}, "already exists"),
        ({"ram": "12"}, "free RAM"),
        ({"vram": ("20000", "400", "19900")}, "three-read min"),
        ({"temp": "31"}, "warm floor"),
    ],
)
def test_preflight_refuses_each_unsafe_condition(monkeypatch, kwargs, match):
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    monkeypatch.setattr(orch, "texec", _Recorder([_probe_output(**kwargs)]))
    with pytest.raises(SystemExit, match=match):
        orch.preflight_gate(_gate_args(), 23160, "ws2s23160")


def test_readiness_probe_neutralises_the_zero_count_exit():
    """``grep -c`` exits 1 on a zero count — a bare probe aborts the launch."""
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    source = pathlib.Path(orch.__file__).read_text()
    probe_block = source[source.index("deadline = time.time() + args.ready_timeout_s"):]
    probe_block = probe_block[: probe_block.index("time.sleep(10)")]
    assert probe_block.count("|| true") == 2, "both grep -c calls must be neutralised"


# ------------------------------------------------------------------
# strict evidence join (no fallback, no vanishing episodes)
# ------------------------------------------------------------------


def test_stale_run_evidence_is_never_substituted():
    """Journal accepted run B, file holds only run A: that is a gap, not a match."""
    from exp.robocasa365.analyze_ws2_vs_ws1 import attempt_rows, join_buckets

    uid = "ws2-iso_a__l1s1_groot_tp__OpenDrawer:eval:0:0"
    rows = [
        {"task_uid": uid, "step_idx": -1, "attempt": 1, "run_id": "runA",
         "prompt": "pre-crash", "seed": 1},
        {"task_uid": uid, "step_idx": 0, "attempt": 1, "run_id": "runA",
         "winner_id": "OpenDrawer/episode_0000_a01:2"},
    ]
    accepted = {uid: {"attempt": 1, "run_id": "runB"}}
    assert attempt_rows(rows, accepted[uid]) == []

    variants = {
        "trajectory_to_bucket": {"OpenDrawer/episode_0000_a01": 0},
        "buckets": [{"bucket_index": 0, "ambiguous": False,
                     "representative": {"prompt": "pre-crash", "status": "resolved"}}],
    }
    joined = join_buckets(rows, accepted, variants)
    assert len(joined) == 1
    assert joined[0]["bucket_variant_status"] == "run_mismatch"
    assert joined[0]["matched"] is None
    assert joined[0]["eval_prompt"] is None


def test_accepted_episode_without_any_rows_is_reported_as_a_gap():
    from exp.robocasa365.analyze_ws2_vs_ws1 import join_buckets

    uid = "ws2-iso_a__l1s1_groot_tp__OpenDrawer:eval:0:3"
    joined = join_buckets([], {uid: {"attempt": 1, "run_id": "runB"}},
                          {"trajectory_to_bucket": {}, "buckets": []})
    assert len(joined) == 1, "an evidence-less episode must not vanish from the denominator"
    assert joined[0]["bucket_variant_status"] == "missing_evidence"
    assert joined[0]["matched"] is None


def test_rows_without_a_header_are_a_gap_not_a_verdict():
    from exp.robocasa365.analyze_ws2_vs_ws1 import join_buckets

    uid = "ws2-iso_a__l1s1_groot_tp__OpenDrawer:eval:0:1"
    rows = [{"task_uid": uid, "step_idx": 0, "attempt": 1,
             "winner_id": "OpenDrawer/episode_0000_a01:2"}]
    variants = {
        "trajectory_to_bucket": {"OpenDrawer/episode_0000_a01": 0},
        "buckets": [{"bucket_index": 0, "ambiguous": False,
                     "representative": {"prompt": "open the drawer", "status": "resolved"}}],
    }
    joined = join_buckets(rows, {uid: {"attempt": 1}}, variants)
    assert joined[0]["bucket_variant_status"] == "missing_header"
    assert joined[0]["matched"] is None


# ------------------------------------------------------------------
# exact-population gate
# ------------------------------------------------------------------


def _pop_cells(cids, tasks, episodes):
    return {c: {(t, i): True for t in tasks for i in range(episodes)} for c in cids}


@pytest.mark.parametrize("extra_in", ["both", "ws2"])
def test_formal_mode_refuses_cells_outside_the_frozen_matrix(extra_in):
    from exp.robocasa365.analyze_ws2_vs_ws1 import require_full_matrix

    cids = ["a", "b"]
    grid = [(t, i) for t in ("T0", "T1") for i in range(2)]
    ws1_cids = cids + (["probe"] if extra_in == "both" else [])
    arms = {"ws1": _pop_cells(ws1_cids, ("T0", "T1"), 2),
            "ws2": _pop_cells(cids + ["probe"], ("T0", "T1"), 2)}
    with pytest.raises(SystemExit, match="outside the frozen matrix"):
        require_full_matrix(arms, {"ws1": grid, "ws2": grid}, cids=cids,
                            episodes=2, tasks=2, label="full-matrix")


def test_matched_mode_projects_before_checking(tmp_path, monkeypatch, capsys):
    """ws1/ws2 legitimately hold the full matrix; only the 12 cells are the population."""
    import argparse

    from exp.robocasa365 import analyze_ws2_vs_ws1 as mod

    grid = [("T0", 0), ("T0", 1)]
    full = _pop_cells(["m1", "m2", "extra"], ("T0",), 2)
    control = _pop_cells(["m1", "m2"], ("T0",), 2)
    monkeypatch.setattr(mod, "load_journals",
                        lambda d, e, p: ({"ws1": full, "ws2": full, "ws2c": control}[p], grid))
    manifest = tmp_path / "m.json"
    manifest.write_text(json.dumps({"segments": {"ws2c": {"cells": ["m1", "m2"]}}}))
    index = tmp_path / "index.json"
    index.write_text(json.dumps({c: {} for c in ("m1", "m2", "extra")}))

    args = argparse.Namespace(
        ws1_dir=str(tmp_path), ws2_dir=str(tmp_path), ws2c_dir=str(tmp_path),
        manifest=str(manifest), episodes=2, tasks=1, index=str(index),
        allow_partial=False, resamples=50, seed=12345, csv="",
    )
    mod.cmd_compare(args)
    out = capsys.readouterr().out
    assert "matched-control decomposition (ONLY the 2 manifest cells" in out
    assert "lib_effect" in out


# ------------------------------------------------------------------
# agent teardown
# ------------------------------------------------------------------


def _agents_down_ns(**over):
    import argparse

    base = dict(worker_node="timan107", worker_home="/home/zixuans8", fleet="0",
                driver_host="ziyanglin.com", driver_port=23180,
                agent_server="ziyanglin.com:23160", tmux_prefix="ws2s", echo=False)
    base.update(over)
    return argparse.Namespace(**base)


def _sweep_predicate(script: str) -> str:
    """The shell condition the sweep applies to one cmdline."""
    return script[script.index("if printf"): script.index("; then")]


def test_agents_down_kills_the_supervisor_before_sweeping_workers(monkeypatch):
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    rec = _Recorder(["", "", "0"])
    monkeypatch.setattr(orch, "texec", rec)
    orch.cmd_agents_down(_agents_down_ns())

    assert len(rec.scripts) == 3
    # The supervisor dies first: sweeping workers first would just make it
    # respawn them.
    assert "kill-session" in rec.scripts[0] and "ws2sagent0" in rec.scripts[0]
    assert "robocasa365.worker_entry" in rec.scripts[1]
    assert "pkill" not in " ".join(rec.scripts), "no broad pkill on a shared host"
    assert "LEFTOVER" in rec.scripts[2], "leftovers must be reported, not assumed gone"


@pytest.mark.parametrize(
    ("cmdline", "expect"),
    [
        # This fleet.
        ("python -m exp.robocasa365.worker_entry --worker-id w0 "
         "--server-key ziyanglin.com:23160 --driver-host ziyanglin.com --driver-port 23180",
         "KILL"),
        # A hostname's dots are regex wildcards unless escaped: an unescaped
        # anchor would sweep this stranger's worker as if it were ours.
        ("python -m exp.robocasa365.worker_entry --worker-id w0 "
         "--server-key ziyanglin.com:23160 --driver-host ziyanglinXcom --driver-port 23180",
         "spare"),
        ("python -m exp.robocasa365.worker_entry --worker-id w0 "
         "--server-key ziyanglinXcom:23160 --driver-host ziyanglin.com --driver-port 23180",
         "spare"),
        # A SIBLING fleet of the same phase: same driver, different endpoint.
        ("python -m exp.robocasa365.worker_entry --worker-id w0 "
         "--server-key ziyanglin.com:23161 --driver-host ziyanglin.com --driver-port 23180",
         "spare"),
        # Another session's driver on this shared node.
        ("python -m exp.robocasa365.worker_entry --worker-id w0 "
         "--server-key ziyanglin.com:23160 --driver-host ziyanglin.com --driver-port 23999",
         "spare"),
        # Same ports, different driver host.
        ("python -m exp.robocasa365.worker_entry --worker-id w0 "
         "--server-key ziyanglin.com:23160 --driver-host other.host --driver-port 23180",
         "spare"),
    ],
)
def test_agents_down_sweeps_exactly_one_fleet(monkeypatch, cmdline, expect):
    """Sibling fleets share the driver; only the server key separates them.

    Runs the generated predicate through a real shell rather than asserting on
    its text, so a quoting or grep-option mistake shows up as a wrong verdict.
    """
    import shlex
    import subprocess

    from exp.robocasa365 import orchestrate_ws_search2 as orch

    rec = _Recorder(["", "", "0"])
    monkeypatch.setattr(orch, "texec", rec)
    orch.cmd_agents_down(_agents_down_ns())
    predicate = _sweep_predicate(rec.scripts[1])

    proc = subprocess.run(
        ["bash", "-c", f"c={shlex.quote(cmdline)}; {predicate}; then echo KILL; else echo spare; fi"],
        capture_output=True, text=True, check=True,
    )
    assert proc.stdout.strip() == expect


def test_agents_down_is_idempotent(monkeypatch):
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    ns = _agents_down_ns(fleet="1")
    first = _Recorder(["", "", "0"])
    monkeypatch.setattr(orch, "texec", first)
    orch.cmd_agents_down(ns)
    second = _Recorder(["", "", "0"])
    monkeypatch.setattr(orch, "texec", second)
    orch.cmd_agents_down(ns)
    assert first.scripts == second.scripts
    assert all("|| true" in s or "true" in s for s in first.scripts[:2])


# ------------------------------------------------------------------
# 13-task identity + cross-process resume (G2 non-blocking follow-up)
# ------------------------------------------------------------------

FULL_TASKS = [
    "CloseBlenderLid", "CloseFridge", "CoffeeSetupMug", "OpenCabinet", "OpenDrawer",
    "OpenStandMixerHead", "PickPlaceCounterToCabinet", "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter", "PickPlaceSinkToCounter", "PickPlaceToasterToCounter",
    "SlideDishwasherRack", "TurnOnSinkFaucet",
]


def _thirteen_task_fixture(tmp_path, cids, episodes=8):
    """Build full views + a complete central journal at real task width."""
    from openpi.conductor.driver import assign_servers
    from openpi.conductor.task import ServerEndpoint

    from exp.robocasa365.run_ws_search import WsSearchStrategy

    tasks = [(name, episodes) for name in FULL_TASKS]
    servers = [ServerEndpoint("h", 23160 + i) for i in range(2)]
    strategies, weights = {}, {}
    for cid in cids:
        s = WsSearchStrategy(cid=cid, run_prefix="ws2", teacher="groot_tp", layout=1, style=1,
                             base_seed=1_000_000, replan_steps=5, tasks=tasks)
        strategies[s.run_id] = s
        for yaml_id, (_, n) in zip(s.yaml_ids, tasks):
            weights[yaml_id] = n
    assignment = assign_servers(weights, servers, None, {s.key: 8 for s in servers})
    graphs = {rid: s.plan(sorted(s.yaml_ids), assignment) for rid, s in strategies.items()}
    return strategies, graphs, tasks


def test_thirteen_tasks_stay_distinct_through_the_real_analyzer(tmp_path):
    """The identity risk is task collapse: 13 tasks x 8 idx must survive intact."""
    from exp.robocasa365.analyze_ws_search_stats import load_journals
    from exp.robocasa365.run_ws_search2 import finalize

    cids = ["iso_vision_2", "grid_vision_2@87_robot_state@12"]
    strategies, graphs, _ = _thirteen_task_fixture(tmp_path, cids)
    expected, lines = {}, []
    for run_id, strategy in strategies.items():
        uids = []
        for yaml_id in strategy.yaml_ids:
            for ep in graphs[run_id].stages[yaml_id].episodes:
                uids.append(ep.task_uid)
                lines.append(json.dumps({
                    "task_uid": ep.task_uid, "yaml_id": yaml_id,
                    # A per-task success pattern: collapse would smear these.
                    "success": ep.task_id % 2 == 0 and ep.episode_idx < 4,
                    "accepted": True, "status": "done", "error": None,
                }))
        expected[run_id] = {"cid": strategy._cid, "uids": uids}  # noqa: SLF001
    central = tmp_path / "journal_central_ws2.jsonl"
    central.write_text("\n".join(lines) + "\n")

    complete = finalize(central, tmp_path, teacher="groot_tp", expected_by_run=expected)
    assert all(complete.values())

    cells, grid = load_journals(tmp_path, 8, "ws2")
    assert set(cells) == set(cids)
    assert sorted({t for t, _ in grid}) == sorted(FULL_TASKS), "all 13 tasks recovered"
    assert len(grid) == 13 * 8
    for outcomes in cells.values():
        assert set(outcomes) == set(grid), "no (task, idx) pair overwrote another"
        # The per-task pattern survived, so tasks did not collapse into one key.
        even = [v for (t, i), v in outcomes.items() if FULL_TASKS.index(t) % 2 == 0 and i < 4]
        odd = [v for (t, _), v in outcomes.items() if FULL_TASKS.index(t) % 2 == 1]
        assert all(even) and not any(odd)


def test_partial_then_resume_in_a_fresh_process(tmp_path):
    """Cross-process: same plan hash, only missing uids dispatched, full products."""
    script = f"""
import json, pathlib, sys
sys.path.insert(0, {str(REPO)!r})
from openpi.conductor.driver import assign_servers
from openpi.conductor.task import ServerEndpoint
from exp.robocasa365.run_collect import build_run_plan, write_run_plan
from exp.robocasa365.run_ws_search import EVAL_NO_COLLECT_ROOT, WsSearchStrategy
from exp.robocasa365.run_ws_search2 import build_cell_specs, finalize

TASKS = [(n, 8) for n in {FULL_TASKS!r}]
tmp = pathlib.Path({str(tmp_path)!r})
cids = ["iso_vision_2", "iso_robot_state"]
servers = [ServerEndpoint("h", 23160 + i) for i in range(2)]
strategies, weights = {{}}, {{}}
for cid in cids:
    (tmp / (cid + ".yaml")).write_text("# " + cid)
    s = WsSearchStrategy(cid=cid, run_prefix="ws2", teacher="groot_tp", layout=1, style=1,
                         base_seed=1_000_000, replan_steps=5, tasks=TASKS)
    strategies[s.run_id] = s
    for yid, (_, n) in zip(s.yaml_ids, TASKS):
        weights[yid] = n
assignment = assign_servers(weights, servers, None, {{s.key: 8 for s in servers}})
graphs = {{rid: s.plan(sorted(s.yaml_ids), assignment) for rid, s in strategies.items()}}
ranks = {{s._cid: i for i, s in enumerate(strategies.values())}}

expected = {{}}
hashes = {{}}
for rid, s in strategies.items():
    payload = build_run_plan(s, graphs[rid], EVAL_NO_COLLECT_ROOT)
    write_run_plan(tmp / ("run_plan_" + rid + ".json"), payload)
    hashes[rid] = payload["plan_hash"]
    expected[rid] = {{"cid": s._cid, "uids": list(payload["uids"])}}

central = tmp / "journal_central_ws2.jsonl"
mode = sys.argv[1]
if mode == "partial":
    # First process: one whole cell plus one task of the other.
    done_rid = list(strategies)[0]
    lines = []
    for yid in strategies[done_rid].yaml_ids:
        for ep in graphs[done_rid].stages[yid].episodes:
            lines.append(json.dumps({{"task_uid": ep.task_uid, "yaml_id": yid, "success": True,
                                     "accepted": True, "status": "done", "error": None}}))
    other = list(strategies)[1]
    first_yaml = strategies[other].yaml_ids[0]
    for ep in graphs[other].stages[first_yaml].episodes:
        lines.append(json.dumps({{"task_uid": ep.task_uid, "yaml_id": first_yaml, "success": False,
                                 "accepted": True, "status": "done", "error": None}}))
    central.write_text("\\n".join(lines) + "\\n")
    print("PARTIAL_WRITTEN", len(lines))
else:
    from openpi.conductor.journal import Journal
    done = Journal(str(central)).replay_done_uids()
    specs = build_cell_specs(strategies, graphs, done, ranks, tmp)
    # Plan hashes must recompute identically (write_run_plan would abort otherwise).
    for rid, s in strategies.items():
        again = build_run_plan(s, graphs[rid], EVAL_NO_COLLECT_ROOT)
        write_run_plan(tmp / ("run_plan_" + rid + ".json"), again)
        assert again["plan_hash"] == hashes[rid], "plan hash drifted across processes"
    done_rid = list(strategies)[0]
    assert done_rid not in specs, "the completed cell must produce no stage"
    active = [ep.task_uid for spec in specs.values()
              for eps in spec["episodes_by_yaml"].values() for ep in eps]
    assert not (set(active) & done), "already-terminal uids must not be dispatched"
    assert len(active) == 12 * 8, "only the missing 12 tasks remain"
    # Finalize still materialises the FULL expected set for both cells.
    complete = finalize(central, tmp, teacher="groot_tp", expected_by_run=expected)
    assert complete[done_rid] is True
    assert complete[list(strategies)[1]] is False
    summary = json.loads((tmp / ("summary_" + list(strategies)[1] + ".json")).read_text())
    assert summary["n_missing"] == 12 * 8
    print("RESUME_OK")
"""
    body = tmp_path / "resume_probe.py"
    body.write_text(script)
    first = subprocess.run([sys.executable, str(body), "partial"], capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    assert "PARTIAL_WRITTEN" in first.stdout
    second = subprocess.run([sys.executable, str(body), "resume"], capture_output=True, text=True)
    assert second.returncode == 0, second.stderr
    assert "RESUME_OK" in second.stdout


def test_ere_literal_escapes_every_metacharacter():
    from exp.robocasa365.orchestrate_ws_search2 import ere_literal

    assert ere_literal("ziyanglin.com:23160") == r"ziyanglin\.com:23160"
    for ch in ".[]*+?{}()|^$":
        assert "\\" + ch in ere_literal(f"a{ch}b")
    # Ordinary characters are untouched.
    assert ere_literal("timan107") == "timan107"


def test_agents_down_fails_when_a_worker_survives(monkeypatch):
    """A surviving worker still holds a CUDA context — never report success."""
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    monkeypatch.setattr(orch, "texec", _Recorder(["", "", "0\nLEFTOVER"]))
    with pytest.raises(SystemExit, match="did not fully clear"):
        orch.cmd_agents_down(_agents_down_ns())


def test_agents_down_fails_when_the_supervisor_survives(monkeypatch):
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    monkeypatch.setattr(orch, "texec", _Recorder(["", "", "1"]))
    with pytest.raises(SystemExit, match="1 tmux session"):
        orch.cmd_agents_down(_agents_down_ns())


def test_agents_down_reports_a_clean_fleet(monkeypatch, capsys):
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    monkeypatch.setattr(orch, "texec", _Recorder(["", "", "0"]))
    orch.cmd_agents_down(_agents_down_ns())
    assert "is clear" in capsys.readouterr().out


def test_worker_node_commands_use_the_worker_account_home(monkeypatch):
    """The worker island runs as a different account than the serving host.

    The tether agent's own HOME is not a user home, so every remote script
    sets it — and using the serving host's value on the worker node points
    tmux, the venv caches and the island paths at an account that does not
    own them.
    """
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    rec = _Recorder(["", "", "0"])
    monkeypatch.setattr(orch, "texec", rec)
    orch.cmd_agents_down(_agents_down_ns())
    assert rec.calls, "teardown must reach the worker node"
    assert all(node == "timan107" and home == "/home/zixuans8" for node, home in rec.calls)


# ------------------------------------------------------------------
# per-teacher serving recipes (the pi0.5 phase reuses this orchestrator)
# ------------------------------------------------------------------


def _serve_ns(**over):
    import argparse

    base = dict(node="weilandserver", repo="/data/openpi_text_ivf_build", teacher="groot_tp",
                tmux_prefix="ws2s", echo=True, ports="23160", cuda_device=0,
                python=None, pythonpath=None, checkpoint=None,
                bootstrap_yaml="exp/robocasa365/config/ws_search2/groot_tp/main/iso_vision_2.yaml",
                min_free_ram_gb=40, min_free_vram_mb=None, min_gpu_temp_c=0,
                ready_timeout_s=900.0)
    base.update(over)
    ns = argparse.Namespace(**base)
    return ns


def _resolved(**over):
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    ns = _serve_ns(**over)
    orch.resolve_serve_defaults(ns)
    return ns


def test_teacher_row_binds_interpreter_checkpoint_and_vram_floor():
    """Every teacher-specific string comes from one table, not from a call site."""
    groot = _resolved(teacher="groot_tp")
    pi05 = _resolved(teacher="pi05")

    assert "gr00t_n15_venv" in groot.python and groot.checkpoint.endswith("checkpoint-60000")
    assert groot.min_free_vram_mb == 8500
    assert pi05.checkpoint == "/home/weiland/ckpt_pi05_robocasa_pytorch"
    # pi0.5 needs a bigger slice of the card than GR00T (round-1 footprints).
    assert pi05.min_free_vram_mb == 10500


def test_every_teacher_grafts_the_serving_clone_in_front_of_its_interpreter():
    """The serving clone has no venv; both teachers borrow one and override src.

    Without the graft the server imports the interpreter's own checkout — a
    different openpi than the one this round's configs were validated against,
    which shows up as a startup rejection at best and a silently different
    retrieval path at worst.
    """
    for teacher in ("groot_tp", "pi05"):
        ns = _resolved(teacher=teacher, repo="/other/clone")
        assert ns.pythonpath.endswith("/other/clone/src:/other/clone"), teacher
        assert not ns.python.startswith("/other/clone"), (
            f"{teacher}: the clone owns no interpreter; the venv path is external"
        )


def test_explicit_flags_still_beat_the_table():
    ns = _resolved(teacher="pi05", python="/custom/python", min_free_vram_mb=1)
    assert ns.python == "/custom/python" and ns.min_free_vram_mb == 1


def test_unknown_teacher_is_refused_with_the_known_set():
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    with pytest.raises(SystemExit, match="no serving recipe"):
        orch.teacher_spec("pi0", "/repo")


def test_pi05_puts_every_flag_before_the_policy_subcommand():
    """tyro parses flags after ``policy:checkpoint`` as the SUBCOMMAND's.

    A ``--cache_config`` placed after it is not a parse error — it binds to the
    wrong parser and the server comes up with no cache at all, which looks like
    a healthy server serving uncached actions.
    """
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    cmd = orch.serve_command(_resolved(teacher="pi05"), 23160, "/tmp/x.log")
    head, _, tail = cmd.partition("policy:checkpoint")
    assert "--port 23160" in head and "--cache_config " in head
    assert "--cache_config" not in tail and "--port" not in tail
    assert "--policy.config pi05_robocasa" in tail and "--policy.dir " in tail


def test_pi05_uses_the_underscore_flag_and_no_dynamic_bundle_flag():
    """``serve_policy`` spells it ``--cache_config`` and has no bundle flag.

    ``WebsocketPolicyServer`` defaults ``allow_dynamic_bundles=True`` and
    ``serve_policy`` never overrides it, so passing an invented flag would only
    make tyro reject the launch.
    """
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    cmd = orch.serve_command(_resolved(teacher="pi05"), 23160, "/tmp/x.log")
    assert "scripts/serve_policy.py" in cmd
    assert "--cache-config" not in cmd
    assert "--allow-dynamic-bundles" not in cmd
    assert "PYTHONPATH=/data/openpi_text_ivf_build/src:" in cmd


def test_groot_keeps_its_verified_recipe():
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    cmd = orch.serve_command(_resolved(teacher="groot_tp"), 23161, "/tmp/x.log")
    assert "exp/robocasa365/serve_groot_n15.py" in cmd
    for flag in ("--concurrent", "--allow-dynamic-bundles", "--cache-config", "--checkpoint"):
        assert flag in cmd
    assert "PYTHONPATH=/home/weiland/gr00t_n15:" in cmd
    assert "policy:checkpoint" not in cmd


@pytest.mark.parametrize("teacher", ["groot_tp", "pi05"])
def test_serve_command_pins_the_card_and_tees_its_log(teacher):
    from exp.robocasa365 import orchestrate_ws_search2 as orch

    cmd = orch.serve_command(_resolved(teacher=teacher, cuda_device=1), 23162, "/tmp/s.log")
    assert "CUDA_VISIBLE_DEVICES=1" in cmd
    # tee, not redirect: the readiness probe greps this log AND an operator
    # attaching to the pane must still see the stream.
    assert cmd.rstrip().endswith("| tee /tmp/s.log")


@pytest.mark.parametrize(
    ("teacher", "cmdline", "expect"),
    [
        ("pi05", "python scripts/serve_policy.py --port 23160 --cache_config x.yaml", True),
        # The other teacher's server on one of the same ports is NOT ours.
        ("pi05", "python exp/robocasa365/serve_groot_n15.py --port 23160", False),
        ("groot_tp", "python exp/robocasa365/serve_groot_n15.py --port 23160", True),
        ("groot_tp", "python scripts/serve_policy.py --port 23160", False),
    ],
)
def test_servers_down_anchors_on_the_teachers_entry_point(
    teacher, cmdline, expect, tmp_path, monkeypatch
):
    """Ports are recycled between phases; the entry point is what disambiguates.

    Runs the generated sweep as real shell against a fake /proc so the ``case``
    pattern is exercised, not merely inspected.
    """
    import subprocess

    from exp.robocasa365 import orchestrate_ws_search2 as orch

    rec = _Recorder(["", "", ""])
    monkeypatch.setattr(orch, "texec", rec)
    orch.cmd_servers_down(_serve_ns(teacher=teacher, ports="23160,23161", echo=False))
    sweep = rec.scripts[-1]

    proc_dir = tmp_path / "proc" / "4242"
    proc_dir.mkdir(parents=True)
    (proc_dir / "cmdline").write_bytes(cmdline.replace(" ", "\0").encode() + b"\0")
    # Neutralise the kill: only the selection is under test.
    script = sweep.replace("/proc", str(tmp_path / "proc")).replace("kill $pid", "true")
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=60)
    assert ("swept 4242" in out.stdout) is expect, out


def test_driver_and_agents_pass_the_chosen_teacher_through(monkeypatch):
    """A pi0.5 phase driven with ``--teacher groot_tp`` would key every uid wrong."""
    import argparse

    from exp.robocasa365 import orchestrate_ws_search2 as orch

    rec = _Recorder(["", ""])
    monkeypatch.setattr(orch, "texec", rec)
    common = dict(node="weilandserver", repo="/repo", tmux_prefix="ws2s", echo=False,
                  teacher="pi05", servers="h:1", run_prefix="ws2p",
                  config_dir="cfg", env_config="env", manifest="")
    orch.cmd_driver_up(argparse.Namespace(
        **common, episodes=104, driver_port=23180, driver_python="py", extra_args=""))
    orch.cmd_agents_up(argparse.Namespace(
        **common, worker_node="timan107", worker_home="/home/zixuans8", worker_repo="/wrepo",
        agent_python="py", agent_server="h:1", fleet="0", workers=9, gpu_ids="0",
        driver_host="h", driver_port=23180))

    assert all("--teacher pi05" in s for s in rec.scripts)
    assert not any("--teacher groot_tp" in s for s in rec.scripts)
