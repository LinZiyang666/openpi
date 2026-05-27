"""Driver conductor tests (M3): placement, retry classification, stage
lifecycle, and the warmup->eval handoff via fake ctl + fake strategy."""

from __future__ import annotations

from openpi.conductor import protocol as P  # noqa: N812
from openpi.conductor import task as T  # noqa: N812
from openpi.conductor.driver import ConductorDriver
from openpi.conductor.driver import assign_servers
from openpi.conductor.driver import is_retriable_error
from openpi.conductor.journal import Journal
from openpi.conductor.scheduler import StageState
from openpi.conductor.strategy import ExperimentStrategy

S1 = T.ServerEndpoint("h1", 8001)
S2 = T.ServerEndpoint("h2", 8002)


# ------------------------------------------------------------------
# placement
# ------------------------------------------------------------------


def test_assign_servers_balances_by_weight():
    a = assign_servers({"y1": 10, "y2": 1, "y3": 1}, [S1, S2])
    # heaviest (y1) lands first; the two light ones balance onto the other server
    assert a["y1"] != a["y2"] or a["y1"] != a["y3"]
    assert set(a.values()) <= {S1, S2}


def test_assign_servers_colocation_same_server():
    a = assign_servers({"y1": 1, "y2": 1}, [S1, S2], colocation={"y1": "g", "y2": "g"})
    assert a["y1"] == a["y2"]  # shared calib group co-located


# ------------------------------------------------------------------
# retry classification (plan §9.1)
# ------------------------------------------------------------------


def test_is_retriable_error():
    assert is_retriable_error("ConnectionResetError: boom") is True
    assert is_retriable_error("TimeoutError") is True
    assert is_retriable_error("ConfigValidationError: write_policy must be never") is False
    assert is_retriable_error(None) is False


# ------------------------------------------------------------------
# fakes
# ------------------------------------------------------------------


class FakeCtl:
    def __init__(self, server):
        self.server = server
        self.calls: list[tuple] = []

    def load_cache_config(self, **kw):
        self.calls.append(("load", kw.get("yaml_id")))
        return {"__ack__": "load_cache_config", "version": 1}

    def fetch_dump(self, warmup_yaml_id):
        self.calls.append(("fetch", warmup_yaml_id))
        return b'{"factor_raw": {"k": 1.0}}\n'

    def preload_normalizer_buffer(self, eval_yaml_id, buffer):
        self.calls.append(("preload", eval_yaml_id))
        return {"__ack__": "preload_normalizer_buffer", "n_keys": len(buffer)}

    def unload_warmup_buffer(self, eval_yaml_id):
        self.calls.append(("unload", eval_yaml_id))
        return {"__ack__": "unload_warmup_buffer"}


class FakeStrategy(ExperimentStrategy):
    """One warmup stage -> one eval stage per yaml."""

    def plan(self, yamls, server_assignment):
        g = T.TaskGraph()
        for y in yamls:
            srv = server_assignment[y]
            wid, eid = f"{y}:warmup", f"{y}:eval"
            g.add_calibration(
                T.CalibrationArtifact(calib_id=y, source="warmup_stage", warmup_stage_id=wid, cleanup_id=y)
            )
            g.add_stage(
                T.Stage(
                    wid,
                    y,
                    "warmup",
                    srv,
                    episodes=[_ep(y, "warmup", 0)],
                    produces_calib_id=y,
                    setup={"warmup_yaml": f"{y}__warmup.yaml"},
                )
            )
            g.add_stage(
                T.Stage(
                    eid,
                    y,
                    "eval",
                    srv,
                    episodes=[_ep(y, "eval", 0), _ep(y, "eval", 1)],
                    consumes_calib_id=y,
                    setup={"eval_yaml": f"{y}.yaml"},
                )
            )
            g.add_dependency(wid, eid)
        return g

    def on_stage_begin(self, stage, ctl, ctx):
        if stage.phase == "warmup":
            ctl.load_cache_config(yaml_path=stage.setup["warmup_yaml"], yaml_id=f"{stage.yaml_id}__warmup")
        else:
            ctl.preload_normalizer_buffer(stage.yaml_id, ctx.get(stage.consumes_calib_id) or {})
            ctl.load_cache_config(yaml_path=stage.setup["eval_yaml"], yaml_id=stage.yaml_id)

    def on_stage_complete(self, stage, ctl, ctx):
        if stage.phase == "warmup":
            ctl.fetch_dump(f"{stage.yaml_id}__warmup")
            ctx.publish(stage.produces_calib_id, {"k": [1.0]})
        else:
            ctl.unload_warmup_buffer(stage.yaml_id)


def _ep(yaml_id, phase, i):
    return T.EpisodeTask(
        task_uid=T.make_task_uid(yaml_id, phase, 0, i),
        yaml_id=yaml_id,
        phase=phase,
        experiment="exp",
        task_id=0,
        episode_idx=i,
        orig_init_state_idx=i,
        server_host=S1.host,
        server_port=S1.port,
        bundle_id=yaml_id,
    )


def _make_driver(tmp_path):
    ctls: dict[str, FakeCtl] = {}

    def factory(server):
        ctls[server.key] = FakeCtl(server)
        return ctls[server.key]

    d = ConductorDriver(
        FakeStrategy(),
        yaml_weights={"y": 3},
        servers=[S1],
        journal_path=str(tmp_path / "j.jsonl"),
        ctl_factory=factory,
    )
    return d, ctls


def _pump(d, server_key, max_iters=100):
    for _ in range(max_iters):
        d.drive_stages_once()
        while True:
            mt, pl = d.handle_pull(server_key)
            # Mirror the real _handle_conn consumer: stop on an idle (none)
            # assignment OR an all-done MSG_SHUTDOWN.
            if mt == P.MSG_SHUTDOWN or pl.get("none"):
                break
            task = P.task_from_wire(pl["task"])
            d.handle_result(P.result_to_wire(T.EpisodeResult(task.task_uid, success=True, n_steps=1)))
        if d.scheduler.all_done():
            return True
    return False


# ------------------------------------------------------------------
# pull / result
# ------------------------------------------------------------------


def test_handle_pull_none_when_nothing_ready(tmp_path):
    d, _ = _make_driver(tmp_path)
    # before any stage setup, nothing is READY
    _, pl = d.handle_pull(S1.key)
    assert pl.get("none") is True


def test_handle_result_journals_terminal(tmp_path):
    d, _ = _make_driver(tmp_path)
    d.drive_stages_once()  # warmup setup -> READY
    _, pl = d.handle_pull(S1.key)
    task = P.task_from_wire(pl["task"])
    d.handle_result(P.result_to_wire(T.EpisodeResult(task.task_uid, success=True, n_steps=1)))
    assert task.task_uid in d.journal.replay_done_uids()


# ------------------------------------------------------------------
# full warmup -> eval lifecycle + ctl call ordering
# ------------------------------------------------------------------


def test_stage_lifecycle_warmup_then_eval(tmp_path):
    d, ctls = _make_driver(tmp_path)
    assert _pump(d, S1.key), "driver did not reach all_done"
    calls = ctls[S1.key].calls
    # ordering: warmup dump cleared (unload cleanup) is via ctl too — but our
    # FakeCtl records unload only from on_stage_complete(eval); the pre-clear
    # in driver uses unload too. Assert the semantic sequence is present.
    kinds = [c[0] for c in calls]
    assert kinds.index("load") < kinds.index("fetch") < kinds.index("preload")
    assert "unload" in kinds
    # warmup loaded before eval loaded
    load_ids = [c[1] for c in calls if c[0] == "load"]
    assert load_ids == ["y__warmup", "y"]


def test_eval_preload_before_load(tmp_path):
    d, ctls = _make_driver(tmp_path)
    _pump(d, S1.key)
    calls = ctls[S1.key].calls
    # eval setup must preload then load (config fail-fasts without the pool)
    preload_i = next(i for i, c in enumerate(calls) if c[0] == "preload")
    eval_load_i = next(i for i, c in enumerate(calls) if c == ("load", "y"))
    assert preload_i < eval_load_i


# ------------------------------------------------------------------
# calibration fan-out across servers (G1R1 Item 3 / plan §7)
# ------------------------------------------------------------------


class _SharedCalibStrategy(ExperimentStrategy):
    """One warmup on S1 produces calib 'c'; two eval stages (S1, S2) consume it."""

    def plan(self, yamls, server_assignment):
        g = T.TaskGraph()
        g.add_calibration(
            T.CalibrationArtifact(calib_id="c", source="warmup_stage", warmup_stage_id="w", cleanup_id="c")
        )
        g.add_stage(T.Stage("w", "yw", "warmup", S1, episodes=[_ep_on("yw", "warmup", 0, S1)], produces_calib_id="c"))
        g.add_stage(T.Stage("eA", "yA", "eval", S1, episodes=[_ep_on("yA", "eval", 0, S1)], consumes_calib_id="c"))
        g.add_stage(T.Stage("eB", "yB", "eval", S2, episodes=[_ep_on("yB", "eval", 0, S2)], consumes_calib_id="c"))
        g.add_dependency("w", "eA")
        g.add_dependency("w", "eB")
        return g

    def on_stage_begin(self, stage, ctl, ctx):
        if stage.phase == "eval":
            ctl.preload_normalizer_buffer(stage.yaml_id, ctx.get("c") or {})

    def on_stage_complete(self, stage, ctl, ctx):
        if stage.phase == "warmup":
            ctx.publish("c", {"k": [1.0]})


def _ep_on(yaml_id, phase, i, server):
    return T.EpisodeTask(
        task_uid=T.make_task_uid(yaml_id, phase, 0, i),
        yaml_id=yaml_id,
        phase=phase,
        experiment="exp",
        task_id=0,
        episode_idx=i,
        orig_init_state_idx=i,
        server_host=server.host,
        server_port=server.port,
        bundle_id=yaml_id,
    )


def test_calibration_fanout_across_servers(tmp_path):
    ctls: dict[str, FakeCtl] = {}

    def factory(server):
        ctls[server.key] = FakeCtl(server)
        return ctls[server.key]

    d = ConductorDriver(
        _SharedCalibStrategy(),
        yaml_weights={"yw": 1, "yA": 1, "yB": 1},
        servers=[S1, S2],
        journal_path=str(tmp_path / "j.jsonl"),
        ctl_factory=factory,
    )

    for _ in range(100):
        d.drive_stages_once()
        for key in (S1.key, S2.key):
            while True:
                mt, pl = d.handle_pull(key)
                if mt == P.MSG_SHUTDOWN or pl.get("none"):
                    break
                task = P.task_from_wire(pl["task"])
                d.handle_result(P.result_to_wire(T.EpisodeResult(task.task_uid, success=True, n_steps=1)))
        if d.scheduler.all_done():
            break
    assert d.scheduler.all_done()
    # The shared calib buffer is preloaded on BOTH servers (fan-out).
    assert ("preload", "yA") in ctls[S1.key].calls
    assert ("preload", "yB") in ctls[S2.key].calls


# ------------------------------------------------------------------
# Round-3 self-review regressions
# ------------------------------------------------------------------


def test_setup_hook_exception_rolls_back_and_retries(tmp_path):
    boom = {"n": 0}

    class BoomStrategy(FakeStrategy):
        def on_stage_begin(self, stage, ctl, ctx):
            if stage.phase == "warmup" and boom["n"] == 0:
                boom["n"] += 1
                raise RuntimeError("transient ctl error")
            super().on_stage_begin(stage, ctl, ctx)

    ctls: dict[str, FakeCtl] = {}

    def factory(server):
        ctls[server.key] = FakeCtl(server)
        return ctls[server.key]

    d = ConductorDriver(
        BoomStrategy(), yaml_weights={"y": 3}, servers=[S1], journal_path=str(tmp_path / "j.jsonl"), ctl_factory=factory
    )
    assert _pump(d, S1.key)  # rolled back to SETUP_PENDING, retried, reached all_done
    assert boom["n"] == 1  # first setup raised exactly once


def test_resume_triggers_on_resume_for_eval(tmp_path):
    jpath = tmp_path / "j.jsonl"
    Journal(jpath).record(
        task_uid=T.make_task_uid("y", "eval", 0, 0), yaml_id="y", phase="eval", status="done", success=True
    )
    resume_calls: list = []

    class ResumeStrategy(FakeStrategy):
        def on_resume(self, stage, ctl, ctx):
            resume_calls.append(stage.stage_id)

    ctls: dict[str, FakeCtl] = {}

    def factory(server):
        ctls[server.key] = FakeCtl(server)
        return ctls[server.key]

    d = ConductorDriver(
        ResumeStrategy(), yaml_weights={"y": 3}, servers=[S1], journal_path=str(jpath), ctl_factory=factory
    )
    _pump(d, S1.key)
    assert "y:eval" in resume_calls  # on_resume fired on the resume path


def test_uid_meta_handles_colon_in_yaml_id():
    assert ConductorDriver._uid_meta("a:b:eval:0:0") == ("a:b", "eval")  # noqa: SLF001


def test_fatal_setup_hook_error_fails_fast_not_infinite_retry(tmp_path):
    # A config error raised inside on_stage_begin must NOT retry forever (G2 R1
    # Blocking 1): the stage goes FAILED + cascades, and all_done() holds.
    class FatalStrategy(FakeStrategy):
        def on_stage_begin(self, stage, ctl, ctx):
            if stage.phase == "warmup":
                raise RuntimeError("ConfigValidationError: bad yaml")
            super().on_stage_begin(stage, ctl, ctx)

    ctls: dict[str, FakeCtl] = {}

    def factory(server):
        ctls[server.key] = FakeCtl(server)
        return ctls[server.key]

    d = ConductorDriver(
        FatalStrategy(),
        yaml_weights={"y": 3},
        servers=[S1],
        journal_path=str(tmp_path / "j.jsonl"),
        ctl_factory=factory,
    )
    for _ in range(5):  # repeated passes must not loop forever
        d.drive_stages_once()
    assert d.scheduler.stage_state("y:warmup") == StageState.FAILED
    assert d.scheduler.stage_state("y:eval") == StageState.FAILED  # cascaded
    assert d.scheduler.all_done()  # no hang
