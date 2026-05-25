"""Shared fakes for conductor integration tests."""

from __future__ import annotations

import threading

from openpi.conductor import task as T  # noqa: N812
from openpi.conductor.strategy import ExperimentStrategy
from openpi.conductor.worker import EpisodeRunner


class FakeCtl:
    """Records control-frame calls a strategy issues during stage setup/teardown."""

    def __init__(self, server):
        self.server = server
        self.calls: list[tuple] = []
        self._lock = threading.Lock()

    def _rec(self, *c):
        with self._lock:
            self.calls.append(c)

    def load_cache_config(self, **kw):
        self._rec("load", kw.get("yaml_id"))
        return {"__ack__": "load_cache_config", "version": 1}

    def fetch_dump(self, warmup_yaml_id):
        self._rec("fetch", warmup_yaml_id)
        return b'{"factor_raw": {"k": 1.0}}\n'

    def preload_normalizer_buffer(self, eval_yaml_id, buffer):
        self._rec("preload", eval_yaml_id)
        return {"__ack__": "preload_normalizer_buffer", "n_keys": len(buffer)}

    def unload_warmup_buffer(self, eval_yaml_id):
        self._rec("unload", eval_yaml_id)
        return {"__ack__": "unload_warmup_buffer"}


def make_episode(yaml_id, phase, i, server):
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


class WarmupEvalFakeStrategy(ExperimentStrategy):
    """One warmup stage -> one eval stage per yaml (mirrors the real
    WarmupEvalStrategy structure, with fake ctl ops)."""

    def __init__(self, n_warmup=2, n_eval=4):
        self._n_warmup = n_warmup
        self._n_eval = n_eval

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
                    episodes=[make_episode(y, "warmup", i, srv) for i in range(self._n_warmup)],
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
                    episodes=[make_episode(y, "eval", i, srv) for i in range(self._n_eval)],
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


class FakeEpisodeRunner(EpisodeRunner):
    """Mock runner: reports a couple progress ticks then returns success.

    ``transient_fail`` maps task_uid -> remaining transient failures (shared
    across workers via the dict so a requeued episode eventually succeeds).
    """

    def __init__(self, transient_fail: dict[str, int] | None = None, lock: threading.Lock | None = None):
        self._fail = transient_fail if transient_fail is not None else {}
        self._lock = lock or threading.Lock()

    def run(self, task, report):
        report(0, 10.0, "MISS")
        report(1, 10.0, "WARM_START")
        with self._lock:
            remaining = self._fail.get(task.task_uid, 0)
            if remaining > 0:
                self._fail[task.task_uid] = remaining - 1
                return T.EpisodeResult(task.task_uid, success=False, n_steps=1, error="TimeoutError")
        return T.EpisodeResult(
            task.task_uid,
            success=True,
            n_steps=2,
            per_step_rows=[
                {
                    "yaml_id": task.yaml_id,
                    "task_id": task.task_id,
                    "episode_idx": task.episode_idx,
                    "step": 0,
                    "hit_type": "MISS",
                }
            ],
        )
