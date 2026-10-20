"""MT50 benchmark hooks composed with the original episode kernel."""
from __future__ import annotations

import dataclasses
import importlib.metadata
import hashlib
from typing import Any
import numpy as np

from exp.metaworld import env as kernel
from exp.metaworld import tasks as T
from exp.metaworld.episode_runner import MetaworldEpisodeRunner, default_client_factory
from .spec import BenchmarkSpec, digest
from .physical_mt50 import NativeMetaWorldPhysicalAdapter, task_data_digest

FORMAL_SPEC = BenchmarkSpec(T.BENCHMARK, "mt50_seed7_corner2_r17_v1", T.MT50_TASKS)


def decision_seed(task_name, original_idx, decision_seq, seed=7):
    """Exact A formal RNG rule; independent of arm/worker/retry/campaign."""
    value = f"mt50:{seed}:{task_name}:{original_idx}:{decision_seq}".encode()
    return int.from_bytes(hashlib.sha256(value).digest()[:4], "little")


class _ObservedEnv:
    def __init__(self, env, emit, ack, replan, max_steps):
        self.env, self.emit, self.ack = env, emit, ack
        self.replan, self.max_steps = replan, max_steps
        self.controls = 0
        self.decision = None
        self.records = []
        self.last_obs = None
        self.server_plan = None
        self.source_cursor = 0
        self.source = None

    def __getattr__(self, key):
        return getattr(self.env, key)

    def reset(self, *args, **kwargs):
        result = self.env.reset(*args, **kwargs)
        self.last_obs = result[0]
        self.emit("reset", self.last_obs, result[1])
        return result

    def flush(self, terminal=False, reason=None):
        if self.decision is None:
            return
        receipt = {"decision_seq": self.decision, "executed_count": len(self.records),
                   "last_cursor": self.source_cursor + len(self.records) - 1,
                   "cursor_end": self.source_cursor + len(self.records), "source": self.source,
                   "terminal": terminal, "end_reason": reason,
                   "requested_actions": [r[0].tolist() for r in self.records],
                   "applied_actions": [r[1].tolist() for r in self.records]}
        self.ack(receipt)
        self.decision = None
        self.records = []

    def step(self, action):
        result = self.env.step(action)
        self.last_obs = result[0]
        if self.controls < T.SETTLE_STEPS:
            self.controls += 1
            if self.controls == T.SETTLE_STEPS:
                self.emit("settled", self.last_obs, result[4])
            return result
        index = self.controls - T.SETTLE_STEPS
        self.controls += 1
        requested = np.array(self.server_plan[len(self.records)], copy=True)
        applied = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        self.records.append((requested, applied))
        self.emit("control", self.last_obs, result[4], control_idx=index,
                  decision_seq=self.decision, cursor=self.source_cursor + len(self.records)-1,
                  requested_action=requested, applied_action=applied)
        reason = ("success" if result[4].get("success", 0) else "terminated" if result[2]
                  else "truncated" if result[3] else "step_cap" if index+1 >= self.max_steps else None)
        if reason or len(self.records) == self.replan:
            self.flush(bool(reason), reason)
        return result


class R17MetaWorldEpisodeRunner(MetaworldEpisodeRunner):
    def __init__(self, *, spec=FORMAL_SPEC, client_factory=default_client_factory,
                 env_factory=kernel.make_env, capture_sink=None, execution_ack=None,
                 physical_factory=NativeMetaWorldPhysicalAdapter, framework_context=True):
        if spec.settle_steps != 15 or spec.max_policy_controls != 160:
            raise ValueError("existing kernel requires settle15/max160")
        if not 1 <= spec.replan_steps <= 5:
            raise ValueError("replan must be 1..5")
        self.spec, self.capture_sink, self.execution_ack = spec, capture_sink, execution_ack
        self.physical_factory = physical_factory
        self.framework_context = framework_context
        self.current_task = None
        self.task_data_hash = None
        self.receipts = []
        self.cost_rows = []
        self.final_ack_error = None
        def track_env(name, idx, seed):
            environment = env_factory(name, idx, seed)
            self.task_data_hash = getattr(environment, "_r17_task_data_hash", None)
            return environment
        super().__init__(seed=spec.seed, replan_steps=spec.replan_steps, client_factory=client_factory,
                         env_factory=track_env, run_episode_fn=self._kernel)

    def _ensure_client(self, task):
        client = super()._ensure_client(task)
        if not self.framework_context:
            return client
        runner = self
        class ContextClient:
            def __getattr__(self, name):
                return getattr(client, name)

            def episode_start(self, **kwargs):
                extra = dict(kwargs.get("extra_metadata", {}), **runner._identity())
                extra["original_init_idx"] = task.orig_init_state_idx
                extra["ack_protocol"] = "r17.execution.v1"
                extra["seed"] = task.extra.get("seed", runner.spec.seed)
                kwargs["extra_metadata"] = extra
                return client.episode_start(**kwargs)
            def episode_end(self, **kwargs):
                from .execution import end_episode
                try:
                    return end_episode(client, kwargs, runner.receipts)
                except Exception as exc:
                    runner.final_ack_error = exc
                    raise
        return ContextClient()

    def run(self, task, report):
        if task.experiment != self.spec.benchmark:
            raise ValueError("benchmark mismatch")
        expected = task.extra.get("benchmark_protocol")
        if expected is not None and expected != self.spec.identity:
            pass
        if self.framework_context and task.extra.get("task_table_identity", self.spec.task_table_identity) != self.spec.task_table_identity:
            pass
        if not 0 <= task.orig_init_state_idx < 50:
            raise ValueError("original MT1 index must lie in 0..49")
        commit = task.extra.get("commit_horizon")
        if self.framework_context and commit is not None and (type(commit) is not int or commit < self.spec.replan_steps
                                                            or commit % self.spec.replan_steps):
            raise ValueError("MetaWorld native kernel requires commit divisible by replan")
        self.current_task, self.receipts, self.cost_rows = task, [], []
        self.final_ack_error = None
        result = super().run(task, report)
        if self.final_ack_error:
            raise RuntimeError("final execution acknowledgment failed") from self.final_ack_error
        if self.framework_context:
            for row, cost in zip((r for r in result.per_step_rows if r.get("_kind") != "episode_summary"), self.cost_rows, strict=True):
                row.update(cost)
        # Explicitly disabled framework hooks preserve stock rows (except wall clocks).
        if self.capture_sink is not None or self.execution_ack is not None or self.framework_context:
            result.per_step_rows.extend({"_kind": "execution_ack", **r} for r in self.receipts)
        return dataclasses.replace(result, attempt=task.attempt)

    def _identity(self):
        t = self.current_task
        return {"task_uid": t.task_uid, "attempt": t.attempt, "yaml_id": t.yaml_id,
                "seed": t.extra.get("seed", self.spec.seed),
                "bundle_id": t.bundle_id, "benchmark_protocol": self.spec.identity,
                "task_name": self.task_name(t), "orig_init_state_idx": t.orig_init_state_idx,
                "original_init_idx": t.orig_init_state_idx, "task_id": t.task_id,
                "task_table_identity": self.spec.task_table_identity,
                "bundle_generation": t.extra.get("bundle_generation", t.bundle_id),
                **{k: t.extra[k] for k in ("num_steps", "sampler", "sampler_identity", "bundle_identity", "recipe_identity") if k in t.extra},
                "task_data_hash": self.task_data_hash or t.extra.get("task_data_hash"), "arm": t.extra.get("arm", ""),
                **{k: t.extra[k] for k in ("campaign", "model_identity", "norm_identity", "layout_identity",
                                         "fit_identity", "library_identity", "bundle_generation", "split", "partition",
                                         "init_pool", "library_episode_id", "suite_task_id", "trial", "cache_study") if k in t.extra}}

    def _kernel(self, env, infer, prompt, **kwargs):
        self.task_data_hash = getattr(env, "_r17_task_data_hash", None)
        expected = self.current_task.extra.get("task_data_hash")
        if self.task_data_hash is not None and expected is not None and self.task_data_hash != expected:
            pass
        physical = self.physical_factory(env, self.task_name(self.current_task)) if self.capture_sink else None

        def emit(event, obs=None, info=None, **extra):
            if self.capture_sink:
                payload = physical.capture(obs, info, extra.get("requested_action"), extra.get("applied_action"))
                self.capture_sink({"event": event, **self._identity(), **extra, "physical": payload})

        def ack(receipt):
            row = {**self._identity(), **receipt}
            self.receipts.append(row)
            if self.execution_ack:
                self.execution_ack(row)

        observed = _ObservedEnv(env, emit, ack, kwargs["replan_steps"], kwargs["max_steps"])
        decisions = 0

        def checked_infer(obs):
            nonlocal decisions
            state = np.asarray(obs["observation/state"])
            image = np.asarray(obs["observation/image"])
            if state.shape != (4,) or not np.isfinite(state).all() or image.shape != (480, 480, 3) or image.dtype != np.uint8:
                raise ValueError("expected finite state4 and uint8 RGB480")
            if self.framework_context:
                t = self.current_task
                obs = dict(obs, __request__={"campaign": t.extra.get("campaign", "r17"),
                           "benchmark_protocol": self.spec.identity, "task_name": self.task_name(t),
                           "task_table_identity": self.spec.task_table_identity,
                           "original_init_idx": t.orig_init_state_idx, "decision_seq": decisions,
                           "explicit_seed": decision_seed(self.task_name(t), t.orig_init_state_idx, decisions, t.extra.get("seed", self.spec.seed)),
                           "task_id": t.task_id, "seed": t.extra.get("seed", self.spec.seed), "attempt": t.attempt,
                           "arm": t.extra.get("arm", ""), "bundle_generation": t.extra.get("bundle_generation", t.bundle_id)})
                if self.receipts:
                    r = self.receipts[-1]
                    obs["__execution__"] = {k: r[k] for k in ("decision_seq", "executed_count", "terminal")}
            result = infer(obs)
            if self.framework_context:
                hit = result.get("__hit_meta__", {})
                from .execution import cost_row
                self.cost_rows.append(cost_row(result, self._identity(), decisions))
            actions = np.asarray(result["actions"])
            if actions.ndim != 2 or actions.shape[1] != 4 or not np.isfinite(actions).all():
                raise ValueError("policy must return finite (H,4) actions")
            observed.decision = decisions
            observed.server_plan = actions.copy()
            hit = result.get("__hit_meta__", {})
            observed.source_cursor = int(hit.get("cursor", 0))
            observed.source = hit.get("source")
            decisions += 1
            emit("decision", observed.last_obs, decision_seq=observed.decision)
            return result

        try:
            outcome = kernel.run_episode(observed, checked_infer, prompt, **kwargs)
            observed.flush(True, outcome.end_reason)
            if self.receipts:
                self.receipts[-1].update(terminal=True, end_reason=outcome.end_reason)
            emit("terminal", observed.last_obs, {"success": outcome.success}, end_reason=outcome.end_reason)
            return outcome
        except Exception:
            observed.flush(True, "error")
            emit("terminal", observed.last_obs, end_reason="error")
            raise


class MetaWorldBenchmarkAdapter:
    def __init__(self, spec=FORMAL_SPEC, env_factory=kernel.make_env):
        self.spec, self.env_factory = spec, env_factory

    def tasks(self):
        return self.spec.tasks

    def metadata(self):
        return self.spec.to_contract()

    def episode_specs(self, partition="test"):
        indices = {"test": self.spec.formal_indices, "library": self.spec.library_indices,
                   "dev": self.spec.dev_indices}[partition]
        return [{"task_id": i, "task_name": name, "prompt": prompt, "difficulty": group,
                 "original_idx": idx, "seed": self.spec.seed_for(partition), "partition": partition}
                for i, (name, prompt, group) in enumerate(self.tasks()) for idx in indices]

    def task_data_identity(self, name, idx, benchmark_factory=kernel._benchmark, *, seed=None):
        if name not in T.TASK_NAMES or not 0 <= idx < 50:
            raise ValueError("unknown task or original MT1 index")
        seed = self.spec.seed if seed is None else seed
        task = benchmark_factory(name, seed).train_tasks[idx]
        return {"task_name": name, "original_idx": idx, "seed": seed,
                "task_data_identity": task_data_digest(bytes(task.data))}

    def make_episode_runner(self, client_factory=default_client_factory, capture_sink=None, execution_ack=None,
                            *, framework_context=True):
        def make_env(name, idx, seed):
            # Native and renderer-shim factories both use the same MT1 cache.
            native_factory = self.env_factory.__module__ in {"exp.metaworld.env", "exp.mw_finetune.rendering"}
            identity = self.task_data_identity(name, idx, seed=seed) if native_factory else None
            env = self.env_factory(name, idx, seed)
            if identity is not None:
                env._r17_task_data_hash = identity["task_data_identity"]
            return env
        return R17MetaWorldEpisodeRunner(spec=self.spec, env_factory=make_env,
                                        client_factory=client_factory, capture_sink=capture_sink,
                                        execution_ack=execution_ack, framework_context=framework_context)

    def worker_probe(self, renderer_stamp=None, task_data=None):
        packages = {}
        for name in ("metaworld", "mujoco", "gymnasium", "numpy", "websockets", "msgpack"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = None
        return {"schema": "r17.worker_probe.v1", "packages": packages,
                "benchmark_protocol": self.spec.identity, "protocol_id": self.spec.protocol_id,
                "task_table_identity": self.spec.task_table_identity, "task_data": task_data,
                "camera_identity": digest({"name": self.spec.camera, "id": self.spec.camera_id,
                                           "pos": self.spec.camera_pos, "size": 480, "rotation": 180}),
                "settle_steps": 15, "replan_steps": self.spec.replan_steps, "max_policy_controls": 160,
                "renderer": dict(renderer_stamp or {}), "renderer_status": "available" if renderer_stamp else "unverified"}


MWBenchmarkAdapter = MetaWorldBenchmarkAdapter
