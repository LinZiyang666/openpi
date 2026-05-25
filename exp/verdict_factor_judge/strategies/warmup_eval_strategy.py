"""WarmupEvalStrategy — the run_phase.py 7-step protocol as an ExperimentStrategy.

Expresses the verdict warmup->eval flow on the conductor engine (plan §11.2):
each yaml becomes a warmup stage (DumpingJudge collection) feeding, via a
calibration artifact, an eval stage. The strategy owns only experiment semantics
(which yaml, when to fetch/preload); the driver core owns scheduling, retry,
resume, and warmup-dump cleanup.

Stage hooks map to the 7 steps:
  warmup.on_stage_begin    -> load_cache_config(<id>__warmup.yaml)        (step 1)
  warmup episodes          -> DumpingJudge writes <id>__warmup dump       (step 2)
  warmup.on_stage_complete -> fetch_dump + aggregate -> ctx.publish       (steps 3-4)
  eval.on_stage_begin      -> preload_normalizer_buffer THEN load eval     (steps 5-6)
  eval episodes            -> the actual evaluation                        (step 7)
  eval.on_stage_complete   -> unload_warmup_buffer                         (cleanup)

The cleanup-id constraint (plan §5.5 / G1R2 Item 1): the warmup dump is named
``<yaml_id>__warmup`` so it is fetch/unload-compatible with the existing server
protocol (no server change).

Coupling map:
  DEPENDS ON:  openpi.conductor (task, strategy), standard library
  CONSUMED BY: thin entry scripts (construct + ConductorDriver.run)
"""

from __future__ import annotations

import contextlib
import json
import math
import pathlib

from openpi.conductor import task as _task
from openpi.conductor.strategy import ExperimentStrategy
from openpi.conductor.strategy import StageContext


def aggregate_dump(
    content: bytes, *, max_per_key: int = 200, declared_keys: set[str] | None = None
) -> dict[str, list[float]]:
    """Aggregate a warmup dump (JSONL of ``{"factor_raw": {key: val}}``) into a
    per-key buffer of finite floats, capped at ``max_per_key`` per key.

    Mirrors run_phase.py ``_parse_dump_to_buffer``: NaN/inf values are dropped;
    if ``declared_keys`` is given only those keys are kept.
    """
    buffer: dict[str, list[float]] = {}
    for raw_line in content.decode("utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        factor_raw = row.get("factor_raw") or row.get("factor_outputs", {}).get("raw") or {}
        for key, val in factor_raw.items():
            if declared_keys is not None and key not in declared_keys:
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(fval):
                continue
            bucket = buffer.setdefault(key, [])
            if len(bucket) < max_per_key:
                bucket.append(fval)
    return buffer


class WarmupEvalStrategy(ExperimentStrategy):
    """Per-yaml warmup->eval, 1:1 calibration (the run_phase.py case)."""

    def __init__(
        self,
        *,
        task_ids: list[int],
        warmup_trials: int,
        eval_trials: int,
        task_suite_name: str,
        yaml_dir: str,
        max_per_key: int = 200,
        skip_warmup: bool = False,
    ) -> None:
        self._task_ids = list(task_ids)
        self._warmup_trials = warmup_trials
        self._eval_trials = eval_trials
        self._suite = task_suite_name
        self._yaml_dir = pathlib.Path(yaml_dir)
        self._max_per_key = max_per_key
        self._skip_warmup = skip_warmup

    # -- graph construction --

    def _episodes(self, yaml_id: str, phase: str, trials: int, server: _task.ServerEndpoint):
        return [
            _task.EpisodeTask(
                task_uid=_task.make_task_uid(yaml_id, phase, task_id, ep),
                yaml_id=yaml_id,
                phase=phase,
                experiment=self._suite,
                task_id=task_id,
                episode_idx=ep,
                orig_init_state_idx=ep,
                server_host=server.host,
                server_port=server.port,
                bundle_id=yaml_id,
            )
            for task_id in self._task_ids
            for ep in range(trials)
        ]

    def plan(self, yamls, server_assignment):
        g = _task.TaskGraph()
        for yaml_id in yamls:
            server = server_assignment[yaml_id]
            eid = f"{yaml_id}:eval"
            consumes = None
            if not self._skip_warmup:
                wid = f"{yaml_id}:warmup"
                g.add_calibration(
                    _task.CalibrationArtifact(
                        calib_id=yaml_id, source="warmup_stage", warmup_stage_id=wid, cleanup_id=yaml_id
                    )
                )
                consumes = yaml_id
            g.add_stage(
                _task.Stage(
                    eid,
                    yaml_id,
                    "eval",
                    server,
                    episodes=self._episodes(yaml_id, "eval", self._eval_trials, server),
                    consumes_calib_id=consumes,
                    setup={"eval_yaml": str(self._yaml_dir / f"{yaml_id}.yaml")},
                )
            )
            if not self._skip_warmup:
                g.add_stage(
                    _task.Stage(
                        f"{yaml_id}:warmup",
                        yaml_id,
                        "warmup",
                        server,
                        episodes=self._episodes(yaml_id, "warmup", self._warmup_trials, server),
                        produces_calib_id=yaml_id,
                        setup={"warmup_yaml": str(self._yaml_dir / f"{yaml_id}__warmup.yaml")},
                    )
                )
                g.add_dependency(f"{yaml_id}:warmup", eid)
        return g

    # -- stage lifecycle --

    def on_stage_begin(self, stage, ctl, ctx: StageContext):
        if stage.phase == "warmup":
            ctl.load_cache_config(
                yaml_content=_read(stage.setup["warmup_yaml"]),
                yaml_id=f"{stage.yaml_id}__warmup",
            )
        else:
            if stage.consumes_calib_id is not None:
                buf = ctx.get(stage.consumes_calib_id) or {}
                if buf:
                    ctl.preload_normalizer_buffer(stage.yaml_id, buf)
            ctl.load_cache_config(yaml_content=_read(stage.setup["eval_yaml"]), yaml_id=stage.yaml_id)

    def on_stage_complete(self, stage, ctl, ctx: StageContext):
        if stage.phase == "warmup":
            content = ctl.fetch_dump(f"{stage.yaml_id}__warmup")
            buf = aggregate_dump(content, max_per_key=self._max_per_key)
            ctx.publish(stage.produces_calib_id, buf)
        else:
            ctl.unload_warmup_buffer(stage.yaml_id)

    def on_resume(self, stage, ctl, ctx: StageContext):
        # eval stage server self-heal (plan §8.3 B): drop any stale pool entry so
        # the (re)built warmup buffer is re-preloaded by the next on_stage_begin.
        if stage.phase == "eval":
            with contextlib.suppress(Exception):
                ctl.unload_warmup_buffer(stage.yaml_id)


def _read(path: str) -> str:
    return pathlib.Path(path).read_text(encoding="utf-8")
