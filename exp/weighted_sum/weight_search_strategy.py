"""WeightSearchStrategy — Phase-2 pure-eval weight search on the conductor engine.

Each emitted YAML (a keybuilder x weight-config cell) becomes a single eval
stage; there is no warmup stage (the Layer-1 normalizers are baked into the YAML
by Phase 1 calibration, so ``consumes_calib_id`` is ``None``). Episodes draw
their ``orig_init_state_idx`` exclusively from the per-task held-out init set
(see init_holdout) to avoid the library-vs-eval init leak.

Coupling map:
  DEPENDS ON:  openpi.conductor (task, strategy)
  CONSUMED BY: run_phase2.py (construct + ConductorDriver.run)
"""

from __future__ import annotations

import pathlib

from openpi.conductor import task as _task
from openpi.conductor.strategy import ExperimentStrategy, StageContext


class WeightSearchStrategy(ExperimentStrategy):
    """Pure-eval strategy: one eval stage per YAML, held-out inits, always_hit."""

    def __init__(
        self,
        *,
        task_ids: list[int],
        eval_trials: int,
        task_suite_name: str,
        yaml_dir: str,
        held_out_inits: dict[int, list[int]],
    ) -> None:
        self._task_ids = list(task_ids)
        self._eval_trials = eval_trials
        self._suite = task_suite_name
        self._yaml_dir = pathlib.Path(yaml_dir)
        self._held_out_inits = held_out_inits

    def _episodes(self, yaml_id: str, server: _task.ServerEndpoint):
        episodes = []
        for task_id in self._task_ids:
            inits = self._held_out_inits.get(task_id, [])
            if len(inits) < self._eval_trials:
                raise ValueError(
                    f"task {task_id}: only {len(inits)} held-out init states < "
                    f"eval_trials={self._eval_trials}; reduce eval_trials or rebuild "
                    f"the library from fewer inits"
                )
            for ep in range(self._eval_trials):
                episodes.append(
                    _task.EpisodeTask(
                        task_uid=_task.make_task_uid(yaml_id, "eval", task_id, ep),
                        yaml_id=yaml_id,
                        phase="eval",
                        experiment=self._suite,
                        task_id=task_id,
                        episode_idx=ep,
                        orig_init_state_idx=inits[ep],  # held-out: disjoint from library
                        server_host=server.host,
                        server_port=server.port,
                        bundle_id=yaml_id,
                    )
                )
        return episodes

    def plan(self, yamls, server_assignment):
        g = _task.TaskGraph()
        for yaml_id in yamls:
            server = server_assignment[yaml_id]
            g.add_stage(
                _task.Stage(
                    f"{yaml_id}:eval",
                    yaml_id,
                    "eval",
                    server,
                    episodes=self._episodes(yaml_id, server),
                    consumes_calib_id=None,  # normalizers baked into the YAML
                    setup={"eval_yaml": str(self._yaml_dir / f"{yaml_id}.yaml")},
                )
            )
        return g

    def on_stage_begin(self, stage, ctl, ctx: StageContext):
        ctl.load_cache_config(
            yaml_content=pathlib.Path(stage.setup["eval_yaml"]).read_text(encoding="utf-8"),
            yaml_id=stage.yaml_id,
        )
