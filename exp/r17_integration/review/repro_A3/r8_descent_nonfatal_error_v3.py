"""Round-3 check of A2 N3: non-retriable dev errors (repr markers) must stop the task cleanly."""
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS
from openpi.conductor.protocol import result_to_wire, task_from_wire
from openpi.conductor.task import EpisodeResult, ServerEndpoint
from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
from exp.r17_integration.conductor.manifest import EndpointSpec, episode_tasks
from exp.r17_integration.conductor.strategy import MetaWorldDescentStrategy, PerTaskDescent
from exp.r17_integration.conductor.driver import MetaWorldConductorDriver

for error in ("ConfigValidationError('bundle select failed')", "SidecarError('x')", "FatalEpisodeError('y')"):
    tmp = Path(tempfile.mkdtemp())
    endpoint = EndpointSpec(ServerEndpoint('local', 1), 'm', 'n', 'l', FORMAL_SPEC.identity)
    controller = PerTaskDescent(['reach-v3', 'push-v3'], ['k2', 'k1'], {n: {30: True, 31: True} for n in ('reach-v3', 'push-v3')}, n_dev=2)
    episodes = []
    for level in controller.levels:
        group = episode_tasks(campaign='c', arm=level, bundle_id=level, endpoint=endpoint.endpoint, model_identity='m',
                              norm_identity='n', layout_identity='l', num_steps=2, partition='dev', names=['reach-v3', 'push-v3'], indices=[30, 31])
        episodes.extend(replace(t, extra={**t.extra, 'descent_level': level}) for t in group)
    strategy = MetaWorldDescentStrategy(episodes, endpoints=[endpoint], controller=controller)
    driver = MetaWorldConductorDriver(strategy, dispatch_path=tmp/'dispatch.jsonl', yaml_weights={t.yaml_id: 4 for t in episodes},
                                      servers=[endpoint.endpoint], journal_path=str(tmp/'journal.jsonl'),
                                      ctl_factory=lambda e: NS(close=lambda: None))
    seen = []
    for _ in range(200):
        driver.drive_stages_once()
        _, payload = driver.handle_pull('local:1')
        if 'task' not in payload:
            if driver.scheduler.all_done():
                break
            continue
        t = task_from_wire(payload['task'])
        seen.append((t.extra['task_name'], t.extra['descent_level'], t.orig_init_state_idx))
        failing = t.extra['task_name'] == 'reach-v3' and t.extra['descent_level'] == 'k2' and t.orig_init_state_idx == 30
        driver.handle_result(result_to_wire(EpisodeResult(t.task_uid, not failing, 3, error if failing else None, attempt=t.attempt)))
    receipt = strategy.freeze({'m': 1})
    print(error.split('(')[0], "| all_done", driver.scheduler.all_done(), "| reach stopped with error:",
          'reach-v3' in controller.errors, "| push chosen:", controller.chosen['push-v3'],
          "| reach k1 dispatched:", any(s[:2] == ('reach-v3', 'k1') for s in seen), "| freeze:", receipt['status'])
    driver.close()
