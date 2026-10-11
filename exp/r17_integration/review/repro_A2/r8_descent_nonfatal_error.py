"""Round-2 repro: a terminal (non-retriable) dev error that is not prefixed FatalEpisodeError
leaves the cohort undecided; the next level still activates and every dispatch raises."""
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

tmp = Path(tempfile.mkdtemp())
endpoint = EndpointSpec(ServerEndpoint('local', 1), 'm', 'n', 'l', FORMAL_SPEC.identity)
controller = PerTaskDescent(['reach-v3'], ['k2', 'k1'], {'reach-v3': {30: True, 31: True}}, n_dev=2)
episodes = []
for level in controller.levels:
    group = episode_tasks(campaign='c', arm=level, bundle_id=level, endpoint=endpoint.endpoint, model_identity='m',
                          norm_identity='n', layout_identity='l', num_steps=2, partition='dev', names=['reach-v3'], indices=[30, 31])
    episodes.extend(replace(t, extra={**t.extra, 'descent_level': level}) for t in group)
strategy = MetaWorldDescentStrategy(episodes, endpoints=[endpoint], controller=controller)
driver = MetaWorldConductorDriver(strategy, dispatch_path=tmp/'dispatch.jsonl', yaml_weights={t.yaml_id: 2 for t in episodes},
                                  servers=[endpoint.endpoint], journal_path=str(tmp/'journal.jsonl'),
                                  ctl_factory=lambda e: NS(close=lambda: None))
driver.drive_stages_once()
for i in range(2):
    _, payload = driver.handle_pull('local:1')
    t = task_from_wire(payload['task'])
    # repr(exc) of a non-retriable worker error, as WorkerLoop reports it.
    err = "ConfigValidationError('bundle select failed')" if i == 0 else None
    driver.handle_result(result_to_wire(EpisodeResult(t.task_uid, True if err is None else False, 3, err, attempt=t.attempt)))
for _ in range(5):
    driver.drive_stages_once()
print("k2 cohort decided:", controller.current, "stopped:", controller.stopped, "errors:", controller.errors)
try:
    driver.handle_pull('local:1')
    print("next pull ok")
except Exception as exc:
    print("next-level pull ->", type(exc).__name__, exc)
print("dispatched-but-unsent k1 tasks:", sum(len(rt.dispatched) for rt in driver.scheduler._rt.values()),
      "| all_done:", driver.scheduler.all_done())
driver.close()
