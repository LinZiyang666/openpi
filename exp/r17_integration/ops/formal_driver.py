"""Frozen full-arm campaign over the R17 conductor (per-episode dispatch)."""
from __future__ import annotations

from dataclasses import asdict
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import time

from .formal_process import write_json


def probe_validator(benchmark, campaign=None):
    from exp.r17_integration.conductor.manifest import validate_worker_probe
    if campaign and campaign.get('library'):
        from .library_campaign import validate_b_probe
        def validate(probe):
            validate_worker_probe(probe, spec=benchmark)
            validate_b_probe(probe,campaign)
        return validate
    if campaign and campaign.get('resume_from'):
        from .formal_resume import validate_probe
        def validate(probe):
            validate_worker_probe(probe, spec=benchmark)
            validate_probe(probe, campaign)
        return validate
    return lambda probe: validate_worker_probe(probe, spec=benchmark)


def plan(campaign):
    if campaign.get('cache_study'):
        from .cache_study import plan as study_plan
        return study_plan(campaign)
    from exp.r17_integration.conductor.manifest import EndpointSpec, episode_tasks, load_benchmark_spec
    from openpi.conductor.task import ServerEndpoint
    from exp.r17_integration.common.contracts import digest
    assets = Path(campaign['local_assets'])
    benchmark = load_benchmark_spec(assets / 'benchmark.json')
    model = json.loads((assets / 'model_spec.json').read_text())
    from exp.r17_integration.conductor.manifest import load_plan_config
    config = load_plan_config(assets / 'full.yaml')
    binding = config.adapter_binding
    from .formal_eval import execution_horizon, execution_steps, selected_suites
    commit = execution_horizon(campaign, model['layout'])
    k = execution_steps(campaign, model['default_sampler']['num_steps'])
    sampler = (json.loads((assets / 'execution_sampler.json').read_text())
               if campaign.get('denoising_steps') is not None else model['default_sampler'])
    if campaign.get('commit_horizon') is not None and bool(binding.get('allow_partial_commit', False)) != bool(commit % model['layout']['replan_steps']):
        pass
    if (benchmark.identity not in model['allowed_benchmark_protocols'] or
        binding['model_id'] != model['model_id'] or
        binding['feature_identity'] != digest(model['feature_spec']) or
        binding['sampler_identity'] != digest(sampler) or sampler['num_steps'] != k or
        binding['checkpoint_identity'] != model['checkpoint_identity'] or
        binding['norm_identity'] != model['norm_identity'] or
        binding['layout_identity'] != digest(model['layout']) or
        binding['benchmark_protocol'] != benchmark.identity or
        binding['num_steps'] != k or
        binding['commit_horizon'] != commit or
        tuple(config.recipe.ordered_tasks) != tuple(benchmark.ordered_tasks) or
        config.recipe.task_table_identity != benchmark.task_table_identity or
        config.enabled or config.checkpoints or config.recipe.enabled):
        pass
    endpoint = ServerEndpoint(campaign['endpoint_host'], campaign.get('endpoint_port', campaign['public_port']))
    names = [benchmark.tasks[i][0] for i in (0, 10, 20, 30)] if campaign['smoke'] and campaign['domain'] == 'libero' else (
        [benchmark.tasks[0][0], benchmark.tasks[1][0]] if campaign['smoke'] else None)
    indices = [0] if campaign['smoke'] and campaign['domain'] == 'libero' else ([0, 1] if campaign['smoke'] else None)
    if campaign['smoke'] and campaign['domain'] == 'libero':
        names = names[:campaign.get('smoke_episodes', 4)]
    if campaign.get('library'):
        names = [r[0] for r in benchmark.tasks if r[2]==campaign['suite']]
        # Local B-pool shim: indices partition distinct from official A table.
        indices = [0] if campaign['smoke'] else list(range(50))
        if campaign['smoke']:
            names = names[:2]
    tasks = episode_tasks(campaign=campaign.get('semantic_campaign', campaign['run_name']), arm='full', bundle_id='formal_full_v1',
        endpoint=endpoint, model_identity=model['checkpoint_identity'], norm_identity=model['norm_identity'],
        layout_identity=digest(model['layout']), num_steps=k,
        spec=benchmark, partition='smoke' if campaign['smoke'] or campaign.get('library') else 'test', names=names, indices=indices,
        yaml_identity=hashlib.sha256((assets / 'full.yaml').read_bytes()).hexdigest(),
        bundle_generation='formal_full_v1', commit_horizon=commit)
    suites = selected_suites(campaign)
    if suites is not None:
        tasks = [task for task in tasks if task.experiment in suites]
    if campaign.get('library'):
        from .library_campaign import teacher_tasks
        tasks = teacher_tasks(campaign,benchmark,tasks)
        if campaign.get('repair_attempts') is not None:
            wanted = set(campaign['repair_attempts'])
            if not wanted or not wanted <= {t.task_uid for t in tasks}:
                pass
            tasks = [t for t in tasks if t.task_uid in wanted]
    if campaign.get('resume_task_uids') is not None:
        wanted = set(campaign['resume_task_uids'])
        if not wanted or len(wanted) != len(campaign['resume_task_uids']) or not wanted <= {t.task_uid for t in tasks}:
            pass
        tasks = [t for t in tasks if t.task_uid in wanted]
    spec = EndpointSpec(endpoint, model['checkpoint_identity'], model['norm_identity'], digest(model['layout']),
        benchmark.identity, allowed_k=(k,), capacity=campaign['workers'])
    return benchmark, model, tasks, [spec]


def run(campaign, directory):
    if campaign.get('cache_study'):
        from .cache_study import run_driver
        return run_driver(campaign, directory)
    from exp.r17_integration.conductor.manifest import validate_worker_probe
    from exp.r17_integration.conductor.strategy import MetaWorldStrategy
    from exp.r17_integration.qualification.driver import ProbeRecordingDriver
    from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    benchmark, model, tasks, endpoints = plan(campaign)
    strategy = MetaWorldStrategy(tasks, endpoints=endpoints,
        bundles={'formal_full_v1': Path(campaign['local_assets']) / 'full.yaml'})
    def append(path, value):
        with (directory / path).open('a') as f:
            f.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
            f.flush()
            os.fsync(f.fileno())
    class CampaignDriver(ProbeRecordingDriver):
        def handle_pull(self, key):
            kind, task = super().handle_pull(key)
            if isinstance(task, dict) and isinstance(task.get('task'), dict):
                append('worker_dispatch.jsonl', dict(worker_id=getattr(self._shutdown_local, 'worker_id', None),
                    task_uid=task['task']['task_uid'], attempt=task['task'].get('attempt', 1), time=time.time()))
            return kind, task
        def handle_result(self, result):
            # P2 accepts wire dictionaries and returns None. Capture its actual
            # generation fence before applying the result, under the same lock.
            with self._scheduler._lock:
                uid = result.get('task_uid')
                entry = self._scheduler._ep_index.get(uid)
                accepted = entry is not None and uid in self._scheduler._rt[entry[0]].dispatched and (
                    result.get('attempt') == self._scheduler._dispatch_gen.get(uid))
                evidence = {}
                forwarded = result
                if campaign.get('library') and accepted:
                    from .library_retry import library_retry_payload, capture_counter
                    forwarded, evidence = library_retry_payload(result, capture_counter(campaign))
                super().handle_result(forwarded)
                append('attempts.jsonl', dict({k:v for k,v in result.items() if k != 'per_step_rows'},
                    **evidence,
                    worker_id=getattr(self._shutdown_local, 'worker_id', None),
                    accepted=accepted, time=time.time()))
    def steps(stage, rows):
        for row in rows:
            append('per_step.jsonl', row)
    driver = CampaignDriver(strategy, dispatch_path=directory / 'dispatch.jsonl',
        rejection_path=directory / 'rejected_worker_probes.json', expected_workers=campaign['workers'],
        shutdown_seconds=60, yaml_weights={t.yaml_id: 1 for t in tasks}, servers=[endpoints[0].endpoint],
        journal_path=str(directory / 'journal.jsonl'),
        ctl_factory=lambda endpoint: WebsocketClientPolicy(endpoint.host, endpoint.port),
        bind_host='0.0.0.0', bind_port=campaign['driver_port'], per_step_writer=steps,
        worker_probe_validator=probe_validator(benchmark,campaign), episode_timeout_s=campaign['episode_seconds'],
        scheduler_kwargs=dict(eval_concurrency=campaign['workers'], max_episode_retries=campaign['retries']))
    from .library_retry import configure_library_driver
    configure_library_driver(driver, campaign)
    if campaign.get('live_fleets'):
        from .live_fleet_driver import configure
        configure(driver,campaign,directory)
    stopped = False
    def stop(*unused):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    lock = (directory / '.eval.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(directory / 'manifest.json', dict(campaign, benchmark=benchmark.manifest(), model=model,
        tasks=[asdict(t) for t in tasks], expected_episodes=len(tasks)))
    started = time.monotonic()
    try:
        driver.run(stop=lambda: stopped or bool(driver.probe_rejections) or time.monotonic()-started > campaign['seconds'])
    finally:
        write_json(directory / 'census.json', driver.worker_census)
        write_json(directory / 'rejected_worker_probes.json', driver.probe_rejections)
        report = dict(status='PASS' if driver.scheduler.all_done() and not driver.probe_rejections and
            driver.worker_shutdown['status'] == 'PASS' else 'PARTIAL',
            worker_shutdown=driver.worker_shutdown, elapsed_seconds=time.monotonic()-started,
            rejected_worker_probes=len(driver.probe_rejections), all_done=driver.scheduler.all_done())
        driver.close()
        lock.close()
        write_json(directory / 'DRIVER_RECEIPT.json', report)
    raise SystemExit(0 if report['status'] == 'PASS' else 2)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', required=True)
    p.add_argument('--directory', required=True)
    a = p.parse_args()
    run(json.loads(Path(a.campaign).read_text()), a.directory)
