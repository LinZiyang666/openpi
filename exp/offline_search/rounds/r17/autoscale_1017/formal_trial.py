"""Explicit prepare/launch entry for real formal queues. Importing never executes it."""
import argparse
from contextlib import contextmanager
import copy
import json
from pathlib import Path
import shutil

from .node import atomic, read
from .policy import Config, NODES
from .release import overlay_worker, require_patch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
EVAL = Path('/data/mw_finetune/eval')


def counts(value):
    result = {}
    for part in value.split(','):
        n, v = part.split('=')
        if n not in NODES or n in result or int(v) < 1:raise ValueError('distinct live nodes and positive desired maxima required')
        result[n] = int(v)
    return result


def configure(campaign, maxima):
    c = copy.deepcopy(campaign)
    from .defaults import desired_maxima
    maxima=desired_maxima(c['domain'],maxima,c.get('worker_hard_limits'))
    # No plan name, endpoint, init index, seed, model, YAML or server knob changes.
    c.update(autoscale=dict(mode='dynamic', decrease=.7, window_s=180),
             autoscale_maximum=sum(maxima.values()), autoscale_node_maxima=dict(maxima),
             worker_nodes=dict(maxima), execution_worker_nodes=dict(maxima),
             workers=sum(maxima.values()), execution_workers=sum(maxima.values()),
             autoscale_overlay=True, prepare_only=True,autoscale_maxima_policy='physical-v2')
    if c.get('worker_egl_devices_by_node'):
        c['worker_egl_devices_by_node'] = {n: v for n, v in c['worker_egl_devices_by_node'].items() if n in maxima}
    if c['domain'] == 'metaworld':c.update(worker_max_episodes=32, worker_max_seconds=1800)
    Config(**c['autoscale'])
    return c


def resolve(name, action, maxima):
    root = EVAL / name
    if not name or '/' in name or root.parent != EVAL:raise ValueError('simple formal campaign name required')
    if action == 'prepare' and not (root/'campaign.json').exists():
        from exp.offline_search.rounds.r17.filter_failed_knob import formal_a_queue as q
        unit = next((u for u in read(q.INDEX)['units'] if u['name'] == name), None)
        if unit is None:raise ValueError('campaign definition not found: '+name)
        if unit['status'] == 'WAITING_OFF':
            # Old queue definitions still contain timan108. Override only the
            # resource input passed to the native materializer, before its
            # topology parser; scientific OFF/levels selection is unchanged.
            from unittest.mock import patch
            native = q.materialize
            def materialize(u, keep, levels):
                return native(dict(u, workers=dict(maxima), autoscale_node_maxima=dict(maxima)), keep, levels) if maxima else native(u, keep, levels)
            with patch.object(q, 'materialize', materialize):
                unit = q.build_levels(unit['model'], unit['mode'], unit['arm'])
        if not unit['jobs']:raise ValueError('OFF decision selected no levels; use the next real formal campaign')
        roots = {Path(j['run']) for j in unit['jobs']}
        if roots != {root}:raise ValueError('queue now has multiple/different jobs; use each actual campaign name')
    if not (root / 'campaign.json').is_file():raise ValueError('formal definition not rebuilt yet: ' + str(root))
    return root


@contextmanager
def longseg_budget(root):
    """filter_failed.stage's /data reader and rsync budget for long-segment campaigns."""
    if not (read(Path(root)/'campaign.json') or {}).get('longseg_1017'):
        yield
        return
    from unittest.mock import patch
    from exp.offline_search.rounds.r17.longseg_1017.io_budget import filesystem_budget
    from exp.r17_integration.ops import direct_transfer
    with filesystem_budget(), patch.object(direct_transfer, 'BWLIMIT_KIBPS', 4096), \
            patch.object(shutil, '_USE_CP_SENDFILE', False):
        yield


def prepare(root, maxima=None, *, reprepare=False, node_config=None, preserve_science=False):
    """Native staging pieces, with a new worker/driver pin and frozen server bytes."""
    from exp.r17_integration.ops import formal_eval as f, driver_release
    from exp.r17_integration.ops.formal_prestage import preparation_lock, seal_prepared
    from exp.r17_integration.ops.immutable_release import install, sha
    from exp.r17_integration.ops.formal_runtime import attach
    from exp.r17_integration.ops.cache_control import attach as attach_control
    from exp.offline_search.rounds.r17.filter_failed import stage, timan1_gpu2, worker_resilience
    from exp.r17_integration.ops.formal_ports import prepare_admission
    from .compare import wait_io
    from .switch import enabled
    if not enabled():
        if maxima is not None:raise ValueError("global OFF: native preparation retains original counts")
        return stage.stage(root)
    wait_io()
    with preparation_lock(root), longseg_budget(root), timan1_gpu2.coordinator(), worker_resilience.coordinator():
        if any((root / n).exists() for n in ('LAUNCH.json', 'RECEIPT.json', 'DONE.json', 'launch.json')):
            raise ValueError('launched campaign cannot be re-prepared')
        if (root/'PREPARED.json').exists() and not reprepare:
            raise ValueError('only a newly rebuilt campaign, or explicit --reprepare, may be changed')
        old = read(root / 'campaign.json')
        if old.get('cache_resume_selection') or old.get('resume_from'):
            raise ValueError('use the next new full formal campaign, not a running/closed continuation')
        plan_before = (root / 'EPISODE_PLAN.json').read_bytes()
        from .defaults import apply
        c = configure(old, maxima) if maxima else apply(copy.deepcopy(old), force=True)
        if node_config is not None:
            c['autoscale_node_config'] = copy.deepcopy(node_config)
            c['autoscale_external_legacy_nodes'] = sorted(node_config)
        maxima = c['autoscale_node_maxima']
        if (root/'PREPARED.json').exists():
            import time
            (root/'PREPARED.json').rename(root/('PREPARED.before_autoscale.'+str(time.time_ns())+'.json'))
        # Reuse the native ff/ffk measured renderer VRAM profile on every node;
        # otherwise timan107 inherits an obsolete1GiB slot and masks RAM pressure.
        c['worker_vram_mib'] = timan1_gpu2.SIZES[c['domain']][0]
        source = Path(root) if preserve_science else Path(c.get('filter_failed_source', root))
        runtime_sha = sha(source / 'runtime.tar')
        if c.get('longseg_1017'):
            # Same as filter_failed.stage._stage: long-segment campaigns run the
            # archive with the segment-boundary fix written into its sources.
            from exp.offline_search.rounds.r17.longseg_1017.runtime_fix import stage as fixed_runtime
            runtime_sha = fixed_runtime(f, c, root)
        server_pin = sha(root/'runtime.tar')
        prepare_admission(c)
        if c.get('autoscale_validation'):
            from .trial_runtime import prepare as prepare_validation_runtime
            prepare_validation_runtime(f,c,root)
        install(root / 'runtime.tar', c['runtime'], runtime_sha, c['runtime_identity'])
        c['coordinator_runtime'] = c['runtime']
        if source.resolve() != root.resolve():
            attach(c, source)
            if c.get('cache_study'):attach_control(c, root)
        driver_release.prepare(c, root, REPO)
        atomic(root / 'campaign.json', c)
        expected = json.loads(plan_before)['tasks']
        if driver_release.planned_tasks(c, root) != expected:raise ValueError('plan changed before staging')
        import uuid
        archive = root / ('autoscale_workers_'+uuid.uuid4().hex+'.tar')
        proof = overlay_worker(source / 'workers.tar', archive, REPO)
        atomic(root / 'AUTOSCALE_RELEASE.json', dict(proof, maxima=maxima, server_runtime_unchanged=True))
        # The native launch path consumes this receipt for GPU2/local routes.
        atomic(root / 'LOCAL_WORKER_ORCHESTRATION.json', dict(proof, schema='r17.autoscale_worker_ops.v1'))
        if source.resolve() != root.resolve():
            shutil.copyfile(source / 'WORKER_DEPLOYMENT.json', root / 'WORKER_DEPLOYMENT.json')
        with f.retry_scope(root):
            deployment = f.prepare_workers(c, root, payload_override=archive)
            stage.stage_server_inputs(f, c, root)
            stage.stage_worker_inputs(f, c, root, deployment)
        atomic(root / 'campaign.json', c)
        if (root / 'EPISODE_PLAN.json').read_bytes() != plan_before or driver_release.planned_tasks(c, root) != expected:
            raise ValueError('plan/UID/init/seed changed during staging')
        atomic(root / 'PREFLIGHT.json', dict(status='PASS', cpu_only=True, episodes=len(expected),
                worker_nodes=list(deployment), replicas=c['replicas'], ports=c['ports'],
                memory_admission='node_local_per_spawn', autoscale_maxima=maxima))
        seal_prepared(c, root, deployment)
        print(json.dumps(dict(status='PREPARED', run=str(root), desired_maxima=maxima, episodes=len(expected))))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'launch'))
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument('--name')
    target.add_argument('--run', type=Path)
    p.add_argument('--reprepare', action='store_true', help='Explicitly reseal an unlaunched prepared campaign')
    p.add_argument('--worker-counts', type=counts)
    a = p.parse_args(argv)
    require_patch(REPO)
    root = a.run.resolve() if a.run else resolve(a.name, a.action, a.worker_counts)
    if a.action == 'prepare':return prepare(root, a.worker_counts, reprepare=a.reprepare)
    c = read(root / 'campaign.json')
    from .defaults import desired_maxima
    expected=desired_maxima(c['domain'],a.worker_counts,c.get('worker_hard_limits')) if a.worker_counts else None
    if (expected and c.get('autoscale_node_maxima') != expected) or (c.get('autoscale') or {}).get('mode') != 'dynamic':
        raise ValueError('launch maxima must match the sealed dynamic preparation')
    from exp.offline_search.rounds.r17.filter_failed_knob.launch import main as launch
    return launch(root)


if __name__ == '__main__':main()
