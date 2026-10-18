"""CPU smoke of the resident recipe-query path of a long-segment campaign.

In the campaign's own server interpreter, with CUDA hidden, run its pinned
ffk_bootstrap.py exactly as the resident does before serve() (longseg method
installer, row-filter/knob runtime install, DP admission bridge) on top of
its frozen runtime with the segment-boundary fix written in (runtime_fix, as
filter_failed.stage does). Then load every cache bundle through the frozen
bundle validator/connection factory and drive ConnectionPolicy.infer with
the worker's request/ack protocol for one episode per planned (bundle, task),
so recipe.query, Q3, the knob and the empty-candidate branch all execute.

The model is a stub that returns the sealed ModelSpec geometry; retrieval
features are rebuilt from fitted library rows. Nothing is staged or written
outside the scratch directory and the JSON report.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import zlib

from .audit import HERE, EVAL, io_ready
from .io_budget import dump
from exp.offline_search.rounds.r17.filter_failed_knob.common import read, sha

SCRATCH = Path('/tmp/r17_recipe_smoke')
MARKER = "ffk_runtime=dict(__name__='__main__')"
# H100-only interpreters; same Python minor/numpy as the local copy (checked
# read-only on H100 10-18: groot 3.11.16/1.26.4 vs local 3.11.15/1.26.4).
LOCAL_INTERPRETER = {'/home/exouser/gr00t_n15_venv/.venv/bin/python':
                     '/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python'}


def child(spec):
    """Runs inside the frozen server interpreter (cwd = fixed runtime)."""
    import copy
    import hashlib
    import json
    import sys
    import traceback
    from collections import Counter
    from pathlib import Path
    from types import SimpleNamespace
    out = dict(name=spec['name'], status='FAIL', stage='bootstrap', bundles=[], errors=[])

    def emit():
        torch = sys.modules.get('torch')
        out['cuda_initialized'] = bool(torch is not None and torch.cuda.is_initialized())
        print('RECIPE_SMOKE_RESULT '+json.dumps(out, sort_keys=True, default=str), flush=True)
    try:
        code = Path(spec['bootstrap']).read_text()
        if code.count(spec['marker']) != 1:
            raise ValueError('bootstrap resident entry marker not found exactly once')
        code = code.replace(spec['marker'], "ffk_runtime=dict(__name__='ffk_recipe_smoke')", 1)
        scope = dict(__name__='ffk_recipe_smoke_bootstrap')
        exec(compile(code, spec['bootstrap'], 'exec'), scope)
        # ffk1015_runtime.main() before serve(), verbatim order.
        scope['ffk_runtime']['install'](spec['selection'])
        if spec['bridge']:
            from exp.offline_search.rounds.r17.filter_failed.server_cli import bridge
            from exp.r17_integration.models.dp import formal_batch_server
            bridge(formal_batch_server)
        out['stage'] = 'provider'
        import numpy as np
        import yaml
        from openpi.cache.config import _substitute_env_vars
        from openpi.serving import websocket_policy_server as registry
        from exp.r17_integration.ops.cache_study import model_spec
        from exp.r17_integration.common.provider import AdapterProvider, register_provider
        from exp.r17_integration.common.benchmarks import register_benchmark
        from exp.r17_integration.common.config import config_from_dict, load_cache_config, build_shared_storage
        from exp.r17_integration.common.serving import make_bundle_validator, make_connection_factory
        from exp.r17_integration.common.recipe import bundle_identity
        from exp.r17_integration.conductor.manifest import load_benchmark_spec
        bench = load_benchmark_spec(Path(spec['benchmark']))
        register_benchmark(bench)
        mspec = model_spec(json.loads(Path(spec['model_spec']).read_text()))
        layout = mspec.layout
        raws = {b['bundle_id']: yaml.safe_load(_substitute_env_vars(Path(b['yaml']).read_text())) or {}
                for b in spec['bundles']}
        samplers, providers = {}, set()
        # The resident default full.yaml (native K) is validated too.
        full = yaml.safe_load(_substitute_env_vars(Path(spec['full_yaml']).read_text())) or {}
        for raw in [full, *raws.values()]:
            binding = raw['adapter_binding']
            samplers[binding['num_steps']] = binding['sampler_identity']
            providers.add(binding['provider'])
        if len(providers) != 1:
            raise ValueError('bundles name several providers: %s' % sorted(providers))
        current = SimpleNamespace(features=None)

        class StubAdapter:
            """Sealed ModelSpec geometry; returns the features being replayed."""
            def metadata(self):
                return mspec
            def extract_state_only(self, wire):
                return np.asarray(wire['smoke/state'], np.float32)
            def prepare(self, wire, request):
                return dict(wire)
            def run_stage1(self, prepared):
                return prepared
            def retrieval_fields(self, s1):
                return {k: np.array(v, np.float32) for k, v in current.features.items()}
            def run_stage2(self, s1, *, capture=False):
                return None
            def sample(self, s2, sampler, rng, capture):
                return SimpleNamespace(normalized_chunk=np.zeros((layout.horizon, layout.action_width), np.float32),
                                       actual_nfe=sampler.num_steps, rng_receipt={}, snapshots=(), capture=None)
            def to_wire(self, normalized, prepared):
                return np.zeros((layout.horizon, layout.valid_action_dim), np.float32)

        provider = AdapterProvider(providers.pop(), mspec, StubAdapter,
            lambda k: SimpleNamespace(num_steps=k, identity=samplers[k], family=mspec.default_sampler.family,
                                      native_timesteps=getattr(mspec.default_sampler, 'native_timesteps', ())))
        # The resident provider's own library admission hook (Fast-WAM/Cosmos),
        # without the loaded-weights manifest; other models require exact specs.
        import importlib
        import inspect
        owner = importlib.import_module('exp.r17_integration.models.%s.provider' % spec['model'])
        configure = getattr(owner, 'configure_library_provider', None)
        if configure is not None:
            kwargs = {'manifest': None} if 'manifest' in inspect.signature(configure).parameters else {}
            provider = configure(provider, **kwargs)
            out['library_validator'] = owner.__name__+'.configure_library_provider'
        register_provider(provider)
        default = load_cache_config(Path(spec['full_yaml']))
        factory = make_connection_factory(provider, default, build_shared_storage(default))
        validator = make_bundle_validator(provider)
        prefix = {'libero': 'libero', 'metaworld_mt50': 'mt50'}[bench.benchmark]

        def decision_seed(task_name, init, seq, seed):
            value = f'{prefix}:{seed}:{task_name}:{init}:{seq}'.encode()
            return int.from_bytes(hashlib.sha256(value).digest()[:4], 'little')
        for b in spec['bundles']:
            out['stage'] = 'bundle '+b['bundle_id']
            row = dict(bundle_id=b['bundle_id'], file=b['file'], episodes=0, deliveries=0, heads=0,
                       sources=Counter(), stall_calls=0, knob_calls=0, empty_candidate_heads=0,
                       empty_tasks=[], recipe=None, errors=[])
            out['bundles'].append(row)
            # WebsocketPolicyServer._load_r17_bundle, then the bound connection.
            config = config_from_dict(raws[b['bundle_id']])
            validator(config, None)
            storage = build_shared_storage(config)
            validator(config, storage)
            if bundle_identity(config) != b['bundle_identity']:
                raise ValueError('bundle content identity differs from the sealed study')
            with registry._bundle_lock:
                registry._bundle_version += 1
                bundle = registry.CurrentCacheBundle('<r17:inline>', config, storage, registry._bundle_version, None)
                registry._bundles[b['bundle_id']] = bundle
            policy = factory(None, b['bundle_id'])
            recipe = policy.recipe
            base = getattr(recipe, 'base', None)
            row['recipe'] = None if recipe is None else type(recipe).__name__+'/'+type(base).__name__
            if base is not None:
                row['base_has_serve_loeo_episodes'] = hasattr(base, 'serve_loeo_episodes')
            for task in b['tasks']:
                try:
                    tid, init = task['task_id'], task['orig_init_state_idx']
                    table = None if base is None else base.tasks.get(tid)
                    empty = base is not None and (table is None or not len(table.rows))
                    if empty:
                        row['empty_tasks'].append(task['extra']['task_name'])
                    if base is not None and not empty:
                        episode = base.episode[table.rows[0]]
                        rows = np.flatnonzero(base.episode == episode)
                        rows = rows[np.argsort(base.step[rows], kind='stable')]
                        state_dim = base.state.shape[1]
                    else:
                        rows, state_dim = None, (base.state.shape[1] if base is not None else len(layout.valid_state_indices))

                    def observation(head):
                        features = {}
                        if base is not None:
                            r = None if rows is None else rows[min(head, len(rows)-1)]
                            offset = 0
                            for name in base.fields:
                                B, shift = base.pca[name]
                                width = B.shape[0]
                                proj = shift*0 if r is None else base.projected[r, offset:offset+width]
                                features[name] = B.T @ (proj + shift)
                                offset += width
                            state = np.zeros(state_dim, np.float32) if r is None else np.array(base.state[r])
                        else:
                            state = np.zeros(state_dim, np.float32)
                            features = {name: np.zeros(dim, np.float32) for name, dim in mspec.feature_spec.fields.items()}
                        features.setdefault('robot_state', state)
                        current.features = features
                        return {'smoke/state': state}
                    extra = task['extra']
                    seed = extra.get('seed', bench.seed)
                    identity = dict(extra, task_uid=task['task_uid'], attempt=task['attempt'], yaml_id=task['yaml_id'],
                        bundle_id=task['bundle_id'], bundle_generation=extra.get('bundle_generation', task['bundle_id']),
                        task_id=tid, task_name=extra['task_name'], original_init_idx=init, orig_init_state_idx=init,
                        task_table_identity=bench.task_table_identity, benchmark_protocol=bench.identity,
                        ack_protocol='r17.execution.v1', partial_tail_protocol='r17.partial_tail.v1', seed=seed)
                    policy.on_episode_start(episode_id=task['episode_idx'], task=extra['task_name'],
                                            extra_metadata=identity)
                    ack, delivery = None, 0
                    for head in range(spec['decisions']):
                        cursor = 0
                        while True:
                            obs = observation(head)
                            obs['__request__'] = dict(campaign=extra.get('campaign', 'r17'),
                                benchmark_protocol=bench.identity, task_table_identity=bench.task_table_identity,
                                task_name=extra['task_name'], task_id=tid, original_init_idx=init,
                                decision_seq=delivery, seed=seed,
                                explicit_seed=decision_seed(extra['task_name'], init, delivery, seed),
                                attempt=task['attempt'], arm=extra.get('arm', ''),
                                bundle_generation=extra.get('bundle_generation', task['bundle_id']))
                            if ack is not None:
                                obs['__execution__'] = ack
                            response = policy.infer(obs)
                            meta = response['__hit_meta__']
                            actions = np.asarray(response['actions'])
                            if actions.ndim != 2 or not len(actions) or not np.isfinite(actions).all():
                                raise ValueError('nonfinite/empty wire actions')
                            row['deliveries'] += 1
                            row['sources'][meta['source']] += 1
                            if meta.get('segment_head', True):
                                row['heads'] += 1
                                row['stall_calls'] += bool(meta.get('stall_call'))
                                row['knob_calls'] += bool(meta.get('knob_call'))
                                row['empty_candidate_heads'] += bool(meta.get('empty_candidates'))
                            count = min(layout.replan_steps, len(actions))
                            ack = dict(decision_seq=meta['decision_seq'], executed_count=count, terminal=False)
                            delivery += 1
                            cursor += count
                            if cursor >= policy.commit_horizon or count == len(actions):
                                break
                    policy.on_episode_end(execution=dict(ack, terminal=True))
                    row['episodes'] += 1
                except Exception as exc:
                    text = ''.join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-6:])
                    row['errors'].append(dict(task_uid=task['task_uid'], error=repr(exc), traceback=text))
                    if len(row['errors']) >= 3:
                        break
            row['sources'] = dict(row['sources'])
            if row['errors']:
                out['errors'].append(dict(bundle_id=b['bundle_id'], **row['errors'][0]))
        out['stage'] = 'done'
        out['status'] = 'FAIL' if out['errors'] else 'PASS'
    except Exception as exc:
        out['errors'].append(dict(error=repr(exc), traceback=''.join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)[-8:])))
    emit()


def fixed_runtime(root):
    """filter_failed.stage's runtime: runtime_fix written into runtime.tar."""
    from .runtime_fix import patched_files
    archive = root/'runtime.tar'
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        data = {m.name: tar.extractfile(m).read() for m in members if m.isfile()}
    fixed = patched_files(lambda member: data[member].decode())
    for name, text in fixed.items():
        data[name] = text.encode()
    identity = hashlib.sha256(b''.join(hashlib.sha256(n.encode()+b'\0'+data[n]).digest()
                                       for n in sorted(data))).hexdigest()
    target = SCRATCH/'runtime'/identity
    if not (target/'READY').exists():
        staging = target.with_name(identity+'.partial')
        if staging.exists():
            import shutil
            shutil.rmtree(staging)
        for name, body in data.items():
            path = staging/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        (staging/'READY').write_text(json.dumps(dict(archive=str(archive), archive_sha256=sha(archive),
                                                     fixed=sorted(fixed))))
        staging.rename(target)
    return target, sorted(fixed)


def preview_bootstrap(root, c):
    """The bootstrap longseg_1017.vfix.repin would pin now (temp copy only)."""
    import copy
    import shutil
    import tempfile
    from .build import pin
    from .recalibrate import pin_plan_evidence
    temp = Path(tempfile.mkdtemp(dir=SCRATCH))
    (temp/'assets').mkdir()
    shutil.copyfile(root/'assets/ffk_evidence.py', temp/'assets/ffk_evidence.py')
    c = copy.deepcopy(c)
    pin(temp, c)
    if c.get('longseg_1017', {}).get('recalibration'):
        pin_plan_evidence(temp, c)
    return temp/'assets/ffk_bootstrap.py'


def bundles(root, c):
    plan = read(root/'EPISODE_PLAN.json')['tasks']
    seal = read(root/'assets/CACHE_STUDY.json')
    result = {}
    for g in seal['groups']:
        b = result.setdefault(g['bundle_id'], dict(bundle_id=g['bundle_id'], bundle_identity=g['bundle_identity'],
                                                   file=g['file'], yaml=str(Path(c['local_assets'])/g['file']), tasks={}))
        if b['bundle_identity'] != g['bundle_identity'] or b['file'] != g['file']:
            raise ValueError('one bundle id names different sealed content: '+g['bundle_id'])
    for t in plan:
        if t['bundle_id'] not in result:
            raise ValueError('planned task outside the sealed study: '+t['task_uid'])
        tasks = result[t['bundle_id']]['tasks']
        if t['task_id'] not in tasks or t['orig_init_state_idx'] < tasks[t['task_id']]['orig_init_state_idx']:
            tasks[t['task_id']] = t
    return [dict(b, tasks=[b['tasks'][k] for k in sorted(b['tasks'])]) for b in result.values() if b['tasks']]


def smoke(name, *, bootstrap=None, preview=False, decisions=4, timeout=1800):
    from exp.r17_integration.ops import formal_eval as f
    root = EVAL/name
    c = read(root/'campaign.json')
    row = dict(name=name, model=c['model'], domain=c['domain'], server_node=c['server_node'],
               source_runtime=c['runtime'], selection_cells=len(c['filter_failed_knob']['selection']))
    if bootstrap is None and preview:
        bootstrap, row['bootstrap'] = preview_bootstrap(root, c), 'preview_repin'
    elif bootstrap is None:
        bootstrap = root/'assets/ffk_bootstrap.py'
        if sha(bootstrap) != c['ffk_bootstrap_sha256']:
            raise ValueError('pinned bootstrap bytes differ from campaign.json')
        row['bootstrap'] = 'pinned'
    else:
        row['bootstrap'] = 'override'
    row['bootstrap_path'], row['bootstrap_sha256'] = str(bootstrap), sha(bootstrap)
    runtime, fixed = fixed_runtime(root)
    row.update(fixed_runtime=str(runtime), runtime_fix_members=fixed)
    layer = Path(c['runtime'])/'exp/r17_integration/common/layers/recipe.py'
    row['frozen_layout_recipe_has_serve_loeo_episodes'] = 'self.serve_loeo_episodes' in layer.read_text()
    spec = dict(name=name, model=c['model'], bootstrap=str(bootstrap), marker=MARKER, selection=c['filter_failed_knob']['selection'],
                bridge=bool(c.get('ffk_dp_capacity_bridge')), benchmark=str(Path(c['local_assets'])/'benchmark.json'),
                model_spec=str(Path(c['local_assets'])/'model_spec.json'),
                full_yaml=str(Path(c['local_assets'])/'full.yaml'), bundles=bundles(root, c), decisions=decisions)
    row['planned_bundle_tasks'] = sum(len(b['tasks']) for b in spec['bundles'])
    code = inspect.getsource(child)+'\nchild('+repr(spec)+')\n'
    packed = base64.b64encode(zlib.compress(code.encode())).decode()
    command = 'import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))'
    python = c['server_python']
    if not Path(python).exists():
        python = LOCAL_INTERPRETER[python]
        row['interpreter_substitute'] = dict(server=c['server_python'], local=python)
    env = dict(os.environ)
    env.update(f.server_env(c, str(runtime)))
    env.update(CUDA_VISIBLE_DEVICES='', JAX_PLATFORMS='cpu', PYTHONDONTWRITEBYTECODE='1')
    row['python'] = python
    io_ready()
    result = subprocess.run([python, '-c', command], cwd=runtime, env=env, capture_output=True, text=True,
                            timeout=timeout)
    lines = [l for l in result.stdout.splitlines() if l.startswith('RECIPE_SMOKE_RESULT ')]
    if not lines:
        row.update(status='FAIL', errors=[dict(error='child produced no result', returncode=result.returncode,
                                               stderr=result.stderr[-3000:])])
        return row
    row.update(json.loads(lines[-1][len('RECIPE_SMOKE_RESULT '):]))
    if row['status'] == 'PASS' and row['cuda_initialized']:
        row.update(status='FAIL', errors=[dict(error='CUDA initialized in a CPU smoke')])
    if row['status'] == 'PASS' and not c['filter_failed_knob']['selection']:
        # Unfiltered: the knob wrapper must equal the native frozen query,
        # i.e. no planned task may reach the empty-candidate branch.
        empty = [(b['bundle_id'], t) for b in row['bundles'] for t in b['empty_tasks']]
        if empty or any(b['empty_candidate_heads'] for b in row['bundles']):
            row.update(status='FAIL', errors=[dict(error='unfiltered campaign reached empty-candidate MISS',
                                                   tasks=empty[:20])])
    row['returncode'] = result.returncode
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--name', action='append', required=True)
    p.add_argument('--bootstrap', type=Path, help='negative control: use this bootstrap file instead')
    p.add_argument('--preview-repin', action='store_true', help='use the bootstrap vfix.repin would pin now')
    p.add_argument('--decisions', type=int, default=4)
    p.add_argument('--out', type=Path, default=HERE/'RECIPE_SMOKE.json')
    a = p.parse_args()
    SCRATCH.mkdir(exist_ok=True)
    rows = []
    for name in a.name:
        try:
            row = smoke(name, bootstrap=a.bootstrap, preview=a.preview_repin, decisions=a.decisions)
        except Exception as exc:
            row = dict(name=name, status='FAIL', errors=[dict(error=repr(exc))])
        rows.append(row)
        summary = {k: row.get(k) for k in ('name', 'status', 'bootstrap', 'planned_bundle_tasks',
                                            'frozen_layout_recipe_has_serve_loeo_episodes')}
        summary['episodes'] = sum(b.get('episodes', 0) for b in row.get('bundles', []))
        summary['deliveries'] = sum(b.get('deliveries', 0) for b in row.get('bundles', []))
        summary['first_error'] = (row.get('errors') or [{}])[0].get('error')
        print(json.dumps(summary), flush=True)
    dump(a.out, dict(status='PASS' if all(r['status'] == 'PASS' for r in rows) else 'FAIL', cpu_only=True,
                     cuda_visible_devices='', rows=rows))


if __name__ == '__main__':
    main()
