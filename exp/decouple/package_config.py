"""Build and validate the local GPU package without starting any process.

Only frozen R16 definitions/fits and B init-state definitions are read. No
outcome, listener, GPU, or remote access is needed by this module.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shlex
import shutil

from exp.decouple.package_deadline import DEFAULT_DEADLINE, parse_deadline

REPO = Path(__file__).resolve().parents[2]
MAIN = Path('/home/weiland/projects/openpi')
R16 = MAIN / 'exp/offline_search/rounds/r16'
TRACE = Path('/home/weiland/trace_runs/os_closed_loop')
PYTHON = str(MAIN / '.venv/bin/python')
GROOT_PYTHON = '/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python'
GROOT = '/home/weiland/projects/openpi_ext/third_party/gr00t_n15'
ISLAND = '/scratch/zixuans8/openpi_r16_conductor_v1'
PI_CHECKPOINT = '/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch'
GR_CHECKPOINT = '/data/ckpt/n15_libero_spatial'
PUBLIC_PORT, BACKEND_PORT, DRIVER_PORT = 23100, 23400, 23179


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':')).encode()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.new')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def model_env(model):
    paths = [str(REPO), str(REPO / 'src'), str(REPO / 'packages/openpi-client/src')]
    if model == 'groot':
        paths = [str(REPO / 'exp/decouple/_vendor'), GROOT, GROOT + '/examples/Libero', *paths]
    return dict(PYTHONPATH=':'.join(paths), PYTHONDONTWRITEBYTECODE='1',
                OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', NO_ALBUMENTATIONS_UPDATE='1',
                BATCHING_MAX_BATCH_SIZE='1', BATCHING_MAX_WAIT_MS='0',
                STAGE1_ONLY='0', GROOT_DENOISING_STEPS='1', OPENPI_MONITOR_LEVEL='BASIC')


def resident_command(case):
    # The same pure argv builder called by R16 local_server.command. That
    # latter function also mkdirs a runtime, so preparation does not call it.
    from exp.offline_search.closed_loop.ops.h100.node import command
    row = case['row']
    plugin = ['--os-method', row['method'], '--os-kwargs', json.dumps(row['kwargs']),
              '--os-cell', row['cell'], *row['plugin_args']]
    argv, _ = command(case['model'], row['suite'], '{backend_port}', '{out}/config/' + row['arm'] + '.yaml',
                      '{out}/runs/bootstrap', row['arm'], plugin,
                      dict(STAGE1_ONLY='0', GROOT_DENOISING_STEPS='1',
                           PI05_CKPT=PI_CHECKPOINT, GROOT_CKPT=GR_CHECKPOINT, OMP='1'))
    prefix = [case['python'], '-B', '-m', 'exp.offline_search.rounds.r16.resident',
              '--registry', '{out}/group.json', '--server-tag', '{backend_port}']
    if case['model'] == 'groot':
        prefix += ['--resident-suite', 'libero_spatial']
    return [*prefix, *argv[2:]]


def build(seed=2026100801, *, smoke=False):
    from exp.decouple.package_assets import prepare_assets
    asset = prepare_assets(REPO,MAIN/'assets')
    roots = json.loads((R16 / 'test_roots.json').read_text())
    frozen = json.loads((R16 / 'dev_results_conductor/levels.json').read_text())
    deployment = json.loads((R16 / 'conductor_deployment/manifest.json').read_text())
    cases = {}
    inputs = {}
    for path in [R16 / 'test_roots.json', R16 / 'dev_results_conductor/levels.json',
                 R16 / 'dev_results_conductor/TABLE_FREEZE.json',
                 R16 / 'conductor_deployment/manifest.json']:
        inputs[str(path)] = sha(path)
    for model in ('groot', 'pi05'):
        cell = f'{model}_spatial_100_k1'
        record = next(r for r in roots if r['cell'] == cell)
        source = Path(record['root'])
        # Definitions only. In particular, never open runs/ or A journals.
        rows = json.loads((source / 'arms.json').read_text())
        assert len(rows) == 1
        row = copy.deepcopy(rows[0])
        assert row['kwargs']['task_levels'] == frozen[cell]
        fit = row['plugin_args'][row['plugin_args'].index('--os-fit-artifact') + 1]
        assert sha(fit) == record['fit_sha256'][fit]
        for name, digest in record['definitions_sha256'].items():
            assert sha(source / name) == digest, name
        for path in [source / 'ROOT.json', source / 'arms.json', Path(row['yaml']), Path(fit)]:
            inputs[str(path)] = sha(path)
        rng = random.Random(seed)  # same cohort across models when fit subsets agree
        pairs, warmup = [], []
        for task in range(10):
            fit_inits = {int(e) - task * 50 for e in row['kwargs']['episode_subset'][str(task)]}
            available = [i for i in range(50) if i not in fit_inits]
            rng.shuffle(available)
            assert len(available) >= 31
            cohort = sorted(available[:30])
            pairs.extend([[task, i] for i in (cohort[:2] if smoke else cohort)])
            warmup.append([task, available[30]])
        # Task 0 exercises every GPU stage; the remaining warmups exercise
        # each frozen cache/periodic branch without ten pure-policy episodes.
        assert row['kwargs']['task_levels']['0'] == 'c10'
        row.update(arm=f'decouple_{cell}_frozen', dev=True, init_pool='B', episodes=20 if smoke else 300)
        manifest = dict(schema='decouple.b_manifest.v1', model=model, suite='libero_spatial',
                        init_pool='B', dev=True, seed=seed, simulator_seed=2026101401,
                        per_task=2 if smoke else 30, pairs=pairs, warmup_pairs=warmup,
                        frozen_cell=cell, fit_sha256=sha(fit), task_levels=frozen[cell])
        dev_group = json.loads((TRACE / f'r16_conductor_dev_{model}_k1_v1/group.json').read_text())
        case = dict(model=model, python=GROOT_PYTHON if model == 'groot' else PYTHON,
                    checkpoint=GR_CHECKPOINT if model == 'groot' else PI_CHECKPOINT,
                    row=row, yaml=Path(row['yaml']).read_text(), manifest=manifest,
                    manifest_sha256=hashlib.sha256(canonical(manifest)).hexdigest(),
                    source_root=str(source), source_seal_sha256=sha(source / 'ROOT.json'),
                    fit_sha256=sha(fit), pool_digests=dev_group['pool_digests'],
                    environment=model_env(model), resident_command=[])
        case['resident_command'] = resident_command(case)
        cases[model] = case
    inputs[str(R16 / 'conductor_deployment/worker.sha256')] = sha(R16 / 'conductor_deployment/worker.sha256')
    for name in ('design.json', 'PROTOCOL.md', 'CONDUCTOR_AMENDMENT.json'):
        assert sha(R16 / name) == sha(REPO / 'exp/offline_search/rounds/r16' / name)
        inputs[str(REPO / 'exp/offline_search/rounds/r16' / name)] = sha(R16 / name)
    cfg = dict(schema='decouple.gpu_package.v1', repo=str(REPO), seed=seed,
               gpu_memory_budget_mb=46000, expected_gpu_total_mb_min=48000,
               gpu_index=0, cpus='10-17,54-61', workers=64,
               public_port=PUBLIC_PORT, backend_port=BACKEND_PORT, driver_port=DRIVER_PORT,
               advertised_host='ziyanglin.com', trace_parent=str(TRACE),
               worker_island=ISLAND, worker_python='/scratch/zixuans8/openpi/.venv/bin/python',
               worker_sha256=(R16 / 'conductor_deployment/worker.sha256').read_text(),
               worker_environment=dict(LIBERO_CONFIG_PATH=ISLAND + '/os_cl/libero',
                   OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   PYTHONPATH=ISLAND + '/packages/openpi-client/src:' + ISLAND + '/src:' + ISLAND,
                   PYTHONDONTWRITEBYTECODE='1', PATH='/scratch/zixuans8/dsp_bin:/usr/local/bin:/usr/bin:/bin'),
               inputs_sha256=inputs, cases=cases, initial_arms=['current_1','split_8'] if smoke else ['current_1', 'current_max', 'split_8', 'split_4', 'split_16'],
               mode='smoke' if smoke else 'full', benchmark_models=['groot'] if smoke else ['groot','pi05'],
               repeats=0 if smoke else 2, batch_size=1, wait_ms=0, wall_limit_s=600 if smoke else 7200,
               wall_deadline=DEFAULT_DEADLINE, shutdown_reserve_s=120,
               admission_safety_factor=1.15, arm_setup_estimate_s=60, arm_cleanup_estimate_s=100,
               assets=dict(path=str(asset),sha256=sha(asset)), expected_episodes=20 if smoke else 300,
               expected_wall_minutes=8 if smoke else 116,
               estimate_assumptions=dict(groot_eps_min=[10,65] if smoke else [10,65,65,50,65],
                                         pi05_eps_min=[40,100] if smoke else [40,100,100,80,100],
                                         startup_gate_warmup_minutes=15),
               source_argv='R16 node.command + conductor_runner.resident_command + conductor agent; unchanged worker protocol',
               worker_changes=[], local_validation_only=True)
    # Resume never mixes measurements/gates produced by different serving or
    # package code. Hash source only, not generated results or environments.
    sources = []
    for directory in ('src/openpi', 'scripts', 'packages/openpi-client/src',
                      'exp/decouple', 'exp/libero_groot', 'exp/offline_search'):
        sources.extend((REPO / directory).rglob('*.py'))
    cfg['source_sha256'] = {str(p):sha(p) for p in sorted(set(sources)) if '__pycache__' not in p.parts
                            and not p.is_relative_to(REPO / 'exp/decouple/results')}
    cfg['source_sha256'][str(REPO/'exp/decouple/run_gpu_package.sh')] = sha(REPO/'exp/decouple/run_gpu_package.sh')
    cfg['inputs_sha256'][str(asset)] = sha(asset)
    from exp.decouple.gate_cache import GateCache, input_signatures
    cfg['gate_inputs'] = input_signatures(cfg)
    for previous_path in (REPO/'exp/decouple/benchmark.local.json', REPO/'exp/decouple/benchmark.smoke.local.json'):
        if previous_path.is_file():
            GateCache().reindex_launcher(cfg, json.loads(previous_path.read_text()))
    cfg['arms'] = [make_arm(cfg, model, kind, f'decouple_bench_prepared_{model}_{kind}')
                   for model in cfg['benchmark_models'] for kind in cfg['initial_arms']]
    cfg['config_sha256'] = hashlib.sha256(canonical(cfg)).hexdigest()
    return cfg


def validate(config, *, check_inputs=True):
    cfg = copy.deepcopy(config)
    digest = cfg.pop('config_sha256')
    if hashlib.sha256(canonical(cfg)).hexdigest() != digest:
        raise ValueError('changed package configuration')
    if cfg['schema'] != 'decouple.gpu_package.v1' or cfg['repo'] != str(REPO):
        raise ValueError('wrong schema/worktree')
    smoke = cfg['mode']=='smoke'
    assert cfg['mode'] in ('full','smoke')
    assert cfg['workers'] == 64 and cfg['batch_size'] == 1 and cfg['repeats'] == (0 if smoke else 2)
    assert cfg['initial_arms'] == (['current_1','split_8'] if smoke else ['current_1', 'current_max', 'split_8', 'split_4', 'split_16'])
    assert cfg['benchmark_models'] == (['groot'] if smoke else ['groot','pi05'])
    assert cfg['public_port'] == 23100 and cfg['driver_port'] == 23179 and cfg['backend_port'] == 23400
    ports = [cfg[k] for k in ('public_port', 'backend_port', 'driver_port')]
    ports += list(range(cfg['backend_port'] + 1, cfg['backend_port'] + 17))
    assert len(set(ports)) == len(ports) and all(0 < p < 65536 for p in ports)
    assert cfg['wall_limit_s'] == (600 if smoke else 7200) and cfg['shutdown_reserve_s'] >= 120
    assert cfg['wall_limit_s'] > cfg['shutdown_reserve_s']
    parse_deadline(cfg['wall_deadline'])
    assert cfg['admission_safety_factor'] >= 1 and cfg['arm_setup_estimate_s'] > 0 and cfg['arm_cleanup_estimate_s'] >= 100
    assert all(len(cfg['estimate_assumptions'][model + '_eps_min']) == len(cfg['initial_arms']) and
               all(rate > 0 for rate in cfg['estimate_assumptions'][model + '_eps_min']) for model in ('groot', 'pi05'))
    assert cfg['gpu_memory_budget_mb'] < cfg['expected_gpu_total_mb_min']
    assert cfg['trace_parent'] == str(TRACE)
    assert cfg['worker_island'] == ISLAND and not cfg['worker_changes']
    assert len(cfg['arms']) == (2 if smoke else 10)
    for arm in cfg['arms']:
        assert arm == make_arm(cfg, arm['model'], arm['logical_arm'], arm['name'])
    for model, case in cfg['cases'].items():
        manifest, row = case['manifest'], case['row']
        assert model in ('pi05', 'groot') and manifest['init_pool'] == row['init_pool'] == 'B'
        assert row['dev'] and manifest['dev'] and len(manifest['pairs']) == cfg['expected_episodes'] == (20 if smoke else 300)
        assert len({tuple(p) for p in manifest['pairs']}) == cfg['expected_episodes']
        assert all(sum(p[0] == t for p in manifest['pairs']) == (2 if smoke else 30) for t in range(10))
        assert not set(map(tuple, manifest['warmup_pairs'])) & set(map(tuple, manifest['pairs']))
        for t, i in manifest['pairs'] + manifest['warmup_pairs']:
            assert 0 <= t < 10 and 0 <= i < 50
            assert t * 50 + i not in row['kwargs']['episode_subset'][str(t)]
        assert hashlib.sha256(canonical(manifest)).hexdigest() == case['manifest_sha256']
        assert row['kwargs']['task_levels'] == manifest['task_levels']
        assert case['environment']['BATCHING_MAX_BATCH_SIZE'] == '1'
        assert '--stage1-only' not in case['resident_command']
        if check_inputs:
            for path in (case['python'], case['checkpoint']):
                if not Path(path).exists():
                    raise FileNotFoundError(path)
            import importlib.util
            assert importlib.util.find_spec(row['method'].partition(':')[0])
    if check_inputs:
        from exp.decouple.package_assets import validate_assets
        validate_assets(cfg['assets']['path'])
        if sha(cfg['assets']['path']) != cfg['assets']['sha256']:
            raise ValueError('norm stats changed since config generation')
        assert Path(PYTHON).is_file() and Path(GROOT).is_dir()
        assert (REPO / 'exp/decouple/_vendor/cloudpickle/__init__.py').is_file()
        assert shutil.which('tether')
        for path, expected in {**cfg['inputs_sha256'], **cfg['source_sha256']}.items():
            if sha(path) != expected:
                raise ValueError(f'changed frozen input: {path}')
        import yaml
        from openpi.cache.config import load_cache_config
        for case in cfg['cases'].values():
            # Parse and check actual paths; do not instantiate a serving policy.
            load_cache_config(case['row']['yaml'])
            parsed = yaml.safe_load(case['yaml'])
            assert parsed['miss']['num_steps'] == 1
        from exp.decouple.recorded import load
        for model in ('pi05', 'groot'):
            assert len(load(model, limit=16).rows) == 16
        from exp.decouple.gate_cache import input_signatures
        if input_signatures(cfg) != cfg['gate_inputs']:
            raise ValueError('model/recorded/runtime qualification inputs changed; regenerate config and requalify')
    return dict(passed=True, models=list(cfg['cases']), mode=cfg['mode'],accepted_episodes_per_arm=cfg['expected_episodes'],
                assets_checked=cfg['assets'],
                warmup_episodes_per_arm=10, workers=64, manifest_identity_per_model=True,
                remote_contacted=False, gpu_used=False, listeners_created=False,
                remote_paths='checked against sealed R16 deployment manifest; live SHA checks deferred to GPU package',
                ports=ports, expected_wall_minutes=cfg['expected_wall_minutes'])


def materialize(case, out):
    """Called only inside a fresh benchmark attempt (or worktree dry-run)."""
    import yaml
    out = Path(out)
    parsed = yaml.safe_load(case['yaml'])
    parsed['miss']['evidence_dir'] = str(out / 'evidence')
    yaml_path = out / 'config' / (case['row']['arm'] + '.yaml')
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(parsed, sort_keys=False))
    manifest_path = out / 'episodes.json'
    write_json(manifest_path, case['manifest'])
    row = copy.deepcopy(case['row'])
    row.update(yaml=str(yaml_path), manifest=str(manifest_path))
    here = REPO / 'exp/offline_search/rounds/r16'
    group = dict(schema='r16.conductor.group.v1', root=str(out), phase='benchmark', model=case['model'], k=1,
                 entries=[dict(row=row, source_root=case['source_root'], source_seal_sha256=case['source_seal_sha256'],
                               fit_sha256=case['fit_sha256'], yaml_sha256=sha(yaml_path), pairs=case['manifest']['pairs'])],
                 definitions_sha256={str(p): sha(p) for p in (yaml_path, manifest_path)},
                 design_sha256=sha(here / 'design.json'), protocol_sha256=sha(here / 'PROTOCOL.md'),
                 amendment_sha256=sha(here / 'CONDUCTOR_AMENDMENT.json'), references=None,
                 pool_digests=case['pool_digests'], recording=False, simulator_seed=2026101401,
                 init_pool='B', benchmark_manifest_sha256=case['manifest_sha256'])
    group['group_sha256'] = hashlib.sha256(canonical(group)).hexdigest()
    write_json(out / 'group.json', group)
    return group


def make_arm(config, model, kind, name):
    case = config['cases'][model]
    resident = case['resident_command']
    split = kind.startswith('split_')
    fronts = int(kind.split('_')[1]) if split else None
    if split:
        server = [case['python'], '-B', '-m', 'openpi.serving.split', '--model', model,
                  '--fronts', str(fronts), '--port', '{backend_port}', '--host', '127.0.0.1',
                  '--batch-size', '1', '--wait-ms', '0',
                  '--runtime-parent', '{out}', '--log-root', '{out}/split_logs',
                  '--gpu-python', case['python'], '--front-python', case['python'], '--', *resident[2:]]
    else:
        server = [PYTHON, '-B', '-m', 'exp.decouple.current_fleet', '--model', model,
                  '--copies', 'auto' if kind == 'current_max' else '1',
                  '--port', '{backend_port}', '--budget-mb', str(config['gpu_memory_budget_mb']),
                  '--out', '{out}', '--', *resident]
    return dict(name=name, logical_arm=kind, model=model, split=split, fronts=fronts,
                model_processes=1 if kind != 'current_max' else 'auto',
                port=config['public_port'], backend_port=config['backend_port'],
                server_command=['taskset', '-c', config['cpus'], *server],
                server_env={**case['environment'], 'HF_MODULES_CACHE': '{out}/hf_modules',
                            'TORCHINDUCTOR_CACHE_DIR': '{out}/inductor', 'TRITON_CACHE_DIR': '{out}/triton'},
                worker_command=[PYTHON, '-B', '-m', 'exp.decouple.package_driver',
                                '--config', str(REPO / 'exp/decouple/benchmark.local.json'),
                                '--model', model, '--out', '{out}', '--port', '{port}'],
                journal='{out}/journal.jsonl', measurement_json='{out}/measurement.json',
                expected_episodes=len(case['manifest']['pairs']), manifest_sha256=case['manifest_sha256'],
                case=case, startup_timeout_s=900, worker_timeout_s=2700)


def worker_command(config, model, lease, token):
    endpoint = f"{config['advertised_host']}:{config['public_port']}"
    if model == 'groot':
        endpoint += '=libero_spatial'
    return [config['worker_python'], '-B', '-m', 'exp.offline_search.rounds.r16.conductor', 'agent',
            '--model', model, '--phase', 'dev', '--servers', endpoint,
            '--driver-host', config['advertised_host'], '--port', str(config['driver_port']),
            '--workers', '64', '--lease-file', lease, '--lease-token', token]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPO / 'exp/decouple/benchmark.local.json')
    parser.add_argument('--seed', type=int, default=2026100801)
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--smoke',action='store_true',help='prepare 20-episode current/1 versus split/8 GR00T smoke')
    args = parser.parse_args()
    if args.smoke and args.out == REPO/'exp/decouple/benchmark.local.json':
        args.out = REPO/'exp/decouple/benchmark.smoke.local.json'
    config = json.loads(args.out.read_text()) if args.validate else build(args.seed,smoke=args.smoke)
    report = validate(config)
    if not args.validate:
        write_json(args.out, config)
    write_json(REPO / ('exp/decouple/results/package_validation_smoke.json' if config['mode']=='smoke' else
                      'exp/decouple/results/package_validation.json'), report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
