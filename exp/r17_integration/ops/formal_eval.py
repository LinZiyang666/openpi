"""Formal SR campaign: resident R17 replicas, stock proxy, episode conductor.

Full campaigns start only with --run. --prepare-only never invokes CUDA.
--smoke --run explicitly runs four complete, original-protocol episodes.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from functools import wraps
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import uuid

from .formal_process import alive, stop_group, ticks, write_json
from .tether_retry import campaign_retries, current_journal, retry, retry_scope, LEASE_SECONDS, RETRY_SECONDS

ROOT = Path(__file__).resolve().parents[3]
CONTROL_PYTHON = str(ROOT / '.venv/bin/python')
FORMAL_ROOT = Path('/data/mw_finetune/eval')
PROBE_SECONDS = 30
WORKER_ROOT = '/scratch/zixuans8/mw_r17/formal'
DEFAULT_CHECKPOINTS = {
    'dp_libero': '/data/mw_finetune/dp/libero_policy_100000',
    'dp_mw': '/data/mw_finetune/dp/mt50_policy_100000',
    'fastwam_libero': '/data/mw_finetune/models/fastwam_libero',
    'fastwam_mw': '/data/mw_finetune/fastwam/mt50_frozen_b128/policy_016001',
    'groot_mw': '/data/mw_finetune/groot/4090_mt50/checkpoint-60000',
    'pi05_mw': '/data/ckpt/pi05_metaworld_rlinf',
    'pi05_libero': '/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch',
    'groot_libero': '/data/ckpt',
    'cosmos_libero': '/data/mw_finetune/models/cosmos_libero',
    'cosmos_mw': '/data/mw_finetune/models/cosmos_mw',
}


def execute(argv, seconds=60, env=None, cwd=None):
    if argv and Path(str(argv[0])).name=='tether' and current_journal.get():
        node=argv[argv.index('--')-1] if '--' in argv else None
        current_journal.get().record_call(' '.join(map(str,argv[:2])),node)
    journal = current_journal.get()
    if journal:
        from .formal_prestage import phase, command_phase
        with phase(journal.directory, command_phase(argv)):
            result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=seconds, env=env, cwd=cwd)
    else:
        result = subprocess.run(list(map(str, argv)), capture_output=True, text=True, timeout=seconds, env=env, cwd=cwd)
    if result.returncode:
        exc = RuntimeError(f'command failed ({result.returncode}): {shlex.join(list(map(str, argv[:8])))}\n'
                           + result.stdout[-3000:] + result.stderr[-4000:])
        exc.stdout, exc.stderr, exc.returncode = result.stdout, result.stderr, result.returncode
        raise exc
    return result.stdout


def remote(node, argv, seconds=60, *, replay_safe=False):
    if node == 'timan108':raise ValueError('timan108 is disabled; no probe allowed')
    if node == 'local':
        return execute(argv, seconds=seconds)
    node = physical_node(node)
    from .tether_retry import transient
    label = 'exec '+(' '.join(argv[argv.index('-m')+1:argv.index('-m')+3]) if '-m' in argv else str(argv[0]))
    spawn_request = None
    if '--import-probe' in argv or argv[:2] == ['mkdir','-p']:
        replay_safe = True
    if 'exp.r17_integration.ops.formal_process' in argv:
        mode = argv[argv.index('exp.r17_integration.ops.formal_process')+1]
        replay_safe = mode in ('status','renew','stop')
        if mode == 'spawn' and '--request' in argv:
            spawn_request = json.loads(argv[argv.index('--request')+1])
    lost_spawn = None
    def operation():
        nonlocal lost_spawn
        if lost_spawn is not None:
            # A lost spawn reply is ambiguous: recover its exact receipt,
            # never execute spawn again or infer that another fleet died.
            prefix = argv[:argv.index('-m')]
            code = ('from pathlib import Path; p=Path('+repr(spawn_request['receipt'])+'); '
                    'print(p.read_text() if p.exists() else "null")')
            value = execute(['tether','exec','--safe','--timeout','10s',node,'--',*prefix,'-c',code],20)
            receipt = json.loads(value)
            if receipt is None:
                raise lost_spawn
            if receipt.get('request') != spawn_request:
                raise ValueError('lost spawn receipt belongs to a different request')
            return value
        try:
            return execute(['tether', 'exec', '--safe', '--timeout', str(seconds)+'s', node, '--', *argv], seconds+10)
        except Exception as exc:
            if spawn_request is not None and transient(exc,replay_safe=True) and not transient(exc):
                lost_spawn = exc
            raise
    return retry(operation, node=node, label=label, replay_safe=replay_safe or spawn_request is not None)


def transfer(argv, *, heartbeat=lambda: None):
    if any(str(value).startswith('timan108:') for value in argv):
        raise ValueError('timan108 is disabled; no transfer allowed')
    from .direct_transfer import transfer as rsync_transfer
    return rsync_transfer(argv,remote=remote,execute=execute,heartbeat=heartbeat)


# ziyanglin.com forwards TCP 23100-23199 1:1 to weilandserver; h100 has a public IP.
DIRECT_PORTS = range(23100, 23200)
H100_NODE, H100_IP = 'h100', '149.165.153.233'
ALIAS_ENV = ('R17_H100_NODE', 'R17_H100_HOST', 'R17_DRIVER_PORT')


def h100_alias():
    """Process-level stand-in machine for node 'h100'; None means the H100 itself.

    R17_H100_NODE (tether node), R17_H100_HOST (its public IP) and R17_DRIVER_PORT
    (this lane's coordinator driver port) are set together. Every tether call,
    spawn, rsync transfer and release install of this process that names 'h100'
    then reaches that machine. Campaigns keep server_node='h100'.
    """
    values = [os.environ.get(key, '') for key in ALIAS_ENV]
    if not any(values):
        return None
    node, host, port = values
    if not all(values) or node == H100_NODE or not port.isdigit() or int(port) not in DIRECT_PORTS:
        raise ValueError('set '+', '.join(ALIAS_ENV)+' together: alias tether node, its public IP, driver port 23100-23199')
    return dict(node=node, host=host, driver_port=int(port))


def physical_node(node):
    alias = h100_alias()
    return alias['node'] if alias and node == H100_NODE else node


def h100_host():
    alias = h100_alias()
    return alias['host'] if alias else H100_IP


H100_HOST = h100_host()


def check_execution_host(campaign):
    """An H100 campaign rebound to an alias machine runs only under that alias, and vice versa."""
    bound = campaign.get('execution_host', H100_NODE)
    if campaign.get('server_node') == H100_NODE and bound != physical_node(H100_NODE):
        raise ValueError(f'campaign is bound to execution host {bound}, but this process serves h100 on '
                         f'{physical_node(H100_NODE)}; set/unset '+', '.join(ALIAS_ENV))


def close_tunnels(tunnels, directory):
    """Kept for call sites: workers connect directly, nothing is exposed through the broker."""


def open_tunnels(campaign, directory):
    """Direct endpoints only; never `tether expose` (owner rule: no data through the VPS broker).

    Local workers always use 127.0.0.1 (local_workers.route); remote workers reach the H100
    server at its public IP and weilandserver services through ziyanglin.com:23100-23199."""
    if campaign.get('resume_source_closure_waiver'):
        from .formal_resume_waiver import refresh
        refresh(campaign,directory)
    remote = [n for n in campaign.get('execution_worker_nodes', campaign['worker_nodes']) if n != 'local']
    port = campaign['public_port']
    if campaign['server_node'] == 'local':
        host = 'ziyanglin.com' if remote else '127.0.0.1'
        if remote and port not in DIRECT_PORTS:
            raise ValueError(f'remote workers need the local server port inside 23100-23199, got {port}')
    else:
        host = h100_host()
    if remote and campaign['driver_port'] not in DIRECT_PORTS:
        raise ValueError(f"remote workers need the driver port inside 23100-23199, got {campaign['driver_port']}")
    campaign.update(endpoint_host=host, endpoint_port=port, endpoint=f'{host}:{port}',
                    driver_host='ziyanglin.com', driver_public_port=campaign['driver_port'])
    if campaign.get('resume_source_closure_waiver'):
        from .formal_resume_waiver import validate_destination, ports, now, record_selection
        proof=campaign['resume_source_closure_waiver']
        validate_destination(campaign,proof)
        proof['destination'].update(all_ports=ports(campaign),endpoint=campaign['endpoint'],
            driver_host=campaign['driver_host'],driver_public_port=campaign['driver_public_port'],isolated_at=now())
        record_selection(campaign,directory)
    return []


def pythonpath(root, extra=''):
    # R16 order: client, src, root; private simulator dependencies follow.
    return ':'.join(filter(None, [root+'/packages/openpi-client/src', root+'/src', root, extra]))


def benchmark(official=False):
    from exp.r17_integration.conductor.manifest import load_benchmark_spec
    spec = load_benchmark_spec(ROOT / 'exp/r17_integration/qualification/specs/libero_benchmark_spec.json')
    if official:
        spec = replace(spec, formal_indices=tuple(range(50)), library_indices=(), dev_indices=(),
                       protocol_id='libero_four_suites_official50_r17_v1')
    return spec


def execution_horizon(campaign, layout):
    """Opt-in policy commitment; the benchmark's transport cadence stays fixed."""
    requested = campaign.get('commit_horizon')
    if requested is None:
        return layout['replan_steps']
    supported = {'libero': ('dp', 'fastwam', 'cosmos', 'pi05', 'groot'), 'metaworld': ('dp', 'fastwam', 'pi05', 'groot')}
    if campaign.get('model') not in supported.get(campaign.get('domain'), ()) or campaign.get('library'):
        raise ValueError('commit-horizon study requires a supported plain model')
    if type(requested) is not int or not layout['replan_steps'] <= requested <= layout['horizon']:
        raise ValueError('commit-horizon must be within native replan and predicted chunk length')
    if campaign.get('domain') == 'metaworld' and requested % layout['replan_steps']:
        raise ValueError('MetaWorld commit must be divisible by native replan steps')
    return requested


def execution_steps(campaign, default_steps):
    """Plain step studies use the provider's native schedule, never a truncated loop."""
    requested = campaign.get('denoising_steps')
    if requested is None:
        return default_steps
    choices = {'dp': (5, 10, 100), 'fastwam': (1, 2, 10), 'cosmos': (1, 2, 3, 5),
               'pi05': (1, 2, 3, 4, 5, 10), 'groot': (1, 2, 3, 4)}
    if campaign.get('domain') == 'metaworld':
        choices = {'dp': (5, 10, 100), 'fastwam': (1, 2, 3, 4, 5, 10), 'pi05': (1, 2, 3, 4, 5, 10), 'groot': (1, 2, 3, 4)}
    if (campaign.get('domain') not in ('libero', 'metaworld') or campaign.get('library') or
        type(requested) is not int or requested not in choices.get(campaign.get('model'), ())):
        raise ValueError('plain steps: DP 5/10/100, Fast-WAM 1/2/10 (MW 1-5/10), pi05 1-5/10, GR00T 1-4, LIBERO Cosmos 1/2/3/5 (native K2 is NFE1)')
    if campaign.get('commit_horizon') is not None or campaign.get('native_parity_only'):
        raise ValueError('plain step study cannot combine horizon or parity-only overrides')
    return requested


def selected_suites(campaign):
    """Select episodes after constructing the original plan, preserving every UID."""
    suites = campaign.get('suites')
    if suites is None:
        return None
    from exp.r17_integration.benchmarks.libero_bridge import SUITES
    if (campaign.get('domain') != 'libero' or not campaign.get('official_50') or
        campaign.get('library') or campaign.get('smoke') or campaign.get('native_parity_only') or
        not isinstance(suites, (list, tuple)) or not suites or
        any(s not in SUITES for s in suites) or len(suites) != len(set(suites))):
        raise ValueError('suites require unique LIBERO suites in a full official-50 plain campaign')
    return tuple(suites)


def full_config(provider, campaign):
    from dataclasses import asdict
    from exp.r17_integration.common.qualification import config_for_probe
    from exp.r17_integration.common.provider import binding_for
    model = provider.model_spec
    k = execution_steps(campaign, model.default_sampler.num_steps)
    config = config_for_probe(provider, k, model.layout.replan_steps)
    if campaign.get('commit_horizon') is not None:
        commit = execution_horizon(campaign, asdict(model.layout))
        config.adapter_binding = binding_for(provider, model.allowed_benchmark_protocols[0],
            num_steps=k, commit_horizon=commit,
            allow_partial_commit=bool(commit % model.layout.replan_steps))
    config.enabled = False
    config.checkpoints = {}
    config.recipe.enabled = False
    return config


def horizon_execution_verified(rows, campaign, model):
    """Verify full policy calls and zero-NFE tails in each accepted episode."""
    commit = execution_horizon(campaign, model['layout'])
    replan = model['layout']['replan_steps']
    k = execution_steps(campaign, model['default_sampler']['num_steps'])
    expected_nfe = k
    if campaign.get('model') == 'cosmos' and k == 2:
        from exp.r17_integration.models.cosmos.manifest import make_sampler
        from exp.r17_integration.common.contracts import digest
        sampler = make_sampler(k)
        sealed = json.loads((Path(campaign['local_assets']) / 'execution_sampler.json').read_text())
        if digest(sealed) != sampler.identity:
            return False
        expected_nfe = sampler.config['expected_nfe']
    slots = (commit + replan - 1) // replan
    episodes = {}
    for row in rows:
        episodes.setdefault(row.get('task_uid'), []).append(row)
    for episode in episodes.values():
        for seq, row in enumerate(sorted(episode, key=lambda r: r.get('decision_seq', -1))):
            cursor = (seq % slots) * replan
            source, calls, nfe = ('policy', [1, 1, expected_nfe], expected_nfe) if cursor == 0 else ('policy_tail', [0, 0, 0], 0)
            if (row.get('decision_seq') != seq or row.get('cursor') != cursor or
                row.get('commit_horizon') != commit or row.get('capture_enabled') is not False or
                row.get('source') != source or row.get('stage_calls') != calls or row.get('actual_nfe') != nfe):
                return False
    return bool(rows)


def seal(campaign):
    """Host-native CPU identity seal; no CUDA allocation or model on GPU."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise ValueError('CPU sealing must hide CUDA')
    from exp.r17_integration.qualification.assets import model_assets
    from exp.r17_integration.common.config import validate_cache_config
    from exp.r17_integration.common.benchmarks import register_benchmark
    from dataclasses import asdict
    import yaml
    assets = Path(campaign['assets'])
    assets.mkdir(parents=True, exist_ok=True)
    if campaign['domain'] == 'libero':
        spec = benchmark(campaign['official_50'])
        if campaign['model'] == 'cosmos':
            from exp.r17_integration.models.cosmos.protocols import benchmark_profile
            spec = benchmark_profile(spec, campaign['official_50'])
    else:
        from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
        spec = FORMAL_SPEC
    register_benchmark(spec)
    write_json(assets / 'benchmark.json', spec.manifest())
    job = dict(campaign, benchmark_spec=str(assets / 'benchmark.json'), diagnostic=False)
    model, provider = model_assets(job, assets)
    if campaign.get('library'):
        model.capabilities.require('debug_capture')
    config = full_config(provider, campaign)
    selected_suites(campaign)
    if campaign.get('denoising_steps') is not None:
        write_json(assets / 'execution_sampler.json', asdict(provider.sampler_for_steps(config.adapter_binding['num_steps'])))
    config.recipe.ordered_tasks = list(spec.ordered_tasks)
    config.recipe.task_table_identity = spec.task_table_identity
    validate_cache_config(config)
    (assets / 'full.yaml').write_text(yaml.safe_dump(asdict(config)))
    if campaign.get('cache_study'):
        from .cache_study import seal
        seal(provider, campaign, spec)
    load_path = assets / 'load.json'
    desc = json.loads(load_path.read_text())
    if campaign['model'] == 'fastwam':
        desc['diagnostic'] = False
        write_json(load_path, desc)
    write_json(assets / 'manifest.json', dict(schema='r17.formal_seal.v1', checkpoint=campaign['checkpoint'],
        model_identity=model.checkpoint_identity, benchmark_protocol=spec.identity,
        cache_enabled=False, debug_capture=bool(campaign.get('library')), full_sampler=asdict(model.default_sampler),
        cuda_graph_required=campaign['model'] == 'dp'))
    if campaign.get('fastwam_batching'):
        from exp.r17_integration.models.fastwam.formal_batching import validate, check_control, receipt_fields
        validate(campaign)
        from .formal_prestage import control_gate
        paired = control_gate(campaign, assets.parent) if campaign['server_node'] == 'local' else None
        write_json(assets / 'batching.json', dict(receipt_fields(campaign), paired_control_seal=paired))
    print(json.dumps({'status': 'PASS', 'assets': str(assets), 'model_identity': model.checkpoint_identity}))


def install_worker_dependencies(node, destination, payload, inherited=None):
    """Publish a verified immutable release; leave legacy live installs alone."""
    if node == 'local':
        from .local_workers import install
        return install(payload)
    if node == 'timan1':
        from .timan1_workers import PYTHON
        return json.loads(remote(node,['env','PYTHONPATH='+pythonpath(payload),PYTHON,'-c',
            'from exp.r17_integration.ops.timan1_workers import install; import json; '
            'print(json.dumps(install('+repr(payload)+')))'],replay_safe=True))
    if inherited and inherited.get('dependencies') and inherited.get('private_config'):
        # Reuse the sealed release/config even when a mutable legacy config
        # was removed by its owner. Never re-install into another live fleet.
        dependency=dict(inherited['dependencies'])
        config=inherited['private_config']
        dependency.update(config_source=config['path']+'/config.yaml',config_sha256=config['sha256'])
        code=('from exp.r17_integration.ops.worker_installs import verify_dependencies,sha; import json; '
              'd='+repr(dependency)+'; verify_dependencies(d); '
              'assert sha(d["config_source"])==d["config_sha256"],"sealed simulator config changed"; '
              'print(json.dumps(d))')
        return json.loads(remote(node,['env','PYTHONPATH='+pythonpath(payload),
            '/scratch/zixuans8/openpi/.venv/bin/python','-c',code],replay_safe=True))
    config = '/home/zixuans8/.libero/config.yaml' if node == 'timan108' else '/scratch/zixuans8/openpi_trace_h100/os_cl/libero/config.yaml'
    code = ('from exp.r17_integration.ops.worker_installs import install_dependencies; import json; '
        'print(json.dumps(install_dependencies("/scratch/zixuans8/libero_sim/lib/python3.8/site-packages",'+
        repr(destination)+','+repr(config)+',assets_manifest='+
        repr(payload+'/exp/r17_integration/qualification/specs/libero_original_assets.json')+
        ',original_assets="/home/zixuans8/.cache/libero/assets")))')
    return json.loads(remote(node, ['env','PYTHONPATH='+pythonpath(payload),
        '/scratch/zixuans8/openpi/.venv/bin/python','-c',code]))


def worker_paths(campaign, node):
    name = campaign['run_name']
    if not name or name in ('.','..') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-' for c in name):
        raise ValueError('campaign name must be a single safe path component')
    if node not in ('timan107', 'timan108', 'local','timan1'):
        raise ValueError('unknown worker node')
    root = WORKER_ROOT
    if node == 'local':
        from .local_workers import ROOT as local_root
        root = str(local_root)
    return dict(stage=(root+'/stage/' if node == 'timan1' else '/tmp/r17_formal/')+name+'/'+node,
        private=root+'/campaigns/'+name+'/'+node,
        run=root+'/runs/'+name+'_'+node)


def _gpu2_coordinator(function):
    """Load worker admission hooks only when starting/preparing a fleet.

    The separately sealed CPU finalizer imports this module for evidence
    functions; it deliberately has no worker-launch packages installed.
    """
    @wraps(function)
    def call(*args, **kwargs):
        from exp.offline_search.rounds.r17.filter_failed import timan1_gpu2
        with timan1_gpu2.coordinator():
            return function(*args, **kwargs)
    return call


@_gpu2_coordinator
def prepare_workers(campaign, directory, *, payload_override=None):
    from .build_client_bundle import worker_payload_files, build_client_bundle
    from exp.offline_search.rounds.r17.autoscale_1017.defaults import apply
    apply(campaign, directory)
    names = worker_payload_files(ROOT)
    if campaign.get('autoscale'):
        from exp.offline_search.rounds.r17.autoscale_1017.bundle import files
        names += files()
    closed_capacity_refresh=bool(campaign.get('cache_capacity_authorization') and campaign.get('prepare_only') is True)
    if closed_capacity_refresh:
        from .cache_capacity import validate_authorization
        validate_authorization(campaign)
        if any((Path(directory)/name).exists() for name in ('LAUNCH.json','RECEIPT.json')):
            pass
    capacity_transfer_options=['--force'] if closed_capacity_refresh else []
    execution_nodes=campaign.get('execution_worker_nodes',campaign['worker_nodes'])
    names += ['exp/r17_integration/ops/'+n+'.py' for n in
              ('formal_process', 'formal_workers', 'formal_resume', 'worker_lifetime', 'worker_gpu', 'worker_disk', 'build_client_bundle', 'sync_back', 'deploy', 'tether_retry', 'remote_json', 'worker_installs', 'worker_memory')]
    names += ['exp/r17_integration/qualification/archive_stage.py']
    if campaign.get('live_fleets'):
        # Include all attachable client routes in the initial immutable pin.
        names += ['exp/r17_integration/ops/'+n+'.py' for n in
                  ('live_fleet_worker','local_workers','local_worker_entry','timan1_workers')]
    if campaign.get('cache_study'):
        names.append('exp/r17_integration/ops/cache_study_worker.py')
    if 'local' in execution_nodes:
        names.append('exp/r17_integration/ops/local_worker_entry.py')
    if 'timan1' in execution_nodes:
        from .timan1_workers import authorize
        campaign['timan1_worker_lease']=authorize(campaign)
        names.append('exp/r17_integration/ops/timan1_workers.py')
    if campaign.get('domain') == 'libero' or not execution_nodes:
        names += ['exp/r17_integration/qualification/specs/libero_original_assets.json']
    if campaign.get('domain') == 'libero' and (campaign.get('library') or campaign.get('eval_pool')=='B'):
        names += ['exp/r17_integration/ops/library_init.py']
    if payload_override is not None:
        from .build_client_bundle import verified_members
        from .immutable_release import sha
        from .live_fleet_worker import authorization
        record=authorization(campaign)
        if record is None or sha(payload_override)!=record['worker_archive_sha256']:
            pass
        manifest,_=verified_members(payload_override,'PAYLOAD.json')
        import shutil
        shutil.copyfile(payload_override,Path(directory)/'workers.tar')
        payload=dict(archive_sha256=sha(payload_override),manifest=manifest)
    elif campaign.get('cache_control'):
        from .cache_control import worker_payload
        payload = worker_payload(campaign, directory, ROOT, sorted(set(names)))
    elif campaign.get('runtime_from'):
        from .formal_runtime import worker_payload
        payload = worker_payload(campaign, directory)
    else:
        payload = build_client_bundle(ROOT, sorted(set(names)), directory / 'workers.tar')
    if payload_override is None and (campaign.get('live_fleets') or {'local','timan1'} & set(execution_nodes)):
        from .local_workers import payload as local_payload
        payload = local_payload(campaign,directory,ROOT,payload)
    if campaign.get('autoscale') and not campaign.get('cache_resume_selection'):
        from exp.offline_search.rounds.r17.autoscale_1017.release import refresh_worker
        payload = refresh_worker(campaign, Path(directory), ROOT, payload)
    digest = payload['archive_sha256']
    if campaign.get('live_fleets'):
        from .live_fleet_worker import check_payload_protocol
        write_json(Path(directory)/'LIVE_WORKER_PROTOCOL.json',check_payload_protocol(Path(directory)/'workers.tar'))
    report = {}
    inherited_path=directory/'WORKER_DEPLOYMENT.json'
    if not inherited_path.exists() and os.environ.get('R17_INHERIT_WORKER_DEPLOYMENT'):
        # Fresh campaign: reuse a sealed dependency release/config from an earlier deployment.
        inherited_path=Path(os.environ['R17_INHERIT_WORKER_DEPLOYMENT'])
    inherited_deployment=json.loads(inherited_path.read_text()) if inherited_path.exists() else {}
    if os.environ.get('R17_INHERIT_WORKER_DEPLOYMENT'):
        # Nodes added on resume reuse the same sealed release (a fresh timan107 install fails).
        for node, item in json.loads(Path(os.environ['R17_INHERIT_WORKER_DEPLOYMENT']).read_text()).items():
            inherited_deployment.setdefault(node, item)
    for node, count in execution_nodes.items():
        paths = worker_paths(campaign, node)
        stage = paths['stage']
        node_root = str(Path(paths['run']).parent.parent)
        destination = node_root+'/payload_'+digest
        from .worker_disk import probe as disk_probe
        disk_probe(campaign,directory,node,paths['run'],stage=stage,phase='prepare')
        remote(node, ['mkdir', '-p', stage])
        from .formal_prestage import client_release
        installed = client_release(sys.modules[__name__],node,directory,stage,destination,payload,
                                   closed_capacity_refresh=closed_capacity_refresh)
        dependency = install_worker_dependencies(node, node_root+'/dependencies', destination,
            **(dict(inherited=inherited_deployment[node]) if node in inherited_deployment else {})) if campaign['domain'] == 'libero' else None
        worker_python = '/scratch/zixuans8/openpi/.venv/bin/python' if campaign['domain'] == 'libero' else '/scratch/zixuans8/mw_r17/envs/sim/bin/python'
        if campaign['domain'] == 'metaworld':
            if node == 'local':
                from .worker_installs import LOCAL_METAWORLD_PYTHON
                worker_python = LOCAL_METAWORLD_PYTHON
            remote(node, ['env','PYTHONPATH='+pythonpath(destination),worker_python,'-c',
                'from exp.r17_integration.ops.worker_installs import prepare_metaworld; '
                'prepare_metaworld(timan1='+repr(node == 'timan1')+
                (',local=True' if node == 'local' else '')+')'], replay_safe=True)
        elif node == 'local':
            from .local_workers import environment
            worker_python = environment()
        extra = dependency['private_dependencies'] if dependency else ''
        env = dict(HOME='/home/zixuans8', PYTHONPATH=pythonpath(destination, extra),
            PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
            MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl', MUJOCO_EGL_DEVICE_ID=str(campaign['worker_egl_device']),
            LIBERO_CONFIG_PATH='/scratch/zixuans8/openpi_trace_h100/os_cl/libero')
        if node == 'local':
            from .local_workers import LOCAL_HOME
            env['HOME'] = LOCAL_HOME
        if campaign.get('wire_zlib'):
            env['OPENPI_WIRE_ZLIB'] = '1'
        if dependency:
            if node == 'local':
                from .local_workers import config as local_config, LOCAL_HOME
                config = local_config(dependency, paths['private']+'/libero_config')
                env['HOME'] = LOCAL_HOME
            else:
                config = json.loads(remote(node, ['env','PYTHONPATH='+pythonpath(destination),worker_python,'-c',
                'from exp.r17_integration.ops.worker_installs import private_config; import json; '
                'print(json.dumps(private_config('+repr(dependency['config_source'])+','+
                repr(paths['private']+'/libero_config')+',assets='+repr(dependency.get('assets'))+')))']))
            env['LIBERO_CONFIG_PATH'] = config['path']
            # Original LIBERO init pools contain NumPy pickle objects. Preserve
            # R16's trusted original-pool loader semantics on torch>=2.6.
            env['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'
        probe_python=worker_python
        if node == 'timan1':
            from .timan1_workers import environment,wrapper
            env=environment(env)
            probe_python=wrapper(campaign)
        probe = json.loads(remote(node, ['env', *[k+'='+v for k,v in env.items()], 'CUDA_VISIBLE_DEVICES=',
            probe_python, '-m', 'exp.r17_integration.conductor.worker_entry', '--worker-id', 'import',
            '--server-key', 'unused:1', '--driver-host', 'unused', '--driver-port', '1', '--import-probe']))
        if dependency:
            validation = remote(node, ['env', *[k+'='+v for k,v in env.items()], 'CUDA_VISIBLE_DEVICES=',
                probe_python, '-c', 'from libero.libero import benchmark; import robosuite; '
                'from examples.libero import main; from libero.libero.envs import OffScreenRenderEnv; '
                'suites=("libero_spatial","libero_object","libero_goal","libero_10"); '
                'pools={s:[len(benchmark.get_benchmark_dict()[s]().get_task_init_states(i)) for i in range(10)] for s in suites}; '
                'assert all(n>=50 for counts in pools.values() for n in counts); print(pools)'],replay_safe=True)
            if campaign.get('resume_from'):
                transfer(['rsync','push', str(Path(campaign['local_assets'])/'benchmark.json'), node+':'+stage+'/benchmark.json'])
                resume_probe = json.loads(remote(node, ['env', *[k+'='+v for k,v in env.items()],
                    'CUDA_VISIBLE_DEVICES=', probe_python, '-c',
                    'from exp.r17_integration.ops.formal_resume import cpu_probe; import json,sys,contextlib\n'
                    'with contextlib.redirect_stdout(sys.stderr):\n p=cpu_probe('+repr(stage+'/benchmark.json')+')\n'
                    'print(json.dumps(p))'],replay_safe=True))
                from .formal_resume import validate_probe
                validate_probe(resume_probe, campaign)
                write_json(directory/(node+'_RESUME_PROBE.json'), resume_probe)
        else:
            validation = remote(node, ['env', *[k+'='+v for k,v in env.items()], 'CUDA_VISIBLE_DEVICES=',
                probe_python, '-c', 'from exp.r17_integration.benchmarks.metaworld import MetaWorldBenchmarkAdapter; '
                'from exp.r17_integration.conductor.manifest import validate_worker_probe; '
                'p=MetaWorldBenchmarkAdapter().worker_probe(); validate_worker_probe(p,require_renderer=False,require_task_data=False); print(p)'],replay_safe=True)
        worker = dict(campaign, worker_node=node, node_workers=count,
            worker_run=paths['run'], worker_private_root=paths['private'],
            worker_benchmark=stage+'/benchmark.json')
        if campaign.get('autoscale'):
            env.update(OPENPI_AUTOSCALE_NODE=paths['run']+'/AUTOSCALE_NODE.json',
                       OPENPI_AUTOSCALE_CONTROL=paths['run']+'/AUTOSCALE_CONTROL.json')
            worker['autoscale_initial'] = count
        if campaign.get('cache_capacity_authorization'):
            from .cache_capacity import renderer_devices
            devices=renderer_devices(campaign,node)
            if devices is not None:worker['worker_egl_devices']=devices
        node_devices = (campaign.get('worker_egl_devices_by_node') or {}).get(node)
        if node_devices:
            worker.update(worker_egl_devices=list(node_devices), worker_egl_device=node_devices[0])
        if node == 'local':
            from .local_workers import route
            route(worker,campaign)
        if node == 'timan1':
            from .timan1_workers import route
            route(worker,campaign)
        if campaign.get('cache_study'):
            from .worker_memory import reservation_mib
            worker['worker_memory_mib']=reservation_mib(worker)
        if campaign['domain'] == 'libero':
            worker.setdefault('worker_memory_mib', 6144)
            worker.setdefault('worker_max_episodes', 4)
        memory_job = {key:worker[key] for key in ('domain', 'node_workers', 'worker_memory_mib',
                                                'worker_memory_headroom_mib') if key in worker}
        if campaign.get('autoscale'):
            memory = dict(admitted=True, deferred=True, desired_maximum=count,
                          reason='Node supervisor must admit every spawn from live memory/PSI/growth')
        else:
            memory = json.loads(remote(node, ['env','PYTHONPATH='+pythonpath(destination),worker_python,'-c',
                'from exp.r17_integration.ops.worker_memory import admission; import json; '
                'print(json.dumps(admission('+repr(memory_job)+','+repr(paths['run'])+')))'], replay_safe=True))
        write_json(directory/(node+'_MEMORY_ADMISSION.json'), memory)
        from .cache_capacity import can_defer_admission
        deferred=bool(campaign.get('autoscale')) or can_defer_admission(campaign)
        if not memory['admitted'] and not deferred and (not campaign.get('prepare_only') or campaign.get('cache_worker_orchestration')):
            raise RuntimeError(node+': '+memory['reason'])
        if campaign.get('cache_worker_orchestration'):
            gpu_job={key:worker[key] for key in ('worker_node','node_workers','worker_egl_device',
                'worker_egl_devices','worker_vram_mib','worker_server_vram_headroom_mib') if key in worker}
            gpu=json.loads(remote(node,['env','PYTHONPATH='+pythonpath(destination),worker_python,'-c',
                'from exp.r17_integration.ops.worker_gpu import admission; import json; '
                'print(json.dumps(admission('+repr(gpu_job)+','+repr(paths['run'])+')))'],replay_safe=True))
            write_json(directory/(node+'_GPU_ADMISSION.json'),gpu)
            if not gpu['admitted'] and not deferred:
                pass
        if campaign.get('domain') == 'libero' and (campaign.get('library') or campaign.get('eval_pool')=='B'):
            init_digest = campaign['bpool_transfer']['archive_sha256']
            transfer(['rsync','push',campaign['bpool_archive'],node+':'+stage+'/bpool_inputs.tar'])
            code = ('from exp.r17_integration.ops.library_init import install_init_archive; import json; '
                'print(json.dumps(install_init_archive('+repr(stage+'/bpool_inputs.tar')+','+
                repr(paths['private']+'/bpool_'+init_digest)+','+repr(init_digest)+')))')
            installed_b = json.loads(remote(node,['env','PYTHONPATH='+pythonpath(destination),worker_python,'-c',code]))
            if installed_b['manifest'] != campaign['bpool_transfer']['manifest']:
                pass
            worker['worker_init_states_dir'] = ','.join(k+'='+v for k,v in sorted(installed_b['directories'].items()))
            write_json(directory/(node+'_BPOOL.json'),installed_b)
        write_json(directory / (node+'.json'), worker)
        live_config=None
        if payload_override is not None:
            from .remote_json import stage_json
            benchmark=stage_json(node,json.loads((Path(campaign['local_assets'])/'benchmark.json').read_text()),
                directory,stage,remote=remote,transfer=transfer,python=worker_python)
            worker['worker_benchmark']=benchmark['path']
            write_json(directory/(node+'.json'),worker)
            live_config=stage_json(node,worker,directory,stage,remote=remote,transfer=transfer,python=worker_python)
        else:
            transfer(['rsync','push', str(directory / (node+'.json')), node+':'+stage+'/campaign.json',*capacity_transfer_options])
        if live_config is None and not (dependency and campaign.get('resume_from')):
            transfer(['rsync','push', str(Path(campaign['local_assets']) / 'benchmark.json'), node+':'+stage+'/benchmark.json',*capacity_transfer_options])
        report[node] = dict(stage=stage, payload=destination, python=worker_python, env=env, workers=count,
            worker_run=worker['worker_run'], installed=installed, import_probe=probe,
            simulator_import=validation, dependencies=dependency, private_config=config if dependency else None,
            worker_memory_mib=worker.get('worker_memory_mib'),worker_vram_mib=worker.get('worker_vram_mib'),
            worker_egl_devices=worker.get('worker_egl_devices'))
        if node == 'timan1':
            report[node]['timan1_worker_lease']=campaign['timan1_worker_lease']
        if live_config is not None:
            report[node].update(campaign_path=live_config['path'],campaign_transfer=live_config,
                                benchmark_transfer=benchmark)
            report[node]['env']['R17_LIVE_FLEET_CONFIG_SHA256']=live_config['sha256']
        if node == 'local':
            report[node].update(transport='direct',host='weilandserver',policy_endpoint=worker['policy_endpoint'],
                worker_memory_mib=worker['worker_memory_mib'],worker_vram_mib=worker['worker_vram_mib'],
                ram_headroom_mib=worker['worker_memory_headroom_mib'],server_vram_headroom_mib=worker['worker_server_vram_headroom_mib'])
    write_json(directory / 'WORKER_DEPLOYMENT.json', report)
    return report


def host_python(campaign):
    if campaign.get('server_python'):
        return campaign['server_python']
    if campaign.get('server_node') == 'h100':
        if campaign['model'] == 'pi05':
            return '/home/exouser/openpi/.venv/bin/python'
        if campaign['model'] == 'groot':
            # CPU asset preparation also consumes this serialized vendor path.
            campaign['vendor'] = '/home/exouser/gr00t_n15'
            return '/home/exouser/gr00t_n15_venv/.venv/bin/python'
    if campaign['model'] == 'dp':
        return '/data/mw_finetune/dp/env/bin/python'
    if campaign['model'] == 'fastwam':
        return '/data/mw_finetune/envs/fastwam/bin/python'
    if campaign['model'] == 'groot':
        return '/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python'
    if campaign['model'] == 'cosmos':
        return campaign['vendor']+'/.venv/bin/python'
    return CONTROL_PYTHON


def server_env(campaign, root):
    h100_native = campaign['server_node'] == 'h100' and campaign['model'] in ('pi05', 'groot')
    extra = '/data/mw_finetune/qualification/python_deps312' if campaign['server_node'] == 'h100' and not h100_native else ''
    if campaign['model'] == 'fastwam':
        extra += ':/data/mw_finetune/vendor/lerobot/src:/data/mw_finetune/vendor/FastWAM/src'
    if campaign['model'] == 'groot':
        vendor = '/home/exouser/gr00t_n15' if h100_native else campaign['vendor']
        extra += ':'+vendor
        if campaign.get('domain') == 'libero':
            extra += ':'+vendor+'/examples/Libero'
    if campaign['model'] == 'cosmos':
        extra = '/data/cosmos/r17_runtime_deps310:'+campaign['vendor']
    env = dict(PYTHONPATH=pythonpath(root, extra.strip(':')), PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', TOKENIZERS_PARALLELISM='false',
        XLA_PYTHON_CLIENT_PREALLOCATE='false')
    if h100_native:
        env['HF_HOME'] = '/home/exouser/.cache/huggingface'
        if campaign['model'] == 'pi05':
            env['OPENPI_DATA_HOME'] = '/home/exouser/.cache/openpi'
        else:
            env['NO_ALBUMENTATIONS_UPDATE'] = '1'
    if campaign.get('fastwam_batching') or campaign.get('dp_batching') or campaign.get('stage_batching'):
        env.update(OPENPI_DISABLE_STAGE_STREAMS='0', BATCHING_STAGE1_WORKERS='1',
                   BATCHING_STAGE2_WORKERS='1', BATCHING_STAGE3_WORKERS='1')
    if campaign['model'] == 'cosmos':
        nvidia = campaign['vendor']+'/.venv/lib/python3.10/site-packages/nvidia'
        env.update(HF_HOME='/data/cosmos/hf_home', DETERMINISTIC='True',
                   LIBERO_CONFIG_PATH='/data/cosmos/r17_runtime_deps310/config', MUJOCO_GL='egl', PYOPENGL_PLATFORM='egl',
                   # TE searches CUDA_HOME recursively for NVRTC before its
                   # ldconfig fallback, which raises on hosts without a CUDA
                   # system install. Use this resident's installed wheel only.
                   CUDA_HOME=nvidia+'/cuda_nvrtc', CUDNN_HOME=nvidia+'/cudnn',
                   LD_LIBRARY_PATH=':'.join(filter(None, [nvidia+'/cuda_nvrtc/lib',
                       nvidia+'/cudnn/lib', nvidia+'/cuda_runtime/lib', os.environ.get('LD_LIBRARY_PATH','')])))
    return env


def require_service_alive(label, status, log_path, read_tail):
    """A closed resident is fatal even if it exited successfully or has a lease."""
    if (status.get('status') in ('CLOSED', 'FAIL') or status.get('child_alive') is False
            or status.get('alive') is False):
        try:
            tail = read_tail()
        except Exception as exc:
            tail = 'could not read owned log: '+repr(exc)
        raise RuntimeError(f"{label} exited before/while serving: exit_code={status.get('exit_code')}, "
                           f"status={status.get('status')}; replica log tail ({log_path}):\n"+tail)


def backend_control_host(campaign):
    # Cache campaigns use fresh ports, independent of the controls' NAT rules.
    # Check the local listener directly before attesting its published route.
    if campaign.get('cache_study'):
        return campaign.get('backend_host') or ('127.0.0.1' if campaign['server_node']=='local' else h100_host())
    return campaign.get('backend_host') or (('127.0.0.1' if campaign['server_node']=='local' else h100_host())
        if campaign['model']=='cosmos' else campaign['endpoint_host'])


def worker_fleet_failed(status, *, dispatch_started):
    """A lease/supervisor cannot attest a lost fleet; startup exit0 is fatal."""
    if not status:
        return False  # Not yet in the next bounded liveness probe.
    if status.get('control_probe_failed'):
        return False  # Transport failure cannot attest fleet death.
    closed = status.get('status') == 'CLOSED'
    if closed:
        return status.get('exit_code') != 0 or not dispatch_started
    return (status.get('status') == 'FAIL' or status.get('child_alive') is False or
            status.get('alive') is False)


def failed_fleet(node, item, status):
    """Read only this campaign's receipt; preserve the worker exit diagnosis."""
    path = item['worker_run']+'/receipt.json'
    result = dict(node=node, process_status=status, receipt_path=path)
    try:
        result['fleet_receipt'] = json.loads(remote(node, [item['python'],'-c',
            'from pathlib import Path; print(Path('+repr(path)+').read_text())'], replay_safe=True))
    except Exception as exc:
        result['receipt_read_error'] = repr(exc)
    return result


@campaign_retries
def prepare(campaign, directory):
    from exp.offline_search.rounds.r17.autoscale_1017.defaults import apply
    apply(campaign, directory)
    from .sync_back import snapshot_run, install_snapshot
    from .build_client_bundle import sha256
    if campaign.get('domain') == 'libero' and campaign.get('eval_pool') == 'B' and not campaign.get('cache_study') and not campaign.get('library'):
        from .library_init import prepare_b_inputs
        prepare_b_inputs(campaign, directory)
    if campaign.get('cache_study'):campaign.setdefault('cache_plan_name',campaign['run_name'])
    if campaign.get('cache_control'):
        from .cache_control import attach
        attach(campaign,directory)
    if campaign.get('runtime_from'):
        # Model-owned launcher hooks (DP batching) populate their campaign
        # fields before entering this function. Validate the complete document.
        from .formal_runtime import attach
        attach(campaign,campaign['runtime_from'])
    write_json(directory / 'campaign.json', campaign)
    if campaign.get('library') and campaign['server_node']=='h100':
        from .library_staging import policy, require_build_host
        require_build_host(campaign)
        limits = policy(campaign)
        # Stdlib admission precedes even the private runtime deployment.
        code = ('import shutil,json; d=shutil.disk_usage("/data").free; t=shutil.disk_usage("/tmp").free; '
            'assert d>='+repr(limits['min_free_data_bytes']+limits['limit_bytes'])+
            ' and t>='+repr(limits['min_free_tmp_bytes']+limits['transfer_reserve_bytes'])+
            ', "H100 free-space admission failed"; print(json.dumps(dict(status="PASS",free_data_bytes=d,free_tmp_bytes=t)))')
        write_json(directory/'H100_DISK_ADMISSION.json',json.loads(remote('h100',[host_python(campaign),'-c',code])))
    from .formal_prestage import runtime_release, phase
    from exp.offline_search.rounds.r17.autoscale_1017.bundle import runtime_sources
    with runtime_sources(campaign), phase(directory, 'runtime_build_verify_transfer'):
        runtime_release(sys.modules[__name__],campaign,directory)
    root = campaign['runtime']
    write_json(directory / 'campaign.json', campaign)
    if campaign.get('cache_study'):
        from .cache_study import stage_inputs
        stage_inputs(campaign, directory, sys.modules[__name__])
        write_json(directory / 'campaign.json', campaign)
    if campaign['server_node'] == 'local':
        env = {**os.environ, **server_env(campaign, root), 'CUDA_VISIBLE_DEVICES': '', 'JAX_PLATFORMS': 'cpu'}
        with phase(directory, 'model_cpu_seal'):
            execute([host_python(campaign), '-m', 'exp.r17_integration.ops.formal_eval', 'seal',
                '--campaign', str(directory / 'campaign.json')], seconds=600, env=env,
                cwd=root if campaign.get('runtime_from') else None)
    else:
        stage = '/tmp/r17_formal/'+campaign['run_name']+'/server'
        campaign['runtime'] = root
        write_json(directory / 'campaign.json', campaign)
        transfer(['rsync','push', str(directory / 'campaign.json'), 'h100:'+stage+'/campaign.json'])
        campaign['server_campaign'] = stage+'/campaign.json'
        env = server_env(campaign, root)
        remote('h100', ['env', *[k+'='+v for k,v in env.items()], 'CUDA_VISIBLE_DEVICES=', 'JAX_PLATFORMS=cpu',
            host_python(campaign), '-m', 'exp.r17_integration.ops.formal_eval', 'seal', '--campaign', stage+'/campaign.json'], seconds=600)
        snapshot = stage+'/assets.tar'
        code = ('from exp.r17_integration.ops.sync_back import snapshot_run; import json; '
            'print(json.dumps(snapshot_run('+repr(campaign['assets'])+','+repr(snapshot)+')))')
        closed = json.loads(remote('h100', ['env', 'PYTHONPATH='+pythonpath(root), host_python(campaign), '-c', code]))
        transfer(['rsync','pull', 'h100:'+snapshot, str(directory / 'assets.tar')])
        write_json(directory / 'SERVER_ASSET_TRANSFER.json', install_snapshot(directory / 'assets.tar',
            campaign['local_assets'], expected_sha256=closed['archive_sha256']))
    if campaign.get('runtime_from'):
        from .formal_runtime import check_assets
        check_assets(campaign,directory)
    from .driver_release import prepare as prepare_orchestration
    prepare_orchestration(campaign,directory,ROOT)
    from .formal_driver import plan
    spec, model, tasks, endpoints = plan(campaign)
    if campaign.get('fastwam_batching'):
        from exp.r17_integration.models.fastwam.formal_batching import check_control
        # Remote CPU sealing cannot access a local control receipt. Validate the
        # complete control on the coordinator, using transferred native seals.
        from .formal_prestage import control_gate
        paired = control_gate(campaign,directory)
        write_json(directory/'BATCHING_CONTROL.json', paired)
    if campaign.get('repair_pin'):
        from .library_repair import verify_repair_seal
        verify_repair_seal(campaign)
    if campaign.get('library'):
        from .library_campaign import disk_estimate, require_fastwam_ab, require_fastwam_g2
        write_json(directory/'DISK_ESTIMATE.json',disk_estimate(model,campaign['suite'],
            wire_size=getattr(spec,'wire_size',spec.render_size if campaign['domain'] != 'libero' else 256),
            capture_images=campaign.get('capture_images',True)))
        if campaign['model']=='fastwam' and campaign['domain']=='libero':
            descriptor = json.loads((Path(campaign['local_assets'])/'load.json').read_text())
            gate = require_fastwam_g2(campaign.get('g2_a_receipt'),campaign.get('g2_b_receipt'),
                campaign['checkpoint'],teacher_descriptor=descriptor,teacher_model=model)
            if campaign.get('ab_qualification') or campaign.get('ab_descriptor'):
                gate['optional_loader_parity'] = require_fastwam_ab(campaign.get('ab_qualification'),
                    campaign.get('ab_descriptor'),descriptor)
            write_json(directory/'GPU_AB_GATE.json',gate)
    episode_plan = dict(expected_episodes=len(tasks),
        benchmark_protocol=spec.identity, model=model, tasks=[__import__('dataclasses').asdict(t) for t in tasks],
        endpoint=campaign['endpoint'], cache_enabled=False, debug_capture=bool(campaign.get('library')))
    if campaign.get('resume_from'):
        from .formal_resume import bind
        episode_plan = bind(campaign, directory, episode_plan)
    write_json(directory / 'EPISODE_PLAN.json', episode_plan)
    deployment = {} if campaign.get('native_parity_only') else prepare_workers(campaign, directory)
    if campaign.get('native_parity_only'):
        write_json(directory/'WORKER_DEPLOYMENT.json',deployment)
    # Server-only overrides (e.g. Fast-WAM text cache, patched runtime) for a fresh preparation.
    campaign.update(json.loads(os.environ.get('R17_CAMPAIGN_EXTRA', '{}')))
    write_json(directory / 'campaign.json', campaign)
    write_json(directory / 'PREFLIGHT.json', dict(status='PASS', cpu_only=True, episodes=episode_plan['expected_episodes'],
        worker_nodes=list(deployment), replicas=campaign['replicas'], ports=campaign['ports'],
        model_identity=model['checkpoint_identity'], protocol=spec.identity,
        memory_admissible=all(json.loads((directory/(node+'_MEMORY_ADMISSION.json')).read_text())['admitted']
                              for node in deployment)))
    from .formal_prestage import seal_prepared
    seal_prepared(campaign,directory,deployment)
    return deployment


def driver_completion(directory, returncode):
    """Rc=2 needs complete outcomes, resolved retries, and explicit drain evidence."""
    if returncode not in (0, 2):
        raise RuntimeError('driver incomplete: '+str(returncode))
    directory = Path(directory)
    tasks = json.loads((directory / 'EPISODE_PLAN.json').read_text())['tasks']
    planned = {task['task_uid'] for task in tasks}
    accepted = {}
    for line in (directory / 'driver/journal.jsonl').read_text().splitlines():
        if line:
            row = json.loads(line)
            if row.get('accepted') is True:
                accepted.setdefault(row.get('task_uid'), []).append(row)
    if (not tasks or len(planned) != len(tasks) or set(accepted) != planned or
        any(len(rows) != 1 or rows[0].get('status') not in ('done', 'failed') or
            rows[0].get('error') is not None for rows in accepted.values())):
        raise RuntimeError('driver incomplete: journal must accept every planned episode exactly once without errors')
    receipt = json.loads((directory / 'driver/DRIVER_RECEIPT.json').read_text())
    campaign_path = directory/'campaign.json'
    campaign = json.loads(campaign_path.read_text()) if campaign_path.exists() else {}
    if campaign.get('driver_mode')=='group-commit':
        # A visible page-cache journal is insufficient after a failed fsync.
        # Resume composites retain the checked raw segment receipt here.
        committed=receipt.get('segment_receipt',receipt)
        if (committed.get('error') is not None or (committed.get('group_commit') or {}).get('closed') is not True or
                committed.get('orchestration')!=campaign.get('orchestration')):
            pass
    recovered = returncode == 2
    errors, resolved = [], []
    recovered_shutdown = recovered and receipt.get('worker_shutdown', {}).get('status') == 'PARTIAL'
    if recovered:
        attempts = [json.loads(line) for line in (directory / 'driver/attempts.jsonl').read_text().splitlines() if line]
        live = [row for row in attempts if row.get('accepted') is True]
        errors = [row for row in live if row.get('error') is not None]
        if (receipt.get('error') is not None or receipt.get('all_done') is not True or receipt.get('rejected_worker_probes') != 0 or
            receipt.get('status') not in ('PASS', 'PARTIAL') or not (errors or recovered_shutdown) or
            any(row.get('task_uid') not in planned for row in live)):
            raise RuntimeError('driver incomplete: 2 is not a completed campaign with resolved retry errors')
        for task in tasks:
            uid = task['task_uid']
            final = accepted[uid][0]
            generation = final.get('attempt', 1)
            cohort = [row for row in live if row['task_uid'] == uid]
            generations = [row.get('attempt', 1) for row in cohort]
            if (type(generation) is not int or generation < task.get('attempt', 1) or
                any(type(g) is not int for g in generations) or
                sorted(generations) != list(range(task.get('attempt', 1), generation+1)) or
                any(row.get('error') is not None if row.get('attempt', 1) == generation else
                    row.get('error') is None for row in cohort) or
                any('success' in row and row['success'] != final.get('success')
                    for row in cohort if row.get('attempt', 1) == generation)):
                pass
        resolved = [dict(task_uid=row['task_uid'], failed_attempt=row.get('attempt', 1),
            resolved_attempt=accepted[row['task_uid']][0].get('attempt', 1), error=row['error'],
            worker_id=row.get('worker_id')) for row in errors]
    elif receipt.get('status') != 'PASS':
        pass
    if campaign.get('library'):
        from .library_repair import outcomes
        from exp.r17_integration.debug.attempts import json_rows
        attempts = json_rows(directory/'driver/attempts.jsonl')
        counts = {(r['task_uid'],r['attempt']):r.get('capture_decisions')
                  for r in attempts if r.get('accepted') is True and r.get('error')}
        try:
            outcomes(directory,tasks,counts,inherited=campaign.get('repair_prior_attempts'))
        except ValueError as exc:
            raise RuntimeError('driver incomplete: unsafe library retry: '+str(exc)) from exc
    return dict(returncode=returncode, verified_completion=True,
        recovered_retried_errors=recovered and bool(errors), recovered_worker_shutdown=recovered_shutdown,
        error_attempts=len(errors), resolved_retry_errors=resolved, accepted_episodes=len(planned))


def drain_completed_workers(owned, completion, managed):
    """Close only a verified complete driver's own fleet; defer snapshot audits."""
    if owned.get('kind',owned.get('label')) != 'workers' or completion.get('verified_completion') is not True:
        pass
    status = managed(owned, 'stop')
    if (status.get('status') != 'CLOSED' or status.get('alive') is not False or type(status.get('exit_code')) is not int or
            status.get('child_alive') is not False or status.get('supervisor_alive') is not False or
            status.get('owned_worker_survivors') != []):
        raise RuntimeError('remote worker close/snapshot incomplete')
    # The supervisor can exhaust its stop grace while the native fleet builds
    # its tar. Closed owners alone are not PASS: the CPU finalizer must verify
    # a quiescent snapshot, the fleet receipt, execution and native audits.
    allowed = (0, 2) if completion.get('recovered_worker_shutdown') else (0,)
    if status['exit_code'] not in allowed:
        status['snapshot_incomplete'] = True
    return status


def aggregate(directory, *, driver_returncode=None):
    """Planned denominators; duplicate/stale attempts never inflate SR."""
    directory = Path(directory)
    if json.loads((directory/'campaign.json').read_text()).get('cache_study'):
        from .cache_study import aggregate as aggregate_study
        return aggregate_study(directory, driver_returncode=driver_returncode)
    completion = driver_completion(directory, driver_returncode) if driver_returncode is not None else None
    plan = json.loads((directory / 'EPISODE_PLAN.json').read_text())
    campaign = json.loads((directory / 'campaign.json').read_text())
    def rows(name):
        path = directory / 'driver' / name
        return [json.loads(line) for line in path.read_text().splitlines() if line] if path.exists() else []
    terminal = {}
    for row in rows('journal.jsonl'):
        if row.get('task_uid') and row.get('accepted') is True and row.get('status') in ('done', 'failed'):
            terminal[row['task_uid']] = row
    attempts = rows('attempts.jsonl')
    suites, task_rates = {}, []
    for task in plan['tasks']:
        suite = task['experiment']
        entry = suites.setdefault(suite, dict(planned=0, completed=0, successes=0, errors=0))
        entry['planned'] += 1
        row = terminal.get(task['task_uid'])
        if row is not None:
            entry['completed'] += 1
            entry['successes'] += int(bool(row.get('success')) and row.get('error') is None)
            entry['errors'] += int(row.get('error') is not None)
    by_task = {}
    for task in plan['tasks']:
        entry = by_task.setdefault(task['extra']['task_name'], dict(planned=0, completed=0, successes=0, errors=0))
        entry['planned'] += 1
        row = terminal.get(task['task_uid'])
        if row:
            entry['completed'] += 1
            entry['successes'] += int(bool(row.get('success')) and row.get('error') is None)
            entry['errors'] += int(row.get('error') is not None)
    for entry in [*suites.values(), *by_task.values()]:
        entry['sr'] = entry['successes']/entry['planned']
    uid_suites = {task['task_uid']:task['experiment'] for task in plan['tasks']}
    for suite, entry in suites.items():
        cohort = [r for r in attempts if r.get('accepted') and uid_suites.get(r.get('task_uid')) == suite]
        entry['error_attempts'] = sum(r.get('error') is not None for r in cohort)
        entry['retry_attempts'] = len({(r['task_uid'],r.get('attempt',1)) for r in cohort if r.get('attempt',1)>1})
    step_rows = [r for r in rows('per_step.jsonl') if not r.get('_kind') and r.get('accepted') is True
                 and r.get('attempt') == terminal.get(r.get('task_uid'), {}).get('attempt')]
    driver_receipt = json.loads((directory / 'driver/DRIVER_RECEIPT.json').read_text()) if (directory / 'driver/DRIVER_RECEIPT.json').exists() else {}
    elapsed = driver_receipt.get('elapsed_seconds', 0)
    n_decisions = len(step_rows)
    k = plan.get('model', {}).get('default_sampler', {}).get('num_steps')
    full_arm_verified = bool(step_rows) and (k is None or all(
        row.get('capture_enabled') is bool(campaign.get('library')) and row.get('source') == 'policy' and
        row.get('actual_nfe') == k and row.get('stage_calls') == [1,1,k] for row in step_rows))
    if campaign.get('commit_horizon') is not None or campaign.get('denoising_steps') is not None:
        full_arm_verified = horizon_execution_verified(step_rows, campaign, plan['model'])
        full_arm_verified = full_arm_verified and {r.get('task_uid') for r in step_rows} == set(uid_suites)
        if campaign.get('denoising_steps') is not None:
            from exp.r17_integration.common.contracts import digest
            sampler = json.loads((Path(campaign['local_assets']) / 'execution_sampler.json').read_text())
            full_arm_verified = full_arm_verified and all(r.get('sampler_identity') == digest(sampler) for r in step_rows)
    output = dict(schema='r17.formal_sr.v1', status='PASS' if sum(e['completed'] for e in suites.values()) == len(plan['tasks'])
        and not sum(e['errors'] for e in suites.values()) and (driver_receipt.get('status') == 'PASS'
            or completion is not None and completion.get('verified_completion') is True)
        and full_arm_verified else 'PARTIAL',
        campaign=campaign['run_name'], smoke=campaign['smoke'], protocol=plan['benchmark_protocol'],
        expected_episodes=len(plan['tasks']), completed_episodes=sum(e['completed'] for e in suites.values()),
        suites=suites, tasks=by_task, task_mean_sr=sum(e['sr'] for e in by_task.values())/max(1,len(by_task)),
        forty_task_mean_sr=sum(e['sr'] for e in by_task.values())/40 if len(by_task) == 40 else None,
        error_attempts=sum(r.get('error') is not None for r in attempts),
        terminal_errors=sum(e['errors'] for e in suites.values()),
        retry_attempts=sum(int(r.get('attempt', 1)) > 1 for r in attempts),
        accepted_worker_ids=sorted({r['worker_id'] for r in attempts if r.get('accepted') and r.get('worker_id')}),
        driver_seconds=elapsed, decisions=n_decisions, observed_infer_per_second=n_decisions/elapsed if elapsed else None,
        full_arm_execution_verified=full_arm_verified,
        cache_enabled=False, debug_capture=bool(campaign.get('library')))
    if campaign.get('commit_horizon') is not None:
        output.update(commit_horizon=campaign['commit_horizon'],
            transport_replan_steps=plan['model']['layout']['replan_steps'],
            policy_calls=sum(r.get('source') == 'policy' for r in step_rows),
            policy_tail_requests=sum(r.get('source') == 'policy_tail' for r in step_rows))
    if campaign.get('denoising_steps') is not None:
        output.update(denoising_steps=campaign['denoising_steps'], execution_sampler=sampler,
            policy_calls=sum(r.get('source') == 'policy' for r in step_rows))
        if sampler['sampler_id'] == 'cosmos_edm_2ab_k2_incl_clean':
            output.update(arm_label=sampler['config']['arm_label'], expected_nfe=sampler['config']['expected_nfe'])
    if campaign.get('suites') is not None:
        output['selected_suites'] = list(selected_suites(campaign))
    if completion is not None:
        output['driver_completion'] = completion
    if campaign.get('fastwam_batching'):
        from exp.r17_integration.models.fastwam.formal_batching import receipt_fields
        output.update(receipt_fields(campaign))
    if (directory/'RESUME_MERGE.json').exists():
        output['resume'] = json.loads((directory/'RESUME_MERGE.json').read_text())
    if campaign.get('orchestration'):
        output['orchestration'] = campaign['orchestration']
    write_json(directory / 'SUMMARY.json', output)
    return output


def backend_evidence(directory):
    directory = Path(directory)
    result = {}
    for path in sorted((directory / 'servers').glob('replica_*/serving.jsonl')):
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        if records:
            result[str(records[0]['replica_port'])] = dict(inferences=len(records),
                task_uids=sorted({r['task_uid'] for r in records if r.get('task_uid')}),
                total_service_seconds=sum(r.get('service_seconds',r['elapsed_seconds']) for r in records))
    write_json(directory / 'BACKEND_EVIDENCE.json', result)
    return result


UNREACHABLE_UNTIL = {}


def refresh_remote_owned(remote_owned, directory, status_cache, control_files, last_status):
    """At most one lease/status session per node per30s, even in a2s loop."""
    if time.monotonic()-last_status < PROBE_SECONDS:
        return last_status
    from .remote_json import stage_json, expression
    remote_owned = [o for o in remote_owned if not o.get('retired_worker_node')]
    for node in sorted({o['node'] for o in remote_owned}):
        if UNREACHABLE_UNTIL.get(node, 0) > time.monotonic():
            continue
        entries=[dict(label=o['label'],request=o['request']) for o in remote_owned if o['node']==node]
        from exp.offline_search.rounds.r17.autoscale_1017.driver import status_entries
        entries=status_entries(entries, directory)
        owned=next(o for o in remote_owned if o['node']==node)
        key=node,json.dumps(entries,sort_keys=True)
        payload=repr(entries)
        if len(payload.encode())>8192 and not any(e.get('autoscale_control') for e in entries) and not (globals().get('worker_resilience') and
                all(o['label']=='workers' for o in entries)):
            if key not in control_files:
                control_files[key]=stage_json(node,entries,directory,str(Path(owned['request']['lease']).parent),
                    remote=remote,transfer=transfer,python=owned['python'])
            payload=expression(control_files[key])
        code=('from exp.r17_integration.ops.formal_process import controls; '
            'from exp.r17_integration.ops.remote_json import load_json; import json; '
            'print(json.dumps(controls('+payload+')))')
        try:
            worker_probe=globals().get('worker_resilience') and all(o['label']=='workers' for o in entries)
            results=json.loads(remote(node,['env','PYTHONPATH='+pythonpath(owned['root']),owned['python'],'-c',code],
                                      seconds=90 if worker_probe else 10,replay_safe=True))
            if set(results)!={o['label'] for o in entries}:
                raise ValueError('remote batched control omitted an owned process')
        except Exception as exc:
            if globals().get('worker_resilience') and all(o['label']=='workers' for o in entries):
                for entry in entries:
                    status_cache[node,entry['label']]=dict(status='UNKNOWN',error=repr(exc),control_probe_failed=True)
                continue
            if not getattr(exc, 'r17_retry_exhausted', False):
                raise
            # Workers reach the driver directly, so a node whose tether agent is unreachable keeps serving
            # (timan107 on 2026-10-14 00:21 aborted cspi_k2 and csgr_h5). Skip its status for 10 minutes.
            UNREACHABLE_UNTIL[node] = time.monotonic()+600
            print('R17 probe: '+node+' unreachable through tether; continuing without its status', file=sys.stderr, flush=True)
            continue
        for label,value in results.items():status_cache[node,label]=value
    from exp.offline_search.rounds.r17.autoscale_1017.driver import collect_status
    collect_status(directory, status_cache)
    return time.monotonic()


@campaign_retries
@_gpu2_coordinator
def run_campaign(campaign, directory, deployment):
    from exp.offline_search.rounds.r17.filter_failed import timan1_gpu2 as _autoscale_gpu2
    class NativeQualificationComplete(Exception):
        pass
    from .sync_back import install_snapshot
    local, remote_owned, handles = [], [], []
    stopped = False
    def stop(*unused):
        nonlocal stopped
        stopped = True
    previous = {sig:signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    report = dict(status='PARTIAL', owned_local=[], owned_remote=[], cleanup=[])
    resilience = globals().get('worker_resilience')
    worker_state = resilience.Running(campaign,directory,deployment,report) if resilience else None
    if campaign.get('cache_study'):
        from .cache_resume import segment_resources
        report.update(segment_resources=segment_resources(campaign,directory),
            execution_workers=campaign.get('execution_workers',campaign['workers']),replicas=campaign['replicas'])
    if campaign.get('local_worker_exception'):
        report.update(local_worker_exception=campaign['local_worker_exception'],worker_nodes=campaign.get('execution_worker_nodes',campaign['worker_nodes']),
            local_worker_orchestration=json.loads((directory/'LOCAL_WORKER_ORCHESTRATION.json').read_text()))
    if campaign.get('cache_worker_orchestration'):
        report.update(cache_worker_orchestration=campaign['cache_worker_orchestration'],
                      worker_nodes=campaign.get('execution_worker_nodes',campaign['worker_nodes']),
                      canonical_worker_nodes=campaign['worker_nodes'])
    if campaign.get('cache_capacity_authorization'):
        report.update(execution_workers=campaign['execution_workers'],replicas=campaign['replicas'],
                      cache_capacity_authorization=campaign['cache_capacity_authorization'])
    if campaign.get('timan1_worker_lease'):
        report['timan1_worker_lease']=campaign['timan1_worker_lease']
    if campaign.get('execution_host'):
        report['execution_host'] = campaign['execution_host']
    if campaign.get('orchestration'):
        report['orchestration'] = campaign['orchestration']
    if campaign.get('resume_source_closure_waiver'):
        report['source_closure_waiver'] = campaign['resume_source_closure_waiver']
    if campaign.get('fastwam_batching'):
        from exp.r17_integration.models.fastwam.formal_batching import receipt_fields
        report.update(receipt_fields(campaign))
        report['paired_control_seal'] = json.loads((directory/'BATCHING_CONTROL.json').read_text())
    if campaign.get('dp_batching'):
        report['dp_batching'] = campaign['dp_batching']
    if campaign.get('stage_batching'):
        report['stage_batching'] = campaign['stage_batching']
    shard_pump = None
    live_fleets = None
    mps_ready = None
    last_status = -float('inf')
    status_cache, control_files = {}, {}
    until = time.monotonic() + campaign['seconds']
    def save():
        report['tether_retry'] = current_journal.get().summary()
        report['worker_disk_admission'] = {node:json.loads(path.read_text()) for node in deployment
            if (path:=directory/(node+'_DISK_ADMISSION.json')).is_file()}
        write_json(directory / 'RECEIPT.json', report)
    def spawn_local(label, argv, env):
        log = (directory / (label+'.log')).open('ab')
        handles.append(log)
        child_env = {**os.environ, **env}
        if label == 'driver' and campaign.get('autoscale'):
            child_env.update(OPENPI_AUTOSCALE_CONFIG=str(directory/'campaign.json'),
                             OPENPI_AUTOSCALE_DRIVER_MODULE='exp.offline_search.rounds.r17.autoscale_1017.driver')
        if label.startswith('replica_') and child_env.get('CUDA_VISIBLE_DEVICES')=='':
            # Slot wrappers deliberately hide CUDA from the CPU conductor.
            # An explicit --run admits only this owned resident to the GPU.
            child_env.pop('CUDA_VISIBLE_DEVICES')
        child_root = (campaign.get('driver_runtime',campaign.get('coordinator_runtime',str(ROOT))) if label=='driver' else
                      campaign['runtime'] if campaign.get('runtime_from') else ROOT)
        child = subprocess.Popen(argv, env=child_env, cwd=child_root, stdout=log,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        entry = dict(label=label, pid=child.pid, pgid=child.pid, start_ticks=ticks(child.pid), argv=argv)
        local.append((child, entry))
        report['owned_local'].append(entry)
        save()
        return child
    def spawn_remote(node, label, argv, env, root, python, stage, ownership=None):
        if node == 'timan1' and label == 'workers' and campaign.get('autoscale'):
            argv = _autoscale_gpu2.bootstrap(argv[0], ['exp.r17_integration.ops.formal_workers', *argv[4:]], 'fleet')
        if ownership:
            from .live_fleets import owned_spawn
            def record(owned,first):
                if first:
                    remote_owned.append(owned);report['owned_remote'].append(owned)
                save()
            return owned_spawn(sys.modules[__name__],node,label,argv,env,root,python,stage,
                seconds=campaign['seconds'],lease_seconds=LEASE_SECONDS,ownership=ownership,record=record)
        request = dict(argv=argv, env=env, cwd=root, receipt=stage+'/'+label+'.process.json',
            log=stage+'/'+label+'.log', lease=stage+'/'+label+'.lease', seconds=campaign['seconds'],
            lease_seconds=LEASE_SECONDS)
        try:
            receipt = json.loads(remote(node, ['env', 'PYTHONPATH='+pythonpath(root), python, '-m',
                'exp.r17_integration.ops.formal_process', 'spawn', '--request', json.dumps(request)]))
        except Exception as exc:
            if worker_state is None or label != 'workers':raise
            # A lost spawn reply is ambiguous. Fence the node and preserve the
            # exact request for later closure; never replay an ambiguous spawn.
            owned=dict(node=node,label=label,request=request,root=root,python=python,receipt={},
                       transport='direct' if node=='local' else 'tether')
            remote_owned.append(owned);report['owned_remote'].append(owned)
            worker_state.retire(node,dict(spawn_error=repr(exc)),owned)
            save()
            return owned
        owned = dict(node=node, label=label, request=request, root=root, python=python, receipt=receipt,
                     transport='direct' if node=='local' else 'tether')
        if ownership:owned.update(ownership)
        remote_owned.append(owned)
        report['owned_remote'].append(owned)
        save()
        return owned
    def managed(owned, mode):
        return json.loads(remote(owned['node'], ['env', 'PYTHONPATH='+pythonpath(owned['root']), owned['python'],
            '-m', 'exp.r17_integration.ops.formal_process', mode, '--request', json.dumps(owned['request'])],
            seconds=10 if mode in ('status', 'renew') else 60))
    def probe():
        nonlocal last_status
        last_status=refresh_remote_owned(remote_owned,directory,status_cache,control_files,last_status)
    def check(required_label=None):
        if live_fleets is not None:
            live_fleets.tick()
        if shard_pump is not None:
            shard_pump.check()
        if stopped or time.monotonic() >= until:
            raise TimeoutError('campaign stopped or hard deadline exceeded')
        for child, entry in local:
            if child.poll() is not None and (entry['label'] != 'driver' or entry['label'] == required_label):
                path = directory / (entry['label']+'.log')
                require_service_alive(entry['label'], dict(status='CLOSED', exit_code=child.returncode),
                                      str(path), lambda: path.read_text(errors='replace')[-6000:])
        probe()
        for owned in remote_owned:
            if owned.get('kind',owned['label'])=='workers':
                status=status_cache.get((owned['node'],owned['label']),{})
                if worker_state is not None and worker_state.check(owned,status):continue
                fleet_key=owned.get('fleet_key',owned['node'])
                if live_fleets is not None and live_fleets.closed_fleet(owned,status):
                    report.setdefault('completed_worker_drains',{})[fleet_key]=status
                if (worker_fleet_failed(status, dispatch_started=any(e['label']=='driver' for _,e in local)) and
                        fleet_key not in report.get('completed_worker_drains', {})):
                    report['worker_fleet_failure']=failed_fleet(owned['node'],deployment[fleet_key],status)
                    detail=report['worker_fleet_failure'].get('fleet_receipt', {})
                    raise RuntimeError('remote worker fleet failed: '+owned['node']+'; exit_code='+
                        str(status.get('exit_code'))+'; '+str(detail.get('reason','receipt unavailable'))+
                        '; worker_failures='+json.dumps(detail.get('worker_failures',[])))
            if owned['label'].startswith('replica_') or owned['label'] in ('proxy','gpu_lock','mps_guard'):
                status = status_cache.get((owned['node'],owned['label']))
                if status is None:continue  # Newly spawned: port/metadata handshake attests startup.
                path = owned['request']['log']
                require_service_alive(owned['label'], status, owned['node']+':'+path,
                    lambda: remote(owned['node'], ['tail', '-c', '6000', path], seconds=10, replay_safe=True))
        if worker_state is not None:worker_state.require_workers()
    def wait_port(host, port, seconds=600, required_label=None):
        bound = min(until, time.monotonic()+seconds)
        while time.monotonic() < bound:
            check(required_label)
            try:
                with socket.create_connection((host, port), timeout=1):
                    return
            except OSError:
                time.sleep(.5)
        raise TimeoutError(f'port {host}:{port} did not become ready')
    server_root = campaign['runtime']
    server_python = host_python(campaign)
    env = server_env(campaign, server_root)
    replica_env = env
    def mps_command(mode, clients=()):
        argv = [server_python, '-m', 'exp.r17_integration.models.fastwam.ops.mps', mode,
                '--ready', mps_ready]
        if clients:
            argv += ['--clients', *map(str, clients)]
        return json.loads(execute(argv, env={**os.environ, **env}) if campaign['server_node']=='local'
                          else remote('h100', ['env', *[k+'='+v for k,v in env.items()], *argv], replay_safe=True))
    if campaign['server_node'] == 'local':
        server_campaign = str(directory / 'campaign.json')
    else:
        server_campaign = campaign['server_campaign']
    try:
        if campaign.get('live_fleets'):
            from .live_fleets import Launcher
            live_fleets=Launcher(sys.modules[__name__],campaign,directory,deployment,spawn_remote)
        if worker_state is None and deployment and (campaign.get('worker_min_free_gib') is not None or
                any((directory/(node+'_DISK_ADMISSION.json')).exists() for node in deployment)):
            from .worker_disk import prepared as disk_admission
            report['worker_disk_admission'] = disk_admission(campaign,directory,deployment)
        if campaign.get('library') and campaign['server_node']=='h100':
            from .library_staging import ShardPump, require_build_host
            require_build_host(campaign)
            from .remote_json import stage_json, expression
            control_reference = None
            argument_references = {}
            def shard_remote(action,*args):
                nonlocal control_reference
                payload = repr(campaign)
                if len(payload.encode())>8192:
                    if control_reference is None:
                        control_reference=stage_json('h100',campaign,directory,
                            '/tmp/r17_formal/'+campaign['run_name']+'/server',remote=remote,transfer=transfer,
                            python=server_python)
                        write_json(directory/'H100_CONTROL_INPUT.json',control_reference)
                    payload=expression(control_reference)
                arguments=repr(args)
                if len(arguments.encode())>8192:
                    if arguments not in argument_references:
                        argument_references[arguments]=stage_json('h100',args,directory,
                            '/tmp/r17_formal/'+campaign['run_name']+'/server',
                            remote=remote,transfer=transfer,python=server_python)
                    arguments=expression(argument_references[arguments])
                code = ('from exp.r17_integration.ops import library_staging as s; import json; '
                    'from exp.r17_integration.ops.remote_json import load_json; '
                    'print(json.dumps(s.'+action+'('+payload+',*'+arguments+')))')
                return json.loads(remote('h100',['env','PYTHONPATH='+pythonpath(server_root),server_python,'-c',code],
                    seconds=30,replay_safe=True))
            write_json(directory/'H100_DISK_START.json',shard_remote('admit'))
            # Q's GPU lock belongs to the execution host. A leased guard is
            # registered first and stopped last, after all owned residents.
            # Never acquire /tmp/r17_gpu.lock on the local 4090 coordinator.
            stage = '/tmp/r17_formal/'+campaign['run_name']+'/server'
            ready = stage+'/GPU_LOCK_READY.json'
            guard = spawn_remote('h100','gpu_lock',[server_python,'-u','-m',
                'exp.r17_integration.ops.formal_gpu_lock','--ready',ready,'--campaign',campaign['run_name']],
                env,server_root,server_python,stage)
            deadline = min(until,time.monotonic()+30)
            while time.monotonic()<deadline:
                check()
                value = json.loads(remote('h100',[server_python,'-c',
                    'from pathlib import Path; p=Path('+repr(ready)+'); print(p.read_text() if p.exists() else "null")'], replay_safe=True))
                if value is not None:
                    if (value.get('status')!='PASS' or value.get('campaign')!=campaign['run_name'] or
                        value.get('pid')!=guard['receipt']['child']['pid'] or value.get('lock')!='/tmp/r17_gpu.lock'):
                        pass
                    write_json(directory/'H100_GPU_LOCK.json',dict(value,node='h100'))
                    break
                time.sleep(.2)
            else:
                raise TimeoutError('H100 GPU lock guard did not become ready')
            shard_pump = ShardPump(campaign,directory,shard_remote,
                lambda path,target: transfer(['rsync','pull','h100:'+path,str(target)]))
            shard_pump.start()
        # Actual reusable listener admission tolerates TCP TIME_WAIT and waits
        # for a prior owned service's release. Never signal an occupied port.
        from .formal_ports import admit
        report['port_release'] = admit(campaign, directory, remote=remote,
                                      python=server_python, heartbeat=check)
        if campaign.get('mps'):
            # A by-value CLI guard serializes the frozen runtime's control
            # queries without overlaying any model/server/worker source.
            from .mps_control_guard import deploy as deploy_mps_control
            control_pin = deploy_mps_control(sys.modules[__name__],campaign,directory)
            env.update(control_pin['env'])
            report['mps_control_orchestration'] = control_pin
            from exp.r17_integration.models.fastwam.ops.mps import config, client_env, comparison_scope
            if campaign['mps'] != config():
                pass
            mps_ready = str(Path(campaign['server_run']).parent/'MPS_READY.json')
            argv = [server_python, '-u', '-m', 'exp.r17_integration.models.fastwam.ops.mps', 'guard',
                    '--campaign-directory', str(Path(campaign['server_run']).parent),
                    '--device', campaign['device'], '--ready', mps_ready]
            if campaign['server_node']=='local':
                spawn_local('mps_guard', argv, env)
            else:
                spawn_remote('h100','mps_guard',argv,env,server_root,server_python,
                             '/tmp/r17_formal/'+campaign['run_name']+'/server')
            deadline = min(until,time.monotonic()+30)
            while time.monotonic()<deadline:
                check()
                try:
                    state = mps_command('state')
                    break
                except Exception:
                    time.sleep(.2)
            else:
                raise TimeoutError('private MPS guard did not become ready')
            if state.get('status')!='RUNNING' or state.get('initial',{}).get('servers'):
                pass
            replica_env = {**env, **client_env(state)}
            report['mps'] = dict(configuration=campaign['mps'], startup=state, attachments=[],
                                 comparison_scope=comparison_scope(campaign))
            save()
        # Workers import/attest while replicas load. Stock WorkerLoop reconnects
        # until the driver starts; dispatch is gated by the later proxy ACK.
        workers_started = not campaign.get('library') and not campaign.get('native_parity_only')
        if workers_started:
            for node,item in deployment.items():
                spawn_remote(node,'workers',[item['python'],'-u','-m','exp.r17_integration.ops.formal_workers',
                    '--campaign',item.get('campaign_path',item['stage']+'/campaign.json')],
                    item['env'],item['payload'],item['python'],item['stage'])
        # Bounded waves retain the existing resident VRAM caps and self-checks.
        # Heavy GR00T/Cosmos loads stay sequential by default.
        width = campaign.get('replica_startup_width', 1)
        from .formal_prestage import phase
        attestations = []
        for index, port in enumerate(campaign['backend_ports']):
            if index % width == 0:
                for sibling in range(index,min(index+width,len(campaign['backend_ports']))):
                    sibling_port=campaign['backend_ports'][sibling]
                    argv = [server_python, '-u', '-m', 'exp.r17_integration.ops.formal_server', '--campaign', server_campaign,
                        '--port', str(sibling_port), '--directory', campaign['server_run']+'/replica_'+str(sibling)]
                    if campaign['model'] == 'dp':
                        argv += ['--cuda-graph']
                    if campaign.get('mps'):
                        from exp.r17_integration.models.fastwam.ops.mps import replica_argv
                        argv = replica_argv(argv, mps_ready)
                    if campaign['server_node'] == 'local':
                        spawn_local('replica_'+str(sibling), argv, replica_env)
                    else:
                        spawn_remote('h100', 'replica_'+str(sibling), argv, replica_env, server_root, server_python,
                            '/tmp/r17_formal/'+campaign['run_name']+'/server')
            control_host = backend_control_host(campaign)
            with phase(directory,'replica_ready_'+str(index)):
                wait_port(control_host, port, required_label='replica_'+str(index))
            from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
            client = WebsocketClientPolicy(control_host, port)
            try:
                metadata = client.get_server_metadata()
            finally:
                client.close()
            expected = json.loads((Path(campaign['local_assets']) / 'model_spec.json').read_text())
            if json.loads(json.dumps(metadata['r17_model_spec'])) != expected:
                pass
            if campaign.get('fastwam_batching'):
                from exp.r17_integration.models.fastwam.formal_batching import check_attestation
                check_attestation(campaign, metadata)
            if campaign['model'] == 'dp' and not metadata.get('cuda_graph_self_check', {}).get('passed'):
                raise ValueError('backend CUDA graph self-check absent or failed')
            if campaign['model'] == 'cosmos' and not metadata.get('native_parity_self_check', {}).get('passed'):
                raise ValueError('Cosmos official/staged parity self-check absent or failed')
            attestations.append(dict(port=port, metadata=metadata))
            write_json(directory / 'SERVER_ATTESTATIONS.json', attestations)
            if (campaign['model'] in ('dp', 'cosmos') or campaign.get('fastwam_batching') or campaign.get('mps')) and (
                    (index+1)%width==0 or index+1==len(campaign['backend_ports'])):
                gpu_command = ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                               '--format=csv,noheader,nounits']
                listing = execute(gpu_command) if campaign['server_node'] == 'local' else remote('h100',gpu_command,replay_safe=True)
                owners = {entry['pid'] for _,entry in local if entry['label'].startswith('replica_')} if (
                    campaign['server_node'] == 'local') else {o['receipt']['child']['pid'] for o in remote_owned
                                                            if o['label'].startswith('replica_')}
                memory = {}
                for line in listing.splitlines():
                    pid, mib = (field.strip() for field in line.split(','))
                    if int(pid) in owners:
                        memory[str(pid)] = memory.get(str(pid),0) + float(mib)
                used_mib = sum(memory.values())
                if campaign.get('mps'):
                    accounting = mps_command('account', sorted(owners))
                    memory, used_mib = accounting['per_pid_mib'], accounting['owned_total_mib']
                    report['mps']['attachments'].append(accounting)
                write_json(directory / 'GPU_MEMORY.json', dict(owned_pids=sorted(owners), per_pid_mib=memory,
                    owned_total_mib=used_mib, budget_mib=campaign['memory_budget_gib']*1024))
                if (not campaign.get('mps') and len(memory) != len(owners)) or used_mib > campaign['memory_budget_gib']*1024:
                    pass
        if campaign.get('native_parity_only'):
            raise NativeQualificationComplete()
        proxy = [server_python, '-u', '-c', 'from openpi.serving.replica_proxy import run_proxy; '
            f'run_proxy({campaign["public_port"]},{campaign["backend_ports"]!r})']
        if campaign['server_node'] == 'local':
            spawn_local('proxy', proxy, env)
        else:
            spawn_remote('h100', 'proxy', proxy, env, server_root, server_python,
                '/tmp/r17_formal/'+campaign['run_name']+'/server')
        endpoint_port = campaign.get('endpoint_port', campaign['public_port'])
        if (campaign['model']=='cosmos' or campaign.get('library') or
                (campaign['endpoint_host'], endpoint_port) != (control_host, campaign['public_port'])):
            # An exposed broker port accepts TCP before its local service is
            # ready. Wait for the actual stock proxy before its public tunnel.
            wait_port(control_host, campaign['public_port'], required_label='proxy')
        wait_port(campaign['endpoint_host'], endpoint_port, required_label='proxy')
        from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
        ctl = WebsocketClientPolicy(campaign['endpoint_host'], endpoint_port)
        try:
            metadata = ctl.get_server_metadata()
            write_json(directory / 'PROXY_METADATA.json', metadata)
            correct_mode = (metadata.get('library_teacher') is True and metadata.get('debug_capture') is True
                if campaign.get('library') else metadata.get('formal_pure_inference') and not metadata.get('debug_capture'))
            if not correct_mode or metadata.get('cache_enabled'):
                pass
            # Publish before worker launch; broadcast must acknowledge every replica.
            ack = ctl.load_cache_config(str(Path(campaign['local_assets'])/'full.yaml'),
                yaml_id='formal_full_v1', bundle_id='formal_full_v1')
            write_json(directory / 'BUNDLE_ACK.json', ack)
            if campaign.get('cache_study'):
                from .cache_study import publish
                write_json(directory / 'STUDY_BUNDLE_ACKS.json', publish(ctl, campaign))
        finally:
            ctl.close()
        if campaign.get('orchestration'):
            from .driver_release import validate as validate_orchestration
            validate_orchestration(campaign,directory)
        driver = spawn_local('driver', [CONTROL_PYTHON, '-u', '-m', campaign.get('driver_module','exp.r17_integration.ops.formal_driver'),
            '--campaign', str(directory / 'campaign.json'), '--directory', str(directory / 'driver')],
            dict(PYTHONPATH=pythonpath(campaign.get('driver_runtime',campaign.get('coordinator_runtime',str(ROOT)))), OMP_NUM_THREADS='1'))
        wait_port('127.0.0.1', campaign['driver_port'], 600, required_label='driver')
        if not workers_started:
            for node, item in deployment.items():
                spawn_remote(node, 'workers', [item['python'], '-u', '-m', 'exp.r17_integration.ops.formal_workers',
                    '--campaign', item.get('campaign_path',item['stage']+'/campaign.json')], item['env'], item['payload'], item['python'], item['stage'])
        while driver.poll() is None:
            check()
            for owned in remote_owned:
                status = status_cache.get((owned['node'],owned['label']),{})
                if owned.get('kind',owned['label']) == 'workers' and not owned.get('retired_worker_node') and status.get('status') == 'CLOSED' and status.get('exit_code'):
                    report['worker_fleet_failure'] = failed_fleet(owned['node'], deployment[owned.get('fleet_key',owned['node'])], status)
                    detail = report['worker_fleet_failure'].get('fleet_receipt', {})
                    raise RuntimeError('remote worker fleet failed: '+owned['node']+'; '+
                        str(detail.get('reason', 'receipt unavailable'))+'; worker_failures='+
                        json.dumps(detail.get('worker_failures', [])))
            time.sleep(2)
        report['driver_completion'] = driver_completion(directory, driver.returncode)
        if live_fleets is not None:live_fleets.freeze()
        save()
        for owned in remote_owned:
            if owned.get('kind',owned['label']) != 'workers':
                continue
            if owned.get('retired_worker_node'):continue
            if report['driver_completion'].get('verified_completion'):
                # All cells/attempts are verified. A recycled worker can miss
                # the final pull; close only this owned fleet and retain its snapshot.
                try:
                    drained = drain_completed_workers(owned, report['driver_completion'], managed)
                except Exception as exc:
                    if worker_state is None:raise
                    worker_state.retire(owned['node'],dict(drain_error=repr(exc)),owned)
                    continue
                report.setdefault('completed_worker_drains', {})[owned.get('fleet_key',owned['node'])] = drained
                if drained.get('snapshot_incomplete'):
                    from .formal_finalize import record_close_gap
                    record_close_gap(report, owned.get('fleet_key',owned['node']), drained)
                status_cache[owned['node'], owned['label']] = drained
                save()
            deadline = min(until, time.monotonic()+90+RETRY_SECONDS)
            while time.monotonic() < deadline:
                check()
                if owned.get('retired_worker_node'):break
                status = status_cache.get((owned['node'],owned['label']),{})
                if status.get('status') == 'CLOSED':
                    break
                time.sleep(.5)
            if owned.get('retired_worker_node'):continue
            allowed_exit = (0, 2) if report['driver_completion'].get('recovered_worker_shutdown') else (0,)
            if (status.get('status') != 'CLOSED' or
                    status.get('exit_code') not in allowed_exit and not status.get('snapshot_incomplete') or
                    report['driver_completion'].get('recovered_worker_shutdown') and status.get('alive') is not False):
                raise RuntimeError('remote worker close/snapshot incomplete')
        # Evidence can be large. Close every owned fleet/resident/MPS daemon
        # below before snapshot transfer, aggregation or scientific sealing.
        report['status'] = 'PENDING_EVIDENCE'
        report['evidence_pending'] = True
    except NativeQualificationComplete:
        report.update(status='PASS', qualification='official/staged native parity only', episodes_dispatched=0)
    except BaseException as exc:
        report.update(status='PARTIAL' if isinstance(exc, TimeoutError) else 'FAIL', error=repr(exc))
        save()
        print(str(exc), file=sys.stderr, flush=True)
    finally:
        if live_fleets is not None:live_fleets.freeze()
        for owned in reversed(remote_owned):
            try:
                result = (worker_state.cleanup(owned,managed) if owned.get('retired_worker_node')
                          else managed(owned, 'stop'))
                report['cleanup'].append(dict(node=owned['node'], label=owned['label'], **result,
                    **({k:owned[k] for k in ('kind','fleet_key','fleet_id') if k in owned}),
                    closure_confirmed=result.get('alive') is False))
            except Exception as exc:
                report['cleanup'].append(dict(node=owned['node'], label=owned['label'], error=repr(exc),
                                              **({k:owned[k] for k in ('kind','fleet_key','fleet_id') if k in owned}),
                                              closure_confirmed=False))
                if report['status'] != 'FAIL':
                    report['status'] = 'PARTIAL'
        for child, entry in reversed(local):
            try:
                result = stop_group(entry, grace=30) if entry['label']=='mps_guard' else stop_group(entry)
                entry['exit_code'] = child.wait(timeout=15)
                report['cleanup'].append(dict(label=entry['label'], **result,closure_confirmed=not alive(entry)))
            except Exception as exc:
                if not campaign.get('mps'):
                    raise
                report['status'] = 'PARTIAL'
                report['cleanup'].append(dict(label=entry['label'],error=repr(exc),closure_confirmed=False))
        for handle in handles:
            handle.close()
        if mps_ready is not None:
            try:
                final = mps_command('state')
                report.setdefault('mps', {})['closure'] = final
                if not final.get('daemon_closed'):
                    report['status'] = 'PARTIAL'
                report['cleanup'].append(dict(label='mps_daemon', closure_confirmed=bool(final.get('daemon_closed'))))
            except Exception as exc:
                report['status'] = 'PARTIAL'
                report['cleanup'].append(dict(label='mps_daemon', closure_confirmed=False, error=repr(exc)))
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        report['all_owned_closed'] = all(not alive(entry) for _,entry in local) and all(
            r['closure_confirmed'] for r in report['cleanup'])
        if worker_state is not None and worker_state.retired:
            report['active_owned_closed'] = all(not alive(entry) for _,entry in local) and all(
                r['closure_confirmed'] for r in report['cleanup'] if r.get('node') not in worker_state.retired)
            if (report['status']=='PENDING_EVIDENCE' and report.get('driver_completion',{}).get('verified_completion')
                    and report['active_owned_closed'] and not report['all_owned_closed']):
                report.update(gpu_released=True,worker_node_closure_pending=True)
        if live_fleets is not None:
            try:live_fleets.seal_owners(report)
            except Exception as exc:
                report.update(status='FAIL',live_fleet_seal_error=repr(exc))
        if report.get('error') == "RuntimeError('remote worker close/snapshot incomplete')" and report['all_owned_closed']:
            from .formal_finalize import completed_close_error
            try:
                if completed_close_error(directory, report):
                    report.update(status='PENDING_EVIDENCE', evidence_pending=True)
            except (ValueError, RuntimeError):
                pass  # Retain the original FAIL unless completion/closure proves the narrow case.
        if shard_pump is not None:
            try:
                if report['status'] in ('PASS','PENDING_EVIDENCE') and report['all_owned_closed']:
                    report.setdefault('transfers',{})['h100_shards'] = shard_pump.finish()
                else:
                    shard_pump.abort()
                    report['h100_partial_shards'] = shard_pump.summary()
                if shard_pump.failure:
                    report['shard_transfer_failure'] = shard_pump.failure
            except Exception as exc:
                report.update(status='PARTIAL',shard_transfer_error=repr(exc))
                if shard_pump.failure:
                    report['shard_transfer_failure'] = shard_pump.failure
                print('H100 shard transfer shutdown failed: '+str(exc),file=sys.stderr,flush=True)
        save()
        if report['status'] in ('PASS','PENDING_EVIDENCE') and report['all_owned_closed']:
            from .formal_finalize import finalize, handoff
            try:
                if not any(campaign.get(k) for k in ('smoke','library','native_parity_only')):
                    # Exit success here means a verified GPU handoff, NOT a
                    # sealed scientific PASS. Receipts remain PENDING_EVIDENCE.
                    report = handoff(campaign, directory, deployment, report)
                else:
                    report = finalize(campaign, directory, deployment, report)
            except Exception as exc:
                report.update(status='FAIL', evidence_pending=False, finalization_error=repr(exc))
                save()
        else:
            report['backend_evidence'] = backend_evidence(directory)
            save()
    print(json.dumps(dict(status=report['status'], run=str(directory), all_owned_closed=report['all_owned_closed'])))
    if (worker_state is not None and report.get('worker_node_closure_pending') and
            report.get('active_owned_closed') and report['status']=='PENDING_EVIDENCE' and
            report.get('driver_completion',{}).get('verified_completion')):
        return 0  # Full plan done, services released; unknown worker closure stays explicit.
    return 0 if report['all_owned_closed'] and (report['status'] == 'PASS' or
        report['status'] == 'PENDING_EVIDENCE' and report.get('gpu_released') is True) else 2


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', nargs='?', choices=('launch', 'seal', 'aggregate'), default='launch')
    p.add_argument('--campaign')
    p.add_argument('--semantic-campaign', help='Stable scientific namespace across paired runtime trials')
    p.add_argument('--model', choices=tuple(DEFAULT_CHECKPOINTS), default='dp_libero')
    p.add_argument('--checkpoint')
    p.add_argument('--server-node', choices=('local', 'h100'))
    p.add_argument('--worker-node', choices=('timan107','local','timan1'))
    from .local_workers import counts
    p.add_argument('--worker-counts', type=counts, help='Desired per-node maxima, e.g. timan1=150,timan107=100; runtime chooses active counts')
    p.add_argument('--local-worker-memory-mib',type=int,default=4096)
    p.add_argument('--local-worker-vram-mib',type=int,default=1024)
    p.add_argument('--worker-autoscale', choices=('dynamic','fixed'), default='dynamic')
    p.add_argument('--worker-memory-floor-gib', type=float, default=20)
    p.add_argument('--worker-memory-critical-gib', type=float, default=10)
    p.add_argument('--extra-workers', type=int, default=0, help='Workers on the other timan node')
    p.add_argument('--replicas', type=int)
    p.add_argument('--mps', action='store_true',
                   help='Opt-in private per-campaign CUDA MPS; replicas only; existing compute mode unchanged')
    p.add_argument('--replica-startup-width', type=int,
                   help='Concurrent load wave (default DP4, Fast-WAM2, other1); resident caps/self-checks remain mandatory')
    p.add_argument('--max-batch', type=int, default=1,
                   help='Native cross-connection batching (pi05/GR00T/Fast-WAM/DP, LIBERO or MT50)')
    p.add_argument('--batch-wait-ms', type=float, default=5., help='Native stage batching window, milliseconds')
    p.add_argument('--wire-zlib', action='store_true', help='Workers zlib-compress large uint8 arrays (lossless; server accepts both)')
    p.add_argument('--node-egl-devices', help='Per-node EGL devices (kernel minors), e.g. timan1=2;timan107=0,1,2,3,4,5,6,7')
    p.add_argument('--paired-control', type=Path,
                   help='Completed same-GPU/batching official-50 Fast-WAM K10/C5 control; required for batched arms')
    p.add_argument('--runtime-from', type=Path,
                   help='Use this baseline run\'s verified frozen server, driver and worker code; never reseal working-tree code')
    p.add_argument('--driver-mode', choices=('legacy','group-commit'),default='legacy',
                   help='Separately pinned orchestration; group-commit leaves frozen model/worker code unchanged')
    p.add_argument('--batch-qualification-observations', type=Path,
                   help='Smoke only: recorded Fast-WAM observations on server for startup B1/B8 parity and throughput')
    p.add_argument('--workers', type=int, default=None, help='Desired maximum; omitted: benchmark renderer hardware limit')
    p.add_argument('--worker-memory-mib', type=int,
                   help='Node RAM reservation per worker (default LIBERO6144, MW2048 MiB)')
    p.add_argument('--worker-min-free-gib',type=float,default=8,
                   help='Minimum available GiB on worker run/stage filesystems before launch/recycle (default8)')
    p.add_argument('--worker-restart-limit', type=int, default=2,
                   help='Bounded replacements per failed worker; 0 restores fail-fast behavior')
    p.add_argument('--worker-max-episodes', type=int,
                   help='Recycle at episode boundaries (default LIBERO4, MW32); 0 disables; separate from failure restart budget')
    p.add_argument('--gpu', default='cuda:0')
    p.add_argument('--worker-egl-device', type=int, default=0)
    p.add_argument('--worker-egl-devices', type=lambda value: [int(x) for x in value.split(',')],
                   help='Optional physical LIBERO GPU allowlist; default all usable GPUs, round-robin; timan108 disabled')
    p.add_argument('--worker-vram-mib', type=int, default=1024,
                   help='LIBERO VRAM reservation per worker; default1024MiB plus per-GPU headroom')
    p.add_argument('--port', type=int)
    p.add_argument('--driver-port', type=int)
    p.add_argument('--official-50', action='store_true', help='Separate sealed original init0..49 campaign')
    p.add_argument('--eval-pool', choices=('A','B'), default='A', help='Pure-inference input pool; B reuses the audited library inits')
    p.add_argument('--init-count', type=int, choices=range(1,51), default=None, metavar='N', help='Pure inference: original indices 0..N-1')
    p.add_argument('--cache-study', type=Path, help='Default-off frozen eight-group YAML manifest; A500/B500 library')
    p.add_argument('--cache-control', type=Path, help='Cache study only: read-only matched complete control')
    p.add_argument('--cache-control-runtime', type=Path, help='Cache study only: native control archive beneath reviewed R17 overlay')
    p.add_argument('--cache-resume-from', type=Path, help='Cache study only: closed segment; retain only complete verified episodes')
    p.add_argument('--live-fleets', action='store_true', help='Default-off pinned live attach/drain worker protocol')
    p.add_argument('--commit-horizon', type=int, default=None,
                   help='Opt-in actions per model inference; reuses policy tails at native transport cadence')
    p.add_argument('--denoising-steps', type=int, default=None,
                   help='Plain step study at native horizon: DP DDIM 5/10 or DDPM100; Fast 1/2/10; Cosmos 1/2/3/5 (K2 includes clean budget; NFE1)')
    p.add_argument('--suites', nargs='+', default=None,
                   help='Optional official-50 LIBERO suite filter; preserves original plan identities')
    p.add_argument('--smoke', action='store_true', help='Four complete original-protocol episodes')
    p.add_argument('--smoke-episodes', type=int, choices=(2,4), default=4, help='LIBERO smoke size; full campaigns unchanged')
    p.add_argument('--native-parity-only', action='store_true', help='Cosmos qualification A: owned residents run startup parity, then close; no episodes')
    p.add_argument('--run', action='store_true', help='Explicitly launch GPU servers and dispatch campaign')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--prepared-run', type=Path, help='Launch a sealed --prepare-only run, without repeating preparation')
    p.add_argument('--defer-control-check', action='store_true',
                   help='Prepare only: stage an arm while its matched control runs; closed PASS gate remains mandatory at launch')
    p.add_argument('--run-name')
    p.add_argument('--resume-from', type=Path, help='Closed identical pure-inference run; retain accepted cells and dispatch only missing cells')
    p.add_argument('--resume-waive-unreachable-source-node', choices=('timan107','timan108'),
                   help='Explicit worker-closure exception: terminal source, closed policy/driver, offline node, isolated destination')
    p.add_argument('--output', type=Path)
    p.add_argument('--seconds', type=int, default=43200)
    p.add_argument('--episode-seconds', type=int, default=1800)
    p.add_argument('--retries', type=int, default=2)
    p.add_argument('--memory-budget-gib', type=float)
    args = p.parse_args(argv)
    if any((args.cache_control,args.cache_control_runtime,args.cache_resume_from)) and not args.cache_study:
        p.error('cache control/resume flags require --cache-study')
    if bool(args.cache_control) != bool(args.cache_control_runtime):
        p.error('--cache-control and --cache-control-runtime are required together')
    if args.cache_study:
        if (args.model not in ('dp_libero','fastwam_libero','cosmos_libero','pi05_libero','groot_libero','dp_mw','fastwam_mw','pi05_mw','groot_mw') or
                args.model.endswith('_libero') and not args.official_50 or
                args.runtime_from or args.resume_from or args.commit_horizon is not None or
                args.denoising_steps is not None or args.native_parity_only):
            p.error('cache study requires a supported model (official-50 for LIBERO) and per-YAML horizons; fresh runtime only')
        args.driver_mode = 'legacy' if args.model == 'cosmos_libero' else 'group-commit'
    if args.runtime_from is not None and args.resume_from is not None:
        p.error('--runtime-from and --resume-from select different source semantics; use one')
    if args.resume_waive_unreachable_source_node and args.resume_from is None:
        p.error('--resume-waive-unreachable-source-node requires --resume-from')
    if args.prepared_run is not None:
        supplied=list(argv if argv is not None else sys.argv[1:])
        if not args.run or args.prepare_only or any(x.startswith('--') and x.split('=')[0] not in ('--run','--prepared-run') for x in supplied):
            p.error('--prepared-run takes only --run; change settings by preparing a new campaign')
        return args
    if args.mode != 'launch':
        return args
    if args.mps and args.gpu != 'cuda:0':
        p.error('--mps requires --gpu cuda:0; select a physical GPU with CUDA_VISIBLE_DEVICES')
    cosmos = args.model in ('cosmos_libero', 'cosmos_mw')
    from exp.r17_integration.models.fastwam.batch_runtime import BatchConfig
    try:
        if args.model in ('dp_libero', 'dp_mw') and args.max_batch > 1:
            from exp.r17_integration.models.dp.batch_runtime import BatchConfig
        elif args.model in ('pi05_mw', 'pi05_libero', 'groot_mw', 'groot_libero'):
            from exp.r17_integration.common.batch_runtime import BatchConfig
        BatchConfig(args.max_batch, args.batch_wait_ms)
        if args.max_batch > 1 and args.model not in ('fastwam_libero', 'dp_libero', 'fastwam_mw', 'dp_mw',
                                                   'pi05_mw', 'pi05_libero', 'groot_mw', 'groot_libero'):
            raise ValueError('--max-batch supports pi05/GR00T/Fast-WAM/DP LIBERO or MT50')
        if args.paired_control and (args.max_batch == 1 or args.model != 'fastwam_libero'):
            raise ValueError('--paired-control requires Fast-WAM batching')
        if args.batch_qualification_observations and (not args.smoke or args.max_batch <= 1 or args.model != 'fastwam_libero'):
            pass
    except ValueError as exc:
        p.error(str(exc))
    study = dict(model=args.model.rsplit('_', 1)[0], domain='libero' if args.model.endswith('_libero') else 'metaworld',
                 denoising_steps=args.denoising_steps, commit_horizon=args.commit_horizon, suites=args.suites,
                 official_50=args.official_50, smoke=args.smoke, native_parity_only=args.native_parity_only)
    try:
        execution_steps(study, None)
        selected_suites(study)
    except ValueError as exc:
        p.error(str(exc))
    if args.commit_horizon is not None:
        geometry = {'dp_libero': (5, 16), 'fastwam_libero': (5, 32), 'cosmos_libero': (16, 16),
                    'dp_mw': (5, 16), 'fastwam_mw': (5, 32), 'pi05_mw': (5, 5), 'groot_mw': (5, 16),
                    'pi05_libero': (5, 10), 'groot_libero': (5, 16)}
        if args.model not in geometry or not geometry[args.model][0] <= args.commit_horizon <= geometry[args.model][1]:
            p.error('--commit-horizon must fit the native chunk: DP/GR00T16, Fast-WAM32, pi05 MT50 5, Cosmos16')
        if args.model.endswith('_mw') and args.commit_horizon % 5:
            p.error('MetaWorld commit must be divisible by native replan steps (5)')
        if args.native_parity_only:
            p.error('--commit-horizon cannot be combined with --native-parity-only')
    if cosmos and getattr(args,'library',False):
        p.error('Cosmos model capture/library qualification is unsupported')
    if args.native_parity_only and not cosmos:
        p.error('--native-parity-only is Cosmos only')
    args.server_node = args.server_node or ('h100' if cosmos or args.model == 'groot_libero' else 'local')
    args.worker_node = args.worker_node or 'timan107'
    if args.workers is None:
        from exp.offline_search.rounds.r17.autoscale_1017.defaults import physical_maximum
        from exp.offline_search.rounds.r17.autoscale_1017.switch import enabled
        args.workers = physical_maximum(args.worker_node, 'libero' if args.model.endswith('_libero') else 'metaworld') if enabled() else 24
    args.port = args.port if args.port is not None else (23240 if cosmos else 23160)
    args.driver_port = args.driver_port if args.driver_port is not None else (
        23112 if cosmos else 23169 if args.model in ('pi05_libero', 'groot_libero') else 23179)
    if args.smoke_episodes != 4 and (not args.smoke or not args.model.endswith('libero')):
        p.error('--smoke-episodes requires a LIBERO --smoke')
    if args.official_50 and not args.model.endswith('libero'):
        p.error('--official-50 is LIBERO only')
    if args.model == 'groot_libero' and args.official_50 and args.suites is None and not args.smoke:
        args.suites = ['libero_spatial', 'libero_10']
    if args.prepare_only and args.run:
        p.error('--prepare-only and --run are mutually exclusive')
    if not args.run and not args.prepare_only:
        p.error('select --prepare-only (CPU) or --run')
    if args.replicas is None:
        args.replicas = 12 if args.model.startswith('dp_') else 1
    if args.replica_startup_width is None:
        args.replica_startup_width = min(args.replicas,4 if args.model.startswith('dp_') else 2 if args.model.startswith('fastwam') else 1)
    if not 1 <= args.replica_startup_width <= args.replicas:
        p.error('startup width must be between1 and replica count')
    if args.defer_control_check and (not args.prepare_only or not args.paired_control or args.model!='fastwam_libero' or args.max_batch<=1):
        p.error('--defer-control-check requires prepare-only, Fast-WAM batching and a paired control')
    if args.workers < 1 or args.extra_workers < 0 or args.replicas < 1 or args.retries < 0:
        p.error('positive replicas/workers and nonnegative retries required')
    if args.worker_counts and args.extra_workers:
        p.error('--worker-counts cannot be combined with --extra-workers')
    if args.local_worker_memory_mib < 1 or args.local_worker_vram_mib < 1:
        p.error('positive local worker RAM/VRAM reservations required')
    if args.worker_memory_mib is None:
        args.worker_memory_mib = 6144 if args.model.endswith('libero') else 2048
    if args.worker_max_episodes is None:
        args.worker_max_episodes = 4 if args.model.endswith('libero') else 32
    if args.worker_memory_mib < 1 or args.worker_restart_limit < 0 or args.worker_max_episodes < 0:
        p.error('positive worker memory and nonnegative worker restart limit required')
    if not 0 < args.worker_min_free_gib < 1024:
        p.error('--worker-min-free-gib must be positive and below1024')
    if args.worker_vram_mib < 1 or args.worker_egl_device < 0 or (args.worker_egl_devices is not None and
            (not args.worker_egl_devices or any(d < 0 for d in args.worker_egl_devices) or
             len(set(args.worker_egl_devices)) != len(args.worker_egl_devices))):
        p.error('positive VRAM reservation and distinct nonnegative EGL devices required')
    ports = [args.port, *range(args.port+1, args.port+1+args.replicas), args.driver_port]
    reserved = (range(23400,23460) if args.cache_study else
                range(23240,23250) if cosmos else range(23180,23190) if args.port >= 23180 else range(23160,23180))
    # The driver serves remote workers through ziyanglin.com, which forwards only 23100..23199.
    # A local server with remote workers listens inside the forwarded 23100..23199 range as well.
    if (len(ports) != len(set(ports)) or any(port not in reserved and port not in DIRECT_PORTS for port in ports[:-1])
            or (args.driver_port not in reserved and args.driver_port not in DIRECT_PORTS)):
        p.error('distinct reserved ports required: '+('23400..23459 (cache study)' if args.cache_study else
            '23240..23249 (Cosmos)' if cosmos else '23160..23179 or 23180..23189')+'; driver may also use 23100..23199')
    max_seconds=172800
    if args.cache_study and args.seconds>max_seconds:
        from .cache_study import validate_manifest
        max_seconds*=len(validate_manifest(args.cache_study).get('library_sizes',[500]))
    if not 60 <= args.seconds <= max_seconds or not 30 <= args.episode_seconds <= args.seconds:
        p.error('campaign bound60..'+str(max_seconds)+'s; episode bound30..campaign seconds')
    if args.memory_budget_gib is None:
        args.memory_budget_gib = 14 if args.model.startswith('dp_') else (36 if args.model.startswith('fastwam') else 10 if cosmos else 20)
    if args.model.startswith('dp_') and args.replicas > args.memory_budget_gib:
        p.error('DP replica count exceeds conservative1GiB/replica memory budget')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.cache_resume_from:
        allowed={'--model','--cache-study','--official-50','--cache-resume-from','--run-name','--prepare-only','--run'}
        supplied=list(argv if argv is not None else sys.argv[1:])
        if any(t.startswith('--') and t.split('=')[0] not in allowed for t in supplied):
            pass
        old=json.loads((args.cache_resume_from/'campaign.json').read_text())
        if old['model']!=args.model.rsplit('_',1)[0] or old['cache_study']!=str(args.cache_study.resolve()):
            pass
        from .cache_resume import prepare as prepare_cache_resume
        from .formal_prestage import launch_prepared
        target=prepare_cache_resume(args.cache_resume_from,args.run_name or 'cc'+str(time.time_ns())[-10:])
        if args.run:return launch_prepared(target)
        print(json.dumps(dict(status='PASS',cpu_only=True,run=str(target))));return 0
    if args.prepared_run is not None:
        from .formal_prestage import launch_prepared
        return launch_prepared(args.prepared_run)
    if args.mode == 'seal':
        seal(json.loads(Path(args.campaign).read_text()))
        return 0
    if args.mode == 'aggregate':
        print(json.dumps(aggregate(args.output)))
        return 0
    name = args.run_name or ('formal_'+args.model+'_'+('smoke'+str(args.smoke_episodes) if args.smoke else 'official2000' if args.official_50 else 'formal400' if args.model.endswith('libero') else 'mt50_500')+'_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
    if not name or len(name) > 150 or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in name):
        raise ValueError('run name must be a unique plain identifier')
    directory = args.output or FORMAL_ROOT / name
    if directory.resolve().parent != FORMAL_ROOT:
        raise ValueError('results must be a direct child of /data/mw_finetune/eval')
    if args.mps:
        from exp.r17_integration.models.fastwam.ops.mps import paths
        if args.output is not None and args.run_name != directory.name:
            raise ValueError('--mps --output requires a matching --run-name')
        if args.run_name is None and args.output is None:
            name = 'mps_'+args.model+'_'+time.strftime('%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
            directory = FORMAL_ROOT / name
        # UUID hashes have fixed width. Reject long custom names during CPU
        # preparation, before creating processes or a daemon.
        paths(directory, 'GPU-placeholder')
    directory.mkdir(parents=True, exist_ok=False)
    model, suffix = args.model.rsplit('_', 1)
    domain = 'libero' if suffix == 'libero' else 'metaworld'
    host = 'ziyanglin.com' if args.server_node == 'local' else h100_host()
    from exp.offline_search.rounds.r17.autoscale_1017.defaults import physical_maximum
    worker_nodes = args.worker_counts or {args.worker_node: args.workers if args.workers is not None else physical_maximum(args.worker_node, domain)}
    if args.extra_workers:
        worker_nodes['timan1' if args.worker_node == 'timan107' else 'timan107'] = args.extra_workers
    campaign = dict(run_name=name, semantic_campaign=args.semantic_campaign or name, model=model, domain=domain,
        checkpoint=args.checkpoint or ('/home/exouser/ckpt' if args.model == 'groot_libero' and args.server_node == 'h100'
                                      else DEFAULT_CHECKPOINTS[args.model]),
        device=args.gpu, server_node=args.server_node, worker_nodes=worker_nodes,driver_mode=args.driver_mode,
        worker_egl_device=args.worker_egl_device, workers=sum(worker_nodes.values()), replicas=args.replicas,
        replica_startup_width=args.replica_startup_width,
        worker_egl_devices=args.worker_egl_devices, worker_vram_mib=args.worker_vram_mib,
        worker_memory_mib=args.worker_memory_mib, worker_restart_limit=args.worker_restart_limit,
        worker_min_free_gib=args.worker_min_free_gib,
        local_worker_memory_mib=args.local_worker_memory_mib,local_worker_vram_mib=args.local_worker_vram_mib,
        worker_max_episodes=args.worker_max_episodes, worker_max_seconds=1800,
        autoscale=dict(mode=args.worker_autoscale, floor=int(args.worker_memory_floor_gib*1024**3),
                       critical=int(args.worker_memory_critical_gib*1024**3)),
        autoscale_maximum=sum(worker_nodes.values()),
        public_port=args.port, backend_ports=list(range(args.port+1,args.port+1+args.replicas)),
        ports=[args.port,*range(args.port+1,args.port+1+args.replicas),args.driver_port],
        driver_host='ziyanglin.com', driver_port=args.driver_port,
        endpoint_host=host, endpoint=host+':'+str(args.port),
        official_50=args.official_50, smoke=args.smoke, smoke_episodes=args.smoke_episodes,
        native_parity_only=args.native_parity_only, seconds=args.seconds,
        prepare_only=args.prepare_only,
        episode_seconds=args.episode_seconds, retries=args.retries, memory_budget_gib=args.memory_budget_gib,
        local_assets=str(directory / 'assets'), assets=str(FORMAL_ROOT / name / 'assets'),
        server_run=str(FORMAL_ROOT / name / 'servers'), runtime=str(ROOT),
        vendor='/home/weiland/projects/openpi_ext/third_party/gr00t_n15', live_k=4)
    if args.eval_pool == 'B':campaign['eval_pool']='B'
    if args.init_count is not None:campaign['init_count']=args.init_count
    if model == 'groot' and args.denoising_steps is not None:
        campaign['live_k'] = args.denoising_steps
    if model == 'groot' and args.cache_study:
        # GR00T's endpoint K is fixed at startup; start it at the study YAMLs' K.
        import yaml
        study = json.loads(args.cache_study.read_text())
        row = study['models'][args.model][0]
        campaign['live_k'] = yaml.safe_load((args.cache_study.parent/row['file']).read_text())['adapter_binding']['num_steps']
    if args.cache_study:
        campaign.update(cache_study=str(args.cache_study.resolve()),
                        driver_module='exp.r17_integration.ops.formal_driver' if args.driver_mode == 'legacy'
                        else 'exp.r17_integration.ops.formal_driver_fast')
        if args.cache_control:
            campaign.update(cache_control=str(args.cache_control.resolve()),
                            cache_control_runtime=str(args.cache_control_runtime.resolve()))
        if args.cache_resume_from:
            campaign['cache_resume_from']=str(args.cache_resume_from.resolve())
    if args.live_fleets:campaign['live_fleets']=True
    if 'local' in worker_nodes:
        from .local_workers import authorize
        campaign['local_worker_exception'] = authorize(campaign)
    if 'timan1' in worker_nodes:
        from .timan1_workers import authorize
        campaign['timan1_worker_lease']=authorize(campaign)
    if args.defer_control_check:
        campaign['defer_control_check']=True
    if args.commit_horizon is not None:
        campaign['commit_horizon'] = args.commit_horizon
    if args.denoising_steps is not None:
        campaign['denoising_steps'] = args.denoising_steps
    if args.suites is not None:
        campaign['suites'] = args.suites
    if model == 'cosmos':
        campaign['vendor'] = '/data/cosmos/cosmos-policy' if args.server_node == 'h100' else '/home/weiland/projects/openpi_ext/lines/cosmos-policy'
    if (args.max_batch > 1 or args.paired_control) and model == 'fastwam':
        from exp.r17_integration.models.fastwam.formal_batching import bind
        if campaign.get('cache_study'):
            # Native full.yaml starts the resident; each study YAML owns C.
            # Absence uses the model-owned admission's native K10/C5 defaults.
            # Never select its frozen-control runtime for this new framework.
            campaign.pop('commit_horizon', None)
            campaign.pop('denoising_steps', None)
        bind(campaign, args.max_batch, args.batch_wait_ms, args.paired_control)
    elif args.max_batch > 1 and model == 'dp':
        from exp.r17_integration.models.dp.batch_runtime import BatchConfig
        campaign['dp_batching'] = BatchConfig(args.max_batch, args.batch_wait_ms).metadata()
    elif args.max_batch > 1 and model in ('pi05', 'groot'):
        from exp.r17_integration.common.batch_runtime import BatchConfig
        campaign['stage_batching'] = BatchConfig(args.max_batch, args.batch_wait_ms).metadata()
    if args.wire_zlib:
        campaign['wire_zlib'] = True
    if args.node_egl_devices:
        campaign['worker_egl_devices_by_node'] = {n: [int(d) for d in v.split(',')] for n, v in (x.split('=') for x in args.node_egl_devices.split(';'))}
    if args.batch_qualification_observations:
        campaign['batch_qualification_observations'] = str(args.batch_qualification_observations)
    if args.mps:
        from exp.r17_integration.models.fastwam.ops.mps import bind
        bind(campaign, True)
    if args.runtime_from is not None or (args.paired_control is not None and args.resume_from is None):
        campaign['runtime_from'] = str(args.runtime_from or args.paired_control)
    if args.resume_from is not None:
        from exp.r17_integration.models.fastwam.ops.mps import check_resume as check_mps_resume
        check_mps_resume(campaign, args.resume_from)
        if args.resume_waive_unreachable_source_node:
            campaign['resume_waive_unreachable_source_node'] = args.resume_waive_unreachable_source_node
        if model == 'fastwam':
            from exp.r17_integration.models.fastwam.formal_batching import check_resume
            check_resume(campaign, args.resume_from)
        from .formal_resume import attach
        attach(campaign, args.resume_from)
    tunnels = []
    try:
        if not args.prepare_only:
            tunnels = open_tunnels(campaign,directory)
        write_json(directory / 'campaign.json', campaign)
        deployment = prepare(campaign, directory)
        if args.prepare_only:
            print(json.dumps(dict(status='PASS', cpu_only=True, run=str(directory))))
            return 0
        return run_campaign(campaign, directory, deployment)
    finally:
        close_tunnels(tunnels,directory)


if __name__ == '__main__':
    raise SystemExit(main())
