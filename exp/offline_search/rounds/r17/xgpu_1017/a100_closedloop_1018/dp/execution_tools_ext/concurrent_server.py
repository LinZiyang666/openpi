"""Isolated formal residents + stock proxy + real concurrent protocol clients.

Default is CPU plan/validation. --run is coordinator-only on 4090/H100.
Only Popen children and this run's private MPS daemon are stopped. No remote
launcher, running lane, shared daemon, driver or worker is changed.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import random
import signal
import shutil
import socket
import string
import subprocess
import sys
import time
from artifacts import digest, sha, write_json
from capture import read_inputs
from replay import verify_runtime


def mps_directory_plan(base, gpu_uuid, frozen_paths, name):
    """Use the pinned runtime's real path builder, including resolve/byte limits."""
    if len(name) != 2 or any(c not in string.ascii_letters + string.digits for c in name):
        raise ValueError('private MPS campaign name must be two ASCII alphanumerics')
    directory = Path(base).resolve() / name
    root, pipe, log = frozen_paths(directory, gpu_uuid)
    pipe_bytes = len(os.fsencode(pipe))
    if pipe_bytes > 85:
        raise ValueError(f'private MPS pipe exceeds 85 bytes: {pipe_bytes}')
    return dict(campaign_directory=str(directory), root=str(root),
                pipe_directory=str(pipe), log_directory=str(log), pipe_bytes=pipe_bytes)


def mps_names():
    alphabet = string.ascii_letters + string.digits
    names = [a+b for a in alphabet for b in alphabet]
    random.SystemRandom().shuffle(names)
    return names


def reserve_mps_directory(base, gpu_uuid, frozen_paths):
    """Atomically reserve a fresh short directory; never adopt another session.

    Two ASCII bytes leave the local 4090 pipe at exactly 85 bytes. Resolve
    through the frozen builder before creating anything; a symlink cannot
    shorten the limit. Existing files/directories/links are left untouched.
    """
    base = Path(base).resolve(strict=True)
    for name in mps_names():
        candidate = base / name
        if os.path.lexists(candidate):
            continue
        plan = mps_directory_plan(base, gpu_uuid, frozen_paths, name)
        try:
            candidate.mkdir(mode=0o700)
        except FileExistsError:
            continue  # Another launcher reserved it after our path check.
        return plan
    raise RuntimeError('private MPS short-name namespace exhausted; no session reused')


def checked_pin(bundle):
    bundle = Path(bundle)
    pin = json.loads((bundle / 'PIN.json').read_text())
    body = dict(pin)
    identity = body.pop('identity')
    if digest(body) != identity:
        raise ValueError('pin identity changed')
    if sha(bundle / 'source_campaign.json') != pin['source_campaign_sha256']:
        raise ValueError('source campaign changed')
    for name, expected in pin['files'].items():
        p = (bundle / 'assets' / name).resolve()
        if not p.is_relative_to((bundle / 'assets').resolve()) or sha(p) != expected:
            raise ValueError('pinned asset changed: ' + name)
    return pin


def campaign_for(bundle, output, runtime, port, python, replicas=None, memory=None):
    c = json.loads((Path(bundle) / 'source_campaign.json').read_text())
    source = copy.deepcopy(c)
    # Only isolated paths/ports and explicit capacity arms change. Scientific
    # fields, sealed bootstrap, batching, sampler, recipe and MPS are retained.
    c.update(assets=str(Path(bundle).resolve() / 'assets'),
        local_assets=str(Path(bundle).resolve() / 'assets'), runtime=str(Path(runtime).resolve()),
        coordinator_runtime=str(Path(runtime).resolve()), server_python=python,
        server_run=str(Path(output).resolve() / 'servers'),
        endpoint_host='127.0.0.1', backend_host='127.0.0.1',
        public_port=port, endpoint_port=port, endpoint=f'127.0.0.1:{port}',
        run_name='xgpu_' + Path(output).name)
    if replicas is not None:
        c['replicas'] = replicas
    if memory is not None:
        c['memory_budget_gib'] = memory
    c['backend_ports'] = list(range(port+1, port+1+c['replicas']))
    c['ports'] = [port, *c['backend_ports']]
    c['xgpu_probe'] = dict(no_workers_started=True, no_lane_mutation=True,
        source_replicas=source['replicas'], topology_changed=c['replicas'] != source['replicas'],
        allocation_budget_changed=c['memory_budget_gib'] != source['memory_budget_gib'])
    return c


def relocate_paths(value, mapping):
    if isinstance(value,dict):
        return {k:relocate_paths(v,mapping) for k,v in value.items()}
    if isinstance(value,list):
        return [relocate_paths(v,mapping) for v in value]
    if isinstance(value,str):
        for old,new in sorted(mapping.items(),key=lambda p:-len(p[0])):
            if value==old or value.startswith(old+'/'):
                return new+value[len(old):]
    return value


def replica_command(c, campaign_path, index, bootstrap, mps_ready):
    argv = [c['server_python'], '-u']
    argv += ['-c', bootstrap] if bootstrap else ['-m', 'exp.r17_integration.ops.formal_server']
    argv += ['--campaign', str(campaign_path), '--port', str(c['backend_ports'][index]),
             '--directory', c['server_run']+'/replica_'+str(index)]
    if c['model'] == 'dp' and 'graphs_off' not in c.get('xgpu_numerics',[]):
        argv.append('--cuda-graph')
    if c.get('mps'):
        argv = [argv[0], '-m', 'exp.r17_integration.models.fastwam.ops.mps',
                'exec', '--ready', str(mps_ready), '--', *argv]
    return argv


class Owned:
    def __init__(self, directory, env, cwd, deadline):
        self.directory, self.env, self.cwd, self.deadline = Path(directory), env, cwd, deadline
        self.children = []

    def spawn(self, label, argv, env=None):
        f = (self.directory / (label+'.log')).open('xb')
        try:
            p = subprocess.Popen(argv, env=env or self.env, cwd=self.cwd,
                stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        finally:
            f.close()
        self.children.append((label, p))
        return p

    def check(self):
        if time.monotonic() > self.deadline:
            raise TimeoutError('owned campaign deadline reached')
        for label, p in self.children:
            if p.poll() is not None:
                raise RuntimeError(f'owned process {label} exited {p.returncode}; see its log')

    def wait_port(self, port):
        while True:
            self.check()
            try:
                with socket.create_connection(('127.0.0.1', port), timeout=.2):
                    return
            except OSError:
                time.sleep(.2)

    def stop(self):
        receipt = []
        for label, p in reversed(self.children):
            if p.poll() is None:
                # These are our unreaped Popen children, each its own session.
                os.killpg(p.pid, signal.SIGTERM)
                try:
                    p.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=10)
            receipt.append(dict(label=label, pid=p.pid, returncode=p.returncode))
        return receipt


def validate_attestation(c, expected_spec, metadata):
    if metadata.get('r17_model_spec') != expected_spec:
        raise ValueError('resident ModelSpec differs from pinned formal ModelSpec')
    if c['model'] == 'dp' and 'graphs_off' not in c.get('xgpu_numerics',[]) and not metadata.get('cuda_graph_self_check', {}).get('passed'):
        raise ValueError('formal DP graph self-check failed/missing')
    if c['model'] == 'cosmos' and not metadata.get('native_parity_self_check', {}).get('passed'):
        raise ValueError('Cosmos native parity failed/missing')


def live_reference(path, pin, declared):
    if not path:return declared,None
    r=json.loads(Path(path).read_text());body=dict(r);expected=body.pop('identity')
    if digest(body)!=expected or r['runtime_identity']!=pin['runtime_identity'] or r['model']!=pin['model']:
        raise ValueError('deployed reference receipt identity/runtime mismatch')
    specs=[x['metadata']['r17_model_spec'] for x in r['attestations']]
    if not specs or any(s!=specs[0] for s in specs):
        raise ValueError('reference deployment replicas disagree on ModelSpec')
    for field in ('model_id','checkpoint_identity','norm_identity','layout','default_sampler','allowed_benchmark_protocols'):
        if specs[0].get(field)!=declared.get(field):
            raise ValueError('deployed reference changes scientific binding: '+field)
    return specs[0],dict(identity=expected,source=r['source'],source_sha256=r['source_sha256'],
        differs_from_declared=specs[0]!=declared,live_spec_digest=digest(specs[0]))


def run(a, pin, c, inputs):
    out = Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    mappings=dict(item.split('=',1) for item in a.relocate)
    if mappings:
        # Path-only private copy. Original asset seals remain unchanged; runtime
        # ModelSpec equality is checked after loading the relocated descriptor.
        relocated=out/'assets'
        shutil.copytree(Path(a.bundle)/'assets',relocated)
        desc=json.loads((relocated/'load.json').read_text())
        write_json(relocated/'load.json',relocate_paths(desc,mappings))
        import yaml
        for path in relocated.rglob('*.yaml'):
            document=yaml.safe_load(path.read_text())
            if isinstance(document,dict) and isinstance(document.get('recipe'),dict):
                for field in ('store_path','fitted_path','stall_path'):
                    if document['recipe'].get(field):
                        document['recipe'][field]=relocate_paths(document['recipe'][field],mappings)
                path.write_text(yaml.safe_dump(document))
        c['vendor']=relocate_paths(c.get('vendor',''),mappings)
        c.update(assets=str(relocated),local_assets=str(relocated))
        write_json(out/'PATH_RELOCATION.json',dict(mapping=mappings,
            original_load_sha256=pin['files']['load.json'],relocated_load_sha256=sha(relocated/'load.json')))
    if a.graph_probe:
        c['graph_probe_observations']=str(Path(a.graph_probe).resolve(strict=True))
    write_json(out / 'campaign.json', c)
    runtime = Path(c['runtime'])
    # Import orchestration from the pinned release, never the mutable checkout.
    sys.path[:0] = [str(runtime), str(runtime/'src'), str(runtime/'packages/openpi-client/src')]
    from exp.r17_integration.ops.formal_eval import server_env
    from exp.r17_integration.models.fastwam.ops.mps import Session, client_env, account, gpu_snapshot, select_gpu, paths as frozen_mps_paths
    env = dict(os.environ, **server_env(c, str(runtime)))
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    env.update(dict(x.split('=',1) for x in a.env))
    if a.numerics or a.batch_telemetry:
        env['PYTHONPATH'] += ':'+str(Path(__file__).parent)
        from numerics_bootstrap import process_environment
        env.update(process_environment(a.numerics,c['model']))
    gpu = select_gpu(c['device'], env.get('CUDA_VISIBLE_DEVICES'), gpu_snapshot())
    if gpu['free_mib'] < a.required_free_gib * 1024:
        raise RuntimeError(f'insufficient free GPU memory: {gpu["free_mib"]} MiB')
    # Admission headroom is physical free memory, distinct from the native
    # allocator cap: multiple CUDA contexts may need more than cap + 2 GiB.
    # Refuse collisions before launching anything; no requests to existing ports.
    for port in c['ports']:
        with socket.socket() as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            probe.bind(('127.0.0.1', port))
    owner = Owned(out, env, runtime, time.monotonic()+a.deadline_seconds)
    mps = None
    report = dict(schema='xgpu.concurrent.run.v1', status='STARTING', pin=pin,
        input_identity=inputs['identity'], campaign=c, gpu=gpu,
        rounds=a.rounds, deliveries=a.deliveries, clients=a.clients or pin['clients'],
        quality_certificate=False, trajectories='repeated real frames with protocol execution ACKs',
        source_environment=env.get('PYTHONPATH'), artifacts=[])
    report['harness_files']={name:sha(Path(__file__).with_name(name)) for name in
        ('concurrent_server.py','concurrent_clients.py','batch_telemetry.py','cache_telemetry.py','numerics_bootstrap.py','attention_control.py')}
    write_json(out/'RUN.json', report)
    try:
        mps_state = None
        mps_ready = out/'MPS_READY.json'
        replica_env = env
        if c.get('mps'):
            mps_plan = reserve_mps_directory(a.mps_root, gpu['uuid'], frozen_mps_paths)
            report['mps_paths'] = mps_plan
            write_json(out/'MPS_PATHS.json', mps_plan)
            mps = Session(Path(mps_plan['campaign_directory']), c['device'])
            mps_state = mps.start()
            write_json(mps_ready, mps_state)
            replica_env = dict(env, **client_env(mps_state))
            report['mps'] = mps_state
        bootstrap_path = Path(a.bundle)/'assets/ffk_bootstrap.py'
        bootstrap = bootstrap_path.read_text() if bootstrap_path.exists() else None
        if a.numerics:
            bootstrap='from numerics_bootstrap import run; run('+repr(str(bootstrap_path.resolve()) if bootstrap_path.exists() else None)+','+repr(a.numerics)+','+repr(c['model'])+')'
        if a.batch_telemetry:
            bootstrap='from batch_telemetry import run; run('+repr(str(bootstrap_path.resolve()) if bootstrap_path.exists() else None)+','+repr(a.numerics)+','+repr(c['model'])+')'
        attestations = []
        from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
        declared = json.loads((Path(a.bundle)/'assets/model_spec.json').read_text())
        spec,reference=live_reference(a.reference_attestation,pin,declared)
        report['live_reference']=reference
        write_json(out/'EXPECTED_LIVE_SPEC.json',spec)
        replicas = []
        for i, port in enumerate(c['backend_ports']):
            p = owner.spawn('replica_'+str(i), replica_command(c, out/'campaign.json', i, bootstrap, mps_ready), replica_env)
            replicas.append(p.pid)
            owner.wait_port(port)
            ctl = WebsocketClientPolicy('127.0.0.1', port)
            try:
                meta = ctl.get_server_metadata()
            finally:
                ctl.close()
            attestations.append(dict(port=port, metadata=meta))
            write_json(out/'SERVER_ATTESTATIONS.json', attestations)
            validate_attestation(c, spec, meta)
        if mps_state:
            report['memory'] = account(mps_state, replicas)
            write_json(out/'GPU_MEMORY.json', report['memory'])
            if report['memory']['owned_total_mib'] > c['memory_budget_gib']*1024:
                raise RuntimeError('formal owned memory budget exceeded')
        owner.spawn('proxy', [c['server_python'], '-u', '-c',
            'from openpi.serving.replica_proxy import run_proxy; '+
            f'run_proxy({c["public_port"]},{c["backend_ports"]!r})'])
        owner.wait_port(c['public_port'])
        ctl = WebsocketClientPolicy('127.0.0.1', c['public_port'])
        groups = json.loads((Path(a.bundle)/'assets/selected_groups.json').read_text())
        selected = [dict(bundle_id='formal_full_v1', file='full.yaml', group='pure')]
        if a.groups != 'pure':
            selected += groups if a.groups == 'all' else [groups[0], groups[-1]] if len(groups)>1 else groups
        acknowledgments = []
        try:
            for group in selected:
                ack = ctl.load_cache_config(str(Path(c['assets'])/group['file']),
                    yaml_id=group['bundle_id'], bundle_id=group['bundle_id'])
                if ack.get('__ack__') == 'error':
                    raise ValueError(ack)
                if group.get('bundle_identity') and ack.get('bundle_identity') != group['bundle_identity']:
                    raise ValueError('bundle identity changed at server')
                acknowledgments.append(dict(group=group, acknowledgment=ack))
        finally:
            ctl.close()
        write_json(out/'BUNDLE_ACKS.json', acknowledgments)
        for repeat in range(a.rounds):
            for n, group in enumerate(selected):
                target = out/f'round_{repeat}'/f'group_{n}'
                # Child clients permit a real wall deadline even if recv blocks.
                argv = [sys.executable, str(Path(__file__).with_name('concurrent_clients.py')),
                    '--inputs', str(Path(a.inputs).resolve()), '--benchmark', str(Path(a.bundle).resolve()/'assets/benchmark.json'),
                    '--output', str(target), '--bundle-id', group['bundle_id'], '--port', str(c['public_port']),
                    '--clients', str(report['clients']), '--deliveries', str(a.deliveries), '--round', str(repeat)]
                p = owner.spawn(f'clients_{repeat}_{n}', argv)
                while p.poll() is None:
                    # Finished earlier client children are expected here.
                    if time.monotonic() > owner.deadline:
                        raise TimeoutError('concurrent client deadline')
                    for label, resident in owner.children:
                        if not label.startswith('clients_') and resident.poll() is not None:
                            raise RuntimeError('owned resident died: '+label)
                    time.sleep(.2)
                if p.returncode:
                    raise RuntimeError(f'concurrent clients failed: {p.returncode}; see clients_{repeat}_{n}.log')
                report['artifacts'].append(str(target.relative_to(out)))
                write_json(out/'RUN.json', report)
        report['status'] = 'COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED', error=repr(exc))
        raise
    finally:
        report['process_cleanup'] = owner.stop()
        if mps is not None:
            report['mps_cleanup'] = mps.close()
        write_json(out/'RUN.json', report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('bundle', 'inputs', 'output', 'python', 'mps-root'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--runtime')
    p.add_argument('--port', type=int, default=29600)
    p.add_argument('--clients', type=int)
    p.add_argument('--replicas', type=int)
    p.add_argument('--memory-gib', type=float)
    p.add_argument('--required-free-gib', type=float, required=True)
    p.add_argument('--deliveries', type=int, default=24)
    p.add_argument('--rounds', type=int, default=2)
    p.add_argument('--deadline-seconds', type=int, default=1200)
    p.add_argument('--groups', choices=['pure', 'screen', 'all'], default='screen')
    p.add_argument('--relocate',action='append',default=[],metavar='OLD=NEW')
    p.add_argument('--env',action='append',default=[],metavar='KEY=VALUE')
    p.add_argument('--graph-probe')
    p.add_argument('--reference-attestation')
    p.add_argument('--batch-telemetry',action='store_true',help='observe native batch members; distinct timing arm')
    p.add_argument('--numerics',action='append',default=[],choices=['tf32_off','highest','sdpa_math','te_unfused','cosmos_math','compile_off','graphs_off','deterministic','full_reduction'])
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    pin = checked_pin(a.bundle)
    live_reference(a.reference_attestation,pin,json.loads((Path(a.bundle)/'assets/model_spec.json').read_text()))
    source = json.loads((Path(a.bundle)/'source_campaign.json').read_text())
    runtime = a.runtime or source['runtime']
    verify_runtime(runtime, pin['runtime_identity'])
    inputs, rows = read_inputs(a.inputs)
    allowed = json.loads((Path(a.bundle)/'assets/model_spec.json').read_text())['allowed_benchmark_protocols']
    if any(r['request']['benchmark_protocol'] not in allowed for r in rows):
        raise ValueError('recorded input protocol not allowed by ModelSpec')
    if a.rounds < 2 or a.deliveries < 2 or a.required_free_gib <= 0:
        p.error('two concurrent rounds and positive resource/length bounds required')
    c = campaign_for(a.bundle, a.output, runtime, a.port, a.python, a.replicas, a.memory_gib)
    c['xgpu_numerics']=a.numerics
    c['xgpu_batch_telemetry']=a.batch_telemetry
    clients = a.clients or pin['clients']
    if clients < c['replicas'] * pin['batch']:
        raise ValueError('clients cannot fill every replica batch')
    plan = dict(model=c['model'], runtime=c['runtime_identity'], bootstrap=pin['bootstrap_sha256'],
        replicas=c['replicas'], max_batch=pin['batch'], clients=clients, port=a.port,
        required_free_gib=a.required_free_gib, memory_budget_gib=c['memory_budget_gib'],
        segment_clock_overlay=pin['segment_clock_overlay'], scope=c['xgpu_probe'], run=a.run)
    print(json.dumps(plan), flush=True)
    if a.run:
        def stop(signum, frame):
            raise KeyboardInterrupt('signal '+str(signum))
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        run(a, pin, c, inputs)


if __name__ == '__main__':
    main()
