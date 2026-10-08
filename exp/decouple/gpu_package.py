"""Fail-first, resumable owner-run numerical gate and 4090 benchmark package."""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid

from exp.decouple.package_config import REPO, build, validate, write_json, make_arm
from exp.decouple.summarize_gpu import summarize
from exp.decouple.package_deadline import (DEADLINE_ENV, DeadlineBudget, PackageDeadlineReached,
    deadline_label, deadline_watchdog, parse_deadline, stop_owned_groups)


def best_arms(completed, model):
    rows = {key.rsplit('/', 1)[-1]:r for key,r in completed.items() if key.startswith(model + '/initial/')}
    if not any(name.startswith('current_') for name in rows) or not any(name.startswith('split_') for name in rows):
        raise ValueError('selection requires a completed current and split arm')
    if len({r['manifest_sha256'] for r in rows.values()}) != 1:
        raise ValueError('initial arms used different manifests')
    current = max((name for name in rows if name.startswith('current_')), key=lambda name:rows[name]['episodes_per_min'])
    split = max((name for name in rows if name.startswith('split_')), key=lambda name:rows[name]['episodes_per_min'])
    return dict(current=current, split=split, current_copies=rows[current]['model_processes'])


def repeat_order(repeat):
    return ['split', 'current'] if repeat % 2 else ['current', 'split']


def resolve_deadline(config, flag=None, environment=None):
    environment = os.environ if environment is None else environment
    return parse_deadline(flag if flag is not None else environment.get(DEADLINE_ENV, config['wall_deadline']))


def check_ports(config, socket_factory=None):
    import socket
    socket_factory = socket_factory or socket.socket
    ports = [config['public_port'], config['driver_port'], config['backend_port']]
    ports += list(range(config['backend_port']+1, config['backend_port']+17))
    for port in ports:
        with socket_factory() as probe:
            try:
                probe.bind(('0.0.0.0', port))
            except OSError as error:
                raise RuntimeError(f'package port {port} occupied; no listener was touched') from error


def gate_argv(case, k, out):
    return [case['python'], '-B', '-m', 'exp.decouple.qualify_gpu', '--model', case['model'],
            '--checkpoint', case['checkpoint'], '--config', 'pi05_libero', '--k', str(k),
            '--batch-size', '1', '--rows', '16', '--out', str(out)]


def run_gate(case, k, out, timeout):
    with out.with_suffix('.log').open('w') as log:
        process = subprocess.Popen(gate_argv(case, k, out), env={**os.environ, **case['environment']},
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
            if code:
                raise RuntimeError(f"{case['model']} k={k} numerical gate exited {code}; see {out.with_suffix('.log')}")
            gate = json.loads(out.read_text())
            if not gate['passed'] or gate['batch_size'] != 1 or not gate['fields'] or not all(f['bit_exact'] for f in gate['fields'].values()):
                raise ValueError('numerical gate did not prove raw-bit equality')
            return dict(passed=True, path=str(out), k=k, batch_size=1)
        finally:
            stop_owned_groups([('gate', process)])


def estimate_arm_seconds(config, completed, model, kind):
    rate = config['estimate_assumptions'][model + '_eps_min'][config['initial_arms'].index(kind)]
    episodes = config['expected_episodes']
    estimates = [60 * episodes / rate + config['arm_setup_estimate_s'] + config['arm_cleanup_estimate_s']]
    for key, result in completed.items():
        if key.startswith(model + '/') and result['logical_arm'] == kind:
            estimates += [result.get('total_wall_s', 0),
                          60 * episodes / result['episodes_per_min'] + config['arm_setup_estimate_s'] + config['arm_cleanup_estimate_s']]
    return max(estimates) * config['admission_safety_factor']


def run_package(config, state, result_dir, *, gate_runner=run_gate, arm_runner=None,
                fleet_context=None, clock=time.monotonic, wall_clock=time.time,
                deadline_epoch=None, watchdog=True, gate_cache=None, gpu_identity=None):
    from exp.decouple.benchmark_gpu import run_arm
    arm_runner = arm_runner or run_arm
    state_path = result_dir / 'state.json'
    budget = DeadlineBudget(parse_deadline(config['wall_deadline']) if deadline_epoch is None else deadline_epoch,
                            config['wall_limit_s'], config['shutdown_reserve_s'], clock=clock, wall_clock=wall_clock)
    def persist():
        write_json(state_path, state)
        summarize(state_path, result_dir)
    remaining = budget.check
    active = None
    phase = 'gate'
    state.update(status='running', mode=config['mode'], expected_episodes=config['expected_episodes'],
                 screening_arms=len(config['initial_arms']),skipped_deadline={}, deadline=dict(
        wall=deadline_label(budget.epoch), shutdown_reserve_s=budget.reserve_s,
        window_s=config['wall_limit_s'], effective_hard_end=deadline_label(wall_clock() + budget.hard_remaining())))
    state.pop('failure', None)
    state.pop('deadline_stop_reason', None)
    persist()
    try:
        with deadline_watchdog(budget, enabled=watchdog):
            remaining()
            for model in ('pi05', 'groot'):
                for k in (1, 2):
                    active = f'{model}_k{k}_b1'
                    if gate_cache:
                        cached = gate_cache.lookup(config,active,gpu_identity)
                        if cached:
                            state['gates'][active]=cached
                            persist()
                            print(f'PKG_GATE_SKIP {active} unchanged_inputs=1',flush=True)
                            continue
                        if config['mode']=='smoke':
                            raise RuntimeError(f'smoke requires a passed unchanged-input gate on this GPU: {active}; run qualification in a full window')
                        state['gates'].pop(active,None)
                    if active not in state['gates']:
                        timeout = min(900, remaining())
                        print(f'PKG_GATE_BEGIN {active}', flush=True)
                        out = result_dir / (active + '.json')
                        state['gates'][active] = gate_runner(config['cases'][model], k, out, timeout)
                        if gate_cache:
                            proof = json.loads(out.read_text())
                            state['gates'][active] = gate_cache.put(config,active,proof,gpu_identity,path=out)
                        persist()
            phase = 'arm'
            active = 'fleet_preflight'
            remaining()
            def skip(key, required, reason='estimated completion plus cleanup does not fit'):
                state['skipped_deadline'][key] = dict(estimated_s=required, available_s=max(0, budget.remaining()), reason=reason)
                persist()
                print(f'PKG_ARM_SKIPPED_DEADLINE {key.replace("/", "_")}', flush=True)
            def arm(model, key, kind, pinned_copies=None):
                nonlocal active
                active = key.replace('/', '_')
                if key in state['completed']:
                    print(f'PKG_ARM_SKIP {active}', flush=True)
                    return
                required = estimate_arm_seconds(config, state['completed'], model, kind)
                if not budget.fits(required):
                    skip(key, required)
                    return
                remaining()
                number = state.setdefault('attempts', {}).get(key, 0) + 1
                state['attempts'][key] = number
                name = f'decouple_bench_{state["package_id"]}_{active}_try{number:02d}'
                spec = make_arm(config, model, kind, name)
                spec['worker_command'][spec['worker_command'].index('--config')+1] = state['config_path']
                spec['startup_timeout_s'] = min(900, remaining())
                spec['worker_timeout_s'] = min(2700, remaining())
                spec.update(budget.arm_fields())
                if pinned_copies is not None and not spec['split']:
                    argv = spec['server_command']
                    argv[argv.index('--copies')+1] = str(pinned_copies)
                    spec['model_processes'] = pinned_copies
                out = Path(config['trace_parent']) / name
                state['active'] = dict(key=key, out=str(out))
                persist()
                print(f'PKG_ARM_BEGIN {active} root={out}', flush=True)
                started = clock()
                try:
                    result = arm_runner(spec, Path(config['trace_parent']), config['gpu_index'], config['gpu_memory_budget_mb'])
                except BaseException as error:
                    state.setdefault('failed_attempts', []).append(dict(key=key, out=str(out), error=str(error)))
                    raise
                state['completed'][key] = {**result, 'out':str(out), 'total_wall_s':clock()-started}
                state.pop('active', None)
                # Make an informative partial comparison visible immediately.
                if '/initial/' in key and not any(k.startswith(model + '/repeat') for k in state['completed']):
                    try:
                        state['selected'][model] = best_arms(state['completed'], model)
                    except ValueError:
                        pass
                persist()
                print(f'PKG_ARM_DONE {active} episodes_min={result["episodes_per_min"]:.2f}', flush=True)
            with fleet_context() if fleet_context else contextlib.nullcontext():
                # Both models' screening comparisons precede every repeat.
                for model in config['benchmark_models']:
                    for kind in config['initial_arms']:
                        arm(model, f'{model}/initial/{kind}', kind)
                for model in config['benchmark_models']:
                    selection = state['selected'].get(model)
                    for repeat in range(1,config['repeats']+1):
                        keys = [(kind, f'{model}/repeat{repeat}/{kind}') for kind in repeat_order(repeat)]
                        required = sum(estimate_arm_seconds(config, state['completed'], model, selection[kind])
                                       for kind, key in keys if key not in state['completed']) if selection else float('inf')
                        if not selection or not budget.fits(required):
                            for kind, key in keys:
                                if key not in state['completed']:
                                    skip(key, required if selection else None, 'complete repeat pair does not fit' if selection else 'screening comparison incomplete')
                            continue
                        for kind, key in keys:
                            arm(model, key, selection[kind], selection['current_copies'] if kind == 'current' else None)
            if state['skipped_deadline']:
                raise PackageDeadlineReached('remaining work skipped to preserve the deadline and complete cohorts')
        state.update(status='done')
        persist()
        print('PKG_DONE', flush=True)
        return 0
    except BaseException as error:
        if isinstance(error, PackageDeadlineReached) or budget.remaining() <= 0:
            state.update(status='deadline_stop', deadline_stop_reason=str(error))
            state.pop('active', None)
            persist()
            print('PKG_DEADLINE_STOP', flush=True)
            return 2
        state.update(status='interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', failure=f'{active}: {error}')
        persist()
        print('PKG_GATE_FAIL' if phase == 'gate' else f'PKG_ARM_FAIL {active}', flush=True)
        print(str(error), flush=True)
        return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--smoke',action='store_true',help='GR00T current/1 and split/8, 20 episodes each, cached gates required, <=10 min')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--new', action='store_true', help='archive the previous package state; start fresh roots')
    parser.add_argument('--deadline', help=f'absolute stop time (ISO 8601 or local time + IANA zone); overrides ${DEADLINE_ENV}; default 2026-10-08 15:15 America/Chicago')
    args = parser.parse_args()
    args.config = args.config or REPO/('exp/decouple/benchmark.smoke.local.json' if args.smoke else 'exp/decouple/benchmark.local.json')
    args.out = args.out or REPO/('exp/decouple/results/gpu_smoke' if args.smoke else 'exp/decouple/results/gpu_comparison')
    if not args.config.exists():
        write_json(args.config, build(smoke=args.smoke))
    config = json.loads(args.config.read_text())
    report = validate(config)
    if args.smoke != (config['mode']=='smoke'):
        parser.error('--smoke/config mode mismatch')
    try:
        deadline_epoch = resolve_deadline(config, args.deadline)
    except (ValueError, KeyError) as error:
        parser.error(f'invalid deadline: {error}')
    report.update(wall_deadline=deadline_label(deadline_epoch), shutdown_reserve_s=config['shutdown_reserve_s'],
                  initial_arm_order=config['initial_arms'], repeat_placement='after both models screening')
    if args.dry_run:
        from exp.decouple.package_dry_run import dry_run
        report.update(dry_run(config, args.config))
        write_json(REPO / ('exp/decouple/results/package_validation_smoke.json' if args.smoke else
                          'exp/decouple/results/package_validation.json'), report)
        print(json.dumps(report, indent=2))
        return
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        print('PKG_GATE_FAIL', flush=True)
        parser.error('owner GPU window only; select the reserved 4090')
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / 'package.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('another package owns these results')
        state_path = args.out / 'state.json'
        if args.new and state_path.exists():
            old = json.loads(state_path.read_text())
            archive = args.out / ('archive_' + old['package_id'])
            archive.mkdir(exist_ok=False)
            for path in args.out.iterdir():
                if path.is_file() and path.name != 'package.lock':
                    path.replace(archive / path.name)
        state = json.loads(state_path.read_text()) if state_path.exists() else dict(
            package_id=time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6],
            config_sha256=config['config_sha256'], config_path=str(args.config.resolve()),
            gates={}, completed={}, selected={})
        if state['config_sha256'] != config['config_sha256']:
            raise ValueError('resume configuration changed; use --new for a new package')
        if state.get('status') == 'done':
            summarize(state_path, args.out)
            print('PKG_DONE', flush=True)
            return
        if time.time() >= deadline_epoch - config['shutdown_reserve_s']:
            # An expired reservation never queries CUDA or contacts workers.
            raise SystemExit(run_package(config, state, args.out, deadline_epoch=deadline_epoch))
        import pynvml as nvml
        nvml.nvmlInit()
        try:
            gpu = nvml.nvmlDeviceGetHandleByIndex(config['gpu_index'])
            name = nvml.nvmlDeviceGetName(gpu)
            name = name.decode() if isinstance(name, bytes) else name
            memory = nvml.nvmlDeviceGetMemoryInfo(gpu)
            if '4090' not in name or memory.total / (1<<20) < config['expected_gpu_total_mb_min']:
                raise ValueError(f'package expects the owner-reserved 48 GB 4090: found {name}, {memory.total/(1<<20):.0f} MiB')
            if memory.used / (1<<20) > 1024:
                raise RuntimeError('reserved card is still occupied; no process was touched')
            import socket
            text = lambda value:value.decode() if isinstance(value,bytes) else value
            gpu_identity = dict(host=socket.gethostname(),index=config['gpu_index'],name=name,
                                uuid=text(nvml.nvmlDeviceGetUUID(gpu)),driver=text(nvml.nvmlSystemGetDriverVersion()))
        finally:
            nvml.nvmlShutdown()
        os.environ.update(SERVER_HOST='local', WORKER_HOST='timan107', OPENPI_BENCH_GPU_INDEX=str(config['gpu_index']))
        from exp.offline_search.closed_loop.ops.h100.control import fleet_lock, remote
        from exp.decouple.package_driver import remote_preflight
        @contextlib.contextmanager
        def fleet():
            with fleet_lock():
                # Refuse occupied ports before starting a model; metadata from
                # an unrelated listener can never masquerade as readiness.
                check_ports(config)
                remote_preflight(config, remote)
                yield
        def interrupt(signum, frame):
            raise KeyboardInterrupt
        for sig in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, interrupt)
        from exp.decouple.gate_cache import GateCache
        raise SystemExit(run_package(config, state, args.out, fleet_context=fleet, deadline_epoch=deadline_epoch,
                                    gate_cache=GateCache(),gpu_identity=gpu_identity))


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as error:
        print('PKG_GATE_FAIL', flush=True)
        print(str(error), flush=True)
        raise SystemExit(1)
