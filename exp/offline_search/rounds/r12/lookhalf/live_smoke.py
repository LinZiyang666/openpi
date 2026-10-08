"""Bounded real policy + LIBERO B-dev engagement smoke; never a chain launch.

One server and one simulator at a time, at most two held-out B episodes per
method, truncated to 150 controls. GPU vision uses CUDA, policy stages use CPU.
The parent enforces a 4096 MiB total GPU ceiling and a 25-minute method deadline.
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[5]
HERE = Path(__file__).resolve().parent
SIM = '/home/weiland/projects/openpi_ext/envs/libero_sim/bin/python'


def server(path):
    """Run the unmodified serving entry with a conservative allocator ceiling."""
    import torch
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(3100 / (torch.cuda.get_device_properties(0).total_memory / 2**20))
    # This AVX2 CPU emulates bf16 GEMMs. Use the same trained weights in fp32
    # for the smoke's CPU policy stages, preserving the genuine GPU encoder.
    # This is engagement evidence, not production-precision equivalence.
    from openpi.policies import policy_config
    original = policy_config.create_trained_policy

    def create_policy(*args, **kwargs):
        policy = original(*args, **kwargs)
        model = policy._model
        tower = model.paligemma_with_expert
        tower.paligemma.model.language_model.layers.float()
        tower.paligemma.model.language_model.norm.float()
        tower.gemma_expert.float()
        backbone = model._stage2_llm_backbone

        def cpu_backbone(prefix, pads, attention, positions):
            return backbone(prefix.float(), pads, attention.float(), positions)

        model._stage2_llm_backbone = cpu_backbone
        prepare_mask = model._prepare_attention_masks_4d

        def fp32_mask(*args, **kwargs):
            return prepare_mask(*args, **kwargs).float()

        model._prepare_attention_masks_4d = fp32_mask
        return policy

    policy_config.create_trained_policy = create_policy
    argv = json.loads(Path(path).read_text())
    sys.argv = argv[4:]  # taskset -c CPUS PYTHON SCRIPT ...
    runpy.run_path(sys.argv[0], run_name='__main__')


def client(path):
    """Run two actual held-out B simulator prefixes with the production audit adapter."""
    from examples.libero import main as m
    from examples.libero.worker_entry import _build_episode_setup
    from examples.libero.episode_runner import LiberoEpisodeRunner
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    from openpi.conductor.task import EpisodeTask
    from exp.offline_search.closed_loop.execution_worker import audited_run
    from exp.offline_search.closed_loop.look_recording import token
    from exp.offline_search.closed_loop.devset import pool_record

    data = json.loads(Path(path).read_text())
    contract = json.loads(Path(data['contract']).read_text())
    _, frozen = pool_record(contract['suite'], 'B')
    directory = REPO / 'exp/common/data/db_init/libero' / contract['suite']
    from exp.offline_search.closed_loop.devset import sha
    if list(directory.glob('*.pruned_init')):
        raise ValueError('unattested init override')
    for p in directory.glob('*.init'):
        if sha(p) != frozen['per_task_digests'][p.stem]:
            raise ValueError('B init digest changed')
    args = m.Args(task_suite_name=contract['suite'], resize_size=224, replan_steps=5,
                  num_workers=1, save_video=False, seed=7)
    setup = _build_episode_setup(args, 7, str(directory))

    def bounded_setup(task):
        env, initial, text, maximum = setup(task)
        return env, initial, text, min(maximum, 150)

    def audited_episode(env, cli, initial, text, opts, maximum, **kwargs):
        return audited_run(m._run_episode, env, cli, initial, text, opts, maximum, **kwargs)

    def client_factory(endpoint):
        cli = WebsocketClientPolicy(host=endpoint.host, port=endpoint.port)
        cli.load_cache_config(yaml_path=data['yaml'], yaml_id=contract['arm'], bundle_id=contract['arm'])
        return cli

    runner = LiberoEpisodeRunner(args, bounded_setup, run_episode_fn=audited_episode, client_factory=client_factory)
    results = []
    try:
        for task, init in contract['pairs']:
            episode = EpisodeTask(task_uid=f"{contract['arm']}:eval:{task}:{init}",
                yaml_id=contract['arm'], phase='eval', experiment=contract['suite'], task_id=task,
                episode_idx=init, orig_init_state_idx=init, server_host='127.0.0.1',
                server_port=data['port'], bundle_id=contract['arm'],
                extra=dict(num_trials_per_task=50, run_id=token(contract)))
            result = runner.run(episode, lambda *a: None)
            results.append(dataclasses.asdict(result))
            print('SMOKE_EPISODE ' + episode.task_uid, flush=True)
    finally:
        runner.close()
    Path(data['result']).write_text(json.dumps(results, indent=2) + '\n')


def gpu_used():
    """Measure total physical GPU memory, including the simulator context."""
    return int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
        '--format=csv,noheader,nounits'], text=True).splitlines()[0])


def stop(proc):
    """Terminate only a process created by this smoke, by its owned PID."""
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=15)


def prepare(out):
    """Emit a separate two-pair B smoke root using the frozen production fits."""
    from exp.offline_search.closed_loop.ops.emit_arms import _emit
    specs = []
    for group, methods in [('lookhalf', ('opus_m1', 'astra_consensus2')), ('guard', ('m2', 'h2'))]:
        record = next(r for r in json.loads((HERE.parent / group / 'pilot_roots.json').read_text())
                      if r['model'] == 'pi05' and r['suite'] == 'spatial')
        source = Path(record['root'])
        selection = json.loads((source / 'dev_selection.json').read_text())
        # The same two B inits are used for each method; no library or A pair.
        selection['pairs'] = sorted(selection['pairs'])[:2]
        path = out / 'dev_selection.json'
        if path.exists() and json.loads(path.read_text()) != selection:
            raise ValueError('method selections differ')
        path.write_text(json.dumps(selection, indent=2) + '\n')
        for row in json.loads((source / 'arms_in.json').read_text()):
            if row['kwargs']['method'] in methods:
                row.update(name='smoke_' + row['kwargs']['method'], manifest=str(path))
                specs.append(row)
    path = out / 'arms_in.json'
    path.write_text(json.dumps(specs, indent=2) + '\n')
    with contextlib.redirect_stdout(io.StringIO()):
        _emit(out, path)
    rows = json.loads((out / 'arms.json').read_text())
    for row in rows:
        row.update(dev=True, init_pool='B', server_host='local')
    (out / 'arms.json').write_text(json.dumps(rows, indent=2) + '\n')
    return rows


def smoke(out):
    """Launch sequential bounded servers, enforce limits and assert real engagement."""
    from exp.offline_search.closed_loop.ops.h100 import control, local_server
    from exp.offline_search.closed_loop.lookhalf_config import value
    from exp.offline_search.closed_loop.lookhalf_ledger import summarize
    if out.exists():
        raise FileExistsError('fresh smoke output required')
    before = gpu_used()
    if before > 1000:
        raise RuntimeError(f'GPU is occupied: {before} MiB')
    port = local_server.choose_ports(requested=[23160])[0]
    out.mkdir(parents=True)
    rows = prepare(out)
    os.environ['SERVER_HOST'] = 'local'
    os.environ['WORKER_HOST'] = 'timan107'
    with contextlib.redirect_stdout(io.StringIO()):
        plan = control.sync(out, [r['arm'] for r in rows], plan_only=True, workdir=out / 'plan')
    report = dict(init_pool='B', dev=True, real_policy=True, real_simulator=True,
                  GPU_before_MiB=before, GPU_ceiling_MiB=4096, server_allocator_ceiling_MiB=3100,
                  servers_at_once=1, simulator_workers_at_once=1, max_episodes_per_method=2,
                  unique_B_pairs=json.loads((out / 'dev_selection.json').read_text())['pairs'],
                  max_controls_per_episode=150, stages=dict(stage1='cuda:0', stage2='cpu', stage3='cpu'),
                  CPU_policy_precision='float32 from trained bf16 weights (AVX2 CPU)',
                  GPU_vision_precision='unchanged bf16',
                  truncated_engagement_check=True, methods=[], status='running')
    def save():
        (out / 'ENGAGEMENT.json').write_text(json.dumps(report, indent=2) + '\n')
    save()
    for row in plan['arms']:
        entry = dict(method=row['kwargs']['method'], status='running', GPU_peak_MiB=before)
        report['methods'].append(entry)
        spec = control.server_spec(out, row, port)
        argv, env = local_server.command(spec)
        # The production argv is unchanged except for this explicit real CPU
        # policy placement; no fake model, altered threshold or synthetic key.
        index = argv.index('policy:checkpoint')
        argv[index:index] = ['--stage1-device', 'cuda:0', '--stage2-device', 'cpu', '--stage3-device', 'cpu']
        argpath = out / (row['arm'] + '_argv.json')
        argpath.write_text(json.dumps(argv, indent=2) + '\n')
        entry.update(argv=str(argpath), server_log=str(Path(spec['out']) / 'live.log'))
        client_spec = out / (row['arm'] + '_client.json')
        result = out / (row['arm'] + '_episodes.json')
        client_spec.write_text(json.dumps(dict(contract=value(argv, '--os-look-contract'), port=port,
                                              yaml=spec['yaml'], result=str(result))))
        server_proc = client_proc = None
        start = time.monotonic()
        try:
            with open(entry['server_log'], 'w') as slog, open(out / (row['arm'] + '_sim.log'), 'w') as clog:
                server_proc = subprocess.Popen([*argv[:4], '-B', '-m', __package__ + '.live_smoke',
                    '--server', str(argpath)], cwd=REPO, env={**os.environ, **env}, stdout=slog, stderr=slog)
                entry['owned_server_pid'] = server_proc.pid
                # The client has no torch GPU allocations; EGL render memory
                # remains included in the physical ceiling monitored below.
                simenv = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'MUJOCO_GL': 'egl',
                          'MUJOCO_EGL_DEVICE_ID': '0', 'PYTHONPATH': f'{REPO}:{REPO}/src:{REPO}/packages/openpi-client/src',
                          'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
                while client_proc is None or client_proc.poll() is None:
                    used = gpu_used()
                    entry['GPU_peak_MiB'] = max(entry['GPU_peak_MiB'], used)
                    if used > 4096:
                        raise RuntimeError(f'smoke GPU ceiling exceeded: {used} MiB')
                    if time.monotonic() - start > 1500:
                        raise TimeoutError('bounded smoke deadline')
                    if server_proc.poll() is not None:
                        raise RuntimeError(f'server exited {server_proc.returncode}; see {entry["server_log"]}')
                    if client_proc is None:
                        try:
                            with socket.create_connection(('127.0.0.1', port), timeout=.3):
                                pass
                        except OSError:
                            pass
                        else:
                            client_proc = subprocess.Popen(['taskset', '-c', '22-37,66-81', SIM, '-B', '-m',
                                __package__ + '.live_smoke', '--client', str(client_spec)],
                                cwd=REPO, env=simenv, stdout=clog, stderr=clog)
                            entry['owned_client_pid'] = client_proc.pid
                    save()
                    time.sleep(1)
                if client_proc.returncode != 0:
                    raise RuntimeError(f'simulator exited {client_proc.returncode}; see {clog.name}')
            decisions = []
            starts = []
            for path in Path(spec['out']).glob('decisions_*.jsonl'):
                for line in path.read_text().splitlines():
                    item = json.loads(line)
                    if item.get('ev') == 'startup':
                        starts.append(item)
                    elif item.get('ev') == 'dec':
                        decisions.append(item)
            assert len(starts) == 1 and starts[0]['lookhalf'] == entry['method']
            assert value(starts[0]['argv'], '--os-lookhalf') == entry['method']
            assert decisions and all(r['execution_audit'] for r in decisions)
            assert all(r.get('ok') is True for r in decisions)
            ledger = summarize('pi05', decisions)
            episodes = json.loads(result.read_text())
            assert 1 <= len(episodes) <= 2 and all(not e.get('error') for e in episodes)
            assert all(e['decisions'] <= 30 for e in ledger['per_episode'].values())
            h2 = [r['lookhalf']['h2'] for r in decisions if 'h2' in (r.get('lookhalf') or {})]
            if entry['method'] in ('m2', 'h2'):
                assert h2 and all(len(h['features']) == 7 for h in h2)
            entry.update(ledger=ledger, decisions=len(decisions), episodes=json.loads(result.read_text()),
                         h2_looks=len(h2),
                         server_startup_method=starts[0]['lookhalf'], all_execution_audited=True)
            if entry['method'] == 'h2':
                assert entry['h2_looks'] > 0
            else:
                assert ledger['one_camera_looks'] > 0, (entry['method'], ledger)
            entry['status'] = 'passed'
            print('ENGAGED ' + entry['method'] + ' ' + json.dumps(dict(
                one_camera_looks=ledger['one_camera_looks'], h2_looks=entry['h2_looks'], GPU_peak_MiB=entry['GPU_peak_MiB'])), flush=True)
        except Exception as error:
            entry.update(status='failed', error=repr(error))
            report['status'] = 'failed'
            save()
            raise
        finally:
            stop(client_proc)
            stop(server_proc)
            entry['elapsed_s'] = round(time.monotonic() - start, 2)
            entry['owned_processes_stopped'] = True
            save()
    report.update(status='passed', GPU_after_MiB=gpu_used())
    save()


def main():
    """Run the parent smoke or its explicitly bounded subprocess roles."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--out', type=Path)
    group.add_argument('--server', type=Path)
    group.add_argument('--client', type=Path)
    args = parser.parse_args()
    if args.server:
        server(args.server)
    elif args.client:
        client(args.client)
    else:
        smoke(args.out.resolve())


if __name__ == '__main__':
    main()
