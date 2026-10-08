"""Benchmark-only owner of unchanged R16 resident model processes.

Starts copies sequentially, using the upper R16 per-copy footprint for the
next-copy admission check. All copies use the ordinary serving path and B=1.
No split hooks are installed. Never invoked during CPU preparation.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from exp.decouple.package_config import canonical, sha, write_json


def replace_option(argv, option, value):
    argv = list(argv)
    if option not in argv:
        raise ValueError(f'missing {option}')
    argv[argv.index(option) + 1] = str(value)
    return argv


def can_admit(used_mb, budget_mb, copy_upper_mb, *, first=False):
    return first or used_mb + copy_upper_mb <= budget_mb


def warm_copy(port, group, model, *, client_factory=None, observation_loader=None):
    """Prime real stage workspace before admitting another weight copy.

    Uses unchanged controls and a recorded B fixture with frozen task-0 k-C10.
    This is not a simulator benchmark episode. Injectable factories permit
    CPU verification of the control sequence and cleanup without a listener.
    """
    if observation_loader is None:
        from exp.decouple.recorded import load, observation
        data = load(model, limit=16)
        index = next(i for i,r in enumerate(data.rows) if int(data.arrays['task_id'][r]) == 0)
        obs = observation(data, index)
        if model == 'groot':
            import cv2
            for key in ('observation/image', 'observation/wrist_image'):
                obs[key] = cv2.resize(obs[key], (256,256))
    else:
        obs = observation_loader(model)
    if client_factory is None:
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        client_factory = lambda port:WebsocketClientPolicy(host='127.0.0.1', port=port)
    row = group['entries'][0]['row']
    assert row['kwargs']['task_levels']['0'] == 'c10'
    client = client_factory(port)
    try:
        client.load_cache_config(row['yaml'], yaml_id=row['arm'], bundle_id=row['arm'])
        client.select_bundle(row['arm'])
        client.episode_start(experiment='libero_spatial', task=obs['prompt'], episode_id=0,
                             extra_metadata=dict(task_id=0, orig_init_state_idx=0,
                                                 task_uid=f'B:decouple_capacity:{port}', attempt=1))
        reply = client.infer(obs)
        if 'actions' not in reply:
            raise ValueError('current copy workspace warmup did not produce actions')
        client.episode_end(success=False)
    finally:
        client.close()


async def route(port, child_ports, children, metadata):
    from openpi.serving.replica_proxy import ReplicaProxy, _MAX_WS_FRAME_BYTES
    from openpi_client import msgpack_numpy
    from websockets.asyncio.server import serve
    proxy = ReplicaProxy('127.0.0.1', child_ports)
    await proxy.prime_metadata()
    greeting = msgpack_numpy.unpackb(proxy._metadata)
    greeting.update(metadata)
    proxy._metadata = msgpack_numpy.packb(greeting)
    async with serve(proxy.handle, '127.0.0.1', port, max_size=_MAX_WS_FRAME_BYTES,
                     compression=None, ping_interval=None):
        while True:
            if any(child.poll() is not None for child in children):
                raise RuntimeError('current fleet child exited')
            await asyncio.sleep(.25)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('groot', 'pi05'), required=True)
    parser.add_argument('--copies', required=True, help='positive count or auto')
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--budget-mb', type=float, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not os.environ.get('CUDA_VISIBLE_DEVICES') or not command:
        parser.error('GPU window only; provide the unchanged resident argv after --')
    if args.copies != 'auto' and (not args.copies.isdigit() or int(args.copies) < 1):
        parser.error('copies must be positive or auto')
    import pynvml as nvml
    from openpi_client import msgpack_numpy
    from websockets.sync.client import connect
    nvml.nvmlInit()
    gpu = nvml.nvmlDeviceGetHandleByIndex(int(os.environ.get('OPENPI_BENCH_GPU_INDEX', '0')))
    upper = 6656 if args.model == 'groot' else 10752
    minimum = 5836.8 if args.model == 'groot' else 9216
    baseline = nvml.nvmlDeviceGetMemoryInfo(gpu).used / (1 << 20)
    max_copies = int(args.copies) if args.copies != 'auto' else int((args.budget_mb - baseline) // minimum)
    children, streams, ports, admissions = [], [], [], []
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    try:
        for index in range(max_copies):
            used = nvml.nvmlDeviceGetMemoryInfo(gpu).used / (1 << 20)
            if not can_admit(used, args.budget_mb, upper, first=index == 0):
                break
            port = args.port + index + 1
            group = json.loads((args.out / 'group.json').read_text())
            # Separate ordinary MISS evidence and resident receipts per copy.
            import yaml
            policy = yaml.safe_load(Path(group['entries'][0]['row']['yaml']).read_text())
            policy['miss']['evidence_dir'] = str(args.out / f'evidence_{port}')
            policy_path = args.out / f'current_{port}' / (group['entries'][0]['row']['arm'] + '.yaml')
            policy_path.parent.mkdir(parents=True, exist_ok=True)
            policy_path.write_text(yaml.safe_dump(policy, sort_keys=False))
            group['entries'][0]['row']['yaml'] = str(policy_path)
            group['entries'][0]['yaml_sha256'] = sha(policy_path)
            group['definitions_sha256'][str(policy_path)] = sha(policy_path)
            group.pop('group_sha256')
            import hashlib
            group['group_sha256'] = hashlib.sha256(canonical(group)).hexdigest()
            group_path = args.out / f'group_{port}.json'
            write_json(group_path, group)
            argv = replace_option(command, '--port', port)
            argv = replace_option(argv, '--server-tag', port)
            argv = replace_option(argv, '--registry', group_path)
            argv = replace_option(argv, '--cache-config', policy_path)
            stream = (args.out / f'current_{port}.log').open('w')
            streams.append(stream)
            child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT)
            children.append(child)
            deadline = time.monotonic() + 900
            peak = used
            while True:
                peak = max(peak, nvml.nvmlDeviceGetMemoryInfo(gpu).used / (1 << 20))
                if peak > args.budget_mb:
                    raise RuntimeError(f'current fleet startup exceeds budget: {peak:.0f} MiB')
                if child.poll() is not None:
                    raise RuntimeError(f'current copy {index + 1} failed; see {stream.name}')
                try:
                    with connect(f'ws://127.0.0.1:{port}', compression=None, open_timeout=2) as client:
                        msgpack_numpy.unpackb(client.recv(timeout=5))
                    break
                except (OSError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise TimeoutError('current model startup timeout')
                    time.sleep(.25)
            ports.append(port)
            warm_copy(port, group, args.model)
            peak = max(peak, nvml.nvmlDeviceGetMemoryInfo(gpu).used / (1 << 20))
            if peak > args.budget_mb:
                raise RuntimeError('current copy workspace warmup exceeded the common budget')
            admissions.append(dict(copy=index + 1, used_before_mb=used, startup_peak_mb=peak,
                                   used_after_mb=nvml.nvmlDeviceGetMemoryInfo(gpu).used / (1 << 20)))
        if not ports:
            raise RuntimeError('budget admits no current model copy')
        if args.copies != 'auto' and len(ports) != int(args.copies):
            raise RuntimeError('repeat could not admit the selected current fleet within budget')
        memory = dict(model_processes=len(ports), benchmark_current=True,
                      budget_mb=args.budget_mb, copy_upper_mb=upper, max_copies_at_lower_footprint=max_copies,
                      next_copy_projected_mb=nvml.nvmlDeviceGetMemoryInfo(gpu).used / (1 << 20) + upper,
                      admission_rule='largest fleet admitted using warmed occupancy + R16 upper footprint for next copy',
                      admissions=admissions)
        write_json(args.out / 'fleet.json', memory)
        asyncio.run(route(args.port, ports, children, memory))
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)
        for stream in streams:
            stream.close()
        nvml.nvmlShutdown()


if __name__ == '__main__':
    main()
