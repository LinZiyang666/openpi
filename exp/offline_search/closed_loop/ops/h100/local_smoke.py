"""One bounded R11 server + one recorded query from timan107; no simulators.

Writes only its supplied smoke root. It neither syncs a worker island nor takes
chain/fleet locks. The probe is executed from an inline Python program, with no
remote files or worker processes installed. Finally stops only its owned PIDs.
"""
import argparse
import base64
import json
import os
from pathlib import Path
import signal
import time
import uuid
import zlib

from . import assets, control, local_server


def probe_code(host, port, episode):
    return f'''
import base64,hashlib,json,sys,time,zlib
import numpy as np
import websockets.sync.client
from openpi_client import msgpack_numpy as msg
with websockets.sync.client.connect('ws://{host}:{port}',compression=None,max_size=None,open_timeout=30,close_timeout=10) as conn:
    metadata=msg.unpackb(conn.recv(timeout=60))
    print('WORKER_CONNECTED '+json.dumps(dict(host='{host}',port={port},metadata_keys=sorted(metadata))),flush=True)
    def request(value):
        conn.send(msg.packb(value))
        reply=conn.recv(timeout=120)
        if isinstance(reply,str): raise RuntimeError(reply)
        return msg.unpackb(reply)
    selected=request(dict(__ctrl__='select_bundle',bundle_id='default'))
    assert selected.get('__ack__')=='select_bundle',selected
    started=request({episode!r})
    assert started.get('__ack__')=='episode_start',started
    observation=msg.unpackb(zlib.decompress(base64.b64decode(''.join(sys.argv[1:]))))
    t=time.monotonic()
    response=request(observation)
    actions=np.asarray(response['actions'])
    assert actions.ndim==2 and actions.shape[1]==7 and np.isfinite(actions).all(),actions.shape
    print('POLICY_ROUNDTRIP_OK '+json.dumps(dict(shape=list(actions.shape),finite=True,queries=1,latency_s=round(time.monotonic()-t,3),actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest())),flush=True)
    request(dict(__ctrl__='episode_end',__success__=False))
'''


def recorded_query(cell):
    import numpy as np
    from openpi_client import msgpack_numpy
    from exp.offline_search.harness.store import QueryCell
    query = QueryCell(assets.STORE, cell)
    rows = set(np.asarray(query.tok_rows).tolist())
    episode = next(e for e in query.episodes if e['start'] in rows)
    row = episode['start']
    index = int(query.tok_index[row])
    observation = {'observation/image': np.asarray(query.tok('img0')[index]),
                   'observation/wrist_image': np.asarray(query.tok('img1')[index]),
                   'observation/state': np.asarray(query.raw_state[row], np.float64), 'prompt': str(episode['task'])}
    packed = base64.b64encode(zlib.compress(msgpack_numpy.packb(observation), 9)).decode()
    if len(packed) > 1000000:
        raise ValueError('recorded probe exceeds bounded inline payload')
    ctrl = dict(__ctrl__='episode_start', __experiment__='r11_localsmoke', __task__=episode['task'],
                __episode_id__=int(episode['init']), __episode_name__='r11_localsmoke',
                __extra__=dict(task_id=int(episode['task_id']), orig_init_state_idx=int(episode['init']),
                               task_uid='r11_localsmoke:recorded_query', attempt=1))
    return packed, ctrl


def smoke(root, source, arm, boot_timeout=600):
    if os.environ.get('SERVER_HOST') != 'local':
        raise ValueError('smoke requires SERVER_HOST=local')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    # Do not write the source run, take its locks, or sync any fleet.
    plan = assets.build_plan(source, [arm], root / 'local_plan', local=True)
    row = plan['arms'][0]
    packed, episode = recorded_query(row['cell'])
    port = local_server.choose_ports(1)[0]
    spec = control.server_spec(root, row, port)
    spec['launch_id'] = uuid.uuid4().hex
    spec_path = root / 'smoke_spec.json'
    if spec_path.exists():
        raise FileExistsError('one-shot smoke already has a spec; inspect it before another launch')
    spec_path.write_text(json.dumps(spec, indent=1))
    before = local_server.gpu_free()
    def interrupted(sig, frame):
        raise TimeoutError(f'bounded smoke interrupted signal={sig}')
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGALRM, interrupted)
    signal.alarm(boot_timeout + 300)
    try:
        print(local_server.start(spec), flush=True)
        deadline = time.monotonic() + boot_timeout
        while True:
            status = local_server.status(spec)
            if status['dead']:
                raise RuntimeError('LOCAL_BOOT_FAILED ' + status['tail'])
            if status['running'] and status['listening']:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('LOCAL_BOOT_TIMEOUT ' + status['tail'])
            time.sleep(2)
        receipt = local_server.matching(spec)
        pid = receipt['child']['pid']
        gpu = control.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.free', '--format=csv,noheader'])
        processes = control.run(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory', '--format=csv,noheader'])
        affinity = sorted(os.sched_getaffinity(pid))
        assert not set(affinity) & (set(range(38, 44)) | set(range(82, 88)))
        print(f'LOCAL_READY pid={pid} port={port} cpus={affinity} gpu={gpu} free_delta_mb={before-local_server.gpu_free()}', flush=True)
        print('GPU_PROCESSES\n' + processes, flush=True)
        code = probe_code('ziyanglin.com', port, episode)
        worker_island = control.island('timan107')
        env = ['env', 'PYTHONDONTWRITEBYTECODE=1', 'CUDA_VISIBLE_DEVICES=', 'OMP_NUM_THREADS=1',
               'OPENBLAS_NUM_THREADS=1', 'MKL_NUM_THREADS=1',
               f'PYTHONPATH={worker_island}/packages/openpi-client/src:{worker_island}/src:{worker_island}']
        # Split argv rather than putting a >128 KiB observation in bash's -c arg.
        chunks = [packed[i:i + 50000] for i in range(0, len(packed), 50000)]
        output = control.run(['tether', 'exec', '--timeout', '240s', 'timan107', '--', *env,
                              '/scratch/zixuans8/openpi/.venv/bin/python', '-B', '-c', code, *chunks],
                             timeout=250, attempts=1)
        print(output, flush=True)
        if 'POLICY_ROUNDTRIP_OK' not in output:
            raise RuntimeError('worker probe did not attest a policy round-trip')
    finally:
        # Disable the timer while proving cleanup, even when boot/probe failed.
        signal.alarm(0)
        print(local_server.stop(spec), flush=True)
        status = local_server.status(spec)
        assert not status['running'] and not status['listening'], status
        print('LOCAL_CLEANUP_OK ' + json.dumps({k: v for k, v in status.items() if k != 'tail'}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/home/weiland/trace_runs/os_closed_loop/r11_localsmoke')
    parser.add_argument('--source', default='/home/weiland/trace_runs/os_closed_loop/r11_devknob_50')
    parser.add_argument('--arm', default='r11_devknob_pi05_l10_50_off')
    parser.add_argument('--boot-timeout', type=int, default=600)
    args = parser.parse_args()
    smoke(args.root, args.source, args.arm, args.boot_timeout)


if __name__ == '__main__':
    main()
