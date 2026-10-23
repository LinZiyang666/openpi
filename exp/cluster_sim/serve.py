"""Start one single-copy π0.5 server on the 4090 for the cluster simulation (exp/cluster_sim/DESIGN.md).

  # full-step pure inference: InferenceInterceptor without a cache (stage 1 -> 2 -> 3, K=10)
  python -m exp.cluster_sim.serve --mode pure --port 8000 --log DIR [knobs]
  # complete system: the R16 resident with the recording group's four arms (K=1, no long segments)
  python -m exp.cluster_sim.serve --mode cache --group /data/mw_finetune/eval/cs_rec_pi05/group.json --port 8000 --log DIR

knobs: --batch 1:32,2:16,3:16  --wait-ms 1:0,2:0,3:0  --workers 1:1,2:1,3:1  --threads 512  --text-tokens 80
       --vram-gb 24  --omp N (unset: library defaults, every core)  --compile vision,lm,expert  --compile-mode M
       --graphs  --buckets 1,2,4,8,12,16,24,32,48,64  --units 1,2,3 | 1,23 | 123

Both modes run the same server tree, model, interceptor and coordinator (serve_entry.py); the cache mode adds the
plugin arms. No CPU pinning. In cache mode the arms' bundles are loaded over one control connection after the port
opens (as the R16 driver does); READY.json is written when the server takes robots. The command blocks until the
server exits; SIGINT/SIGTERM stop the server (SIGTERM, then SIGKILL after 15 s).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

TREE = Path('/home/weiland/projects/openpi_r16_q3_server')
PI05_CKPT = '/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch'
ENTRY = Path(__file__).with_name('serve_entry.py')
RESIDENT = 'exp.offline_search.rounds.r16.resident'
SHARED_CACHE = Path('/nvme/weiland/openpi/cluster_sim/compile_cache')  # inductor/triton caches shared by every server start


def command(args):
    policy = ['policy:checkpoint', '--policy.config', 'pi05_libero', '--policy.dir', PI05_CKPT]
    if args.mode == 'pure':
        return [str(ENTRY), '--port', str(args.port), '--replicas', '1', '--cache', *policy], 'scripts.serve_policy', None
    group = json.loads(Path(args.group).read_text())
    row = group['entries'][0]['row']
    log = Path(args.log) / 'plugin'
    argv = [str(ENTRY), '--registry', args.group, '--server-tag', str(args.port),
            '--os-method', row['method'], '--os-kwargs', json.dumps(row['kwargs']), '--os-cell', row['cell'],
            *row['plugin_args'], '--os-log-dir', str(log), '--os-tag', f'cluster_sim_{args.port}',
            '--port', str(args.port), '--replicas', '1', '--cache-config', row['yaml'], *policy]
    return argv, RESIDENT, group


def environment(args, module):
    env = {k: v for k, v in os.environ.items()
           if k not in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'PYTHONPATH', 'VIRTUAL_ENV')}
    if args.omp:
        env.update(OMP_NUM_THREADS=str(args.omp), MKL_NUM_THREADS=str(args.omp), OPENBLAS_NUM_THREADS=str(args.omp))
    runtime = Path(args.log) / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    workers = dict(item.split(':') for item in args.workers.split(','))
    env.update(PYTHONPATH=f'{TREE}:{TREE}/src:{TREE}/packages/openpi-client/src',
               CUDA_VISIBLE_DEVICES=os.environ.get('CLUSTER_SIM_GPU', '0'),
               HOME='/home/weiland', PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               OPENPI_SERVER_GPU_MEMORY_LOCK='0', STAGE1_ONLY='0', TMPDIR=str(runtime),
               TORCHINDUCTOR_CACHE_DIR=str(SHARED_CACHE / 'inductor'), TRITON_CACHE_DIR=str(SHARED_CACHE / 'triton'),
               CLUSTER_SIM_MODULE=module, CLUSTER_SIM_LOG=str(args.log), CLUSTER_SIM_BATCH=args.batch,
               CLUSTER_SIM_WAIT_MS=args.wait_ms, CLUSTER_SIM_THREADS=str(args.threads),
               CLUSTER_SIM_TEXT_TOKENS=str(args.text_tokens), CLUSTER_SIM_VRAM_GB=str(args.vram_gb),
               CLUSTER_SIM_COMPILE=args.compile, CLUSTER_SIM_COMPILE_MODE=args.compile_mode,
               CLUSTER_SIM_GRAPHS='1' if args.graphs else '0', CLUSTER_SIM_BUCKETS=args.buckets,
               CLUSTER_SIM_UNITS=args.units, CLUSTER_SIM_KEYS='1' if args.mode == 'cache' else '0',
               CLUSTER_SIM_FUSE23='1' if args.fuse23 else '0', CLUSTER_SIM_PRIORITY=args.priority,
               **{f'BATCHING_STAGE{s}_WORKERS': n for s, n in workers.items()})
    return env


def wait_port(port, proc):
    while proc.poll() is None:
        with socket.socket() as s:
            if s.connect_ex(('127.0.0.1', port)) == 0:
                return
        time.sleep(2)
    raise RuntimeError(f'server exited with {proc.returncode} before listening')


def preload(port, group):
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    ctl = WebsocketClientPolicy(host='127.0.0.1', port=port)
    try:
        for entry in group['entries']:
            row = entry['row']
            ctl.load_cache_config(row['yaml'], yaml_id=row['arm'], bundle_id=row['arm'])
    finally:
        ctl.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('pure', 'cache'), required=True)
    parser.add_argument('--group', default='/data/mw_finetune/eval/cs_rec_pi05/group.json')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--log', required=True)
    parser.add_argument('--batch', default='1:32,2:16,3:16')
    parser.add_argument('--wait-ms', default='1:0,2:0,3:0')
    parser.add_argument('--workers', default='1:1,2:1,3:1')
    parser.add_argument('--threads', type=int, default=512)
    parser.add_argument('--text-tokens', type=int, default=80)
    parser.add_argument('--vram-gb', type=float, default=24)
    parser.add_argument('--omp', type=int, default=0)
    parser.add_argument('--compile', default='', help='comma list of vision,lm,expert (torch.compile)')
    parser.add_argument('--compile-mode', default='max-autotune-no-cudagraphs')
    parser.add_argument('--graphs', action='store_true', help='CUDA graphs per padded batch size (stages 1, 2, denoise step)')
    parser.add_argument('--buckets', default='1,2,4,8,12,16,24,32,48,64')
    parser.add_argument('--priority', default='', help='CUDA stream priority per stage, e.g. 1:0,3:-3 (lower first)')
    parser.add_argument('--fuse23', action='store_true', help='stage 2 runs inside the stage-3 batch (no hand-off)')
    parser.add_argument('--units', default='1,2,3', help='stage groups sharing the GPU one batch at a time, e.g. 1,23 or 123')
    args = parser.parse_args()
    Path(args.log).mkdir(parents=True, exist_ok=True)
    argv, module, group = command(args)
    env = environment(args, module)
    (Path(args.log) / 'SERVER.json').write_text(json.dumps(dict(vars(args), argv=argv, module=module), indent=1) + '\n')
    with (Path(args.log) / 'server.log').open('a') as out:
        # CLUSTER_SIM_PYSPY=<file>: run the server under py-spy (ptrace_scope 1 only lets a parent attach).
        pyspy = os.environ.get('CLUSTER_SIM_PYSPY')
        flags = os.environ.get('CLUSTER_SIM_PYSPY_FLAGS', '--threads --idle').split()
        prefix = [os.path.expanduser('~/.local/bin/py-spy'), 'record', *flags, '-r', '50', '-f', 'raw',
                  '-o', pyspy, '--'] if pyspy else []
        proc = subprocess.Popen([*prefix, str(TREE / '.venv/bin/python'), *argv], cwd=TREE, env=env, stdout=out,
                                stderr=out, start_new_session=True)
    def stop(signum, frame):
        # The server ignores SIGINT (its stage threads keep it alive); SIGTERM ends it, SIGKILL if it lingers.
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    wait_port(args.port, proc)
    if group is not None:
        preload(args.port, group)
    (Path(args.log) / 'READY.json').write_text(json.dumps(dict(time=time.time(), pid=proc.pid, port=args.port)) + '\n')
    print(f'READY pid={proc.pid} port={args.port}', flush=True)
    sys.exit(proc.wait())


if __name__ == '__main__':
    main()
