"""One server mode, every N: start serve.py, replay the library once per robot count, stop, analyze.

  python -m exp.cluster_sim.sweep --mode pure --run /nvme/weiland/openpi/cluster_sim/runs/pure \
      --library /nvme/weiland/openpi/cluster_sim/library --robots 5,10,20,30,40,50,60,70,80,90,100 [serve knobs]

Unknown options go to serve.py unchanged (--batch, --wait-ms, --workers, --threads, --text-tokens, --omp, ...).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import analyze, replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('pure', 'cache'), required=True)
    parser.add_argument('--run', required=True)
    parser.add_argument('--library', required=True)
    parser.add_argument('--robots', default='5,10,20,30,40,50,60,70,80,90,100')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--episodes', default='0', help='first K episodes per replay (tuning), or N:K per robot count; 0 = all 500')
    parser.add_argument('--stride', type=int, default=1, help='replay every stride-th infer frame (long execution segments)')
    parser.add_argument('--warmup', type=int, default=40, help='episodes replayed at 20 robots before measuring (not analyzed)')
    args, serve_args = parser.parse_known_args()
    run = Path(args.run)
    server_log = run / 'server'
    server_log.mkdir(parents=True, exist_ok=True)
    server = subprocess.Popen([sys.executable, '-m', 'exp.cluster_sim.serve', '--mode', args.mode, '--port', str(args.port),
                               '--log', str(server_log), *serve_args], start_new_session=True)
    try:
        while not (server_log / 'READY.json').exists():
            if server.poll() is not None:
                raise RuntimeError(f'server exited with {server.returncode}; see {server_log}/server.log')
            time.sleep(2)
        if args.warmup:
            asyncio.run(replay.run(argparse.Namespace(library=args.library, url=f'ws://127.0.0.1:{args.port}', robots=20,
                                                      seed=args.seed + 1, interval=replay.interval_for(args.stride), stride=args.stride,
                                                      out=str(run / 'warmup'), episodes=args.warmup)))
        per = dict(map(int, x.split(':')) for x in args.episodes.split(',')) if ':' in args.episodes else {}
        for n in map(int, args.robots.split(',')):
            asyncio.run(replay.run(argparse.Namespace(library=args.library, url=f'ws://127.0.0.1:{args.port}', robots=n,
                                                      seed=args.seed, interval=replay.interval_for(args.stride), stride=args.stride,
                                                      out=str(run / 'replay'),
                                                      episodes=per.get(n, 0) if per else int(args.episodes))))
            time.sleep(3)  # last request rows reach the server log
    finally:
        os.killpg(server.pid, signal.SIGINT)
        server.wait()
    analyze.analyze(run)


if __name__ == '__main__':
    main()
