"""Step 2: short sweeps over candidate server configs for one mode, one summary table.

Each config is a list of serve.py options; every config runs sweep.py with the same short replay (default N=5 for
latency at light load and N=40 near saturation, 60 episodes each, 20 warm-up episodes) into <out>/<name>/.

  python -m exp.cluster_sim.tune --mode pure --library LIB --out /nvme/weiland/openpi/cluster_sim/tune/pure \
      --configs exp/cluster_sim/tune_pure.json

configs JSON: {"b32": ["--batch", "1:32,2:32,3:32"], "b16_s3x2": ["--batch", "1:16,2:16,3:16", "--workers", "1:1,2:1,3:2"], ...}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('pure', 'cache'), required=True)
    parser.add_argument('--library', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--configs', required=True)
    parser.add_argument('--robots', default='5,40')
    parser.add_argument('--episodes', default='60', help='K, or N:K per robot count')
    parser.add_argument('--warmup', type=int, default=20)
    parser.add_argument('--only', default='', help='comma list of config names to run (default all)')
    args = parser.parse_args()
    configs = json.loads(Path(args.configs).read_text())
    only = set(filter(None, args.only.split(',')))
    out = Path(args.out)
    table = {}
    for name, options in configs.items():
        if only and name not in only:
            continue
        run = out / name
        if not (run / 'RESULTS.json').exists():
            subprocess.run([sys.executable, '-m', 'exp.cluster_sim.sweep', '--mode', args.mode, '--run', str(run),
                            '--library', args.library, '--robots', args.robots, '--episodes', args.episodes,
                            '--warmup', str(args.warmup), *options], check=False)
        if (run / 'RESULTS.json').exists():
            table[name] = dict(options=options, results=json.loads((run / 'RESULTS.json').read_text()))
    (out / 'TUNE.json').write_text(json.dumps(table, indent=1) + '\n')
    for name, row in table.items():
        cells = '  '.join(f"N={r['robots']}: {r['requests_per_s']} req/s p50 {r['latency_ms']['p50']} p99 {r['latency_ms']['p99']} ms"
                          for r in row['results'])
        print(f'{name:24s} {cells}', flush=True)


if __name__ == '__main__':
    main()
