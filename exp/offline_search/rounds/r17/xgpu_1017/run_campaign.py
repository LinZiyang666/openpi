"""Coordinator launcher: read one sealed campaign; no lane or campaign mutation."""
import argparse
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('job', 'inputs', 'output', 'python'):
        p.add_argument('--'+key, required=True)
    p.add_argument('--cache-yaml', required=True)
    p.add_argument('--memory-gib', type=float, required=True)
    p.add_argument('--profile', default='native')
    p.add_argument('--vendor-path', action='append', default=[])
    a = p.parse_args()
    job = json.loads(Path(a.job).read_text())
    argv = [a.python, str(Path(__file__).with_name('replay.py')), '--job', a.job,
        '--runtime', job['runtime'], '--runtime-identity', job['runtime_identity'],
        '--inputs', a.inputs, '--output', a.output, '--cache-yaml', a.cache_yaml,
        '--device', 'cuda:0', '--memory-gib', str(a.memory_gib), '--repeats', '3',
        '--deadline-seconds', '300', '--profile', a.profile]
    for vendor in a.vendor_path:
        argv += ['--vendor-path', vendor]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
        OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1', XLA_PYTHON_CLIENT_PREALLOCATE='false')
    print(json.dumps({'argv':argv,'scope':'new serial B1 replay; coordinator admission required'}),flush=True)
    raise SystemExit(subprocess.call(['timeout','--signal=TERM','--kill-after=10s','310s',*argv],env=env))


if __name__ == '__main__':
    main()
