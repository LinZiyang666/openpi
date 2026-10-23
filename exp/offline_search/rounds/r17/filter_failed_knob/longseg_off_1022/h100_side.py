"""Runs ON THE H100 with python3 (stdlib only): start / status / stop of one R16 resident.

Copied into the H100 code tree /data/oscl_h100/openpi_r16_q3_server at this same relative path (h100.code()).
It is the 4090's local_server (receipts, supervisor, port and GPU checks unchanged) with the H100's paths:
GR00T checkout/venv and checkpoints under /home/exouser, the tree's .venv -> /home/exouser/openpi/.venv,
no CPU pinning list of the 4090, HOME=/home/exouser. The served entry is conductor_runner.resident_command's:
rounds.r16.resident with the group registry.

  python3 h100_side.py start|status|stop|argv '<spec json>'
"""
import json
import os
from pathlib import Path
import sys
import time

TREE = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(TREE))
from exp.offline_search.closed_loop.ops.h100 import local_server as ls  # noqa: E402

CKPT = '/home/exouser/ckpt'
ls.GROOT = Path('/home/exouser/gr00t_n15')
ls.GROOT_PY = '/home/exouser/gr00t_n15_venv/.venv/bin/python'
ls.PI05_CKPT = CKPT + '/pi05_libero_pytorch'
ls.cpu_affinity = lambda: '0-%d' % (os.cpu_count() - 1)  # the supervisor's taskset; the 4090 list does not exist here
_command = ls.command


def command(spec):
    argv, env = _command(dict(spec, env=dict(spec['env'], GROOT_CKPT=f"{CKPT}/n15_{spec['suite']}")))
    assert argv[:2] == ['taskset', '-c']
    argv = argv[3:]
    i = next(i for i, v in enumerate(argv) if str(v).endswith(('serve_pi05.py', 'serve_groot.py')))
    argv[i:i+1] = ['-m', 'exp.offline_search.rounds.r16.resident', '--registry', spec['registry'],
                   '--server-tag', str(spec['port'])]
    if spec['model'] == 'groot':
        argv[i+6:i+6] = ['--resident-suite', spec['suite']]
    env['HOME'] = '/home/exouser'
    return argv, env


ls.command = command


def status(spec, wait_s=30):
    """Poll here while the resident is starting, so the launcher's 1 s loop makes few tether calls."""
    deadline = time.monotonic() + wait_s
    while True:
        state = ls.status(spec)
        if state['listening'] or state['dead'] or not state['running'] or time.monotonic() > deadline:
            return state
        time.sleep(2)


def stop(spec):
    try:
        return ls.stop(spec)
    except RuntimeError as error:
        if 'LOCAL_PORT_RELEASE_TIMEOUT' not in str(error):
            raise
        receipt = ls.matching(spec)
        ls.wait_port_release(spec, [receipt] if receipt else [], time.monotonic() + 240)
        return ls.stop(spec, timeout_s=0)


def main():
    action, spec = sys.argv[1], json.loads(sys.argv[2])
    if action == 'start':
        print(ls.start(spec))
    elif action == 'status':
        print(json.dumps(status(spec)))
    elif action == 'stop':
        print(stop(spec))
    elif action == 'argv':  # what start would exec; writes only the spec's out/runtime directory
        argv, env = command(spec)
        print(json.dumps(dict(argv=argv, env=env)))
    else:
        raise SystemExit('start, status, stop or argv')


if __name__ == '__main__':
    main()
