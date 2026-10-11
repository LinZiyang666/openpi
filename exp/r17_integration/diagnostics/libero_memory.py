"""Read-only LIBERO host-memory census; imports no simulator or GPU library.

Run on a worker node with an exact campaign name. Keep snapshots locally on
the coordinator; compare the same PID generation and dispatch time window.
The residual is an accounting estimate, not a NVIDIA allocation counter.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time


def kib_fields(text):
    fields = {}
    for line in text.splitlines():
        key, sep, value = line.partition(':')
        tokens = value.split()
        if sep and len(tokens) == 2 and tokens[1] == 'kB':
            fields[key] = int(tokens[0]) * 1024
    return fields


def residual_bytes(fields):
    # Cached includes Shmem; Slab includes SReclaimable. KReclaimable can also
    # include allocations outside Slab, so subtract only that additional part.
    # VmallocUsed is a virtual mapping total and may overlap these allocations.
    accounted = ('MemFree', 'AnonPages', 'Cached', 'Buffers', 'Slab',
                 'KernelStack', 'PageTables', 'Percpu', 'Hugetlb')
    other_reclaimable = max(0, fields.get('KReclaimable', 0) - fields.get('SReclaimable', 0))
    return fields['MemTotal'] - sum(fields.get(key, 0) for key in accounted) - other_reclaimable


def snapshot(campaign, *, proc_root=Path('/proc')):
    proc_root = Path(proc_root)
    meminfo = kib_fields((proc_root / 'meminfo').read_text())
    result = dict(schema='r17.libero_memory.v1', time=time.time(), campaign=campaign,
                  hostname=os.uname().nodename, meminfo_bytes=meminfo,
                  residual_bytes=residual_bytes(meminfo), workers=[], errors=[])
    modules = {'exp.r17_integration.conductor.worker_entry',
               'exp.r17_integration.ops.worker_lifetime', 'examples.libero.worker_entry'}
    for proc in sorted(proc_root.iterdir()):
        if not proc.name.isdigit():
            continue
        try:
            argv = (proc / 'cmdline').read_bytes().decode().split('\0')
            if '-m' not in argv or argv[argv.index('-m') + 1] not in modules:
                continue
            worker = argv[argv.index('--worker-id') + 1]
            # Formal IDs are campaign_node_index. A prefix match alone would
            # also include a different campaign named campaign_other.
            identity = worker.rsplit('_', 2)
            if len(identity) != 3 or identity[0] != campaign or not identity[2].isdigit():
                continue
            row = dict(pid=int(proc.name), worker_id=worker)
            # stat's comm field can itself contain spaces / parentheses.
            row['start_ticks'] = (proc / 'stat').read_text().rsplit(')', 1)[1].split()[19]
            row['status_bytes'] = kib_fields((proc / 'status').read_text())
            try:
                row['smaps_bytes'] = kib_fields((proc / 'smaps_rollup').read_text())
            except OSError as exc:
                row['smaps_error'] = str(exc)
            row['nvidia_maps'] = sum('/dev/nvidia' in line for line in (proc / 'maps').read_text().splitlines())
            fds = list((proc / 'fd').iterdir())
            row['fds'] = len(fds)
            row['nvidia_fds'] = sum('/dev/nvidia' in os.readlink(fd) for fd in fds)
            result['workers'].append(row)
        except (OSError, ValueError, IndexError) as exc:
            result['errors'].append(dict(pid=int(proc.name), error=str(exc)))
    return result


def compare(before, after, *, dispatched_episodes):
    if before['campaign'] != after['campaign'] or before['hostname'] != after['hostname']:
        raise ValueError('compare snapshots of the same campaign and node')
    if type(dispatched_episodes) is not int or dispatched_episodes < 1:
        raise ValueError('positive number of episode dispatches required')
    generation = lambda value: {(w['pid'], w['start_ticks']) for w in value['workers']}
    if not generation(before) or generation(before) != generation(after):
        raise ValueError('worker generations changed; segment the measurement at recycle boundaries')
    delta = after['residual_bytes'] - before['residual_bytes']
    return dict(residual_delta_bytes=delta,
                estimated_driver_bytes_per_worker_episode=delta / dispatched_episodes,
                elapsed_seconds=after['time'] - before['time'],
                workers=len(after['workers']),
                anon_delta_bytes=after['meminfo_bytes']['AnonPages'] - before['meminfo_bytes']['AnonPages'],
                note='node residual / episode dispatches; includes unrelated node allocations; not a pinned-page counter')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True)
    args = parser.parse_args()
    print(json.dumps(snapshot(args.campaign), sort_keys=True))


if __name__ == '__main__':
    main()
