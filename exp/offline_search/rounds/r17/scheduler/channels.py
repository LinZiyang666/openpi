"""Launcher <-> node control channels: the 30-second lease/status call carries the
driver's control to the node and the node's snapshot back."""
from pathlib import Path

from .node import atomic, read


def status_entries(entries, directory):
    """Piggyback on the existing 30-second lease/status call, including N=0."""
    control = read(Path(directory) / 'AUTOSCALE_CONTROL.json')
    return [dict(e, autoscale_control=control) if e['label'] == 'workers' else e for e in entries]


def status_result(request, result):
    path = request.get('env', {}).get('OPENPI_AUTOSCALE_NODE')
    if path:
        result['autoscale_node'] = read(path, {})
    return result


def control_entry(entry):
    path = entry['request'].get('env', {}).get('OPENPI_AUTOSCALE_CONTROL')
    control = entry.get('autoscale_control')
    if path and control and Path(path).parent.is_dir():
        with Path(path + '.lock').open('a') as lock:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX)
            old = read(path, {})
            if control.get('time', 0) > old.get('time', 0):
                atomic(path, control)


def collect_status(directory, status_cache):
    atomic(Path(directory) / 'AUTOSCALE_NODES.json', {n: s['autoscale_node'] for (n, label), s in status_cache.items()
                                                     if label == 'workers' and s.get('autoscale_node')})
