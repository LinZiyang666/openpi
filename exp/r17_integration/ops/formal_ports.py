"""Bounded orchestration-only port admission; never signal an existing listener."""
import errno
import inspect
import json
import socket
import time
from pathlib import Path


def local_ports(campaign):
    """Server ports belong to server_node; the campaign driver is local."""
    return sorted(set(campaign['ports'] if campaign['server_node'] == 'local'
                      else [campaign['driver_port']]))


def prepare_admission(campaign):
    """Fail before staging on a local collision, without starting a listener.

    Server ports of the lane's registered kept resident are skipped: its launch attaches to it or stops it."""
    from exp.offline_search.rounds.r17.seeds_1023.resident_attach import held_ports
    held = held_ports(campaign)
    busy = []
    for port in local_ports(campaign):
        if port in held:
            continue
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('0.0.0.0', port))
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                busy.append(port)
    if busy:
        raise ValueError('LOCAL_PORT_BUSY before staging: '+str(busy)+
                         '; leave existing owners running and choose disjoint campaign ports')


def probe_ports(ports):
    """Test the actual reusable listener shape, holding all binds together."""
    sockets, busy = [], []
    try:
        for port in ports:
            sock = socket.socket()
            sockets.append(sock)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(('0.0.0.0', port))
                sock.listen(1)
            except OSError as exc:
                if exc.errno != errno.EADDRINUSE:
                    raise
                busy.append(port)
        return dict(free=not busy, ports=list(ports), busy=busy)
    finally:
        for sock in sockets:
            sock.close()


def wait_release(probe, *, seconds=120, heartbeat=lambda: None):
    deadline, samples = time.monotonic()+seconds, []
    while True:
        heartbeat()
        result = probe()
        samples.append(dict(elapsed_seconds=round(time.monotonic()-(deadline-seconds), 3), **result))
        if result['free']:
            return samples
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            raise TimeoutError('ports did not release within '+str(seconds)+'s: '+str(result['busy']))
        time.sleep(min(.5, remaining))


def admit(campaign, directory, *, remote, python, heartbeat=lambda: None, driver_only=False):
    """Send the same CPU probe by value; model/runtime pins remain unchanged.

    driver_only: an attached kept resident already serves the public/backend ports."""
    from .formal_process import write_json
    directory = Path(directory)
    result = dict(schema='r17.port_release.v1', status='WAITING', cpu_only=True,
                  seconds=120, server_node=campaign['server_node'], samples={}, driver_only=driver_only)
    started = time.monotonic()
    try:
        if driver_only:
            result['samples']['local_driver'] = wait_release(
                lambda: probe_ports([campaign['driver_port']]), heartbeat=heartbeat)
        elif campaign['server_node'] == 'local':
            result['samples']['local'] = wait_release(lambda: probe_ports(campaign['ports']),
                                                     heartbeat=heartbeat)
        else:
            code = ('import socket,errno,json;\n'+inspect.getsource(probe_ports)+
                    '\nprint(json.dumps(probe_ports('+repr(campaign['ports'][:-1])+')))')
            result['samples']['h100'] = wait_release(lambda: json.loads(remote(
                'h100', [python, '-c', code], seconds=10, replay_safe=True)), heartbeat=heartbeat)
            result['samples']['local_driver'] = wait_release(
                lambda: probe_ports([campaign['driver_port']]), heartbeat=heartbeat)
        result['status'] = 'PASS'
        return result
    except BaseException as exc:
        result.update(status='FAIL', error=repr(exc))
        raise
    finally:
        result['elapsed_seconds'] = time.monotonic()-started
        write_json(directory/'PORT_RELEASE.json', result)
