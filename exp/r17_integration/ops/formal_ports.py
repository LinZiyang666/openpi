"""Bounded orchestration-only port admission; never signal an existing listener."""
import errno
import inspect
import json
import socket
import time
from pathlib import Path


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


def admit(campaign, directory, *, remote, python, heartbeat=lambda: None):
    """Send the same CPU probe by value; model/runtime pins remain unchanged."""
    from .formal_process import write_json
    directory = Path(directory)
    result = dict(schema='r17.port_release.v1', status='WAITING', cpu_only=True,
                  seconds=120, server_node=campaign['server_node'], samples={})
    started = time.monotonic()
    try:
        if campaign['server_node'] == 'local':
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
