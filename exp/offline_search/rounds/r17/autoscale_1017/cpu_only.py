"""Pytest guard: fail an unmocked remote transfer/query before subprocess spawn."""
import os
from pathlib import Path
import socket
import subprocess
import signal
import faulthandler
import pytest


def pytest_runtest_logreport(report):
    if report.failed:
        import sys
        # Emit immediately, so a later slow case cannot hide the first stack.
        sys.__stderr__.write('\n'+str(report.longrepr)+'\n')


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Bound setup, call and teardown together, with the blocked stack recorded."""
    from .compare import wait_io
    wait_io()  # Do not begin another fixture read while full avg10 exceeds20.
    seconds = float(os.environ.get('AUTOSCALE_TEST_TIMEOUT', '60'))
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        faulthandler.dump_traceback()
        pytest.fail(f'per-test timeout ({seconds:g}s): {item.nodeid}', pytrace=True)
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def pytest_sessionstart(session):
    xml = getattr(session.config.option, 'xmlpath', None)
    if xml:
        import hashlib
        import json
        root=Path(__file__).resolve().parents[5]
        baseline=json.loads((Path(__file__).parent/'PATCH_BASE.json').read_text())
        names=list(baseline)+['exp/r17_integration/common/serving.py',
            'exp/r17_integration/ops/cache_study.py','exp/r17_integration/ops/cache_knob.py',
            'exp/r17_integration/ops/library_campaign.py']
        Path(xml).with_suffix('.sources.json').write_text(json.dumps(dict(
            baseline_overlay=bool(os.environ.get('AUTOSCALE_BASELINE_ROOT')),
            hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}),indent=2)+'\n')
    native = subprocess.Popen
    class CPUOnlyPopen(native):
        def __init__(self, args, *a, **kw):
            import shlex
            words = shlex.split(args) if isinstance(args,str) else list(map(str,args))
            if any(Path(w).name in ('tether','rsync','ssh','scp','nvidia-smi') for w in words):
                raise AssertionError('CPU test attempted unmocked remote/GPU/transfer command: '+str(words[:3]))
            super().__init__(args,*a,**kw)
    subprocess.Popen = CPUOnlyPopen
    native_bind = socket.socket.bind
    def guarded_bind(sock, address):
        if isinstance(address, tuple) and len(address) >= 2 and 23100 <= address[1] <= 23199:
            raise AssertionError("CPU test must not bind production port: " + str(address))
        return native_bind(sock, address)
    socket.socket.bind = guarded_bind
    connect=socket.create_connection
    def local_only(address,*a,**kw):
        if address[0] not in ('127.0.0.1','localhost','::1'):
            raise AssertionError('CPU test attempted remote socket: '+str(address))
        return connect(address,*a,**kw)
    socket.create_connection=local_only
