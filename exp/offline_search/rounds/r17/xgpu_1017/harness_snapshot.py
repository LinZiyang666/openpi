"""Freeze the small executing harness before any resident/client subprocess.

The model/runtime/cache PIN is separate. Snapshots contain source code only,
never checkpoints, inputs, environments or credentials. Old snapshots remain.
"""
import fcntl
import hashlib
import json
from pathlib import Path


FILES = ('run_concurrent.py','concurrent_server.py','concurrent_clients.py',
    'batch_telemetry.py','cache_telemetry.py','numerics_bootstrap.py',
    'attention_control.py','inventory.py','artifacts.py','capture.py',
    'replay.py','harness_snapshot.py','cache_files.py','pi05_checkpoint_view.py')


def seal(source, parent, names=FILES):
    source, parent = Path(source).resolve(), Path(parent).resolve()
    if any(Path(n).name != n for n in names):
        raise ValueError('flat explicit source files required')
    contents = {name:(source/name).read_bytes() for name in names}
    hashes = {name:hashlib.sha256(body).hexdigest() for name,body in contents.items()}
    identity = hashlib.sha256((json.dumps(hashes,sort_keys=True,separators=(',',':'))+'\n').encode()).hexdigest()
    parent.mkdir(parents=True,exist_ok=True)
    target = parent/identity
    with (parent/'.snapshot.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not target.exists():
            target.mkdir()
            for name,body in contents.items():
                path=target/name;path.write_bytes(body);path.chmod(0o444)
            receipt=dict(schema='xgpu.harness.snapshot.v1',identity=identity,files=hashes)
            path=target/'SNAPSHOT.json';path.write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n');path.chmod(0o444)
            target.chmod(0o555)
    return target
