"""Run-scoped JSON control files sent through the existing /tmp transfer."""
from pathlib import Path
import json

from .build_client_bundle import json_bytes
from .sync_back import atomic_bytes


def stage_json(node, value, directory, stage, *, remote, transfer, python='python3'):
    raw = json_bytes(value)
    import hashlib
    digest = hashlib.sha256(raw).hexdigest()
    local = Path(directory)/'control_inputs'/(digest+'.json')
    atomic_bytes(local, raw)
    target = stage+'/control_'+digest+'.json'
    transfer(['rsync','push',str(local),node+':'+target])
    return dict(path=target,sha256=digest,bytes=len(raw))


def load_json(reference):
    path = Path(reference['path'])
    return json.loads(path.read_text())


def expression(reference):
    return 'load_json('+repr(reference)+')'
