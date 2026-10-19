"""Run-scoped JSON control files, verified after the existing /tmp transfer."""
from pathlib import Path
import json

from .build_client_bundle import json_bytes, sha256
from .sync_back import atomic_bytes


def stage_json(node, value, directory, stage, *, remote, transfer, python='python3'):
    raw = json_bytes(value)
    import hashlib
    digest = hashlib.sha256(raw).hexdigest()
    local = Path(directory)/'control_inputs'/(digest+'.json')
    atomic_bytes(local, raw)
    target = stage+'/control_'+digest+'.json'
    transfer(['tether','push',str(local),node+':'+target])
    code = ('from pathlib import Path;import hashlib,json; p=Path('+repr(target)+'); '
            's=hashlib.sha256(p.read_bytes()).hexdigest(); '
            'assert s=='+repr(digest)+',"staged control JSON SHA mismatch"; print(json.dumps({"sha256":s}))')
    verified = json.loads(remote(node,[python,'-c',code]))
    if verified != {'sha256':digest}:
        raise ValueError('remote control transfer verification differs')
    return dict(path=target,sha256=digest,bytes=len(raw))


def load_json(reference):
    path = Path(reference['path'])
    if path.is_symlink() or path.resolve()!=path or sha256(path)!=reference['sha256']:
        raise ValueError('staged control JSON SHA/path mismatch')
    return json.loads(path.read_text())


def expression(reference):
    return 'load_json('+repr(reference)+')'
