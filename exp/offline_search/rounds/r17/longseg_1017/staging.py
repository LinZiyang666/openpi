"""Small server metadata snapshot + reuse of already sealed immutable inputs."""
import json
import os
from pathlib import Path


def metadata_tree(root):
    root=Path(root);dest=root/'longseg_server_assets'
    dest.mkdir(exist_ok=True)
    def link(src,dst):
        if src.is_symlink():raise ValueError('Unexpected serving-asset symlink: '+str(src))
        if src.is_dir():
            dst.mkdir(exist_ok=True)
            for child in src.iterdir():link(child,dst/child.name)
        else:
            if dst.exists() and not dst.samefile(src):
                import time
                dst.rename(dst.with_name(dst.name+'.previous_'+str(time.time_ns())))
            if not dst.exists():os.link(src,dst)
    for src in (root/'assets').iterdir():
        if src.name not in ('cache_inputs','.eval.lock','TRANSFER_SHA256.json'):
            link(src,dest/src.name)
    return dest


def reusable_inputs(launch,path):
    """Trust the prior immutable install's seal; read metadata, never fit arrays.

    Missing source roots still use the native verified snapshot installer.
    Conflicting seals are rejected by that installer rather than overwritten.
    """
    path=Path(path)
    if not (path/'manifest.json').exists():return False
    local=json.loads((path/'manifest.json').read_text())
    code=('import json;from pathlib import Path\np=Path('+repr(str(path))+')\n'
        'm=json.loads((p/"manifest.json").read_text()) if (p/"manifest.json").exists() else {}\n'
        'print(json.dumps(dict(digest=m.get("digest"),files=m.get("files"),present=all((p/n).is_file() for n in m.get("files",{})))))')
    result=json.loads(launch.remote('h100',['python3','-c',code],seconds=90,replay_safe=True).splitlines()[-1])
    return result['digest']==local['digest'] and result['files']==local['files'] and result['present']
