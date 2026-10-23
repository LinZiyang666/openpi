"""Small server metadata snapshot + reuse of already sealed immutable inputs."""
import filecmp
import json
import os
from pathlib import Path
import shutil


def stall_file(path):
    """A Q3 stall artifact file. Servers cache its (inode, size, mtime, ctime) stamp and refuse a
    changed one; a new hard link bumps ctime on every link, so a prepare that linked a shared stall
    artifact stopped running batches at their next bundle load (10-20 01:41, fa_ls_v01_0)."""
    path=Path(path)
    return any((path.parent/name).exists() for name in ('stall.json','stall.pkl'))


def metadata_tree(root):
    root=Path(root);dest=root/'longseg_server_assets'
    dest.mkdir(exist_ok=True)
    def link(src,dst):
        if src.is_dir():
            dst.mkdir(exist_ok=True)
            for child in src.iterdir():link(child,dst/child.name)
        else:
            if dst.exists() and not dst.samefile(src):
                if stall_file(src) and filecmp.cmp(src,dst,shallow=False):return
                import time
                dst.rename(dst.with_name(dst.name+'.previous_'+str(time.time_ns())))
            if not dst.exists():
                if stall_file(src):shutil.copy2(src,dst)  # new inode; see stall_file
                else:os.link(src,dst)
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
