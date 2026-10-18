"""CPU-only private checkpoint view; preserve the frozen identity and shared files."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import uuid

DEFAULT_VIEW = '/data/mw_finetune/xgpu_1017/pi05_ckpt_4008289bc944'


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def fingerprint(path):
    s=Path(path).stat()
    return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns)


def file_hash(path, mib_per_second=64):
    h=hashlib.sha256();start=time.monotonic();count=0
    with Path(path).open('rb') as stream:
        while block:=stream.read(4*1024*1024):
            h.update(block);count+=len(block)
            if mib_per_second:
                delay=count/(mib_per_second*1024*1024)-(time.monotonic()-start)
                if delay>0:time.sleep(delay)
    return h.hexdigest()


def verify(view, manifest, *, full=False, mib_per_second=64):
    view=Path(view); receipt_path=view.with_name(view.name+'_receipt.json')
    receipt=json.loads(receipt_path.read_text())
    expected=manifest['content_digests']
    if receipt.get('checkpoint_identity')!=manifest['checkpoint_identity'] or receipt.get('hashes')!=expected:
        raise ValueError('checkpoint view receipt differs from frozen manifest')
    if view.is_symlink():raise ValueError('checkpoint view must be a real directory')
    actual={p.relative_to(view).as_posix() for p in view.rglob('*') if p.is_file()}
    if actual!=set(expected):raise ValueError('checkpoint view has missing/extra files')
    for name,wanted in expected.items():
        path=view/name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents if p!=view.parent):
            raise ValueError('checkpoint view cannot contain symlinks')
        if fingerprint(path)!=receipt['fingerprints'][name]:raise ValueError('checkpoint view changed: '+name)
        if full or path.stat().st_size<1024*1024:
            if file_hash(path,mib_per_second)!=wanted:raise ValueError('checkpoint hash mismatch: '+name)
    return receipt


def create(view, source, small, manifest, *, mib_per_second=64):
    view,source,small=map(Path,(view,source,small))
    expected=manifest['content_digests'];weight='model.safetensors'
    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in expected):raise ValueError('unsafe manifest path')
    calculated=digest(dict(config='pi05_metaworld',files=expected,tokenizer=manifest['tokenizer']['sha256']))
    if calculated!=manifest['checkpoint_identity']:raise ValueError('sealed manifest identity inconsistent')
    if view.exists():return verify(view,manifest,full=True,mib_per_second=mib_per_second)
    view.parent.mkdir(parents=True,exist_ok=True)
    if source.is_symlink() or source.stat().st_dev!=view.parent.stat().st_dev:
        raise ValueError('private checkpoint must share the source filesystem; no weight copy fallback')
    temporary=view.with_name(view.name+'_building_'+uuid.uuid4().hex[:10])
    temporary.mkdir()
    # Only the immutable weight is hard-linked. Never chmod/write that inode.
    os.link(source/weight,temporary/weight)
    for name,wanted in expected.items():
        if name==weight:continue
        p=small/name
        if p.is_symlink() or p.stat().st_size>=1024*1024 or file_hash(p,0)!=wanted:
            raise ValueError('unsealed small checkpoint file: '+name)
        dest=temporary/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest)
    hashes={}
    for name,wanted in expected.items():
        path=temporary/name;before=fingerprint(path);hashes[name]=file_hash(path,mib_per_second)
        if before!=fingerprint(path) or hashes[name]!=wanted:raise ValueError('checkpoint content differs: '+name)
    if view.exists():raise ValueError('another process created checkpoint view')
    temporary.rename(view)
    receipt=dict(schema='xgpu.pi05.checkpoint_view.v1',checkpoint_identity=calculated,
                 source=str(source),view=str(view),hashes=hashes,
                 fingerprints={n:fingerprint(view/n) for n in expected},
                 weight_hardlinked=(source/weight).stat().st_ino==(view/weight).stat().st_ino,
                 additional_file_bytes=sum((view/n).stat().st_size for n in expected if n!=weight),
                 logical_bytes=sum((view/n).stat().st_size for n in expected),
                 scope='frozen identity preserved; shared checkpoint bytes untouched; no model load')
    with view.with_name(view.name+'_receipt.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
    return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',default='/home/exouser/xgpu_1017/concurrent/bundles/pi05')
    p.add_argument('--view',default=DEFAULT_VIEW)
    p.add_argument('--small',default='/home/exouser/xgpu_1017/pi05_checkpoint_small')
    p.add_argument('--source',default='/data/ckpt/pi05_metaworld_rlinf')
    p.add_argument('--create',action='store_true')
    a=p.parse_args();manifest=json.loads((Path(a.bundle)/'assets/model_manifest.json').read_text())
    result=create(a.view,a.source,a.small,manifest) if a.create else verify(a.view,manifest)
    print(json.dumps(result))


if __name__=='__main__':main()
