"""Resolve <RUN>, verify exact R3 fit contracts, copy fits atomically, emit arm config.
No server, remote sync or rollout is started. Use --smoke for the g50 replicate pair.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.closed_loop.ops.emit_arms import main as emit

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-root',type=Path,required=True)
    ap.add_argument('--smoke',action='store_true')
    a=ap.parse_args()
    spec=json.loads((Path(__file__).with_name('arms_rand.json')).read_text().replace('<RUN>',str(a.run_root.resolve())))
    if a.smoke: spec=spec[:2]
    audited={}
    for arm in spec:
        src=Path(arm['_reuse_fit'])
        if str(src) not in audited:
            load_method_class(arm['method'])
            with src.open('rb') as f: blob=pickle.load(f)
            want=dict(spec=arm['method'],kwargs=arm['kwargs'],cell='pi05_l10_cache')
            assert {k:blob.get(k) for k in want}==want, (src,'fit contract mismatch')
            audited[str(src)]=dict(bytes=src.stat().st_size,sha256=hashlib.file_digest(src.open('rb'),'sha256').hexdigest(),contract=want)
            del blob
        args=arm['plugin_args']; dst=Path(args[args.index('--os-fit-artifact')+1]); dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():
            fd,tmp=tempfile.mkstemp(prefix='.k5_fit_',dir=dst.parent); os.close(fd)
            try: shutil.copyfile(src,tmp); os.replace(tmp,dst)
            finally:
                if Path(tmp).exists(): Path(tmp).unlink()
        assert hashlib.file_digest(dst.open('rb'),'sha256').hexdigest()==audited[str(src)]['sha256'], f'{dst}: existing fit differs'
    a.run_root.mkdir(parents=True,exist_ok=True)
    path=a.run_root/'arms_rand_resolved.json'; path.write_text(json.dumps(spec,indent=2)+'\n')
    emit(['--run-root',str(a.run_root),'--spec',str(path)])
    (a.run_root/'k5_fit_audit.json').write_text(json.dumps(audited,indent=2)+'\n')
    print(json.dumps(audited,indent=2))
if __name__=='__main__': main()
