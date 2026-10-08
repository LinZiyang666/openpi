"""Bind final implementation source bytes to the completed numerical fits."""
import json
import pickle
from pathlib import Path
from .common import *

def main():
    files=[*KNOB.glob('*.py'),HERE/'build.py',HERE/'common.py',HERE/'allocation_audit.py']
    code={str(p.relative_to(REPO)):sha(p) for p in sorted(files)}
    records=json.loads((HERE/'fits.json').read_text())
    for rec in records:
        rec['implementation_sources_sha256']=code
        p=Path(rec['path'])
        if p.is_relative_to(KNOB):
            blob=pickle.loads(p.read_bytes())
            if 'r14_manifest' in blob:
                obj=blob['method']
                if obj.r14_head=='pool':
                    obj.inner.base.head_meta=dict(obj.inner.base.head_meta,task_onehot=False,per_task=False)
                manifest=blob['r14_manifest']
                manifest['implementation_sources_sha256']=code
                manifest.pop('manifest_sha256',None)
                manifest['manifest_sha256']=hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest()
                blob['method'].r14_fit=manifest
                rec['manifest']=manifest
            else:
                blob['r14_reference_manifest']['implementation_sources_sha256']=code
            dump_pickle(p,blob)
            rec.update(sha256=sha(p),bytes=p.stat().st_size)
    write_json(HERE/'fits.json',records)
    write_json(HERE/'implementation_sources.json',code)
    print('SEALED',len(records),sum(r['bytes'] for r in records))

if __name__=='__main__':main()
