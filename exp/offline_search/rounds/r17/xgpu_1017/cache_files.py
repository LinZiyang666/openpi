"""Independent byte identity of the fitted cache files, outside model execution.

The native YAML/recipe PIN remains unchanged. This extra seal records actual
files, verified before and after each new run; it is not a retrospective claim
about old runs. Reads are serial and deduplicated when preparing all bundles.
"""
import argparse
import json
from pathlib import Path
from artifacts import digest,sha,write_json


def checked_file(path):
    path=Path(path);before=path.stat()
    value=dict(bytes=before.st_size,sha256=sha(path))
    after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
        raise ValueError('cache dependency changed while hashing: '+str(path))
    return value


def seal_all(root):
    seen={};reports=[]
    for bundle in sorted(Path(root).iterdir()):
        if not (bundle/'PIN.json').is_file():continue
        pin=json.loads((bundle/'PIN.json').read_text())
        paths=(bundle/'data.files').read_text().splitlines()
        for name in paths:
            if not Path(name).is_relative_to('/data/mw_finetune') or '..'in Path(name).parts:
                raise ValueError('only prepared cache dependency paths accepted')
            if name not in seen:seen[name]=checked_file(name)
        body=dict(schema='xgpu.cache.files.v1',serving_pin=pin['identity'],
            files={name:seen[name] for name in paths},
            scope='current dependency bytes; not retrospective verification of old runs')
        doc=dict(body,identity=digest(body));target=bundle/'CACHE_FILES.json'
        if target.exists() and json.loads(target.read_text())!=doc:
            raise ValueError('cache seal changed; publish a new bundle instead of overwriting')
        if not target.exists():write_json(target,doc)
        reports.append(dict(bundle=bundle.name,identity=doc['identity'],files=len(paths),bytes=sum(seen[x]['bytes']for x in paths)))
    return dict(bundles=reports,unique_files=len(seen),unique_bytes=sum(x['bytes']for x in seen.values()))


def verify(bundle,relocations=()):
    bundle=Path(bundle);doc=json.loads((bundle/'CACHE_FILES.json').read_text())
    body=dict(doc);identity=body.pop('identity')
    if digest(body)!=identity or doc['serving_pin']!=json.loads((bundle/'PIN.json').read_text())['identity']:
        raise ValueError('cache file seal/PIN changed')
    for name,expected in doc['files'].items():
        if not Path(name).is_relative_to('/data/mw_finetune') or '..'in Path(name).parts:
            raise ValueError('unsafe cache dependency path')
        mapped=name
        for old,new in sorted(relocations,key=lambda x:-len(x[0])):
            if name==old or name.startswith(old+'/'):
                mapped=new+name[len(old):];break
        if checked_file(mapped)!=expected:raise ValueError('cache dependency bytes differ: '+name)
    return dict(identity=identity,serving_pin=doc['serving_pin'],files=len(doc['files']),
        bytes=sum(x['bytes']for x in doc['files'].values()),status='PASS')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seal-all',required=True)
    a=p.parse_args();print(json.dumps(seal_all(a.seal_all),indent=2))
