"""Generate a narrow installed-distribution allowlist; no download or install."""
import argparse
import importlib.metadata as md
from pathlib import Path
import json
import sys
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--allow-missing-metadata',action='store_true',help='copy installed closure only; log missing declarations, no package installation')
    a=p.parse_args()
    base=Path(md.distribution('torch').locate_file('')).resolve()
    pending=a.roots.split(',');versions={};files=set();missing=set()
    while pending:
        name=canonicalize_name(pending.pop())
        if name in versions or name in missing:continue
        try:dist=md.distribution(name)
        except md.PackageNotFoundError:
            if not a.allow_missing_metadata:raise
            missing.add(name);continue
        versions[name]=dist.version
        for requirement in dist.requires or []:
            req=Requirement(requirement)
            if req.marker is None or req.marker.evaluate({'extra':''}):pending.append(req.name)
        for f in dist.files or []:
            path=Path(dist.locate_file(f))
            if path.is_file() and path.resolve().is_relative_to(base) and '__pycache__' not in path.parts and path.suffix!='.pyc':
                files.add(path.resolve())
    includes=set()
    for f in files:
        rel=f.relative_to(base)
        includes.add('/'+str(rel))
        includes.update('/'+str(x)+'/' for x in rel.parents if x!=Path('.'))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'site.include').write_text(''.join(x+'\n' for x in sorted(includes)))
    (out/'site.files').write_text(''.join(str(x)+'\n' for x in sorted(files)))
    receipt=dict(site=str(base),python=sys.version,base_prefix=sys.base_prefix,packages=versions,
        bytes=sum(x.stat().st_size for x in files),files=len(files),missing_distribution_metadata=sorted(missing))
    (out/'ISLAND.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='packages'}))


if __name__=='__main__':main()
