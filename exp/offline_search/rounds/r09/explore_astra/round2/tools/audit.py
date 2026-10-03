"""Provenance and locked-split inventory; only allowlisted compact inputs."""
import json
from pathlib import Path
import numpy as np
from .data import HERE, COMPACT, CELLS, load, dataset, dump, sha


def main():
    sources=[];counts=[]
    for cell in CELLS:
        row=dict(cell=cell,paths={})
        for variant in ['A','CU','IP']+(['W10'] if cell.startswith('pi05') else []):
            z=load(cell,variant)
            if z is None:continue
            p=COMPACT/f'r8_{cell}_{variant}.npz'
            sources.append(dict(path=str(p),sha256=sha(p),metadata_sha256=sha(p.with_suffix('.json'))))
            d=dataset(cell,variant)
            row['paths'][variant]=dict(fit_anchors=int((d['init']<20).sum()),
                eval_anchors=int((d['init']>=20).sum()),eval_sr=float(z['success'][:,20:30].mean()),
                eval_ir=float(z['cost'][:,20:30].sum()/z['decisions'][:,20:30].sum()))
        counts.append(row)
    frozen=json.loads((HERE/'FROZEN_CANDIDATES.json').read_text())
    for c in frozen['candidates']:
        assert sha(c['artifact'])==c['artifact_sha256']
        assert sha(c['kwargs']['head_path'])==c['head_sha256']
    dump(HERE/'results/population.json',counts)
    outputs=[]
    for path in sorted(HERE.rglob('*')):
        if path.is_file() and path.name not in ['PROVENANCE.json','audit.log'] and '__pycache__' not in path.parts:
            outputs.append(dict(path=str(path.relative_to(HERE)),bytes=path.stat().st_size,sha256=sha(path)))
    dump(HERE/'PROVENANCE.json',dict(source_policy='Existing discovery-only compact store, identity checked BEFORE payload; raw physical files explicitly addressed only for 20..29',
        fit_inits=list(range(20)),eval_inits=list(range(20,30)),read_other_researcher_round2=False,
        sources=sources,outputs=outputs,candidate_hashes_match=True,closed_loop_run=False,gpu_used=False))
    print('AUDIT_OK',len(sources),'compact sources;',len(frozen['candidates']),'frozen candidates')


if __name__=='__main__':main()
