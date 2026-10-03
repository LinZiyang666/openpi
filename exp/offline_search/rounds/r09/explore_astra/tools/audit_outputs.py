"""Check the final handoff, then hash owned source/results/artifacts for provenance."""
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
from .common import HERE,DERIVED,dump,sha,load_compact


def main():
    expected={'extract':70,'routing':216,'student':8,'student_audit':8,'shadow_memory':8,
              'horizon_metric':8,'synthesis':8,'call_value':4,'routing_uncertainty':8,'deployment_checks':8}
    counts={}
    for name,n in expected.items():
        values=json.loads((HERE/'results'/f'{name}.json').read_text())
        assert len(values)==n,(name,len(values),n);counts[name]=n
    for m in json.loads((HERE/'results/extract.json').read_text()):
        z=load_compact(m['arm'])
        assert z['success'].shape==(10,30) and z['cost'].shape==(10,30)
        assert len(z['task'])==m['decisions'] and z['decisions'].sum()==m['decisions']
        assert np.isin(z['success'],[0,1]).all()
        assert np.isfinite(z['policy_shadow_chunk']).all()
    root=Path('/home/weiland/trace_runs/os_closed_loop/r09_astra_confirmation')
    arms=json.loads((root/'arms.json').read_text())
    assert len(arms)==34 and len({a['arm'] for a in arms})==34
    for a in arms:
        manifest=json.loads(Path(a['manifest']).read_text())
        pairs={(p['task'],p['init']) for p in manifest['selected']}
        assert pairs=={(t,i) for t in range(10) for i in range(30)}
    hashes={}
    for folder in [HERE, HERE/'tools',HERE/'results',HERE/'artifacts']:
        for p in sorted(folder.iterdir()):
            if p.is_file() and p.name!='PROVENANCE.json' and p.suffix in ['.py','.sh','.md','.json','.npz','.pkl']:
                hashes[str(p.relative_to(HERE))]=dict(sha256=sha(p),bytes=p.stat().st_size)
    dump(HERE/'PROVENANCE.json',dict(time_utc=datetime.now(timezone.utc).isoformat(),
        audit='PASS',counts=counts,confirmation_arms=34,discovery_pairs=300,
        no_closed_loop_or_remote_launch=True,owned_files=hashes,
        run_arms_sha256=sha(root/'arms.json'),manifest_sha256=sha(root/'manifests/discovery300.json')))
    print('PASS',counts,'34 discovery-only confirmation arms',len(hashes),'hashed owned files')


if __name__=='__main__':main()
