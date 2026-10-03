"""Integrity inventory and split contracts. No remote operation or holdout read."""
import json
import numpy as np
from .data import HERE,COMPACT,DERIVED,CELLS,load,dump,sha,library
from .prepare_confirmation import OUT
from .bounds import policy_references


def main():
    sources=[];populations=[]
    for cell in CELLS:
        for variant in ['A','CU','IP']:
            d=load(cell,variant)
            if d is None:continue
            p=COMPACT/f'r8_{cell}_{variant}.npz'
            label=DERIVED/f'{cell}_{variant}_labels.npz'
            sources.append(dict(path=str(p),sha256=sha(p),metadata_sha256=sha(p.with_suffix('.json')),
                labels=str(label),labels_sha256=sha(label)))
            populations.append(dict(cell=cell,variant=variant,fit_episodes=int(len(np.unique(d['episode'][d['init']<20]))),
                eval_episodes=int(len(np.unique(d['episode'][d['init']>=20]))),
                fit_anchors=int((d['init']<20).sum()),eval_anchors=int((d['init']>=20).sum())))
    frozen=json.loads((HERE/'FROZEN_CANDIDATES.json').read_text())
    for reference in policy_references():
        p=COMPACT/f"{reference['arm']}.npz"
        sources.append(dict(path=str(p),sha256=sha(p),metadata_sha256=sha(p.with_suffix('.json')),
            purpose='Historical P10 reference, outcomes sliced to 20..29 before aggregation'))
    for c in frozen['candidates']:
        assert sha(c['artifact'])==c['artifact_sha256']
        assert sha(c['kwargs']['head_path'])==c['head_sha256']
        assert sha(c['kwargs']['phase_path'])==c['phase_sha256']
    manifest=json.loads((OUT/'manifests/eval100.json').read_text())
    assert {(r['task'],r['init']) for r in manifest['selected']}=={(t,i) for t in range(10) for i in range(20,30)}
    dependencies=set()
    for c in frozen['candidates']:dependencies.add(c['kwargs']['base_fit'])
    dependencies.add(str(HERE.parent/'round2/inference.py'))
    for cell in CELLS:
        for name in ['step.npy','ep_len.npy','next.npy','action.npy']:dependencies.add(str(library(cell)/name))
    dependency_hashes=[dict(path=p,sha256=sha(p)) for p in sorted(dependencies)]
    dump(HERE/'results/populations.json',populations)
    outputs=[]
    for p in sorted(HERE.rglob('*')):
        if p.is_file() and p.name not in ['PROVENANCE.json','audit.log'] and '__pycache__' not in p.parts:
            outputs.append(dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)))
    dump(HERE/'PROVENANCE.json',dict(fit_inits=list(range(20)),eval_inits=list(range(20,30)),
        source_policy='Validate discovery compact identities before payload. Address single raw episodes only via accepted discovery IDs. No mixed raw journal reader used.',
        sources=sources,dependencies=dependency_hashes,outputs=outputs,other_researchers_read=False,closed_loop_launched=False,
        gpu_used=False,remote_mutation=False,candidate_hashes_match=True,
        cpu_affinity='10-21,54-65',thread_environment='OMP/OPENBLAS/MKL=1'))
    print('AUDIT_OK',len(sources),'discovery sources',len(frozen['candidates']),'frozen candidates')


if __name__=='__main__':main()
