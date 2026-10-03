"""Emit only CLIP specs; prove existing other arms/configs are preserved."""
import hashlib
import json
from pathlib import Path

from exp.offline_search.closed_loop.ops import emit_arms
from .make_arms import HERE,RUN,build
from .prepare import write_json


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    # The common emitter locks .emit_arms.lock across read/merge/config/write.
    # arms_in.json belongs to the other supplementary-ablation task. Do not write it.
    rows,prov=build()
    spec=HERE/'arms_clip.json'
    write_json(spec,rows)
    own={r['name'] for r in rows}
    previous=json.loads((RUN/'arms.json').read_text()) if (RUN/'arms.json').exists() else []
    foreign={r['arm']:r for r in previous if r['arm'] not in own}
    cfgs={str(p):sha(p) for r in foreign.values() for p in (r['yaml'],r['matrix'])}
    emit_arms.main(['--run-root',str(RUN),'--spec',str(spec)])
    emitted={r['arm']:r for r in json.loads((RUN/'arms.json').read_text())}
    assert own<=emitted.keys()
    assert all(emitted[n]==r for n,r in foreign.items())
    assert all(sha(p)==digest for p,digest in cfgs.items())
    artifacts={r['name']:(RUN/'fits'/f'{r["name"]}.pkl') for r in rows}
    ready=[n for n,p in artifacts.items() if p.exists()]
    blocked=sorted(own-set(ready))
    evidence={'PASS':True,'clip_arms':len(own),'prior_other_arms_preserved':len(foreign),
              'prior_other_config_files_byte_equal':len(cfgs),'total_emitted_arms':len(emitted),
              'ready_arms':ready,'blocked_arms':blocked,'merge':'common emitter preserves rows by name; flock serializes merge and atomic replace',
              'unchanged_other_rows':foreign,'other_config_sha256':cfgs}
    write_json(HERE/'results/emission.json',evidence)
    write_json(RUN/'clip/ready_arms.json',[emitted[n] for n in ready])
    write_json(RUN/'clip/blocked_arms.json',[{'arm':n,'reason':'Exact GR00T library images unavailable; no fit artifact. Do not schedule.'} for n in blocked])
    write_json(RUN/'clip/pairing.json',{'pairs':500,'task_ids':list(range(10)),'inits':list(range(50)),
                                      'mode':'standard; no debug capture','arms':[{k:p[k] for k in ('arm','A_references')} for p in prov]})
    print(json.dumps({k:v for k,v in evidence.items() if k not in ('unchanged_other_rows','other_config_sha256')}),flush=True)


if __name__=='__main__': main()
