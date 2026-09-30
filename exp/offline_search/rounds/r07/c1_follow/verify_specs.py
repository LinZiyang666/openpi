"""Check every delivered fit against both emit_arms input files, CPU only."""
import json
import pickle
from pathlib import Path

import numpy as np

from exp.offline_search.closed_loop.blind import BlindQueryView, LookReason
from exp.offline_search.closed_loop.ops.emit_arms import main as emit_arms
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import fingerprint, sha
from .prefit import HERE, OUT


def main():
    checked = []
    for phase in ('profile', 'eval500'):
        rows = json.loads((HERE/f'arms_{phase}.json').read_text())
        assert len(rows) == 24 and len({r['name'] for r in rows}) == 24
        for row in rows:
            args = row['plugin_args']
            path = Path(args[args.index('--os-fit-artifact')+1].replace('<RUN>',str(OUT)))
            with path.open('rb') as f:
                blob = pickle.load(f)
            assert blob['spec'] == row['method'] and blob['kwargs'] == row['kwargs']
            assert blob['cell'] == f"{row['model']}_{row['suite']}_cache"
            method = blob['method']
            assert fingerprint(method) == method.follow_table.retrieval_fingerprint
            assert method.follow_extend_blocks == row['kwargs']['extend_blocks']
            assert not hasattr(method, 'policy_tail_step')
            if phase == 'eval500':
                # A policy-origin predecessor cannot enter follow continuation.
                rs = np.zeros_like(method.blind_rs[0])
                bq = BlindQueryView(1,0,'policy',rs,rs,False,None,
                                    np.empty((0,*method.act.shape[1:]),np.float32),np.zeros(1,np.int8),
                                    np.zeros((1,len(rs)),np.float32),np.ones(1,bool),0)
                assert isinstance(method.blind_step(bq),LookReason)
                assert method._anchor is None
            checked.append(dict(phase=phase,name=row['name'],artifact=str(path),sha256=sha(path),
                                kwargs=row['kwargs'],bytes=path.stat().st_size))
        emit_arms(['--run-root',str(OUT/f'emitted_{phase}'),'--spec',str(HERE/f'arms_{phase}.json')])
    note=dict(arms_checked=len(checked),unique_fits=24,policy_origin_rejections=24,
              emitted_yaml=48,emitted_matrices=48,checks=checked)
    (OUT/'packaging.json').write_text(json.dumps(note,indent=2)+'\n')
    print(json.dumps({k:v for k,v in note.items() if k!='checks'}))


if __name__=='__main__':
    main()
