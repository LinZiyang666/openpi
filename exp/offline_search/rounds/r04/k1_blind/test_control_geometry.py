"""Document the intentional CSL differences and unchanged fresh/step-zero action branches."""
import json,pickle
import numpy as np
from exp.offline_search.harness import api,store
from exp.offline_search.rounds.r04.k1_blind.checks import ROOT,OUT,view,get_method


def main():
    reports=[]
    for scale in (50,500):
        for key in ('pi05_l10','pi05_spatial','groot_l10','groot_spatial'):
            with open(f'/tmp/k1_blind_fits/cslG_{key}_{scale}.pkl','rb') as f:g=pickle.load(f)['method']
            with open(f'/tmp/k1_blind_fits/cslGS_{key}_{scale}.pkl','rb') as f:gs=pickle.load(f)['method']
            awm=get_method(key,scale)
            for arm in ('inf','cache'):
                qc=store.QueryCell(ROOT,key+'_'+arm);arrays=api.QueryArrays(qc)
                counts=dict(queries=0,unchanged_branch=0,stale=0,nonzero_offset=0,changed_parent=0,changed_splice=0)
                for i in sorted(set([0]+np.linspace(1,qc.N-1,32,dtype=int).tolist())):
                    q=view(qc,arrays,i)
                    a,b,c=g.query(q),gs.query(q),awm.query(q)
                    counts['queries']+=1
                    np.testing.assert_array_equal(a.topk,b.topk)
                    np.testing.assert_array_equal(a.scores,b.scores)
                    if not q.step or q.prev_hit is False:
                        counts['unchanged_branch']+=1
                        for r in (a,b):
                            np.testing.assert_array_equal(r.topk,c.topk)
                            np.testing.assert_array_equal(r.scores,c.scores)
                            np.testing.assert_array_equal(r.action,c.action)
                    else:
                        counts['stale']+=1
                        counts['nonzero_offset']+=int(a.extras['offset_nonzero_mass']>0)
                        counts['changed_parent']+=int(not np.array_equal(a.topk,c.topk))
                        counts['changed_splice']+=int(not np.array_equal(a.action[:5,:7],b.action[:5,:7]))
                        assert len(np.unique(a.topk))==len(a.topk)
                reports.append(dict(key=key,scale=scale,arm=arm,**counts))
    (OUT/'control_geometry.json').write_text(json.dumps(reports,indent=2));print(json.dumps(reports),flush=True)

if __name__=='__main__':main()
