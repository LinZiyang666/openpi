"""Two direct PCA arms x four complete token-subsample episodes through the real plugin."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from exp.offline_search.harness import store
from exp.offline_search.rounds.r06.p1_groot_commit import replay
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE,RUN,STORE,artifact


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['pi05','groot'],required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    r,=[r for r in json.loads((HERE/'arms_pca.json').read_text()) if r['model']==a.model and r['suite']=='spatial' and r['kwargs']['lib']=='current']
    qc=store.QueryCell(STORE,f'{a.model}_spatial_cache')
    eps=[i for i,e in enumerate(qc.episodes) if np.all(qc.tok_index[e['start']:e['end']]>=0)]
    selected=[eps[i] for i in np.linspace(0,len(eps)-1,4,dtype=int)]
    replay.main(['--method',r['method'],'--kwargs',json.dumps(r['kwargs']),'--cell',f'{a.model}_spatial_cache',
        '--replay-cell',f'{a.model}_spatial_cache','--fit-artifact',artifact(r).replace('<RUN>',str(RUN)),
        '--episodes','='+','.join(map(str,selected)),'--tag','pca_smoke','--out',str(a.out)])
    report=json.loads((a.out/'report.json').read_text())
    data=np.load(a.out/'served.npz')
    assert report['episodes']==4 and report['misses']==0 and report['max_blind_run']==1
    assert report['vision']==report['stage1_calls']
    tails=0
    for i in range(len(data['step'])):
        if data['vision'][i]:continue
        assert i>0 and data['vision'][i-1]
        assert np.array_equal(data['served'][i,:5,:7],data['served'][i-1,5:10,:7])
        tails+=1
    report.update(PASS=True,selected_episodes=selected,tail_checks=tails)
    (HERE/'results/pca'/f'smoke_{a.model}.json').write_text(json.dumps(report,indent=1)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
