"""Simple library-only coverage audit, not the full R6 Q1 quality score."""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.distance import cdist

OUT=Path(__file__).resolve().parent
STORE=Path('/home/weiland/trace_runs/offline_search_store/library')
PRIOR=OUT.parents[1]/'r05/ideation_B'

def main():
    result=[]
    for cell in ['pi05_l10','pi05_spatial','groot_l10','groot_spatial']:
        for size in ([50,100,200,300,500] if cell.startswith('pi05') else [50,500]):
            lib='current' if size==50 else (('bpool_cs' if cell.startswith('pi05') else 'bpool_all') if size==500 else f'demo{size}')
            p=STORE/cell/lib;meta=json.loads((p/'manifest.json').read_text())
            rs=np.load(p/'rs.npy',mmap_mode='r')[:,:meta['rs_valid_dims']]
            ts=np.load(p/'task_id.npy',mmap_mode='r');ep=np.load(p/'episode.npy',mmap_mode='r')
            prior_path=PRIOR/f'verified_library_{cell}_{size}.json'
            prior=json.loads(prior_path.read_text()) if prior_path.exists() else None
            for t in np.unique(ts):
                idx=np.flatnonzero(ts==t);x=np.asarray(rs[idx],float);e=ep[idx];sd=x.std(0);active=sd>0
                x=x[:,active]/sd[active];ds=cdist(x,x,'sqeuclidean')/max(1,active.sum());ds[e[:,None]==e[None,:]]=np.inf
                near=np.sqrt(ds.min(1));per_ep=[float(near[e==j].mean()) for j in np.unique(e)]
                # Squashing is descriptive only; raw distance is the regression predictor.
                rr=dict(cell=cell,lib=str(size),task=int(t),library=str(p),episodes=len(np.unique(e)),rows=len(idx),
                        median_decisions=float(np.median([sum(e==j) for j in np.unique(e)])),
                        mean_decisions=float(np.mean([sum(e==j) for j in np.unique(e)])),
                        mean_anchors_b2=float(np.mean([np.ceil(sum(e==j)/2) for j in np.unique(e)])),
                        state_loeo_distance=float(np.mean(per_ep)),coverage=float(np.mean([np.exp(-a) for a in per_ep])),
                        inverse_sqrt_episodes=1/np.sqrt(len(np.unique(e))),
                        action_loeo=None,state_residual_loeo=None,prior_source=str(prior_path) if prior else None)
                if prior:
                    rr['action_loeo']=prior['table']['base']['action_RMS']['tasks'][str(t)]
                    rr['state_residual_loeo']=prior['table']['base']['state_RMS']['tasks'][str(t)]
                result.append(rr)
            print(cell,size,'done',flush=True)
    (OUT/'library_quality.json').write_text(json.dumps(result,indent=1))

if __name__=='__main__':main()
