"""Top-1 library-stage proxy and NumPy kernel timing, both descriptive."""
import json
import time
import numpy as np
from analyze_lazy import OUT, SCRATCH, load_bank, loeo_neighbours, residuals


def main():
    results=[]
    for model in ['pi05', 'groot']:
        for cell in ['l10_50','l10_500','sp_50','sp_500']:
            key=model+'_'+cell
            awm,lib,rs,scale,succ,_,_=load_bank(model,cell)
            cam=json.loads((OUT/f'cameras_{key}.json').read_text())
            q=json.loads((OUT/f'evidence_{key}.json').read_text())['calibration']['q_row95']
            motion=np.zeros(len(rs)); valid=succ[1]>=0
            motion[valid]=np.sqrt(np.mean(((rs[succ[1,valid]]-rs[valid])/scale)**2,axis=1))
            signs=np.asarray(lib['action'][:,:,6])>=0
            prev=lib['prev']
            event=(signs!=signs[:,:1]).any(axis=1) | ((prev>=0)&(signs[:,0]!=signs[np.maximum(prev,0),4]))
            stages=np.where(event,'gripper_change',np.where(motion<=cam['motion_median'],'low_motion','moving'))
            result=dict(cell=key,stage_proxy='top-1 library row; low-motion uses library median; not semantic ground truth')
            for stream in ['bval','p3_A_r0']:
                with np.load(SCRATCH/f'{key}_{stream}_residuals.npz') as z:
                    labels=stages[z['members'][:,0]]
                    out={}
                    for stage in sorted(set(labels)):
                        mask=(labels==stage)&z['support'][:,4]
                        if not mask.any():continue
                        alert=np.nanmax(z['delta'][mask,1:5],axis=1)>q
                        out[stage]=dict(n=int(mask.sum()),alert20=float(alert.mean()),
                                       median_delta20=float(np.median(z['delta'][mask,4])),
                                       share_windows_from_failed_episodes=float((z['success'][mask]==0).mean()))
                    result[stream]=out
                    members=z['members'][0]; w=z['weights'][0]; current=z['states'][0,2]
            starts, members_cal, weights=loeo_neighbours(awm,lib)
            fut=succ[:,starts].T
            states=rs[np.maximum(fut,0)].copy();states[fut<0]=np.nan
            d,_,support=residuals(states,members_cal,weights,rs,scale,succ)
            result['library_stage_continuation']={}
            for stage in sorted(set(stages)):
                result['library_stage_continuation'][stage]={}
                for h in [2,4,6,8]:
                    mask=(stages[starts]==stage)&support[:,h]
                    maxima=np.nanmax(d[mask,1:h+1],axis=1)
                    result['library_stage_continuation'][stage][str(5*h)]=dict(n=int(mask.sum()),
                       p95_max_delta=float(np.quantile(maxima,.95)),alert=float((maxima>q).mean()))
            # Cost excludes memory I/O, Python serving, stage inference, telemetry,
            # SE(3) conversion and fitting. Report only this tiny arithmetic kernel.
            m0=w@rs[members]
            times=[]
            for k in range(1100):
                t=time.perf_counter_ns()
                advanced=succ[2,members]
                if (advanced<0).any():
                    advanced=members
                pred=w@rs[advanced]
                err=np.sqrt(np.mean(((current-rs[members[0]]-pred+m0)/scale)**2))
                action=np.einsum('k,khd->hd',w,awm.act[advanced])
                if k>=100:times.append((time.perf_counter_ns()-t)/1e6)
            result['numpy_tube_plus_synth_ms_p50_p95']=np.quantile(times,[.5,.95]).tolist()
            results.append(result)
            print(key,result['bval'],result['numpy_tube_plus_synth_ms_p50_p95'],flush=True)
    (OUT/'stage_cost_evidence.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
