"""Read-only R7 PROFILE audit; no serving changes or new evaluation episodes."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from exp.offline_search.rounds.r07.c4_profile.common import accepted_arm

HERE=Path(__file__).parent
ROOT=Path('/home/weiland/trace_runs/os_closed_loop/r07_profile_bval1')
REPORT=Path('exp/offline_search/rounds/r07/profile_results/closed_loop/profile_report.json')
STORE=Path('/home/weiland/trace_runs/offline_search_store/library')


def variant(name):
    if name.startswith('r7_sw_'):return 'SW'
    return name.rsplit('_',2)[1]


def main():
    original=json.loads(REPORT.read_text())
    refs={r['cell']:r for r in original['reports'] if variant(r['arm'])=='A'}
    results=[];banks={}
    for r in original['reports']:
        arm,cell,kind=r['arm'],r['cell'],variant(r['arm'])
        model,suite,size=cell.split('_')
        if cell not in banks:
            bankname='current' if size=='50' else ('bpool_cs' if model=='pi05' else 'bpool_all')
            path=STORE/f'{model}_{suite}'/bankname
            banks[cell]={k:np.load(path/f'{k}.npy',mmap_mode='r') for k in ['rs','action','next','success','episode','step']}
        lib=banks[cell]
        states=np.asarray(lib['rs'][:,:8],float)
        scale=np.std(states[lib['success']],axis=0);active=scale>np.finfo(float).eps*max(scale.max(),1);scale[~active]=1
        episodes,audit=accepted_arm(ROOT/'runs'/arm)
        counts=Counter();bad=Counter();fire_ages=Counter();sources=Counter();max_action=0.;max_delta=0.
        error_samples=[]
        for e in episodes:
            anchor=None;anchor_action=None
            for d in e['decisions']:
                counts['N']+=1;counts['V']+=int(d['vision']);counts['M']+=int(not d['hit'])
                if d['vision']:
                    counts['grants']+=int(d.get('extras',{}).get('os_sf_granted',0)>0)
                ex={**d.get('extras',{}),**d.get('blind_extras',{})}
                if kind in ['SF1','SF2','UF1'] and anchor is not None:
                    age=d['step']-anchor['step']
                    rows=np.asarray(anchor['rows'],int)
                    for h in range(age):
                        rows=np.where(rows>=0,lib['next'][np.maximum(rows,0)],-1)
                    w=np.asarray(anchor['weights'],np.float32)
                    if ex.get('os_sf_valve_checked'):
                        counts['valve_checks']+=1
                        if (rows<0).any():bad['valve_support']+=1
                        else:
                            expected=np.einsum('k,kd->d',w.astype(float),states[rows]-states[np.asarray(anchor['rows'],int)])
                            dd=((np.asarray(d['robot_state'][:8])-np.asarray(anchor['robot_state'][:8])-expected)/scale)[active]
                            stat=float(np.sqrt(np.mean(dd**2)))
                            max_delta=max(max_delta,abs(stat-ex['os_sf_delta']))
                            bad['valve_verdict']+=int((stat>ex['os_sf_radius'])!=bool(ex.get('os_sf_valve_fire')))
                    if ex.get('os_sf_valve_fire') and d['vision']:
                        counts['valve_looks']+=1;fire_ages[str(age)]+=1
                    if not d['vision']:
                        counts['blind_prefixes_checked']+=1
                        if age==1:
                            expected_action=anchor_action[5:10,:7]
                        elif model=='groot' and age==2:
                            expected_action=anchor_action[10:15,:7]
                        else:
                            if (rows<0).any():bad['served_missing_successor']+=1;continue
                            expected_action=np.tensordot(w,np.asarray(lib['action'][rows,:5,:7]),1)
                        diff=float(np.max(np.abs(expected_action-np.asarray(d['served_head']))))
                        max_action=max(max_action,diff)
                        if diff>1e-6:
                            bad['served_prefix_over_1e_6']+=1
                            if len(error_samples)<3:error_samples.append([d['uid'],d['step'],diff])
                        bad['changed_weights']+=int(not np.array_equal(np.asarray(d['weights']),np.asarray(anchor['weights'])))
                        if age>=2:
                            counts['extra_blocks_served']+=1;sources[str(ex.get('os_sf_source'))]+=1
                cam=d.get('camera_mode')
                if kind=='SW':
                    counts['wrist_looks']+=int(d['vision'] and cam=='wrist_only')
                    counts['completion_calls']+=d.get('camera_completion_calls',0)
                    counts['camera_stage1_calls']+=d.get('camera_stage1_calls',0)
                    bad['camera_on_blind']+=int(not d['vision'] and cam!='blind')
                    bad['first_camera']+=int(d['step']==0 and cam!='full')
                    price=(.152 if cam=='full' else .055198 if cam=='wrist_only' else 0)+.049890*d.get('camera_completion_calls',0)+.848*(not d['hit'])
                    bad['owner_cost']+=int(abs(price-d['owner_cost'])>1e-8)
                if d['vision']:
                    anchor=d
                    anchor_action=np.tensordot(np.asarray(d['weights'],np.float32),lib['action'][np.asarray(d['rows'],int)],1)
        for key in ['N','V','M']:assert counts[key]==r[key],(arm,key,counts[key],r[key])
        assert len(episodes)==20
        ref=refs[cell]
        by_pair={(e['task'],e['init']):e for e in ref['episodes']}
        gains=losses=0;ir_deltas=[]
        for e in r['episodes']:
            other=by_pair[e['task'],e['init']]
            gains+=int(e['Y']>other['Y']);losses+=int(e['Y']<other['Y'])
            ir_deltas.append(e['IR']-other['IR'])
        summary=json.loads((ROOT/'runs'/arm/'summary.json').read_text())
        out=dict(arm=arm,cell=cell,variant=kind,successes=sum(e['Y'] for e in episodes),
            IR=r['IR'],equal_episode_IR=float(np.mean([e['IR'] for e in r['episodes']])),
            saved_vs_A=ref['IR']-r['IR'],paired_episode_saved_vs_A=-float(np.mean(ir_deltas)),
            gains_vs_A=gains,losses_vs_A=losses,counts=dict(counts),
            eligibility=counts['grants']/counts['V'],wrist_share=counts['wrist_looks']/counts['V'],
            fire_ages=dict(fire_ages),extension_sources=dict(sources),bad=dict(bad),
            max_prefix_difference=max_action,max_valve_stat_difference=max_delta,
            error_samples=error_samples,summary_eager_IR=summary['cost_ledger']['ir_per_request'],
            audit=audit)
        results.append(out)
        print(kind,cell,'IR',round(out['IR'],6),'saved',round(out['saved_vs_A'],6),
              'paired',round(out['paired_episode_saved_vs_A'],6),'Y',out['successes'],
              'elig',round(out['eligibility'],3),'valve',counts['valve_looks'],dict(fire_ages),
              'extra',counts['extra_blocks_served'],'bad', {k:v for k,v in bad.items() if v},flush=True)
    (HERE/'profile_audit.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
