"""Freeze the implementation payload, exact paired grid and prospective forecasts."""
from common import *
fence()
import numpy as np
from replay_study import DOSES, library_pack

WIDTH=1.0
EPISODES=500
PRIMARY_DOSES=(.32,.40)
REPLICATES={.25:2,.32:4,.40:4,.45:2}
HOSTS={'pi05':'h100','groot':'local_4090'}

def main():
    fits=[r for r in read(HERE/'robust_fits.json') if r['width']==WIDTH]
    periodic=read(HERE/'periodic_fits.json')
    power=read(HERE/'power.json')
    selected=next(r for r in read(HERE/'robust_summary.json') if r['width']==WIDTH)
    stamp=now()
    old=read(HERE/'freeze.json') if (HERE/'freeze.json').exists() else None
    if old: stamp=old['timestamp']
    prediction=[];grid=[];calpairs=[]
    sources={}
    for cell in CELLS:
        model,suite,size=cell.split('_');p=library_pack(cell)
        cv,cm=COSTS[model]
        lib_off=(cv*p['alive'].sum()+cm*p['guard'].sum())/p['N'].sum()
        refs=[read(mp) for mp in sorted((HERE/'data').glob('*.json'))]
        src=next((r for r in refs if r['cell']==cell and r['root'].startswith('r12_') and r['method']=='off'),None)
        offir=src['ir'] if src else float(lib_off)
        offsr=src['sr'] if src else None
        for target in DOSES:
            df=next(r for r in fits if r['cell']==cell and r['target']==target)
            pf=next(r for r in periodic if r['cell']==cell and r['target']==target)
            # This is explicitly a subjective prospective effect prior, not an
            # estimate of benefit from an exogenous replay or historical A data.
            raw_prior_pp=(1.0 if model=='pi05' and suite=='l10' else -1.0 if suite=='l10' else -.5)
            prior_pp=raw_prior_pp*(1 if target<=.32 else .5 if target==.40 else 0)
            for rep in range(REPLICATES[target]):
                for method in ('periodic','distance_feedback'):
                    arm=f'r13_{cell}_{method}_ir{round(target*100):02d}_rep{rep}'
                    row=dict(arm=arm,model=model,suite=suite,size=int(size),cell=cell,method=method,
                        target=target,replicate=rep,episodes=EPISODES,host=HOSTS[model],
                        batch=f'{model}_{suite}_rep{rep}',pair_group=f'{cell}_ir{round(target*100):02d}_rep{rep}',
                        init_pool='A',analysis_role='primary' if target in PRIMARY_DOSES else 'curve_extension',
                        knob_coin_seed=0,rollout_replicate_seed=2026100400+rep,
                        prerequisite='B_pilot_pass_or_explicit_Q1_failure_only',
                        config_key=f'{cell}:{method}:{target}')
                    grid.append(row)
                    prediction.append(dict(arm=arm,cell=cell,method=method,target=target,
                        predicted_ir=target if method=='distance_feedback' else pf['pred_ir_library'],
                        ir_planning_band_lo=max(0,target-.02),ir_planning_band_hi=target+.02,
                        ir_band_type='pilot acceptance aim; not statistical coverage',
                        sr_delta_distance_minus_periodic_pp=prior_pp,sr_prior_halfwidth_pp=3.,
                        sr_interpretation='subjective same-realized-IR prior; zero is plausible in every cell',
                        calibration='B behavior paths' if df['sources'][0][1]!='library' else 'held-out B library; unvalidated transfer'))
        for rep in range(4):
            arm=f'r13_{cell}_off_rep{rep}'
            grid.append(dict(arm=arm,model=model,suite=suite,size=int(size),cell=cell,method='off',target=None,
                replicate=rep,episodes=EPISODES,host=HOSTS[model],batch=f'{model}_{suite}_rep{rep}',
                pair_group=f'{cell}_off_rep{rep}',init_pool='A',analysis_role='same_batch_baseline',
                knob_coin_seed=0,rollout_replicate_seed=2026100400+rep,prerequisite='frozen_lower_layers',config_key=f'{cell}:off'))
            prediction.append(dict(arm=arm,cell=cell,method='off',predicted_ir=offir,
                prediction_source=src['root'] if src else 'library heldout guard replay',predicted_sr=offsr))
        sources[cell]=dict(reference_off_source=src['root'] if src else 'library',reference_off_ir=offir,
                           reference_off_sr=offsr,library_allcall_ir=float(p['alive'].sum()/p['N'].sum()))
    # Both pure controls are model/suite/replicate controls shared by sizes.
    # C10 is the all-call endpoint of layer 4. C5 is a separate contextual reference.
    for model in COSTS:
        for suite in ('l10','spatial'):
            for rep in range(4):
                for method in ('pure_c10','pure_c5'):
                    arm=f'r13_{model}_{suite}_{method}_rep{rep}'
                    grid.append(dict(arm=arm,model=model,suite=suite,size=None,cell=f'{model}_{suite}_all_sizes',
                        method=method,target=None,replicate=rep,episodes=EPISODES,host=HOSTS[model],
                        batch=f'{model}_{suite}_rep{rep}',pair_group=f'{model}_{suite}_{method}_rep{rep}',
                        init_pool='A',analysis_role='endpoint' if method=='pure_c10' else 'contextual_reference',
                        knob_coin_seed=0,rollout_replicate_seed=2026100400+rep,
                        prerequisite='pure_c10_action_lifecycle_equivalence' if method=='pure_c10' else 'standard_pure_policy',
                        config_key=f'{model}_{suite}:{method}'))
                    prediction.append(dict(arm=arm,method=method,cell=f'{model}_{suite}_all_sizes',
                        predicted_ir=1. if method=='pure_c5' else sources[f'{model}_{suite}_50']['library_allcall_ir'],
                        prediction_source='every decision calls' if method=='pure_c5' else 'library terminal-truncation estimate'))
    # Adjacent method pairs, seeded within-block ordering; no outcome-based arm order.
    rng=np.random.default_rng(20261004)
    groups={}
    for r in grid: groups.setdefault((r['batch'],r['pair_group']),[]).append(r)
    ordered=[]
    for batch in sorted({r['batch'] for r in grid},key=lambda s:(int(s.split('rep')[-1]),s)):
        keys=[k for k in groups if k[0]==batch]
        for k in rng.permutation(len(keys)):
            group=groups[keys[k]]
            for j in rng.permutation(len(group)): ordered.append(group[j])
    for i,r in enumerate(ordered):r['global_order']=i
    grid=ordered
    configs=[]
    for r in fits:
        configs.append(dict(config_key=f'{r["cell"]}:distance_feedback:{r["target"]}',method='distance_feedback',**r))
    for r in periodic:
        configs.append(dict(config_key=f'{r["cell"]}:periodic:{r["target"]}',method='periodic',**r))
    pilot=[]
    for cell in [c for c in CELLS if not c.endswith('_500')]:
        model=cell.split('_')[0]
        for target in DOSES:
            for method in ('periodic','distance_feedback'):
                pilot.append(dict(arm=f'r13_validate_{cell}_{method}_ir{round(target*100):02d}',cell=cell,
                    method=method,target=target,episodes=100,init_pool='B',replicate=0,host=HOSTS[model],
                    selection='new_K10_disjoint_from_used_B_identity_union_and_served_subset',
                    shadow=cell.endswith('_50') and target==.32,config_key=f'{cell}:{method}:{target}'))
                if cell.endswith('_50') and target==.32:
                    pilot.append(dict(arm=f'r13_validate_{cell}_{method}_ir32_rep1',cell=cell,method=method,
                        target=target,episodes=100,init_pool='B',replicate=1,host=HOSTS[model],
                        selection='same_new_K10_as_validation_rep0',shadow=False,config_key=f'{cell}:{method}:{target}'))
    static=[]
    for r in read(HERE/'static_b_fits.json'):
        if r['cell'].endswith('_50') and r['target'] in (.32,.4):
            static.append(dict(arm=f'r13_validate_{r["cell"]}_distance_static_b_ir{round(r["target"]*100):02d}',
                episodes=100,host=HOSTS[r['cell'].split('_')[0]],init_pool='B',**r))
    payload=dict(timestamp=stamp,status='frozen algorithm/configs/grid; A primary interpretation gated on B validation',
        score_width=WIDTH,distance_fits=fits,periodic_fits=periodic,grid=grid,pilot=pilot,static_pilot=static,
        predictions=prediction,selection_evidence=selected,power=power,source_context=sources,
        primary_doses=PRIMARY_DOSES,replicates=REPLICATES,episodes_per_arm=EPISODES,
        sources_sha256={name:sha(HERE/name) for name in ('controller.py','robust_fits.json','periodic_fits.json',
            'static_b_fits.json','robust_replay.csv','development_splits.csv','selfcheck.json')},
        no_new_variant_closed_loop_results_accessed=True,no_A_inputs=True)
    if old and clean(payload)!=old: raise RuntimeError('Frozen payload differs; create a separately named version before new results')
    dump(HERE/'freeze.json',payload);dump(HERE/'configs.json',configs)
    csvout(HERE/'test_grid.csv',grid);csvout(HERE/'predictions.csv',prediction)
    dump(HERE/'pilot_grid.json',pilot);dump(HERE/'static_pilot_grid.json',static)
    # Identity union includes development validation because it selected width.
    used={}
    for mp in sorted((HERE/'data').glob('*.json')):
        m=read(mp)
        if m['cell'] not in CELLS: continue
        with np.load(mp.with_suffix('.npz'),allow_pickle=False) as z:
            for t,i in z['keys']:used.setdefault((m['model'],m['suite']),set()).add((int(t),int(i)))
    dump(HERE/'used_b_identities.json',{f'{m}_{s}':sorted(v) for (m,s),v in used.items()})
    counts=dict(timestamp=stamp,arms=len(grid),episodes=len(grid)*EPISODES,
        active_arms=sum(r['method'] in ('periodic','distance_feedback') for r in grid),
        off_arms=sum(r['method']=='off' for r in grid),pure_arms=sum(r['method'].startswith('pure') for r in grid),
        pilot_arms=len(pilot),static_pilot_arms=len(static),pilot_episodes=100*(len(pilot)+len(static)),
        primary_pairs=sum(r['method']=='periodic' and r['analysis_role']=='primary' for r in grid)*EPISODES,
        projected_days=[len(grid)/70,len(grid)/40],
        mde4_by_replicate_correlation=[dict(rho=rho,mde_pp=power['mde80_conservative_pp']*np.sqrt((1+3*rho)/4)) for rho in (0,.25,.5,1)])
    dump(HERE/'grid_counts.json',counts);provenance('freeze');print(json.dumps(clean(counts),indent=2))

if __name__=='__main__':main()
