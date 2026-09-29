"""Fit R -> same-state shadow disagreement and model C's owner cost.

The cost model is budget.py's exact cadence DAG under R6-C-v2 + SELECTION §7b.

Success/outcome columns are never loaded. Official pilot data require --dry-run
and produce artifacts rejected by CalibratedRescue.fit and the validation specs.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from .common import SCHEMA, CONTROLLER_VERSION, OUT, read_bank, sha, write_json, output_path
from .stall_bridge import load_stall, InactiveStallTracker, verify_stall, AMBIGUOUS_RULE
from .budget import replay_cadence, solve, expected_cost

EKEY = ['arm','uid','attempt']
AKEY = EKEY + ['step']


def vector(s):
    return np.asarray(json.loads(s), float)


def verify_resets(ep, manifest, client_root):
    if client_root is None:
        raise ValueError('production calibration requires --client-root to attest actual B-val reset states')
    wanted={(r.uid,int(r.attempt)):(int(r.task_id),int(r.init)) for r in ep.itertuples()}
    hashes={(r['task'],r['init']):r['state_sha256'] for r in manifest['selected']}
    found={}
    for path in Path(client_root).rglob('controls.jsonl'):
        with path.open() as f:
            first=json.loads(next(f));key=(first['task_uid'],int(first.get('attempt',1)))
            if key not in wanted:continue
            if key in found:raise ValueError('duplicate accepted reset provenance')
            resets=[first] if first.get('ev')=='reset' else []
            for line in f:
                r=json.loads(line)
                if r.get('ev')=='reset':resets.append(r)
        if len(resets)!=1 or resets[0]['init_state_sha256']!=hashes[wanted[key]]:
            raise ValueError('actual simulator reset is not the frozen non-test B-val state')
        found[key]=dict(path=str(path),init_state_sha256=resets[0]['init_state_sha256'])
    if found.keys()!=wanted.keys():raise ValueError('missing accepted B-val reset provenance')
    return list(found.values())


def fit(tables, bank_path, out, c1, targets, dry_run=False, bval_manifest=None, stall_model_path=None,client_root=None,cooldown_scope='stall'):
    if cooldown_scope not in ('stall','all') or (cooldown_scope=='all' and not dry_run):
        raise ValueError('deployment requires stall cooldown; all scope is dry-run/test only')
    tables, bank_path, out = Path(tables), Path(bank_path), output_path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'calibration.json').exists():
        raise FileExistsError('refuses overwriting a calibration artifact')
    bank, arrays = read_bank(bank_path)
    b = bank['interface']['block_controls']; h=bank['interface']['commit_controls']
    dims = bank['interface']['valid_action_indices']
    ca = set(AKEY+['task_id','init','catalog_sha256','assignment.cohort','retrieval.rows',
                   'retrieval.weights','retrieval.metric_code','retrieval.metric','provenance.run_block'])
    anchors = pd.read_csv(tables/'anchors.csv', usecols=lambda k:k in ca)
    anchors = anchors[anchors['assignment.cohort']=='A']
    if dry_run:
        anchors = anchors[(anchors['provenance.run_block']==0)&(anchors['init']==0)]
        role = 'DRYRUN_TEST_INITS'
    else:
        if bval_manifest is None:
            raise ValueError('non-test calibration requires the frozen B-val manifest')
        manifest = json.loads(Path(bval_manifest).read_text())
        if manifest['role'] != 'NONTEST_BVAL' or manifest['cell'] != bank['cell']:
            raise ValueError('B-val manifest identity mismatch')
        selected = {(r['task'],r['init']) for r in manifest['selected']}
        if set(map(tuple,anchors[['task_id','init']].drop_duplicates().to_numpy())) != selected:
            raise ValueError('recordings differ from the frozen outcome-blind B-val selection')
        role = 'NONTEST_BVAL'
    ep = anchors[EKEY+['task_id','init']].drop_duplicates()
    if not 1<=len(ep)<=10 or ep.task_id.nunique()!=len(ep) or ep.duplicated(['task_id','init']).any():
        raise ValueError('calibration requires at most ten episodes, one initialization per recorded task')
    reset_attestation=[] if dry_run else verify_resets(ep,manifest,client_root)
    ids = set(ep.uid)
    dc = set(AKEY+['task_id','init','vision','absolute_input_archive','catalog_sha256','actual_controls',
                  'blind_features.retrieval.rows','blind_features.retrieval.weights',
                  'blind_features.retrieval.metric_code','blind_features.retrieval.metric'])
    decisions = pd.read_csv(tables/'decisions.csv', usecols=lambda k:k in dc)
    decisions = decisions[decisions.uid.isin(ids)&decisions.arm.isin(ep.arm)]
    if decisions.duplicated(AKEY).any():
        raise ValueError('duplicate accepted decision')
    # Catalogs are read only from servers referenced by these accepted episodes.
    from exp.offline_search.rounds.r06.ideation_Q1.pilot_q1 import catalogs
    cats, catalog_paths = catalogs(decisions)
    for cat in cats.values():
        if cat['library_manifest']['sha256'] != bank['library_hashes']['manifest.json']:
            raise ValueError('recorded library does not match R bank')
        if not dry_run:
            supplied = cat['supplied']
            for key,want in [('q1_recording_role','NONTEST_BVAL'),('q1_pool_sha256',manifest['pool_sha256']),
                             ('q1_selection_sha256',sha(bval_manifest))]:
                if supplied.get(key)!=want:
                    raise ValueError('recording provenance mismatch: '+key)
            if cat['base_kwargs'].get('exclusions_sha256') != manifest['exclusions_sha256']:
                raise ValueError('recording did not bind the required source-episode exclusions')
            if cat['fit_info'].get('q1_r_bank_sha256')!=sha(bank_path):
                raise ValueError('recording did not bind the frozen bank and retrieval fit')
    action = pd.read_csv(tables/'action_steps.csv',usecols=lambda k:k in set(AKEY+['chunk_step']) or k.startswith(('cache.','policy.')))
    action = action[action.uid.isin(ids)&action.arm.isin(ep.arm)]
    chunks = action.groupby(AKEY,sort=False)
    labels=[]
    for row in anchors.itertuples(index=False,name=None):
        r=dict(zip(anchors.columns,row));key=tuple(r[k] for k in AKEY)
        chunk=chunks.get_group(key).sort_values('chunk_step').iloc[:h]
        if len(chunk)!=h:raise ValueError('short primary shadow chunk')
        delta=chunk[[f'cache.{d}' for d in dims]].to_numpy()-chunk[[f'policy.{d}' for d in dims]].to_numpy()
        E=float(np.sqrt(np.mean((delta/arrays['sigma'])**2)))
        ix=vector(r['retrieval.rows']).astype(int); w=vector(r['retrieval.weights']);w/=w.sum()
        R=float(w@arrays['r'][ix])
        labels.append({**{k:r[k] for k in AKEY+['task_id','init']},'R':R,'E':E})
    labels=pd.DataFrame(labels)
    # One episode/task here; episode-normalized anchor weights generalize without
    # allowing long trajectories to dominate the two-parameter fit.
    weights=1/labels.groupby(EKEY).R.transform('size').to_numpy()/len(ep)
    beta=nnls(np.column_stack([np.ones(len(labels)),labels.R])*np.sqrt(weights[:,None]),labels.E*np.sqrt(weights))[0]
    labels['Ehat']=beta[0]+beta[1]*labels.R
    labels.to_csv(out/'calibration_anchor_labels.csv',index=False)
    am={tuple(r[k] for k in AKEY):r for r in anchors.to_dict('records')}
    episodes=[]
    for key,g in decisions.groupby(EKEY,sort=True):
        g=g.sort_values('step')
        if list(g.step)!=list(range(len(g))):raise ValueError('decision grid is not contiguous')
        if (g.actual_controls.iloc[:-1]!=b).any() or not g.actual_controls.between(1,b).all():
            raise ValueError('recorded control protocol differs from the five-control plugin contract')
        rows=[];control=0
        for r in g.to_dict('records'):
            rec=am.get(tuple(r[k] for k in AKEY))
            prefix='retrieval.'
            if rec is None:
                rec=r;prefix='blind_features.retrieval.'
            if not isinstance(rec.get(prefix+'metric_code'),str):
                raise ValueError('stall/extra-LOOK cost replay needs blind_shadow=true codes at every decision')
            ix=vector(rec[prefix+'rows']).astype(int);w=vector(rec[prefix+'weights']);w/=w.sum()
            R=float(w@arrays['r'][ix])
            rows.append(dict(step=r['step'],control_index=control,R=R,Ehat=float(beta[0]+beta[1]*R),
                key=dict(metric_code=vector(rec[prefix+'metric_code']),metric=rec[prefix+'metric'])))
            control+=int(r['actual_controls'])
        episodes.append(dict(uid=key[1],task=int(g.task_id.iloc[0]),init=int(g.init.iloc[0]),weight=1/len(ep),
            block_controls=b,commit_controls=h,decisions=rows))
    model,tracker,stall_fingerprint=load_stall(stall_model_path)
    if model is not None:
        from .common import load_base
        verify_stall(model,load_base(bank['source'])[0],bank)
    modes={'no_stall':[replay_cadence(e,cooldown_scope=cooldown_scope) for e in episodes]}
    fingerprints={'no_stall':'none','stall':None}
    if stall_model_path:
        modes['stall']=[replay_cadence(e,model,tracker,cooldown_scope) for e in episodes]
        fingerprints['stall']=stall_fingerprint
    solutions={}
    for mode,recordings in modes.items():
        solutions[mode]={}
        floor=expected_cost(recordings,0.,'uniform',c1,cooldown_scope)['IR']
        for placement in ['uniform','R']:
            sols={format(rho,'.12g'):solve(recordings,rho,placement,c1,cooldown_scope) for rho in [*targets,floor]}
            # SELECTION 7b: an infeasible target runs at the calibrated ceiling under an explicit C.max
            # label; store that pseudo-target (key = its 12-digit value) so no refit is needed for it.
            ceiling=next(iter(sols.values()))['ceiling']
            key=format(ceiling,'.12g')
            if key not in sols:sols[key]=solve(recordings,float(key),placement,c1,cooldown_scope)
            solutions[mode][placement]=sols
    # Relocatable artifact: coordinator copies the cell directory as a unit.
    import shutil
    if bank_path.parent.resolve()!=out.resolve():
        shutil.copy2(bank_path,out/'r_bank.json');shutil.copy2(bank_path.parent/bank['arrays'],out/bank['arrays'])
    from exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.budget import solve as q2_solve
    cal=dict(schema=SCHEMA+'.calibration',status=role,cell=bank['cell'],intercept=float(beta[0]),slope=float(beta[1]),
        controller_version=CONTROLLER_VERSION,cooldown_scope=cooldown_scope,ambiguous_rule=AMBIGUOUS_RULE,
        calibration_episodes=ep.to_dict('records'),calibration_anchors=len(labels),
        baseline_E=float(np.average(labels.E,weights=weights)),r_bank_path='r_bank.json',
        r_bank_sha256=sha(out/'r_bank.json'),base_source=bank['source'],c1=c1,
        stall_fingerprint=fingerprints,solutions=solutions,
        library_uniform_budget={format(rho,'.12g'):q2_solve(bank['library_lengths'],rho,c1,'uniform') for rho in targets},
        tables={str(tables/name):sha(tables/name) for name in ['anchors.csv','decisions.csv','action_steps.csv','audit.json']},
        catalog_paths=catalog_paths,reset_attestation=reset_attestation,coefficient_weighting='equal task/init/episode, equal anchors inside episode; no success labels',
        cost_model='equal-episode mean owner IR on fixed recordings; exact expectation over the cadence DAG (SELECTION 7b: slow_ambiguous draws the lottery, extra LOOK only without a call), stall-call cooldown, mandatory stall calls and extra LOOKs; nominal b-control requests',
        cadence_nodes={mode:int(sum(len(e['nodes']) for e in recs)) for mode,recs in modes.items()},
        limitations=['Recorded observations and retrieval proposals do not react to hypothetical calls.',
                    'A source episode is excluded during B-val recording; the deployed full bank is unchanged.',
                    'No success label or simulator state enters the fit; measured live IR must validate the model.'])
    if bval_manifest:cal['bval_manifest_sha256']=sha(bval_manifest)
    write_json(out/'calibration.json',cal)
    # JSON-safe replay product for independent rate tests; contains codes only.
    write_json(out/'cost_replay.json',dict(status=role,controller_version=CONTROLLER_VERSION,cooldown_scope=cooldown_scope,
        ambiguous_rule=AMBIGUOUS_RULE,modes=modes))
    print(json.dumps(dict(cell=bank['cell'],status=role,episodes=len(ep),anchors=len(labels),a=beta[0],b=beta[1],
        solutions=solutions)),flush=True)
    return cal


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--tables',type=Path,required=True);ap.add_argument('--bank',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--c1',type=float,required=True)
    ap.add_argument('--rho',type=float,nargs='+',default=[.18,.30,.45])
    ap.add_argument('--dry-run',action='store_true');ap.add_argument('--bval-manifest',type=Path)
    ap.add_argument('--stall-model-path',type=Path)
    ap.add_argument('--client-root',type=Path)
    ap.add_argument('--cooldown-scope',choices=['stall','all'],default='stall')
    a=ap.parse_args();fit(a.tables,a.bank,a.out,a.c1,a.rho,a.dry_run,a.bval_manifest,a.stall_model_path,a.client_root,a.cooldown_scope)


if __name__=='__main__':main()
