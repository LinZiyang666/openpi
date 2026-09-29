"""Additional integration checks: fitted recording specs and non-test attestation.

v2b (R6-C-v2 + SELECTION §7b): replays from replays_v2b; recording prefits
from final_recording_prefits (final CAL paths); U/Bmech prefits regenerated in
prefits_v2b with the v2 kwargs; production specs re-emitted/re-rendered in
memory and required to equal the files on disk.
"""
import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd
from .common import HERE,OUT,sources,sha,write_json
from .fit_calibration import verify_resets
from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.rounds.r06.p1_groot_commit.judge import GrootCommitJudge,gripper_closed
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='v2b')
    ap.add_argument('--recording-prefits',type=Path,default=OUT/'final_recording_prefits')
    ap.add_argument('--baseline-prefits',type=Path,default=OUT/'prefits_v2b')
    args=ap.parse_args()
    facts={}
    def decs(cell,case):
        p=next((OUT/f'replays_{args.tag}'/f'{cell}_{case}').glob('decisions_*.jsonl'))
        return [r for s in p.read_text().splitlines() if (r:=json.loads(s))['ev']=='dec']
    dropped=set()
    for cell in ['pi05_l10_50','groot_l10_50']:
        for a,f in zip(decs(cell,'A'),decs(cell,'floor')):
            for field in ['lib','top1','topk','scores','conf','hit','src','vision','rows','weights','served_head']:
                assert a.get(field)==f.get(field),(cell,field)
            for k,v in a.get('extras',{}).items():
                if k in f['extras']:assert f['extras'][k]==v
                else:dropped.add(k)
    assert GrootCommitJudge.policy_tail_step is CommitJudge.policy_tail_step
    assert gripper_closed(-1,'groot') and not gripper_closed(1,'groot')
    assert gripper_closed(1,'pi05') and not gripper_closed(-1,'pi05')
    facts['retrieval_identity']='A/floor top-k, scores, confidence, mixture rows/weights and retained diagnostics match exactly'
    facts['log_capacity']=dict(note='Existing plugin retains at most 40 scalar extras; C telemetry can evict baseline blind diagnostics without changing actions/retrieval.',dropped_baseline_keys=sorted(dropped))
    facts['GR00T_judge']='C delegates the exact inherited lifecycle tail function; normalized negative-close convention retained'
    # Verify the reset-attestation gate without any simulator or outcomes.
    folder=OUT/'reset_gate_test';folder.mkdir(exist_ok=True)
    p=folder/'controls.jsonl';row=dict(arm='synthetic',uid='synthetic:0:0',attempt=1,task_id=0,init=0)
    ep=pd.DataFrame([row]);manifest=dict(selected=[dict(task=0,init=0,state_sha256='audited')])
    p.write_text(json.dumps(dict(ev='reset',task_uid=row['uid'],attempt=1,init_state_sha256='audited'))+'\n')
    assert len(verify_resets(ep,manifest,folder))==1
    manifest['selected'][0]['state_sha256']='wrong-test-pool'
    try:verify_resets(ep,manifest,folder)
    except ValueError as e:assert 'reset' in str(e)
    else:raise AssertionError('wrong init pool accepted')
    facts['reset_provenance']='matching actual init hash accepted; wrong actual pool rejected'
    from exp.ablation_study.cache_size.run_size_eval import load_apool_digest
    from exp.ablation_study.cache_size.verify_apool import load_init_states
    import hashlib
    import yaml
    for suite,full in [('l10','libero_10'),('spatial','libero_spatial')]:
        record=yaml.safe_load((HERE/f'bval_pool_{suite}.yaml').read_text())
        record['apool_dir']=str(HERE.parents[5]/f'exp/common/data/db_init/libero/{full}')
        path=OUT/f'bval_pool_{suite}_local_test.yaml';path.write_text(yaml.safe_dump(record))
        verified=load_apool_digest(str(path));assert verified['total_inits']==500
        audit=json.loads((HERE/'bval_membership_audit.json').read_text())[suite]
        for r in audit['tasks'].values():
            state=load_init_states(Path(record['apool_dir'])/(r['init_stem']+'.init'))[r['init']]
            assert hashlib.sha256(np.asarray(state).tobytes()).hexdigest()==r['state_sha256']
    facts['driver_pool_binding']='existing driver loader rehashed all 20 local B files and accepted both 500-state records'
    notes=[]
    all_specs=json.loads((HERE/'recording_prefit_specs.json').read_text())+json.loads((HERE/'validation_prefit_specs.json').read_text())
    for spec in all_specs:
        path=(args.recording_prefits if spec['name'].startswith('r6c_cal_') else args.baseline_prefits)/(spec['name']+'.pkl')
        if spec['method'].endswith(':CalibratedRescue'):
            assert not path.exists();continue
        cls,_=load_method_class(spec['method']);cls(**spec['kwargs'])
        with path.open('rb') as f:blob=pickle.load(f)
        assert blob['spec']==spec['method'] and blob['kwargs']==spec['kwargs']
        method=blob['method']
        if spec['name'].startswith('r6c_cal_'):
            cell=spec['name'][len('r6c_cal_'):]
            assert method.catalog['fit_info']['q1_r_bank_sha256']==sha(OUT/cell/'r_bank.json')
            assert method.enabled and method.p==0 and method.blind_shadow and method.resample_p==0
        notes.append(dict(name=spec['name'],artifact=str(path),sha256=sha(path),bytes=path.stat().st_size))
    assert len(notes)==20
    facts['prefits']=dict(note='8 A calibration-recording (final CAL paths), 8 original uniform controls, 4 Q3 Bmech; kwargs equal the current specs; no C calibration from test inits',
        recording_prefits=str(args.recording_prefits),baseline_prefits=str(args.baseline_prefits))
    specs=json.loads((HERE/'emit_arms_c32.json').read_text())
    assert len(specs)==32
    for row in specs:
        cls,_=load_method_class(row['method']);cls(**row['kwargs'])
        assert '--os-no-shadow-native' in row['plugin_args']
        assert row['plugin_args'][row['plugin_args'].index('--os-policy-tail-blocks')+1]=='1'
    # Production specs are current: the emitter/renderer reproduce the files on disk exactly,
    # and every C row carries the R6-C-v2 kwargs (7b changes no kwarg).
    from . import emit_specs, render_specs
    from .common import CONTROLLER_VERSION
    captured={}
    with patch.object(emit_specs,'write_json',lambda path,value:captured.__setitem__(Path(path).name,json.loads(json.dumps(value)))):
        emit_specs.main()
    for name,value in captured.items():
        assert json.loads((HERE/name).read_text())==value,name
    loc=json.loads((HERE/'rendered_locations.json').read_text())
    rendered={}
    with patch.object(render_specs,'write_json',lambda path,value:rendered.__setitem__(Path(path).name,json.loads(json.dumps(value)))):
        render_specs.render(cal_root=loc['cal_root'],stall_root=loc['stall_root'],recording_run_root=loc['recording_run_root'],
            validation_run_root=loc['validation_run_root'],smoke_run_root=loc['smoke_run_root'])
    for name,value in rendered.items():
        if name!='rendered_locations.json':assert json.loads((HERE/name).read_text())==value,name
    c_rows=0
    for fname in ['emit_arms_c32.json','emit_arms_smoke2.json','validation_prefit_specs.json','smoke_prefit_specs.json']:
        for row in json.loads((HERE/fname).read_text()):
            k=row['kwargs'];cell=f"{row['model']}_{row['suite']}_{row['name'].split('_')[3]}"
            if row['method'].endswith(':CalibratedRescue'):
                assert k['cooldown_scope']=='stall' and k['randomization_key']==CONTROLLER_VERSION+'/'+cell and k['random_seed']==26092903,(fname,row['name'])
                assert set(k)=={'rho','placement','cooldown_scope','stall_model_path','calibration_path','random_seed','randomization_key'},(fname,row['name'])
                c_rows+=1
            elif row['method'].endswith(':RiskLottery'):
                assert k['randomization_key']==CONTROLLER_VERSION+'/'+cell,(fname,row['name'])
    for row in json.loads((HERE/'recording_prefit_specs.json').read_text()):
        assert row['kwargs']['provenance']['q1_controller_version']==CONTROLLER_VERSION
    facts['production_specs']=dict(note='emit_specs and render_specs (rendered_locations.json roots) reproduce emit_arms_c32/emit_arms_smoke2/eval500/smoke4/spec_audit and recording/validation/smoke prefit specs byte-for-value; all C rows carry R6-C-v2 kwargs',
        checked_C_rows=c_rows,files=sorted(captured)+sorted(n for n in rendered if n!='rendered_locations.json'))
    write_json(HERE/f'prefit_manifest_{args.tag}.json',notes)
    write_json(HERE/f'packaging_tests_{args.tag}.json',dict(status='PASS',tag=args.tag,checks=facts,checked_validation_specs=32))
    print(json.dumps(dict(status='PASS',tag=args.tag,prefits=20,validation_specs=32,C_spec_rows=c_rows)))

if __name__=='__main__':main()
