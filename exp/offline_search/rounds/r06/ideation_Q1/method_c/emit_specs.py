"""Emit frozen configuration identities; placeholders are never runnable fits."""
import copy
import json
from .common import HERE,OUT,STORE,CONTROLLER_VERSION,sources,sha,write_json

def main():
    q2=HERE.parents[1]/'ideation_Q2/frontier/adapters'
    q3=HERE.parents[1]/'ideation_Q3/stall'
    rows=[]
    for cell,source in sources().items():
        model,suite,size=cell.split('_')
        rho=.30 if size=='50' else .18
        for label,target in [('U',rho),('R',rho),('C',rho)]+([('C',.45)] if size=='50' else []):
            name=f'r6c_{cell}_{label}{int(round(target*100)):02d}'
            if label=='U':
                method='exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.methods:RiskLottery'
                kwargs={**source['kwargs'], 'rho':target,'allocation':'uniform',
                    'calibration_path':str(q2/'calibration.json'), 'calibration_sha256':sha(q2/'calibration.json')}
            else:
                method='exp.offline_search.rounds.r06.ideation_Q1.method_c.methods:CalibratedRescue'
                kwargs=dict(rho=target,placement='R',cooldown_scope='stall',stall_model_path=f'<STALL>/{cell.replace("spatial","sp")}' if label=='C' else None,
                    calibration_path=f'<CAL>/{cell}/calibrated/calibration.json')
            kwargs.update(random_seed=26092903,randomization_key=CONTROLLER_VERSION+'/'+cell)
            rows.append(dict(name=name,model=model,suite=suite,mode='plugin',method=method,kwargs=kwargs,
                full_model=True,cost_ledger=True,manifest='<RUN>/manifests/eval500.json',
                client_overrides=dict(replan_steps=5,resize_size=256 if model=='groot' else 224),
                plugin_args=['--os-root',str(STORE),'--os-blind','--os-policy-tail','--os-policy-tail-blocks','1',
                    '--os-judge','guard_only','--os-no-shadow-native','--os-fit-artifact',f'<RUN>/fits/{name}.pkl']))
    bmech=json.loads((q3/'emit_arms_bmech.json').read_text())
    assert len(bmech)==4
    for row in bmech:
        row['manifest']='<RUN>/manifests/eval500.json'
        if '--os-policy-tail-blocks' not in row['plugin_args']:
            row['plugin_args']+=['--os-policy-tail-blocks','1']
    rows+=bmech
    assert len(rows)==32 and len({r['name'] for r in rows})==32
    write_json(HERE/'emit_arms_c32.json',rows)
    smoke=[]
    for model in ['pi05','groot']:
        row=copy.deepcopy(next(r for r in rows if r['name']==f'r6c_{model}_l10_500_C18'))
        row['manifest']='<RUN>/manifests/smoke4.json';smoke.append(row)
    write_json(HERE/'emit_arms_smoke2.json',smoke)
    write_json(HERE/'eval500_manifest.json',dict(selected=[dict(task=t,init=i) for t in range(10) for i in range(50)]))
    write_json(HERE/'smoke4_manifest.json',dict(selected=[dict(task=t,init=i) for t in range(2) for i in range(2)]))
    write_json(HERE/'spec_audit.json',dict(controller_version=CONTROLLER_VERSION,cooldown_scope='stall',arms=len(rows),episodes=16000,smoke_arms=2,smoke_episodes=8,
        bmech_source=str(q3/'emit_arms_bmech.json'),bmech_sha256=sha(q3/'emit_arms_bmech.json'),
        deployment_gate='Production NONTEST_BVAL fit and feasible target required; no test-init dryrun artifact may be prefitted.',
        uniform_control='The specified U arms use original RiskLottery(uniform), including its episode mixture. Lottery calls have no cooldown in either adapter. C uniform is separately tested.'))
    print(json.dumps(dict(arms=len(rows),smoke_arms=2)))

if __name__=='__main__':main()
