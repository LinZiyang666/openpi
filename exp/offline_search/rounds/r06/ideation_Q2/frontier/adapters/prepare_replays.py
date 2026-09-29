"""Prepare CPU replay cases and zero-dose artifacts; never collect new episodes."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from .budget import digest
from .prefit import write_fit

HERE = Path(__file__).resolve().parent
MOD = 'exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters.methods:'


def main():
    baselines = json.loads((HERE/'replay_baselines.json').read_text())
    cases = []
    cal = HERE/'calibration.json'
    for b in baselines:
        cell = b['cell']
        model,suite,size = cell.split('_')
        label = b['name'].rsplit('_',1)[-1]
        baseline = dict(name=b['name'],model=model,suite=suite,method=b['method'],kwargs=b['kwargs'])
        source = dict(source_spec=b['method'],source_kwargs=b['kwargs'],source_arm=b['source_arm'],source_artifact=b['source_artifact'])
        cases.append(dict(name=b['name'],cell=cell,kind=label,spec=baseline,artifact=b['source_artifact'],
                          judge='guard_only' if label=='B' else None,policy_tail=label=='B'))
        variants = [('B0',0.),('Bextra',.5)] if label=='B' else [('A0',0.),('Risk',.2)]
        if label=='A' and suite=='spatial' and size=='50':
            variants.append(('Fixed',.25))
        if label=='A' and suite=='l10' and size=='50':
            variants.append(('Uniform',.2))
        for kind,value in variants:
            spec = copy.deepcopy(baseline)
            spec['name'] = f'replay_{cell}_{kind}'
            kw = spec['kwargs']
            if label=='B':
                spec['method'] = MOD+('ExtraDosePi05' if model=='pi05' else 'ExtraDoseGroot')
                kw['dose'] = value
            elif kind=='Fixed':
                spec['method'] = MOD+'CacheDose'
                kw['dose'] = value
            else:
                spec['method'] = MOD+'RiskLottery'
                kw.update(rho=value,allocation='uniform' if kind=='Uniform' else 'risk',
                          calibration_path=str(cal),calibration_sha256=digest(cal))
            kw.update(random_seed=26092901,randomization_key='Q2-replay-v1/'+cell)
            note = write_fit(spec,source)
            case = dict(name=spec['name'],cell=cell,kind=kind,spec=spec,artifact=note['artifact'],judge='guard_only',policy_tail=True)
            cases.append(case)
            if suite=='l10' and size=='50' and kind in ('Bextra','Risk'):
                repeat = copy.deepcopy(case)
                repeat.update(name=case['name']+'_reverse',kind=kind+'_reverse',reverse=True)
                cases.append(repeat)
    (HERE/'replay_cases.json').write_text(json.dumps(cases,indent=2)+'\n')
    print(json.dumps(dict(replay_cases=len(cases),recorded_episode_streams_per_case=20)))


if __name__ == '__main__':
    main()
