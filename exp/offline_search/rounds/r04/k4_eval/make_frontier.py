"""Write the approved first-batch/single-policy controls; does not launch or fit."""
import json
from pathlib import Path

AWM = 'exp/offline_search/rounds/r02/g1_awm/awm.py:AWM'
JUDGE = 'exp/offline_search/rounds/r03/h3_judge/judge.py:MixedJudge'
rows = []
for suite, short in [('l10', 'l10'), ('spatial', 'sp')]:
    for seed in [1001, 2001]:
        name = f'r4f_p_{short}_inf_s{seed}'
        rows.append(dict(name=name, model='pi05', suite=suite, mode='plugin', pure_inference=True,
                         server_seed=seed, plugin_args=['--os-fit-artifact', f'<RUN>/fits/{name}.pkl'],
                         _batch=1, _library_episodes=None,
                         _prefit='SeededInference B0 proposal, all MISS. Reuse only same SeededInference spec/kwargs/cell. '
                                 'Library not used for executed actions; plot on both 50/500 panels.'))
for suite, short, scale, guard, period, noprog in [
    ('l10','l10',500,True,None,4), ('l10','l10',500,False,8,None), ('l10','l10',500,False,12,None),
    ('spatial','sp',500,True,None,3), ('spatial','sp',500,False,12,None),
    ('l10','l10',50,False,6,None), ('l10','l10',50,True,None,4)]:
    base_kw = {} if scale == 500 else {'lib':'current','kref':5}
    kwargs = dict(base=AWM, base_kwargs=base_kw, guards=True, events='none') if guard else base_kw
    if guard and noprog != 3:
        kwargs['noprog_n'] = noprog
    suffix = f'g{scale}' + (f'_np{noprog}' if noprog != 3 else '') if guard else f'lib{scale}_per{period}'
    name = f'r4f_p_{short}_{suffix}'
    method = JUDGE if guard else AWM
    if guard and noprog != 3:
        note = 'New fit required: noprog_n=4 changes kwargs; do not relabel an R3 noprog_n=3 pickle.'
    elif guard:
        note = 'Reuse matching spec/kwargs/cell R4 frontier guard fit if available; l10 R3 fit cannot be reused for spatial.'
    else:
        note = f'Reuse R2 r02_g{scale}/fits/oscl{scale}_p_{short}_cl2.pkl (exact AWM spec/kwargs/cell); copy to this path.'
    rows.append(dict(name=name, model='pi05', suite=suite, mode='plugin', method=method, kwargs=kwargs,
                     full_model=True, plugin_args=['--os-fit-artifact',f'<RUN>/fits/{name}.pkl',
                                                  '--os-judge','guard_only' if guard else f'periodic:{period}'],
                     _batch=1, _library_episodes=scale, _prefit=note))
for suite, short in [('l10','l10'),('spatial','sp')]:
    for k, length in [(10,10),(2,10),(2,5)]:
        name = f'r4f_p_{short}_inf_k{k}_L{length}'
        rows.append(dict(name=name, model='pi05', suite=suite, mode='plugin', pure_inference=True,
                         server_seed=3001, yaml_patch={'miss':{'num_steps':k}},
                         client_overrides={'replan_steps':length},
                         plugin_args=['--os-fit-artifact',f'<RUN>/fits/{name}.pkl'],
                         _batch=2 if length==5 else 3, _library_episodes=None,
                         _prefit='Reuse same SeededInference spec/kwargs/cell pickle from first batch; K/L/seed do not change fit. '
                                 'Library not used for executed actions; plot on both 50/500 panels.'))
for row in rows:
    row['cost_ledger'] = True
    row['plugin_args'].append('--os-log-r4')
Path(__file__).with_name('arms_frontier.json').write_text(json.dumps(rows,indent=2)+'\n')
print(f'{len(rows)} rows: 11 first batch, 4 L=10, 2 K2-only')
