"""Build deployable specs/calibration/CPU commands without emitting a run root."""
from __future__ import annotations
import copy
import hashlib
import json
from pathlib import Path
import shlex
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
FRONT = HERE.parent
Q2 = FRONT.parent
REPO = Q2.parents[4]
ROOT = Path('/home/weiland/trace_runs/offline_search_store')
MOD = 'exp.offline_search.rounds.r06.ideation_Q2.frontier.adapters'
CPU = ['taskset', '-c', '18-21,62-65', 'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
       'MKL_NUM_THREADS=1', 'CUDA_VISIBLE_DEVICES=', 'PYTHONDONTWRITEBYTECODE=1', 'PYTHONPATH=.:src',
       f'TMPDIR={HERE}/.tmp', f'MPLCONFIGDIR={HERE}/.mplconfig', '.venv/bin/python']


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def replace_arg(args, flag, value=None):
    args = list(args)
    if flag in args:
        i = args.index(flag)
        if value is not None:
            args[i+1] = str(value)
    else:
        args += [flag] + ([] if value is None else [str(value)])
    return args


def make_calibration():
    source = Q2/'library_quality.json'
    banks = {}
    for r in json.loads(source.read_text()):
        if r['lib'] not in ('50', '500'):
            continue
        path = Path(r['library'])
        key = f'{path.parent.name}/{path.name}'
        if key not in banks:
            banks[key] = dict(c1=.152 if r['cell'].startswith('pi05') else .148,
                c1_provenance='owner stage-price convention from Q2 PREREG; not a newly measured latency',
                commitment_blocks=2, request_controls=5, source_library=str(path),
                risk_recipe='Q2 library_proxy.py: task standardized valid-state nearest other-episode RMS; episode then task mean',
                library_sha256={f: sha(path/f) for f in ['manifest.json','rs.npy','task_id.npy','episode.npy']}, tasks={})
        banks[key]['tasks'][str(r['task'])] = dict(risk=r['state_loeo_distance'], h=r['mean_decisions'],
            a=r['mean_anchors_b2'], traffic=1., source_episodes=r['episodes'])
    save('calibration.json', dict(schema='Q2-library-budget-v1', source=str(source), source_sha256=sha(source),
        risk_is_SR_certificate=False, libraries=banks))
    return HERE/'calibration.json'


def base_ids(cell):
    m, s, size = cell.split('_')
    sh = 'sp' if s == 'spatial' else 'l10'
    a = f'r05_x/r5x_g_{sh}_{size}_tail1u' if m == 'groot' else f'r05_ptail/r5t_p_{sh}_{size}_tail1uc'
    if cell == 'pi05_spatial_500':
        a = 'r04_blind/r4b3_p_sp_500_tail1uc'
    b = f'r06_paper/r6p1_c10_g_{sh}_{size}' if m == 'groot' else f'r05_q1/r5q1_c10_p_{sh}_{size}'
    return a, b


def main():
    (HERE/'.tmp').mkdir(exist_ok=True)
    cal = make_calibration()
    plan = json.loads((FRONT/'completion_plan.json').read_text())
    source = json.loads((FRONT/'source_evidence.json').read_text())
    old = {x['name']: x for x in json.loads((FRONT/'emit_arms_existing.json').read_text())}
    pending = {x['name']: x for x in json.loads((FRONT/'needs_code_configs.json').read_text())}
    manifest = json.loads((FRONT/'eval500_manifest.json').read_text())
    save('eval500_manifest.json', manifest)
    manifest_sha = sha(HERE/'eval500_manifest.json')
    specs, reuse, changes = [], {}, []
    for row in plan['arms']:
        name, cell = row['name'], row['cell']
        model, suite, size = cell.split('_')
        a_id, b_id = base_ids(cell)
        if name in old:
            spec = copy.deepcopy(old[name])
            src = row['source']
            reference = source[src]['spec']
        else:
            cfg = pending[name]
            src = cfg['source_arm']
            reference = source[src]['spec']
            spec = {k: copy.deepcopy(reference[k]) for k in ['model','kwargs','plugin_args','client_overrides','yaml_patch'] if k in reference}
            kw = spec['kwargs']
            if 'rho' in cfg:
                spec['method'] = MOD+'.methods:RiskLottery'
                kw.update(rho=cfg['rho'], allocation='risk', calibration_path=str(cal), calibration_sha256=sha(cal))
                if cfg['commitment_blocks'] != 2:
                    changes.append(dict(arm=name, prior_controls=5*cfg['commitment_blocks'], controls=10,
                        reason='owner now explicitly requires 10-control lottery commitment; rho unchanged, infeasibility logged'))
            elif cfg['baseline'] == 'B':
                spec['method'] = MOD+'.methods:'+('ExtraDosePi05' if model == 'pi05' else 'ExtraDoseGroot')
                kw['dose'] = cfg['dose']
            else:
                spec['method'] = MOD+'.methods:CacheDose'
                kw['dose'] = cfg['dose']
            kw.update(random_seed=26092901, randomization_key=f'{manifest_sha}/{cell}')
            args = spec.setdefault('plugin_args', [])
            for flag, value in [('--os-blind',None),('--os-policy-tail',None),('--os-policy-tail-blocks',1),('--os-judge','guard_only')]:
                args = replace_arg(args, flag, value)
            spec['plugin_args'] = args
        spec.update(name=name, model=model, suite=suite, mode='plugin', full_model=True, cost_ledger=True,
                    manifest='<RUN>/manifests/eval500.json')
        spec.setdefault('client_overrides', {})['replan_steps'] = 5
        if model == 'groot':
            spec['client_overrides']['resize_size'] = 256
        args = spec.setdefault('plugin_args', [])
        for flag, value in [('--os-root', ROOT),('--os-no-shadow-native',None),('--os-fit-artifact',f'<RUN>/fits/{name}.pkl')]:
            args = replace_arg(args, flag, value)
        spec['plugin_args'] = args
        specs.append(spec)
        pargs = reference.get('plugin_args', [])
        fit = pargs[pargs.index('--os-fit-artifact')+1] if '--os-fit-artifact' in pargs else None
        reuse[name] = dict(source_arm=src, source_spec=reference['method'], source_kwargs=reference['kwargs'],
                           source_artifact=fit, target_changes=spec['kwargs'] != reference['kwargs'])
    assert len(specs) == 43 and len({s['name'] for s in specs}) == 43
    save('emit_arms_all43.json', specs)
    save('fit_sources.json', reuse)
    save('plan_changes.json', changes)
    smoke_names = ['r6q2_pi05_l10_50_B_dose0p5','r6q2_groot_l10_50_risk_rho0p35']
    smoke = [copy.deepcopy(next(s for s in specs if s['name'] == n)) for n in smoke_names]
    for s in smoke:
        s['manifest'] = '<RUN>/manifests/smoke4.json'
    save('emit_arms_smoke2.json', smoke)
    save('smoke4_manifest.json', dict(selected=[dict(task=t, init=i) for t in (0,1) for i in (0,1)]))
    commands = [shlex.join(CPU+['-m', MOD+'.prefit', '--name', s['name']]) for s in specs]
    (HERE/'prefit_all43.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\n'+'\n'.join(commands)+'\n')
    save('prefit_commands.json', dict(commands=commands, output_root='/tmp/q2_adapter_fits', launches=False))
    # Also declare zero-dose and nonzero replay configurations, without allocating run roots.
    replay = []
    for model in ('pi05','groot'):
        for suite in ('l10','spatial'):
            for size in (50,500):
                cell = f'{model}_{suite}_{size}'
                a_id,b_id = base_ids(cell)
                for label, srcid in [('A',a_id),('B',b_id)]:
                    ref = source[srcid]['spec']
                    replay.append(dict(name=f'test_{cell}_{label}', cell=cell, method=ref['method'], kwargs=ref['kwargs'], source_arm=srcid,
                        source_artifact=ref['plugin_args'][ref['plugin_args'].index('--os-fit-artifact')+1]))
    save('replay_baselines.json', replay)
    print(json.dumps(dict(arms=len(specs), adapters=len(pending), calibration_libraries=len(json.loads(cal.read_text())['libraries']),
                         smoke_arms=smoke_names, changes=changes)))


if __name__ == '__main__':
    main()
