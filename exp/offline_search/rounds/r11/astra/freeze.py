"""Freeze the arm grid and prospective predictions, before reading any R11 result.

Numeric SR forecasts are explicitly subjective priors, NOT learned from B error
capture or fitted to A. Historical REPORT aggregates contextualize forecasts only;
they never enter score fitting, thresholds, arm selection or IR calibration.
"""
from __future__ import annotations
import csv
from datetime import datetime, timezone
import json
import re
import numpy as np
from exp.offline_search.rounds.r11.astra.boundary import HERE, dump, install, sha
from exp.offline_search.rounds.r11.astra.experiment import CELLS
from exp.offline_search.rounds.r11.astra.signals import simulate


def old_aggregates():
    report=HERE.parents[1]/'r10'/'REPORT.md'
    table={};model=suite=None
    for line in report.read_text().splitlines():
        if line.startswith('### ') and len(line.split())==3:
            _,model,suite=line.split()
        if model in ('pi05','groot') and suite in ('l10','spatial') and re.match(r'^\| \d+ \|',line):
            fields=[x.strip() for x in line.strip('|').split('|')]
            sr,ir=map(float,fields[-1].split(' @ '))
            table[f'{model}_{suite}_{fields[0]}']=dict(sr=sr,ir=ir)
    return table,report


def markdown_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    install()
    if (HERE/'freeze.json').exists():
        raise FileExistsError('Prediction freeze already exists; do not silently rewrite a preregistration')
    timestamp=datetime.now(timezone.utc).isoformat()
    old,source=old_aggregates()
    pure={'pi05_l10':.908,'groot_l10':.898,'pi05_spatial':.988,'groot_spatial':.940}
    priors=dict(recovery_slope=.65,sparse_uncertainty_pp=4.,larger_uncertainty_pp=2.,
                signal_advantage_over_random_pp=0.,pure_policy_sr=pure,
                explanation='Subjective forecasts; no causal error-to-success fit is possible from B policy states.')
    arms=[]
    for model,suite,size in CELLS:
        tag=f'{model}_{suite}_{size}'
        a=json.loads((HERE/'data'/tag/'analysis.json').read_text())
        with np.load(HERE/'data'/tag/'signals.npz') as z:
            signals={k:np.asarray(z[k]) for k in z.files}
        with np.load(HERE/'data'/tag/'pack.npz') as z:
            pack={k:np.asarray(z[k]) for k in z.files}
        cfg=[]
        for c in a['calibrations']:
            if c['method']=='distance' and c['beta']==1.:
                cfg.append(('distance',c))
            elif c['method']=='predicted_error' and c['beta']==.5:
                cfg.append(('error_hybrid',c))
            elif size==50 and c['method']=='disagreement' and c['beta']==1. and c['target']==.32:
                cfg.append(('disagreement',c))
        if size==50:
            cfg.extend(('adaptive_error_hybrid',c) for c in a['adaptive'])
        for method,c in cfg:
            target=c['target'];adaptive=method.startswith('adaptive')
            arm=f'astra_{tag}_{method}_ir{int(round(100*target)):02d}'
            if adaptive:
                point=c['validation']['owner_ir_mean']
                scenarios=[dict(stress=s['stress'],**s['adaptive']) for s in c['scenarios']]
                rate=c['validation']['miss_per_look']
                threshold=None;tie=None;beta=.5
            else:
                point=c['owner_ir'];rate=c['miss_per_look'];threshold=c['threshold'];tie=c['tie_probability'];beta=c['beta']
                score=signals[c['method']]
                scenarios=[]
                for shift,stress in ((0.,False),(.2,False),(-.2,False),(0.,True)):
                    r=simulate(pack,score,c['dose'],target,model,beta=beta,seeds=64,
                               shift=shift,stress=stress,seed=20261006)
                    scenarios.append(dict(**r))
            low=min(s['owner_ir_mean'] for s in scenarios)
            high=max(s['owner_ir_mean'] for s in scenarios)
            baseline=old[tag];gap=max(0.,pure[f'{model}_{suite}']-baseline['sr'])
            spend=float(np.clip((point-baseline['ir'])/(.5-baseline['ir']),0,1))
            center=100*priors['recovery_slope']*spend*gap
            uncertainty=priors['sparse_uncertainty_pp'] if size==50 else priors['larger_uncertainty_pp']
            cfgrow=dict(arm=arm,cell=tag,model=model,suite=suite,size=size,method=method,
                target_ir=target,dose=c['dose'],threshold=threshold,tie_probability=tie,beta=beta,
                eta=.2 if adaptive else 0.,random_floor_initial=(1-beta)*c['dose'],
                predicted_realized_ir=point,ir_scenario_low=low,ir_scenario_high=high,
                predicted_miss_per_look=rate,predicted_sr_delta_pp=center,
                sr_subjective_low_pp=center-uncertainty,sr_subjective_high_pp=center+uncertainty,
                predicted_sr=baseline['sr']+center/100,historical_base=baseline,
                calibration_feasible=c['feasible'],scenarios=scenarios,
                predictor_path=str(HERE/'data'/tag/'predictor.json') if 'error' in method else None,
                score_reference_path=str(HERE/'data'/tag/'signals.npz') if adaptive else None,
                calibration_sha256=sha(HERE/'data'/tag/'analysis.json'))
            assert c['feasible']
            arms.append(cfgrow)
        print('frozen',tag,len(cfg),flush=True)
    for tag in {a['cell'] for a in arms}:
        for method in {a['method'] for a in arms}:
            assert sum(a['cell']==tag and a['method']==method for a in arms)<=3
    counts={m:sum(a['method']==m for a in arms) for m in sorted({a['method'] for a in arms})}
    freeze=dict(timestamp_utc=timestamp,arms=arms,method_counts=counts,arm_count=len(arms),
        forecast_priors=priors,historical_aggregate_source=str(source),historical_report_sha256=sha(source),
        historical_use='Forecast context only; no fitting, calibration or selection.',
        fitting_data='B libraries only; no B-val recordings used; no closed-loop outputs read.',
        status='Prospective freeze, explorer has not read any R11 closed-loop result.',
        source_sha256={p.name:sha(p) for p in HERE.glob('*.py')})
    dump(HERE/'freeze.json',freeze)
    cols=['arm','cell','method','target_ir','dose','threshold','tie_probability','beta','eta',
          'random_floor_initial','predicted_realized_ir','ir_scenario_low','ir_scenario_high',
          'predicted_miss_per_look','predicted_sr_delta_pp','sr_subjective_low_pp','sr_subjective_high_pp']
    with (HERE/'arm_grid.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore');writer.writeheader();writer.writerows(arms)
    grid=['# Proposed state-dependent arm grid',f'Frozen {timestamp}.',
          f'{len(arms)} proposed arms, excluding the coordinator’s same-batch knob-off controls and opus’s schedules.',
          'All cell-size parameters are shared across tasks. Target levels follow the brief’s small grid; no A outcomes select parameters.',
          markdown_table(['Method','Cells and targets','Arms','Rationale'],[
            ['distance','Every size-50 cell: .25/.32/.40; L10-200/500: .25',counts['distance'],
             'Requested simple state trigger; conservative benchmark for richer signals.'],
            ['error_hybrid','Every size-50 cell: .25/.32/.40; L10-200/500: .25',counts['error_hybrid'],
             'Strictly nested error score plus half-uniform floor; strongest consistent error proxy.'],
            ['adaptive_error_hybrid','Every size-50 cell: .32/.40',counts['adaptive_error_hybrid'],
             'Track actual committed cost; tested only at feasible higher targets.'],
            ['disagreement','Every size-50 cell: .32',counts['disagreement'],
             'Cheap nonlearned comparator; the predictor’s incremental gain is often small.']]),
          'Standalone gripper triggers, pure learned-error thresholds, and dense low-budget adaptive arms are not proposed. '
          'Pure learned thresholds remain an offline baseline, to keep the live grid small. '
          'The knob remains off unless a spend target is explicitly selected; the library cannot establish positive causal spend utility.',
          'If the combined arm budget needs trimming, remove disagreement ablations first, then adaptive Spatial arms. '
          'Keep the distance and hybrid comparisons plus same-batch knob-off controls.',
          'Exact numeric parameters and per-arm predictions: [arm_grid.csv](arm_grid.csv), [freeze.json](freeze.json), [PREDICTION.md](PREDICTION.md).']
    (HERE/'ARM_GRID.md').write_text('\n\n'.join(grid)+'\n')
    pred=['# Prospective predictions',f'Frozen UTC: **{timestamp}**. Generated by `freeze.py`; full immutable payload: `freeze.json`.',
          'No R11 closed-loop result was read. No closed-loop job was launched. '
          'All cost calibration and candidate selection used B-library evidence. Historical R10 REPORT aggregates below enter subjective success forecasts only.',
          'IR point predictions assume the B-held-out state distribution and ordinary ten-control cadence. '
          'The interval is the range across explicit offline scenarios (nominal ranks, rank shifts ±.20, and synthetic persistent stalls), '
          'not a confidence interval or guaranteed live bound. Adaptation cannot defeat a mandatory guard floor.',
          'Success forecasts deliberately award **zero extra success gain over random at matched IR** to signal placement. '
          'The earlier disagreement prior argues against translating error capture directly into success. '
          'The forecast is subjective: ΔSR = .65 × clip((predicted IR − historical base IR)/(.5 − historical base IR),0,1) '
          '× max(pure-policy SR − historical base SR,0). The uncertainty half-width is 4 percentage points for size-50 '
          'and 2 for larger libraries. These constants are declared priors in `freeze.py`, not measured effects.',
          'Sparse-library direction: modest improvement as spending rises, with harm still plausible. '
          'Libraries whose three-layer base is already near pure policy: approximately neutral; do not spend by default.',
          markdown_table(['Arm','Target','Predicted IR','Offline scenario range','ΔSR pp','Subjective ΔSR range pp'],[
            [r['arm'],f"{r['target_ir']:.2f}",f"{r['predicted_realized_ir']:.3f}",
             f"{r['ir_scenario_low']:.3f}–{r['ir_scenario_high']:.3f}",f"{r['predicted_sr_delta_pp']:+.1f}",
             f"{r['sr_subjective_low_pp']:+.1f} to {r['sr_subjective_high_pp']:+.1f}"] for r in arms]),
          'Every number in this document is emitted by the executed `freeze.py`, either from B analysis, '
          'from permitted historical aggregate markdown, or from the explicitly declared forecast priors. '
          'Final SR comparisons require the coordinator’s same-batch knob-off and matched-IR random arms.']
    (HERE/'PREDICTION.md').write_text('\n\n'.join(pred)+'\n')
    print('FREEZE',timestamp,len(arms),counts,flush=True)


if __name__=='__main__':
    main()
