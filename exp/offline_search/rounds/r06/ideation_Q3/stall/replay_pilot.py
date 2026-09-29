"""Descriptive/engineering replay on all A/B pilot anchors; never fits constants."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .stall import StallModel, StallTracker

HERE=Path(__file__).resolve().parent
TABLES=Path('/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables')
KEY=['arm','uid','attempt']
STATES=['inactive','ok','slow_confirmed','slow_ambiguous']


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True);args=ap.parse_args()
    cell=args.cell;model=StallModel.load(Path('/tmp/q3_stall_fits')/cell)
    root=TABLES/cell
    dec=pd.read_csv(root/'decisions.csv',usecols=KEY+['step','actual_controls'])
    dec=dec.sort_values(KEY+['step'])
    dec['control_index']=dec.groupby(KEY).actual_controls.cumsum()-dec.actual_controls
    cols=KEY+['step','task_id','init','assignment.cohort','retrieval.metric_code','retrieval.metric','guards.inputs_outputs.os_flags']
    a=pd.read_csv(root/'anchors.csv',usecols=cols)
    a=a[a['assignment.cohort'].isin(['A','B'])].merge(dec[KEY+['step','control_index']],on=KEY+['step'],validate='one_to_one')
    records=[]
    for identity,group in a.groupby(KEY,sort=True):
        group=group.sort_values('step');tracker=StallTracker(model,int(group.task_id.iloc[0]))
        for r in group.to_dict('records'):
            code=json.loads(r['retrieval.metric_code'])
            key=dict(metric_code=code,metric=r['retrieval.metric'])
            cpu_start=time.process_time_ns();t=time.perf_counter_ns()
            tracker.observe(key,int(r['control_index']))
            elapsed=time.perf_counter_ns()-t;cpu_elapsed=time.process_time_ns()-cpu_start
            status=tracker.status()
            assert status.get('reason')!='invalid_observation',(identity,r['step'])
            records.append(dict(cell=cell,arm=identity[0],uid=identity[1],attempt=identity[2],task_id=r['task_id'],
                init=r['init'],step=r['step'],control_index=r['control_index'],cohort=r['assignment.cohort'],
                noprog_flag=bool(int(r['guards.inputs_outputs.os_flags'])&8),observe_us=elapsed/1000,
                observe_cpu_us=cpu_elapsed/1000,
                **{k:v for k,v in status.items() if k!='reference_windows'}))
    frame=pd.DataFrame(records)
    frame.to_csv(HERE/f'pilot_status_{cell}.csv',index=False)
    summaries=[]
    for cohort,g in frame.groupby('cohort'):
        assert len(g[KEY].drop_duplicates())==60
        for scope,f in [('all_anchors',g),('noprog_flagged',g[g.noprog_flag])]:
            counts=f.state.value_counts().reindex(STATES,fill_value=0)
            summaries.append(dict(cell=cell,cohort=cohort,scope=scope,n=len(f),episodes=len(g[KEY].drop_duplicates()),
                counts=counts.to_dict(),rates={s:int(counts[s])/len(f) if len(f) else None for s in STATES},
                observe_median_us=float(f.observe_us.median()) if len(f) else None,
                observe_p99_us=float(f.observe_us.quantile(.99)) if len(f) else None,
                observe_cpu_median_us=float(f.observe_cpu_us.median()) if len(f) else None,
                observe_cpu_p99_us=float(f.observe_cpu_us.quantile(.99)) if len(f) else None))
    result=dict(cell=cell,model_fingerprint=model.fingerprint,anchors=len(frame),
        episodes=len(frame[KEY].drop_duplicates()),observe_input='precomputed deployed A metric_code; excludes JSON parse/status copy',
        observe_median_us=float(frame.observe_us.median()),observe_p99_us=float(frame.observe_us.quantile(.99)),
        observe_cpu_median_us=float(frame.observe_cpu_us.median()),observe_cpu_p99_us=float(frame.observe_cpu_us.quantile(.99)),
        active_observe_median_us=float(frame[frame.state.ne('inactive')].observe_us.median()),
        active_observe_p99_us=float(frame[frame.state.ne('inactive')].observe_us.quantile(.99)),
        scope='official test inits; descriptive and engineering only; no tuning or fitting',summaries=summaries)
    previous=HERE/f'pilot_replay_{cell}.json'
    if previous.exists():
        old=json.loads(previous.read_text())
        assert old['model_fingerprint']==result['model_fingerprint']
        assert [(s['cohort'],s['scope'],s['counts']) for s in old['summaries']]==[(s['cohort'],s['scope'],s['counts']) for s in summaries]
        result['status_counts_match_previous_run']=True
    (HERE/f'pilot_replay_{cell}.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
