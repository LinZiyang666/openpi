"""Lossless union of the columns consumed by the frozen Q1 runner.

Reads only the eight coordinator-verified table directories, never failed1 or
repair copies. No change to score definitions, fits, samples or inference.
"""
import json
from pathlib import Path
import pandas as pd
import pilot_q1 as q

CELLS=[f'{m}_{s}_{n}' for m in ['pi05','groot'] for s in ['l10','sp'] for n in [50,500]]
SOURCE=Path('/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/tables')
OUT=q.HERE/'joint_tables'
OUT.mkdir(exist_ok=False)
manifest={'frozen_runner_sha256':q.sha(q.HERE/'pilot_q1.py'),'cells':{},'sources':{}}
for c in CELLS:
    p=SOURCE/c
    audit=json.loads((p/'audit.json').read_text())
    assert audit['episodes']==540,(c,audit)
    manifest['cells'][c]=audit
for name in ['episodes','anchors','decisions','action_steps','attempts']:
    frames=[]
    for c in CELLS:
        p=SOURCE/c/(name+'.csv')
        manifest['sources'][str(p)]=dict(sha256=q.sha(p),bytes=p.stat().st_size)
        frames.append(q.table_read(SOURCE/c,name))
    frame=pd.concat(frames,ignore_index=True)
    if name=='episodes':
        assert len(frame)==4320 and frame.arm.nunique()==216
        assert not frame.duplicated(q.EKEY).any()
    frame.to_csv(OUT/(name+'.csv'),index=False)
    print(json.dumps({'table':name,'rows':len(frame),'columns':len(frame.columns)}),flush=True)
q.dump(OUT/'source_manifest.json',manifest)
