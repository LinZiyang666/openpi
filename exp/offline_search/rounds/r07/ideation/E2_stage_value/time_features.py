"""Arithmetic-only timing on recorded queries; excludes retrieval and JSON I/O."""
import json
import time
import numpy as np
import pandas as pd
from analyze_stage import OUT,RUN,canon,load_lib,stage_features,source,finish_sources

rows=[]
for root in sorted((RUN/'r06_c_cal/tables').iterdir()):
    if not root.is_dir():continue
    cell=canon(root.name);lib=load_lib(cell)
    cols=['task_id','retrieval.rows','retrieval.weights','state.normalized','guards.inputs_outputs.gexec']
    samples=[]
    for r in pd.read_csv(source(root/'anchors.csv'),usecols=cols,nrows=100).to_dict('records'):
        rr,w=json.loads(r['retrieval.rows']),json.loads(r['retrieval.weights'])
        cache=np.einsum('i,ijk->jk',np.asarray(w)/sum(w),lib['action'][rr,:,:7])
        samples.append((lib,int(r['task_id']),rr,w,json.loads(r['state.normalized']),cache,r['guards.inputs_outputs.gexec']))
    for args in samples:stage_features(*args)
    elapsed=[]
    for _ in range(10):
        for args in samples:
            t=time.process_time_ns();stage_features(*args);elapsed.append((time.process_time_ns()-t)/1000)
    rows.append(dict(cell=cell,n=len(elapsed),median_us=np.median(elapsed),p99_us=np.quantile(elapsed,.99)))
pd.DataFrame(rows).to_csv(OUT/'feature_cpu.csv',index=False)
finish_sources('feature_cpu')
print(pd.DataFrame(rows).to_string(index=False))
