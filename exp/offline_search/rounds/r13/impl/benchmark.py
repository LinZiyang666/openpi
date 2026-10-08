"""Incremental CPU query overhead on recorded B keys; no policy latency claim."""
import json
from types import MethodType
import time
import numpy as np

from .prepare import HERE, write_json
from .mechanics import Harness, load, plugin


def measure(h, method):
    times=[];inner_times=[]
    original_connection=h.connection
    def connection(source,**kwargs):
        conn=original_connection(source,**kwargs);m=conn._osp_sessions[0].method
        query=m.query;inner=m.inner.query;last=[0.]
        def base(owner,q):
            begin=time.perf_counter_ns();result=inner(q);last[0]=(time.perf_counter_ns()-begin)/1e3
            return result
        def outer(owner,q):
            begin=time.perf_counter_ns();result=query(q);total=(time.perf_counter_ns()-begin)/1e3
            times.append(total);inner_times.append(last[0]);return result
        m.inner.query=MethodType(base,m.inner);m.query=MethodType(outer,m)
        return conn
    h.connection=connection
    try:h.run(method,steps=24)
    finally:h.connection=original_connection
    overhead=np.array(times)-np.array(inner_times)
    return dict(anchors=len(times),query_median_us=float(np.median(times)),
                wrapper_median_us=float(np.median(overhead)),wrapper_p95_us=float(np.quantile(overhead,.95)))


if __name__=='__main__':
    plugin._git_head=lambda:None
    rows=[]
    for model in ('pi05','groot'):
        for size in (50,500):
            cell=f'{model}_l10_{size}';h=Harness(cell,HERE/'benchmark'/cell)
            off=measure(h,load(cell,'off'));feedback=measure(h,load(cell,'distance_feedback',.32))
            rows.append(dict(cell=cell,off=off,feedback=feedback,
                             incremental_wrapper_median_us=feedback['wrapper_median_us']-off['wrapper_median_us']))
            h.rt.flush_all();print('BENCH',cell,rows[-1]['incremental_wrapper_median_us'],flush=True)
    write_json(HERE/'benchmark.json',dict(PASS=True,records=rows,
               caveat='CPU replay wrapper time minus inherited inner query; includes row mapping, ledger checks, raw distance and proposal deepcopy; no model/policy latency measured'))
