"""CPU: compare a host's complete native/concurrent run with published 4090 data."""
import argparse
import json
from pathlib import Path
import subprocess
from concurrent_compare import compare
from artifacts import write_json


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['dp','pi05','groot','fastwam','cosmos'])
    p.add_argument('--host',required=True,choices=['a100','h100'])
    p.add_argument('--work',default='/home/exouser/xgpu_1017')
    p.add_argument('--pull-reference',action='store_true')
    a=p.parse_args();work=Path(a.work);reference=work/'reference_results'
    if a.pull_reference:
        reference.mkdir(parents=True,exist_ok=True)
        subprocess.run(['rsync','-a','--bwlimit=16384','--include=/4090_'+a.model+'_concurrent_*/***',
            '--exclude=*','rsync://ziyanglin.com:23177/xc_local_results/',str(reference)+'/'],check=True)
    def complete(parent,host):
        rows=[]
        for r in sorted(parent.glob(host+'_'+a.model+'_concurrent_*')):
            f=r/'RUN.json'
            if f.exists():
                d=json.loads(f.read_text())
                if (d.get('status')=='COMPLETE' and d.get('schema')=='xgpu.concurrent.run.v1'
                    and not d['campaign'].get('xgpu_numerics')):rows.append((r,d))
        return rows
    local=complete(work/'results',a.host);remote=complete(reference,'4090')
    matches=[(l,r) for l,ld in local for r,rd in remote if
        ld['pin']['identity']==rd['pin']['identity'] and
        bool(ld['campaign'].get('xgpu_batch_telemetry'))==bool(rd['campaign'].get('xgpu_batch_telemetry'))]
    if not matches:raise SystemExit('No compatible COMPLETE pair; run/publish coordinator reference first.')
    l,r=matches[-1]
    def cold_start(candidates,selected):
        selected_data=dict(candidates)[selected]
        compatible=[]
        for path,data in candidates:
            if path==selected:continue
            if any(data.get(k)!=selected_data.get(k) for k in
                ('pin','input_identity','clients','deliveries','live_reference','harness_files')):continue
            if any(data['campaign'].get(k)!=selected_data['campaign'].get(k) for k in
                ('replicas','memory_budget_gib','xgpu_numerics','xgpu_batch_telemetry')):continue
            if data.get('gpu',{}).get('uuid')!=selected_data.get('gpu',{}).get('uuid'):continue
            compatible.append(path)
        return compatible[-1:]
    result=compare(r,l,.000345707 if a.model=='dp' else .001,
        left_repeats=cold_start(remote,r),right_repeats=cold_start(local,l))
    output=work/'evidence'/('cross_4090_'+a.host+'_'+a.model+'.json')
    output.parent.mkdir(parents=True,exist_ok=True);write_json(output,result)
    print(json.dumps(dict(report=str(output),status=result['status'],max_abs=result['max_abs'],
        max_concurrent_floor=result['max_concurrent_floor'],outside_budget=result['outside_budget'])))
