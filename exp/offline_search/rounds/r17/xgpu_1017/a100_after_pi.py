"""Serial, dedicated-A100 continuation after BOTH pi05 cold starts close.

Default is a plan. On failure, retain the current model for investigation.
Never signal a predecessor or infer completion from elapsed time. After a
successful pi05 pair, reuse its environment for the new DP recipe, then test
TF32-off only if the native DP pair exceeds the preregistered action budget.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from a100_stage import WORK,audit


def process_alive(pid,ticks):
    try:
        text=(Path('/proc')/str(pid)/'stat').read_text();fields=text[text.rfind(')')+2:].split()
        return fields[19]==ticks and fields[0]!='Z'
    except FileNotFoundError:return False


def terminal_pair(name):
    paths=[WORK/'results'/(name+suffix)/'RUN.json' for suffix in ('','_cold2')]
    rows=[json.loads(p.read_text())if p.exists()else{} for p in paths]
    if any(r.get('status')=='FAILED' for r in rows):
        raise RuntimeError('pi05 capacity run failed; retain its files, do not start DP')
    return paths[-1].parent if all(r.get('status')=='COMPLETE' and r.get('mps_cleanup',{}).get('daemon_closed') for r in rows) else None


def cold_floor(name):
    root=WORK/'results'/name;repeat=WORK/'results'/(name+'_cold2')
    command=[str(WORK/'env/bin/python'),str(WORK/'tools/concurrent_compare.py'),str(root),str(root),
        '--left-repeat',str(repeat),'--right-repeat',str(repeat),'--absolute-budget','0.000345707',
        '--output',str(root/'COLD_FLOOR.json')]
    with (root/'cold_floor.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((root/'COLD_FLOOR.json').read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pi-name',required=True);p.add_argument('--dp-name',required=True)
    p.add_argument('--after-pid',type=int,required=True);p.add_argument('--after-ticks',required=True)
    p.add_argument('--port',type=int,default=29800);p.add_argument('--run',action='store_true')
    a=p.parse_args()
    if any(Path(x).name!=x for x in (a.pi_name,a.dp_name)) or a.after_pid<=0:
        raise ValueError('owned run basenames and a predecessor PID required')
    print(json.dumps(dict(pi_pair=a.pi_name,dp_pair=a.dp_name,after_pid=a.after_pid,
        after_ticks=a.after_ticks,run=a.run,scope='serial A100-only, shared environment, native then conditional TF32-off')),flush=True)
    if not a.run:return
    names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
    if 'A100'not in names or 'H100'in names:raise ValueError('dedicated A100 only')
    deadline=time.monotonic()+3*3600
    while time.monotonic()<deadline:
        # Wait for the supervisor too: its final CPU summary still uses the env.
        if process_alive(a.after_pid,a.after_ticks):time.sleep(10);continue
        previous=terminal_pair(a.pi_name)
        if previous is None:raise RuntimeError('predecessor exited without a complete pi05 pair; no cleanup')
        break
    else:raise TimeoutError('predecessor still active; no cleanup or DP launch')
    command=[sys.executable,str(WORK/'tools/a100_openpi_handoff.py'),'--pi-run',str(previous),
        '--output-name',a.dp_name,'--port',str(a.port),'--run']
    audit('queued_dp_handoff',argv=command);subprocess.run(command,check=True)
    floor=cold_floor(a.dp_name)
    needs_tf32=floor['outside_budget']>0 or any(
        field['observed'] and field['agreement']<1 for field in floor['head_agreement'].values())
    audit('native_dp_pair_complete',test_tf32_off=needs_tf32)
    if needs_tf32:
        command=[sys.executable,str(WORK/'tools/a100_run_pair.py'),'--model','dp',
            '--deployment','longseg_recal1018','--output-name',a.dp_name+'_tf32off',
            '--port',str(a.port),'--numerics','tf32_off','--run']
        audit('queued_dp_tf32off',argv=command);subprocess.run(command,check=True)
        effect=cold_floor(a.dp_name+'_tf32off')
        audit('dp_tf32off_measured',max_abs=effect['max_abs'],outside_budget=effect['outside_budget'],
            top1=effect['head_agreement']['top1_neighbour'],restored_cross_gpu_equivalence=False)
    audit('queued_a100_work_complete',dp_pair=a.dp_name,tf32off_tested=needs_tf32,
        cross_gpu_evidence=False,closedloop_evidence=False)


if __name__=='__main__':main()
