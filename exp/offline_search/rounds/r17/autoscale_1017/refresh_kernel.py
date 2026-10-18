"""Read-only R15 node evidence refresh; write audit ONLY to a separate output."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess

from . import kernel_evidence as k, trial_lane


def refresh(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    if output == root or root in output.parents:raise ValueError('audit output must be outside the completed lane')
    record = trial_lane.read(root/'AUTOSCALE_VALIDATION.json')
    if record['path'] != 'r15':raise ValueError('this refresh is for the R15 acceptance lane')
    journals = list(root.glob('runs/*/client/journal.jsonl'))
    if len(journals) != 1:raise ValueError('one collected arm required')
    scope = journals[0].parent/'autoscale'
    old = trial_lane.read(scope/'AUTOSCALE_KERNEL.json')
    launch = trial_lane.read(journals[0].parent.parent/'h100_launch.json')
    host = launch['worker_host']
    if host not in ('timan107','timan1'):raise ValueError('unsupported worker node')
    remote_run = str(Path(launch['driver']['out'])/'autoscale')
    # Only standard-library reads and journalctl/dmesg; no installation or writes.
    code = 'from pathlib import Path\nimport json,re,subprocess,time\nOOM=re.compile('+repr(k.OOM.pattern)+',re.I)\n'
    code += '\n'.join(inspect.getsource(f) for f in (k.counters,k.counter_evidence,k.collect))
    code += '\nprint(json.dumps(collect('+repr(remote_run)+','+repr(old['started'])+',ended_at='+repr(old['ended'])+',persist=False)))\n'
    result = subprocess.run(['tether','exec','--safe','--timeout','90s',host,'--','/usr/bin/python3','-c',code],
                            text=True,capture_output=True,timeout=95,check=True)
    proof=json.loads(result.stdout.strip().splitlines()[-1])
    if proof['counter_end']['host'] != host:raise ValueError('node identity mismatch')
    output.mkdir(parents=True,exist_ok=True)
    trial_lane.write(output/'AUTOSCALE_KERNEL.json',proof)
    trial_lane.write(output/'ORIGINAL_KERNEL.json',old)
    return trial_lane.audit(root,output=output,kernel_overrides={str(scope):proof})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();print(json.dumps(refresh(a.run,a.out),indent=2))


if __name__=='__main__':main()
