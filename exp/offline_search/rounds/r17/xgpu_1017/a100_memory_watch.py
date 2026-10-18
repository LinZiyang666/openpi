"""Bounded, read-only A100 memory sampling; never signals GPU processes."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-name', required=True)
    p.add_argument('--minutes', type=int, default=95)
    p.add_argument('--run', action='store_true')
    a = p.parse_args()
    if Path(a.run_name).name != a.run_name or not 1 <= a.minutes <= 180:
        raise ValueError('owned basename and 1..180 minutes required')
    root = Path('/home/exouser/xgpu_1017/results')
    output = root / (a.run_name + '_memory.jsonl')
    print(json.dumps(dict(output=str(output),interval_seconds=2,run=a.run)),flush=True)
    if not a.run:
        return
    names = subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
    if 'A100' not in names or 'H100' in names:
        raise ValueError('dedicated A100 only')
    end = time.monotonic() + a.minutes * 60
    peak, samples, closed = 0, 0, False
    with output.open('x') as f:
        while time.monotonic() < end:
            raw = subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used,memory.free',
                '--format=csv,noheader,nounits'],text=True).strip()
            peak = max(peak, int(raw.split(',')[1])); samples += 1
            f.write(json.dumps(dict(time=time.time(),gpu=raw))+'\n'); f.flush()
            for suffix in ('', '_cold2'):
                path = root / (a.run_name+suffix) / 'RUN.json'
                r = json.loads(path.read_text()) if path.exists() else {}
                if r.get('mps_cleanup',{}).get('daemon_closed') and (
                    r.get('status') == 'FAILED' or suffix and r.get('status') == 'COMPLETE'):
                    closed = True
            if closed:
                break
            time.sleep(2)
    summary = dict(samples=samples,peak_mib=peak,terminal_pair_observed=closed,
        sampling_seconds=2,scope='whole dedicated A100; sampled lower bound on true peak')
    output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
