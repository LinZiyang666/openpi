"""At most three CPU replay subprocesses plus this coordinator (four total)."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from .prepare_adapters import CPU, MOD

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--names', nargs='*')
    args = ap.parse_args()
    if not args.out.resolve().is_relative_to(HERE):
        raise ValueError('output must be inside adapters/')
    args.out.mkdir(parents=True,exist_ok=False)
    cases = json.loads((HERE/'replay_cases.json').read_text())
    if args.names:
        cases = [c for c in cases if c['name'] in args.names]
        if len(cases) != len(args.names):
            raise ValueError('unknown case')
    def one(case):
        cmd = CPU+['-m',MOD+'.replay','--case',case['name'],'--out',str(args.out/case['name'])]
        logpath = args.out/(case['name']+'.log')
        with logpath.open('w') as f:
            proc = subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=False)
        return dict(name=case['name'],exit_code=proc.returncode,log=str(logpath),command=cmd)
    with ThreadPoolExecutor(max_workers=3) as executor:
        results = []
        for r in executor.map(one,cases):
            results.append(r)
            print(json.dumps({k:r[k] for k in ('name','exit_code','log')}),flush=True)
    (args.out/'commands.json').write_text(json.dumps(results,indent=2)+'\n')
    if any(r['exit_code'] for r in results):
        raise SystemExit('replay failures; see command logs')


if __name__ == '__main__':
    main()
