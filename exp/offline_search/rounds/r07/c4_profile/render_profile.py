"""Render C4's audited plan into owned scratch, with final RUN paths in specs."""
from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from . import common as C


def recipes(out,run='<RUN>'):
    """Persist one exact prefit line per arm, retaining final-path placeholders."""
    counts={}
    for phase in ['profile','eval500']:
        rows=json.loads((out/f'arms_{phase}.json').read_text())
        counts[phase]=len(rows)
        commands=[]
        prefix="taskset -c 10-13,54-57 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src .venv/bin/python"
        for row in rows:
            commands.append(f'{prefix} -m exp.offline_search.rounds.r07.c4_profile.prefit --spec {shlex.quote(run+"/arms_"+phase+".json")} --name {shlex.quote(row["name"])} --out /tmp/r7_C4/prefits_{phase}')
        (out/f'prefit_{phase}.sh').write_text('set -euo pipefail\n'+'\n'.join(commands)+'\n')
        (out/f'arm_names_{phase}.txt').write_text(''.join(r['name']+'\n' for r in rows))
    return counts


def render(source,out,run):
    out=Path(out).resolve();run=str(Path(run).resolve())
    if C.SCRATCH not in out.parents and out!=C.SCRATCH:raise ValueError('render destination must be /tmp/r7_C4')
    if any(c in run for c in ('\n','\r',"'",'`','$',';')):raise ValueError('run path must be shell-safe')
    out.mkdir(parents=True,exist_ok=True)
    for path in Path(source).rglob('*'):
        if path.is_file():
            dest=out/path.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_text(path.read_text().replace('<RUN>',run))
    C.write(out/'manifests/eval500.json',dict(role='OFFICIAL_TEST',selected=[dict(task=t,init=i) for t in range(10) for i in range(50)]))
    counts=recipes(out,run)
    C.write(out/'render_audit.json',dict(run_root=run,counts=counts,executed=False,source=str(source),
        source_specs_sha256={phase:C.sha(Path(source)/f'arms_{phase}.json') for phase in counts}))
    return counts


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,default=C.HERE/'prepared')
    p.add_argument('--out',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    a=p.parse_args();print(json.dumps(render(a.source,a.out,a.run_root)))


if __name__=='__main__':main()
