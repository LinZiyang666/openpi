"""Stage plugin-generated artifacts where this sandbox permits writes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
from exp.offline_search.rounds.r04.k7_guard.prepare import HERE, PREFIX, SPEC, RUN

STAGE = Path('/tmp/k7_guard_fits')
STAGE.mkdir(exist_ok=True)
commands = json.loads((HERE/'results/prefit_commands.json').read_text())
# Spatial 50 is a verification fit, not a sixth deployment arm.
commands.append(PREFIX+['-m','exp.offline_search.closed_loop.plugin','--os-method',SPEC,
    '--os-kwargs',json.dumps(dict(base_kwargs=dict(lib='current',kref=5,budget=2,gates='all',serving='phase_particles'),
        progress_guard='noprog_span',events='none',stuck_guard='vision_confirmed')),
    '--os-cell','pi05_spatial_cache','--os-root','/home/weiland/trace_runs/offline_search_store',
    '--os-log-dir',str(HERE/'results/prefit/verification_pi05_spatial_50'), '--os-blind',
    '--os-judge','guard_only','--os-fit-artifact',str(RUN/'fits/verification_pi05_spatial_50.pkl')])
commands = [[x.replace(str(RUN/'fits'), str(STAGE)) for x in cmd] for cmd in commands]
(HERE/'prefit_staged.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd /home/weiland/projects/openpi\n'
    +'\n'.join(shlex.join(c) for c in commands)+'\n')
(HERE/'results/prefit_staged_commands.json').write_text(json.dumps(commands,indent=2))

def job(cmd):
    artifact = Path(cmd[cmd.index('--os-fit-artifact')+1])
    if not artifact.exists():
        with (HERE/'results'/('fit_'+artifact.stem+'.log')).open('w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    report = dict(artifact=str(artifact), requested=str(RUN/'fits'/artifact.name), bytes=artifact.stat().st_size,
                  sha256=hashlib.file_digest(artifact.open('rb'),'sha256').hexdigest(),command=cmd)
    print(json.dumps(report), flush=True)
    return report
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    result = list(pool.map(job, commands))
(HERE/'results/fits.json').write_text(json.dumps(result,indent=2))
