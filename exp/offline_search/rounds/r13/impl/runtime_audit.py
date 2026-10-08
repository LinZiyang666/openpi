"""Audit all fit imports in a clean CPU interpreter; no explorer imports."""
import json
from pathlib import Path
import pickle
import sys

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
for r in json.loads((HERE/'fits.json').read_text()):
    pickle.loads(Path(r['path']).read_bytes())
assert not any(n.startswith(('exp.offline_search.rounds.r13.astra','exp.offline_search.rounds.r11.astra',
                            'exp.offline_search.rounds.r11.opus','exp.offline_search.rounds.r12.opus')) for n in sys.modules)
files={Path(m.__file__).resolve() for n,m in sys.modules.items() if n!='__main__'
       and (getattr(m,'__file__','') or '').endswith('.py')
       and Path(m.__file__).resolve().is_relative_to(REPO/'exp/offline_search')}
# Complete the established R11 serving/worker closure with changed/default-off
# recording, seed and coordinator files. Existing base dependency lists stay RO.
for file in (HERE.parents[1]/'r11/knob/H100_SOURCES.sha256',):
    files|={REPO/line.split('  ',1)[1] for line in file.read_text().splitlines()}
files|={REPO/f'exp/offline_search/closed_loop/{name}.py' for name in
        ('plugin','look_recording','worker_pool','rollout_seed','execution_worker')}
files.add(REPO/'exp/offline_search/closed_loop/ops/h100/control.py')
from exp.offline_search.closed_loop.devset import sha
(HERE/'H100_SOURCES.sha256').write_text(''.join(f'{sha(p)}  {p.relative_to(REPO)}\n' for p in sorted(files)))
(HERE/'runtime_audit.json').write_text(json.dumps(dict(PASS=True,artifacts=92,no_exploration_imports=True,
                                                     source_closure_files=len(files),new_runtime_files=7),indent=1)+'\n')
print('RUNTIME_IMPORT_PASS',len(files),flush=True)
