"""Read-only schema inspection of explicitly authorized B-dev recordings."""
from .common import *
DEV.update({'r12_devlooks_50L', 'r12_devlooks_50S', 'r12_devlooks_200', 'r12_devlooks_400', 'r12_devlooks_knob'})
install()
import pickle
import numpy as np
from exp.offline_search.closed_loop.look_recording import read_looks

out = {'timestamp': now(), 'prediction_sha256': sha(HERE/'PREDICTION.md'), 'arms': []}
root = RUNS/'r12_devlooks_50L'
for arm in json.loads((root/'arms.json').read_text()):
    run = root/'runs'/arm['arm']
    args = arm['plugin_args']; fit = Path(args[args.index('--os-fit-artifact')+1])
    with fit.open('rb') as f: method = pickle.load(f)['method']
    base = method.inner.base
    look = next(read_looks(next(run.glob('server_*/looks'))))
    sample = {k: v if isinstance(v, dict) else {'shape': list(v.shape), 'dtype': str(v.dtype)} for k,v in look.items()}
    decisions = [json.loads(s) for s in next(run.glob('server_*/decisions_*.jsonl')).read_text().splitlines()]
    sample_dec = [a for a in decisions if a.get('ev')=='dec'][:5]
    out['arms'].append(dict(arm=arm['arm'], sample=sample, sample_decisions=sample_dec,
        method_type=str(type(method)), guard_type=str(type(method.inner)), base_type=str(type(base)),
        base_fields={k: {'shape':list(v.shape)} if isinstance(v,np.ndarray) else str(type(v)) for k,v in vars(base).items()},
        guard_fields={k: v for k,v in vars(method.inner).items() if isinstance(v,(str,float,int,bool,type(None)))},
        fit_info=method.fit_info, contract=json.loads((run/'pool_contract.json').read_text()),
        journal_first=json.loads((run/'client/journal.jsonl').read_text().splitlines()[0])))
dump(HERE/'dev_inspect.json',out)
print(json.dumps(out,indent=2))
