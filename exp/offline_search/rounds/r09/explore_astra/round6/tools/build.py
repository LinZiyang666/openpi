"""Freeze eight arms with the standard emitter and the standard store. No launch."""
from datetime import datetime, timezone
import importlib
import json
import pickle
import shutil
from types import SimpleNamespace
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from .data import HERE, RUN, STORE, CELLS, control_row, artifact, dump, sha, owned

METHODS = 'exp.offline_search.rounds.r09.explore_astra.round6.tools.methods'


def main():
    if (RUN / 'arms.json').exists():
        raise ValueError('already emitted; do not regenerate frozen arms')
    prediction = HERE / 'PREDICTION.md'
    text = prediction.read_text()
    assert 'UTC' in text and '20–29' in text
    manifest = RUN / 'manifests/eval100.json'
    dump(manifest, dict(role='EVAL_ONLY_DISJOINT_FIT_0_19',
        selected=[dict(task=t, init=i) for t in range(10) for i in range(20, 30)]))
    arms, audit = [], []
    for cell in CELLS:
        row = control_row(cell)
        model, suite, _ = cell.split('_')
        kwargs = row['kwargs']
        assert kwargs['max_calls'] == 0 and kwargs['force_trigger_at'] == []
        for variant in ('control', 'taskfree'):
            name = f'r9a6_{cell}_{variant}'
            dest = owned(RUN / 'fits' / f'{name}.pkl')
            source = artifact(row)
            if variant == 'control':
                shutil.copyfile(source, dest)
                with dest.open('rb') as f:
                    blob = FitUnpickler(f).load()
                spec, kw = row['method'], kwargs
                assert blob['spec'] == spec and blob['kwargs'] == kw
                assert sha(source) == sha(dest)
                method = blob['method']
                control_base = method.base
            else:
                clsname = 'TaskFreeEsc' if cell == 'pi05_l10_50' else (
                    'TaskFreeStack' if model == 'pi05' else 'TaskFreeStackGroot')
                cls = getattr(importlib.import_module(METHODS), clsname)
                kw = dict(kwargs, residual_path=str(HERE / 'artifacts' / f'{cell}.npz'))
                spec = METHODS + ':' + clsname
                method = cls(**kw)
                method.prof = api.NULL_PROFILER
                method.fit(None, SimpleNamespace(cell=row['cell'], model=model))
                assert np.array_equal(method.base.act, control_base.act)
                blob = dict(method=method, registered={}, spec=spec, kwargs=kw, cell=row['cell'], fit_s=0.,
                    provenance=dict(fit_inits=list(range(20)), eval_inits=list(range(20,30)),
                        prediction_sha256=sha(prediction), head_sha256=sha(kw['residual_path'])))
                with dest.open('wb') as f:
                    pickle.dump(blob, f, protocol=4)
            plugin = list(row['plugin_args'])
            plugin[plugin.index('--os-fit-artifact') + 1] = str(dest)
            assert plugin[plugin.index('--os-root') + 1] == str(STORE)
            arms.append(dict(name=name, model=model, suite=suite, mode='plugin', full_model=True,
                method=spec, kwargs=kw, plugin_args=plugin, cost_ledger=True,
                client_overrides=row['client_overrides'], manifest=str(manifest)))
            audit.append(dict(arm=name, artifact=str(dest), sha256=sha(dest), variant=variant,
                source_arm=row['arm'], control_source=str(source), control_source_sha256=sha(source),
                fit_info=method.fit_info))
            print(name, 'FROZEN', flush=True)
    dump(HERE / 'arm_specs.json', arms)
    dump(HERE / 'FROZEN.json', dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        prediction_sha256=sha(prediction), selection_sha256=sha(HERE / 'SELECTION.json'), arms=audit))
    emit(['--run-root', str(RUN), '--spec', str(HERE / 'arm_specs.json')])


if __name__ == '__main__':
    main()
