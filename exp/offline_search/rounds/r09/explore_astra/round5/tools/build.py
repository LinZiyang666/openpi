"""Prepare six frozen arms and safe local metadata. Never launch or sync."""
import argparse
import importlib
import json
import pickle
import shutil
from datetime import datetime, timezone
from types import SimpleNamespace

from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from .methods import StackTakeover
from .safe import HERE, RUN, STORE, ROOTS, EVAL, dump, sha, episode_metadata

MOD = 'exp.offline_search.rounds.r09.explore_astra.round5.tools.methods:StackTakeover'
SAFE_STORE = RUN/'serving_store'


def sources():
    m = importlib.import_module('exp.offline_search.rounds.r09.explore_fable.round3.tools.methods')
    modules = {m.__name__, 'exp.offline_search.rounds.r09.explore_fable.round2.tools.methods',
               'exp.offline_search.rounds.r09.explore_opus.round2.methods',
               'exp.offline_search.rounds.r09.explore_astra.round2.inference',
               'exp.offline_search.rounds.r09.explore_astra.round2b.inference'}
    for cls in [m.NpGraspStack3, m.NpGraspStackGroot3, m.CorrectedCacheJ]:
        modules.update(c.__module__ for c in cls.__mro__ if c.__module__.startswith('exp.'))
    return {n:sha(importlib.import_module(n).__file__) for n in sorted(modules)}


def prepare_store():
    SAFE_STORE.mkdir(parents=True, exist_ok=True)
    link = SAFE_STORE/'library'
    if not link.exists():
        link.symlink_to(STORE/'library', target_is_directory=True)
    for model in ['pi05','groot']:
        cell=model+'_l10_cache'
        eps=list(episode_metadata(STORE/'queries'/cell/'episodes.json'))
        assert {(int(e['task_id']),int(e['init'])) for e in eps} == EVAL
        # Serving only needs task identity. Strip outcome/trajectory metadata.
        identity=[{k:e[k] for k in ('uid','task_id','task','init')} for e in eps]
        dump(SAFE_STORE/'queries'/cell/'episodes.json', identity)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--emit',action='store_true');args=ap.parse_args()
    prediction=HERE/'PREDICTION.md'
    if not prediction.exists():
        raise ValueError('timestamped predictions must precede any arm emission')
    if (RUN/'arms.json').exists():
        raise ValueError('already frozen: use verify/selftest; do not silently regenerate')
    prepare_store()
    source={a['arm']:a for a in json.loads((ROOTS['r09_fable_r3c']/'arms.json').read_text())}
    arms=[]; audit=[]
    pins=sources()
    dump(RUN/'manifests/eval100.json',dict(role='EVAL_ONLY_DISJOINT_FIT_0_19',
         selected=[dict(task=t,init=i) for t,i in sorted(EVAL)]))
    for model in ['pi05','groot']:
        src=source[f'r9f3c_{model}_l10_50_np_corr05'+('_esc' if model=='pi05' else '')]
        cell=src['cell'];prefix=f'r9a5_{model}_l10_50_'
        monitor=HERE.parent/f'round2b/artifacts/{model}_l10_50_monitor.npz'
        phase=monitor.with_name(f'{model}_l10_50_phase.npy')
        meta=json.loads(monitor.with_suffix('.json').read_text())
        assert meta['training_inits']==list(range(20))
        kw=dict(src['kwargs'])
        kw.pop('lag_threshold',None);kw.pop('deadline',None)
        assert kw['max_calls']==0 and kw['force_trigger_at']==[]
        assets={str(p):sha(p) for p in [monitor,phase,kw['onlynp_fit'],kw['corrected_fit']]}
        for mode in ['control','latch','pace12']:
            name=prefix+mode
            dest=RUN/'fits'/f'{name}.pkl';dest.parent.mkdir(parents=True,exist_ok=True)
            if mode=='control':
                original=src['plugin_args'][src['plugin_args'].index('--os-fit-artifact')+1]
                shutil.copyfile(original,dest)
                with open(dest,'rb') as f:blob=FitUnpickler(f).load()
                assert blob['spec']==src['method'] and blob['kwargs']==src['kwargs']
                spec=src['method'];kwargs=src['kwargs'];method=blob['method']
                control_sha=sha(original)
                assert sha(dest)==control_sha
            else:
                kwargs=dict(stack_class='NpGraspStack3' if model=='pi05' else 'NpGraspStackGroot3',
                    stack_kwargs=kw,head_path=str(monitor),phase_path=str(phase),threshold=meta['threshold'],
                    pins=dict(sources=pins,assets=assets),mode=mode,force_at=[])
                method=StackTakeover(**kwargs);method.prof=api.NULL_PROFILER
                method.fit(None,SimpleNamespace(cell=cell,model=model))
                spec=MOD
                blob=dict(method=method,registered={},spec=spec,kwargs=kwargs,cell=cell,fit_s=0.,
                    provenance=dict(fit_inits=list(range(20)),eval_inits=list(range(20,30)),
                        prediction_sha256=sha(prediction),source_pins=pins,assets=assets))
                with dest.open('wb') as f:pickle.dump(blob,f,protocol=4)
            # Identical model/client/topology within each three-arm block.
            plugin=list(src['plugin_args'])
            plugin[plugin.index('--os-fit-artifact')+1]=str(dest)
            plugin[plugin.index('--os-root')+1]=str(SAFE_STORE)
            arms.append(dict(name=name,model=model,suite='l10',mode='plugin',full_model=True,
                method=spec,kwargs=kwargs,plugin_args=plugin,cost_ledger=True,
                client_overrides=src['client_overrides'],manifest=str(RUN/'manifests/eval100.json')))
            audit.append(dict(arm=name,artifact=str(dest),sha256=sha(dest),
                control_source_sha256=control_sha if mode=='control' else None,
                fit_info=method.fit_info,source_arm=src['arm'],mode=mode))
            print(name,'FROZEN',flush=True)
    dump(HERE/'arm_specs.json',arms)
    dump(HERE/'FROZEN.json',dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        prediction_sha256=sha(prediction),fit_inits=list(range(20)),eval_inits=list(range(20,30)),arms=audit))
    if args.emit:
        emit(['--run-root',str(RUN),'--spec',str(HERE/'arm_specs.json')])


if __name__=='__main__':main()
