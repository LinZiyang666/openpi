"""Fit only from deployed banks/fits. All writes stay in the assigned output root."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import time

import numpy as np

from exp.offline_search.harness import store
from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel, _digest

HERE=Path(__file__).resolve().parent
OUT=Path('/tmp/q3_stall_fits')
ROOT=Path('/home/weiland/trace_runs/offline_search_store')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while data:=f.read(8<<20):h.update(data)
    return h.hexdigest()


def load_source(cell):
    source=json.loads((HERE/'source_manifest.json').read_text())[cell]
    cls,_=load_method_class(source['source_spec']);cls(**source['source_kwargs'])
    with open(source['source_artifact'],'rb') as f:blob=pickle.load(f)
    assert blob['spec']==source['source_spec'] and blob['kwargs']==source['source_kwargs']
    awm=blob['method']
    lib=store.LibraryView(ROOT,store.lib_key(blob['cell']),awm.cand_name)
    assert np.array_equal(awm.lib_ep,lib.episode) and np.array_equal(awm.lib_step,lib.step)
    return source,blob,awm,lib


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--cell',required=True)
    ap.add_argument('--verify-reproducible',action='store_true')
    args=ap.parse_args();start=time.perf_counter()
    source,blob,awm,lib=load_source(args.cell)
    meta=lib.meta;L=min(int(meta['H']),2*int(meta['exec_steps']))
    model=StallModel.fit(lib,metric=awm,commit_controls=L)
    # Hash actual files used by the deployed projection/bank, not a cell name.
    needed=['manifest.json','task_id.npy','episode.npy','step.npy','ep_len.npy','success.npy','rs.npy']
    needed += sorted(p.name for p in lib.dir.glob('key_v*.npy'))
    provenance=dict(cell=args.cell,bank=str(lib.dir),bank_files={k:sha(lib.dir/k) for k in needed},
                    source_fit=source['source_artifact'],source_fit_sha256=sha(source['source_artifact']),
                    source_spec=source['source_spec'],source_kwargs=source['source_kwargs'])
    model.provenance['source']=provenance
    model.fingerprint=_digest(model._payload())
    path=OUT/args.cell
    if args.verify_reproducible:
        old=StallModel.load(path)
        assert old.fingerprint==model.fingerprint,(args.cell,old.fingerprint,model.fingerprint)
        report=dict(cell=args.cell,status='INDEPENDENT_REFIT_IDENTICAL',fingerprint=model.fingerprint,
                    seconds=time.perf_counter()-start)
        (HERE/('refit_'+args.cell+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    else:
        model.save(path)
        loaded=StallModel.load(path)
        assert loaded.fingerprint==model.fingerprint
        report=dict(cell=args.cell,status='FITTED_AND_RELOADED',fingerprint=model.fingerprint,
            seconds=time.perf_counter()-start,artifact=str(path),bytes=(path/'stall.pkl').stat().st_size,
            tasks={k:dict(E=d['E'],W=d['W'],K=d['K'],windows=len(d['calibration']),
                         references=len(d['references'])) for k,d in model.tasks.items()})
        (HERE/('fit_'+args.cell+'.json')).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
