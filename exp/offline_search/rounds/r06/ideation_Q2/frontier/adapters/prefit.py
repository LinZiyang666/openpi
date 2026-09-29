"""CPU-only prefits, written exclusively under /tmp/q2_adapter_fits.

Reuses an audited deployed fit when only serving/randomization changes. This
avoids refitting A/B differently merely because an inference budget changed.
No PluginRuntime/install/main, server, git metadata, or run root is invoked.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import pickle
import time

from exp.offline_search.harness import api, store
from exp.offline_search.closed_loop.plugin import load_method_class
from .budget import digest
from .methods import CacheDose, RiskLottery, ExtraDosePi05, ExtraDoseGroot

HERE = Path(__file__).resolve().parent
OUT = Path('/tmp/q2_adapter_fits')
ROOT = Path('/home/weiland/trace_runs/offline_search_store')


def load_source(row):
    cls, _ = load_method_class(row['source_spec'])
    cls(**row['source_kwargs'])  # resolves nested file-module classes before unpickling
    path = Path(row['source_artifact'])
    with path.open('rb') as f:
        blob = pickle.load(f)
    if blob['spec'] != row['source_spec'] or blob['kwargs'] != row['source_kwargs']:
        raise ValueError('source fitted artifact differs from audited emitted spec')
    return blob, path


def transplant(new, old):
    """Retain fit arrays/memos, restore only newly declared serving settings."""
    before = dict(new.__dict__)
    new.__dict__.update(old.__dict__)
    for k, v in before.items():
        if k.startswith('q2_') or k.startswith('_policy_gate_anchor'):
            new.__dict__[k] = v
    return new


def fit_spec(spec, source, *, fresh=False):
    cell = f'{spec["model"]}_{spec["suite"]}_cache'
    cls, class_path = load_method_class(spec['method'])
    method = cls(**spec['kwargs'])
    scratch = OUT/'scratch'/spec['name']
    scratch.mkdir(parents=True, exist_ok=True)
    ctx = api.Context(root=ROOT, cell=cell, seed=0, scratch=scratch)
    lib = store.LibraryView(ROOT, store.lib_key(cell), 'current')
    provenance = dict(source='fresh CPU fit', method_source=class_path, method_sha256=digest(class_path))
    if not fresh and source.get('source_artifact') and Path(source['source_artifact']).exists():
        blob, path = load_source(source)
        if blob['cell'] != cell:
            raise ValueError('source fit cell mismatch')
        old = blob['method']
        if isinstance(method, (ExtraDosePi05, ExtraDoseGroot)):
            expected = {k:v for k,v in spec['kwargs'].items() if k not in ('dose','random_seed','randomization_key')}
            if expected != source['source_kwargs']:
                raise ValueError('extra-dose baseline kwargs must exactly equal deployed B')
            method = transplant(method, old)
        elif isinstance(method, CacheDose):
            extra = {'dose','random_seed','randomization_key','rho','allocation','calibration_path','calibration_sha256'}
            expected = {k:v for k,v in spec['kwargs'].items() if k not in extra}
            if expected != source['source_kwargs']:
                raise ValueError('cache adapter baseline kwargs must exactly equal deployed A')
            method = transplant(method, old)
            method.finish_adapter_fit(lib, ctx)
        elif spec['kwargs'] == source['source_kwargs'] and spec['method'] == source['source_spec']:
            method = old
        else:
            changed = {k for k in set(spec['kwargs']) | set(source['source_kwargs'])
                       if spec['kwargs'].get(k) != source['source_kwargs'].get(k)}
            allowed = {'budget'} if cls.__name__ == 'BlindAWM' else {'cycle_k','tail_blocks'} if cls.__name__ == 'CycleTail' else set()
            if spec['method'] != source['source_spec'] or not changed <= allowed:
                raise ValueError('unapproved fit reuse; use --fresh: '+str(changed))
            configured = {k:copy.deepcopy(getattr(method,k)) for k in ('name','budget','cycle_k','tail_blocks') if hasattr(method,k)}
            method.__dict__.update(old.__dict__)
            method.__dict__.update(configured)
            # These attributes are serving-only: BlindAWM._fit_blind and CycleTail.fit do not use them.
        provenance.update(source='reused deployed fit; randomization/serving configuration only',
            source_artifact=str(path), source_sha256=digest(path), source_arm=source['source_arm'])
        registered = blob.get('registered', {})
    else:
        method.fit(lib, ctx)
        registered = ctx.registered
    method.prof = api.NULL_PROFILER
    api.check_method_attrs(method)
    return method, registered, cell, provenance


def write_fit(spec, source, *, fresh=False):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/(spec['name']+'.pkl')
    note_path = OUT/(spec['name']+'.json')
    want = dict(spec=spec['method'], kwargs=spec['kwargs'], cell=f'{spec["model"]}_{spec["suite"]}_cache')
    if path.exists():
        if not note_path.exists():
            raise ValueError('existing fit has no provenance sidecar; choose another name')
        note = json.loads(note_path.read_text())
        if note['identity'] != want or note['sha256'] != digest(path):
            raise ValueError('existing artifact identity/digest differs; refuses overwrite')
        print(json.dumps(dict(name=spec['name'], status='EXISTING_VERIFIED', artifact=str(path))), flush=True)
        return note
    t0 = time.monotonic()
    method, registered, cell, provenance = fit_spec(spec, source, fresh=fresh)
    blob = dict(method=method, registered=registered, **want, fit_s=time.monotonic()-t0, q2_provenance=provenance)
    temporary = path.with_suffix('.pkl.tmp')
    with temporary.open('xb') as f:
        pickle.dump(blob, f, protocol=4)
    os.replace(temporary, path)
    note = dict(name=spec['name'], artifact=str(path), bytes=path.stat().st_size, sha256=digest(path),
                identity=want, provenance=provenance, fit_s=blob['fit_s'], budget=getattr(method,'q2_budget',None))
    note_path.write_text(json.dumps(note, indent=2)+'\n')
    print(json.dumps(dict(name=spec['name'], artifact=str(path), bytes=note['bytes'], seconds=round(note['fit_s'],3),
                         budget=note['budget'] and {k:v for k,v in note['budget'].items() if k != 'tasks'})), flush=True)
    return note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--fresh', action='store_true', help='fit library anew instead of reusing the deployed fit')
    args = ap.parse_args()
    specs = json.loads((HERE/'emit_arms_all43.json').read_text())
    spec = next(s for s in specs if s['name'] == args.name)
    source = json.loads((HERE/'fit_sources.json').read_text())[args.name]
    write_fit(spec, source, fresh=args.fresh)


if __name__ == '__main__':
    main()
