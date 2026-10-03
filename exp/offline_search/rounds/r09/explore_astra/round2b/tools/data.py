"""Identity-first admission. Never enumerate raw mixed-init episodes/journals.

The existing compact source was filtered to 0..29. Validate identity members
BEFORE loading trajectory/outcome members, and apply an explicit 0..29 filter.
Raw control files are addressed ONLY through these admitted decision IDs.
All fitted statistics must subsequently use 0..19, evaluation uses 20..29.
"""
import json
import hashlib
from pathlib import Path
import numpy as np
from ...round2.tools.data import (COMPACT, RUN, STORE, dump, sha, validate_identity,
    masks, subset, balanced_weights, epmeans, interval)

HERE = Path(__file__).resolve().parents[1]
DERIVED = STORE/'derived/r09_astra/round2b'
CELLS = [f'{m}_l10_{n}' for m in ['pi05','groot'] for n in [50,500]]


def source(cell, variant):
    if cell not in CELLS or variant not in ['A','CU','IP']:
        raise ValueError('Population is not allowlisted')
    path = COMPACT/f'r8_{cell}_{variant}.npz'
    if not path.exists(): return None
    meta = json.loads(path.with_suffix('.json').read_text())
    if meta['episodes'] != 300 or 'inits 0..29' not in meta['split']:
        raise ValueError('Discovery-only provenance missing')
    with np.load(path,allow_pickle=False) as z:
        task, init = z['task'], z['init']
        validate_identity(task,init)  # firewall BEFORE payload access
        admitted = (init>=0)&(init<30)
        vision = z['vision'][admitted]
        names = ['seq','decision_id','state_norm','shadow_look_rows','shadow_look_weights',
            'shadow_look_cache_chunk','shadow_look_keys_pca_third','shadow_look_keys_pca_wrist',
            'policy_shadow_chunk','d1_rel','dst','eligible','treatment','p','coin']
        d = {k:z[k][admitted][vision] for k in names}
        d['task'],d['init'] = task[admitted][vision],init[admitted][vision]
        for k in ['success','cost','decisions']:
            if z[k].shape!=(10,30): raise ValueError('Non-discovery ledger rejected')
        d['success'] = z['success'][:, :30][d['task'],d['init']]
        d['cost'] = z['cost'][:, :30][d['task'],d['init']]
        d['decisions'] = z['decisions'][:, :30][d['task'],d['init']]
    d['episode'] = d['task'].astype(int)*30+d['init']
    d['base'] = d.pop('shadow_look_cache_chunk')
    d['teacher'] = d.pop('policy_shadow_chunk')
    d['visual'] = np.c_[d.pop('shadow_look_keys_pca_third'),d.pop('shadow_look_keys_pca_wrist')]
    d['rows'],d['weights'] = d.pop('shadow_look_rows'),d.pop('shadow_look_weights')
    d['state'] = d.pop('state_norm')
    for k in ['base','teacher','visual','state','weights']:
        if not np.isfinite(d[k]).all(): raise ValueError(f'Missing {k}')
    return d


def library(cell):
    model,suite,size=cell.split('_')
    name='current' if size=='50' else ('bpool_cs' if model=='pi05' else 'bpool_all')
    return STORE/'library'/f'{model}_{suite}'/name


def safe_episode(cell,variant,task,init,decision_ids):
    validate_identity(np.array([task]),np.array([init]))
    if cell not in CELLS or variant not in ['A','CU','IP']: raise ValueError('population')
    keys={str(v).split(':')[0] for v in decision_ids}
    if len(keys)!=1: raise ValueError('conflicting accepted episode')
    arm=f'r8_{cell}_{variant}'
    directory=RUN/'runs'/arm/'debug/client'/keys.pop()
    meta=json.loads((directory/'episode.json').read_text())
    if meta['task_uid']!=f'{arm}:eval:{task}:{init}': raise ValueError('identity mismatch')
    # Every control file is single-episode and identity has already been admitted.
    chunks=[];sources=[directory/'episode.json']
    for path in sorted(directory.glob('controls_*.npz')):
        sources.append(path)
        with np.load(path,allow_pickle=False) as z:
            chunks.append({k:z[k] for k in ['predicates','decision_seq','is_settle']})
    out={k:np.concatenate([c[k] for c in chunks]) for k in chunks[0]}
    if len(out['predicates'])!=meta['n_controls']: raise ValueError('incomplete controls')
    meta['source_stat_sha256']=hashlib.sha256(json.dumps([
        (str(p),p.stat().st_size,p.stat().st_mtime_ns) for p in sources]).encode()).hexdigest()
    return meta,out


def load(cell,variant):
    d=source(cell,variant)
    if d is None:return None
    path=DERIVED/f'{cell}_{variant}_labels.npz'
    with np.load(path,allow_pickle=False) as z:
        validate_identity(z['task'],z['init'])  # before labels
        admitted=(z['init']>=0)&(z['init']<30)
        if not np.array_equal(z['decision_id'][admitted],d['decision_id']):raise ValueError('label join mismatch')
        d.update({k:z[k][admitted] for k in z.files if k not in ['task','init','decision_id']})
    return d
