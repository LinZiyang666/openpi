"""B-pool development selection and fail-closed run/worker contracts.

No fitting, query corpus, closed-loop outcomes, or model inference is used.
The driver still uses run_gtp's digest loader; its historical 'apool' argument
names describe the interface, not the pool selected by this module.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle

REPO = Path(__file__).resolve().parents[3]
STORE = Path('/home/weiland/trace_runs/offline_search_store')
PARENT = {'pi05': 'bpool_cs', 'groot': 'bpool_all'}
ALIASES = {'libero_10': 'l10', 'libero_spatial': 'spatial', 'sp': 'spatial'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def suite_short(suite):
    value = ALIASES.get(suite, suite)
    if value not in ('l10', 'spatial'):
        raise ValueError('unsupported LIBERO suite')
    return value


def suite_long(suite):
    return {'l10': 'libero_10', 'spatial': 'libero_spatial'}[suite_short(suite)]


def pool_for(row):
    dev = row.get('dev', False)
    if type(dev) is not bool:
        raise ValueError('dev must be a JSON boolean')
    pool = 'B' if dev else 'A'
    if row.get('init_pool', pool) != pool:
        raise ValueError('POOL_MISMATCH: dev roots require B; non-dev roots require A')
    # Selection is a root property; environment overrides cannot change it.
    if os.environ.get('OSCL_INIT_POOL', pool) != pool:
        raise ValueError('POOL_MISMATCH: OSCL_INIT_POOL cannot override the run root')
    return pool


def root_pool(root):
    rows = json.loads((Path(root) / 'arms.json').read_text())
    pools = {pool_for(r) for r in rows}
    if len(pools) != 1:
        raise ValueError('POOL_MISMATCH: every arm in a root must use the same pool')
    return pools.pop()


def check_manifest_pool(manifest, pool):
    data = manifest['data']
    if pool == 'B':
        if data.get('init_pool') != 'B' or data.get('dev') is not True:
            raise ValueError('POOL_MISMATCH: dev manifest must explicitly declare dev=true, init_pool=B')
    elif data.get('init_pool', 'A') != 'A' or data.get('dev', False) is not False:
        raise ValueError('POOL_MISMATCH: non-dev manifest cannot select B')


def integer(value, name):
    if type(value) is not int:
        raise ValueError(f'{name} must be an integer, got {value!r}')
    return value


def library_episodes(model, suite, library, store=STORE):
    if model not in PARENT or not isinstance(library, str) or not library or '/' in library or library in ('.', '..'):
        raise ValueError('invalid model/library')
    path = Path(store) / 'library' / f'{model}_{suite_short(suite)}' / library / 'episodes.json'
    # Reject escaped/symlinked sources; only library metadata is eligible.
    if not path.resolve().is_relative_to((Path(store) / 'library').resolve()):
        raise ValueError('library metadata escaped store/library')
    eps = json.loads(path.read_text())
    if any(e.get('orig_init_state_idx') is None for e in eps):
        identity_path = REPO / 'exp/offline_search/rounds/r11/devset/identities' / f'{model}_{suite_short(suite)}_{library}.json'
        if identity_path.is_file():
            identity = json.loads(identity_path.read_text())
            if identity['episodes_sha256'] != sha(path) or identity['rs_sha256'] != sha(path.parent / 'rs.npy'):
                raise ValueError('certified library identities changed; rebuild identity proof')
            evidence = identity['evidence']
            if 'parent_episodes_sha256' in evidence:
                parent = path.parent.with_name(PARENT[model])
                if evidence['parent_episodes_sha256'] != sha(parent / 'episodes.json') or evidence['parent_rs_sha256'] != sha(parent / 'rs.npy'):
                    raise ValueError('certified identity B-parent evidence changed')
            if len(identity['identities']) != len(eps):
                raise ValueError('certified identity count differs from library')
            for idx, e in enumerate(eps):
                entry = identity['identities'][idx]
                if entry['episode_index'] != idx or entry['task_id'] != e['task_id'] or entry['stem'] != e['stem']:
                    raise ValueError('certified identity does not match episode')
                init = integer(entry['orig_init_state_idx'], 'certified orig_init_state_idx')
                if not 0 <= init < 50 or e.get('orig_init_state_idx') not in (None, init):
                    raise ValueError('certified identity conflicts with source metadata')
                e['orig_init_state_idx'] = init
                e['_identity_sha256'] = sha(identity_path)
    ids = {}
    for pos, e in enumerate(eps):
        task = integer(e['task_id'], 'task_id')
        if not 0 <= task < 10:
            raise ValueError('task outside 0..9')
        # IDs in R10 subset specs address episode.npy / metadata position,
        # not the historical episode_id attribute of the collection.
        ids.setdefault(task, []).append(pos)
    return path, eps, ids


def subset_record(model, suite, library, episode_subset=None, store=STORE):
    path, eps, by_task = library_episodes(model, suite, library, store)
    if episode_subset is not None:
        if not isinstance(episode_subset, dict) or set(map(str, episode_subset)) != set(map(str, range(10))):
            raise ValueError('subset must explicitly list all ten tasks')
        by_task = {int(t): list(v) for t, v in episode_subset.items()}
    excluded = {}
    for t in range(10):
        ids = by_task.get(t, [])
        if not ids or len(set(ids)) != len(ids):
            raise ValueError(f'task {t}: empty/duplicate library episode selection')
        values = []
        for idx in ids:
            idx = integer(idx, 'parent episode index')
            if not 0 <= idx < len(eps) or eps[idx]['task_id'] != t:
                raise ValueError(f'task {t}: invalid parent episode index {idx}')
            init = eps[idx].get('orig_init_state_idx')
            if init is None:
                raise ValueError(f'UNKNOWN_LIBRARY_INIT: {path} episode index {idx} has no orig_init_state_idx; dev refused')
            init = integer(init, 'orig_init_state_idx')
            if not 0 <= init < 50:
                raise ValueError('library init outside B-pool 0..49')
            values.append(init)
        excluded[str(t)] = sorted(set(values))
    record = dict(model=model, suite=suite_short(suite), library=library,
                episodes_sha256=sha(path), episode_ids_by_task={str(t): list(by_task[t]) for t in range(10)},
                excluded_inits_by_task=excluded)
    if eps and eps[0].get('_identity_sha256'):
        record['identity_sha256'] = eps[0]['_identity_sha256']
    return record


def r10_subset(model, suite, size, store=STORE):
    path = REPO / 'exp/offline_search/rounds/r10/subsets' / f'{model}_{suite_short(suite)}_{size}.json'
    data = json.loads(path.read_text())
    library = PARENT[model]
    parent = Path(store) / 'library' / f'{model}_{suite_short(suite)}' / library
    if data['model'] != model or suite_short(data['suite']) != suite_short(suite) or data['size'] != size:
        raise ValueError('R10 subset cell/size mismatch')
    if Path(data['parent']).resolve() != parent.resolve() or data['episodes_sha256'] != sha(parent / 'episodes.json'):
        raise ValueError('R10 subset parent metadata changed')
    return subset_record(model, suite, library, data['episode_ids_by_task'], store)


def build_manifest(model, suite, *, r10_size=None, library=None, per_task=5, seed=0, store=STORE):
    import numpy as np
    if (r10_size is None) == (library is None):
        raise ValueError('choose exactly one of r10_size/library')
    if type(per_task) is not int or per_task < 1 or type(seed) is not int or seed < 0:
        raise ValueError('per_task must be positive; seed must be nonnegative')
    subset = r10_subset(model, suite, r10_size, store) if r10_size is not None else subset_record(model, suite, library, store=store)
    pairs = []
    for t in range(10):
        available = sorted(set(range(50)) - set(subset['excluded_inits_by_task'][str(t)]))
        if len(available) < per_task:
            raise ValueError(f'task {t}: only {len(available)} held-out B inits, requested {per_task}')
        rng = np.random.Generator(np.random.PCG64(seed + t))
        pairs.extend([[t, int(i)] for i in sorted(rng.choice(available, per_task, replace=False))])
    return dict(dev=True, init_pool='B', model=model, suite=suite_short(suite),
                seed=seed, per_task=per_task, subset=subset, pairs=pairs)


def require_disjoint(manifest, subset):
    check_manifest_pool(manifest, 'B')
    overlap = sorted((t, i) for t, i in manifest['selected'] if i in subset['excluded_inits_by_task'][str(t)])
    if overlap:
        raise ValueError(f'DEV_LIBRARY_OVERLAP: {overlap}')


def arm_contract(root, row, manifest, store=STORE):
    """Bind the actual prefit's selection, not a user-provided exclusion claim."""
    from .ops.h100.assets import flag
    from .ops.remote.run_gtp_subset import check_manifest
    pool = pool_for(row)
    if ((Path(root) / 'arms.json').exists() or pool == 'B') and root_pool(root) != pool:
        raise ValueError('POOL_MISMATCH: row differs from root')
    if manifest:
        check_manifest_pool(manifest, pool)
        check_manifest(manifest, row['model'], row['suite'])
    if pool == 'A':
        return None
    if manifest is None:
        raise ValueError('dev requires an exact B manifest; Cartesian selection is refused')
    if row.get('mode') != 'plugin':
        raise ValueError('dev requires an attested library-subset plugin prefit')
    path = flag(row.get('plugin_args', []), '--os-fit-artifact')
    if not path:
        raise ValueError('dev requires --os-fit-artifact')
    path = Path(path.replace('<RUN>', str(root)))
    with path.open('rb') as f:
        blob = pickle.load(f)
    if any(blob.get(k) != row.get(k, {}) for k in ('cell', 'kwargs')) or blob.get('spec') != row['method']:
        raise ValueError('dev prefit metadata differs from arm')
    kwargs = row.get('kwargs') or {}
    library = kwargs.get('library')
    method = blob['method']
    info = getattr(method, 'fit_info', None)
    if not library or not isinstance(info, dict) or 'episode_ids_by_task' not in info or not info.get('library_only'):
        raise ValueError('dev prefit must attest library_only and episode_ids_by_task')
    if getattr(method, 'library', None) != library:
        raise ValueError('dev fitted library differs from constructor')
    subset = subset_record(row['model'], row['suite'], library, kwargs.get('episode_subset'), store)
    actual = {str(t): list(v) for t, v in info['episode_ids_by_task'].items()}
    if {t: sorted(v) for t, v in actual.items()} != {t: sorted(v) for t, v in subset['episode_ids_by_task'].items()}:
        raise ValueError('dev fitted episode selection differs from arm')
    require_disjoint(manifest, subset)
    return dict(dev=True, init_pool='B', arm=row['arm'], model=row['model'], suite=suite_long(row['suite']),
                manifest_sha256=manifest['sha256'], subset=subset, fit_sha256=sha(path))


def pure_reference(manifest, store=STORE):
    from .ops.remote.run_gtp_subset import check_manifest
    check_manifest_pool(manifest, 'B')
    data = manifest['data']
    model, suite = data['model'], suite_short(data['suite'])
    check_manifest(manifest, model, suite)
    subset = data.get('subset')
    if not subset or subset['model'] != model or subset['suite'] != suite:
        raise ValueError('dev reference needs matching subset provenance')
    live = subset_record(model, suite, subset['library'], subset['episode_ids_by_task'], store)
    if live != subset:
        raise ValueError('dev subset provenance changed')
    require_disjoint(manifest, live)
    path, eps, _ = library_episodes(model, suite, PARENT[model], store)
    outcomes = {}
    for e in eps:
        p = (integer(e['task_id'], 'task_id'), integer(e['orig_init_state_idx'], 'orig_init_state_idx'))
        if p in outcomes or type(e.get('success')) is not bool:
            raise ValueError('parent must have one boolean outcome per B init')
        outcomes[p] = e['success']
    if set(outcomes) != {(t, i) for t in range(10) for i in range(50)}:
        raise ValueError('parent does not cover all 500 B inits')
    rows = [dict(task_id=t, orig_init_state_idx=i, success=outcomes[t, i]) for t, i in sorted(manifest['selected'])]
    return dict(init_pool='B', dev=True, model=model, suite=suite, manifest_sha256=manifest['sha256'],
                parent_library=PARENT[model], parent_episodes_sha256=sha(path), reference=rows,
                complete=len(rows), success=sum(r['success'] for r in rows),
                sr=sum(r['success'] for r in rows) / len(rows))


def validate_journal_pool(rows, pool):
    for r in rows:
        # Old unlabelled journals can only mean A. New B rows must be explicit.
        if r.get('init_pool', 'A') != pool:
            raise ValueError('POOL_MISMATCH: journal contains a different or unlabelled init pool')


def pool_record(suite, pool='B', repo=REPO):
    import yaml
    path = Path(repo) / 'exp/offline_search/closed_loop/ops/h100' / f'bpool_{suite_long(suite)}.yaml' if pool == 'B' else Path(repo) / 'exp/ablation_study/cache_size/config' / f'apool_{suite_long(suite)}.yaml'
    record = yaml.safe_load(path.read_text())
    if pool == 'B' and (record.get('init_pool') != 'B' or record.get('dev') is not True):
        raise ValueError('POOL_MISMATCH: bad frozen B record')
    return path, record


def main(argv=None):
    import argparse
    from .ops.remote.run_gtp_subset import load_manifest
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest='action', required=True)
    b = sub.add_parser('dev_manifest')
    b.add_argument('--model', choices=PARENT, required=True)
    b.add_argument('--suite', choices=('l10', 'spatial', 'libero_10', 'libero_spatial'), required=True)
    group = b.add_mutually_exclusive_group(required=True)
    group.add_argument('--r10-size', type=int)
    group.add_argument('--library')
    b.add_argument('--per-task', type=int, required=True)
    b.add_argument('--seed', type=int, required=True)
    b.add_argument('--output')
    r = sub.add_parser('reference')
    r.add_argument('--manifest', required=True)
    r.add_argument('--output')
    a = ap.parse_args(argv)
    result = build_manifest(a.model, a.suite, r10_size=a.r10_size, library=a.library, per_task=a.per_task, seed=a.seed) if a.action == 'dev_manifest' else pure_reference(load_manifest(a.manifest))
    content = json.dumps(result, indent=1) + '\n'
    if a.output:
        with Path(a.output).open('x') as f:
            f.write(content)
    else:
        print(content, end='')


if __name__ == '__main__':
    main()
