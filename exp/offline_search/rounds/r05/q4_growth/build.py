"""Publish successful acquisition episodes as a new library in both stores.

Never updates an existing library. No raw tokens are needed by AWM; pooled keys
and complete policy chunks are copied in the existing library.v1 layout.
"""
import argparse
import datetime
import json
import shutil
import time
from pathlib import Path
import numpy as np
from exp.offline_search.harness import store
from exp.offline_search.r0.build_library_store import arrays_info, check
from exp.offline_search.rounds.r05.q4_growth.common import (
    COLD, SHM, OUT, FIELDS, base_artifact, load_base, features, fingerprint, sha256,
)


def topology(episode, step, ep_len):
    """No bridge across omitted decisions; missing next never defines terminal."""
    index = {(int(e), int(s)): i for i, (e, s) in enumerate(zip(episode, step))}
    prev = np.array([index.get((int(e), int(s)-1), -1) for e, s in zip(episode, step)], np.int32)
    nxt = np.array([index.get((int(e), int(s)+1), -1) for e, s in zip(episode, step)], np.int32)
    terminal = step == ep_len - 1
    return dict(prev=prev, next=nxt, next_valid=nxt >= 0,
                terminal_known=np.ones(len(step), bool), is_terminal=terminal)


def build(suite, name):
    start = time.time()
    key = f'pi05_{suite}'
    L, Q = store.LibraryView(COLD, key), store.QueryCell(COLD, key + '_inf')
    destinations = [root / 'library' / key / name for root in (COLD, SHM)]
    for p in destinations:
        check(not p.exists() and not p.with_name(name + '.partial').exists(), f'refusing existing destination {p}')
    M = load_base(suite)
    eps = sorted([(i, e) for i, e in enumerate(Q.episodes) if 0 <= e['init'] < 25],
                 key=lambda ie: (ie[1]['init'], ie[1]['task_id']))
    check({(e['task_id'], e['init']) for _, e in eps} == {(t, i) for t in range(10) for i in range(25)}, 'acquisition split incomplete')
    check(len(eps) == 250, 'duplicate acquisition pairs')
    successful = [(i, e) for i, e in eps if e['success']]
    source_rows = np.concatenate([np.arange(e['start'], e['end']) for _, e in successful])
    check(np.array_equal(Q.a_inf[source_rows, :, :7], Q.a_exec[source_rows, :, :7]), 'source is not policy execution')
    Xb, Xs = features(M, L), features(M, Q, source_rows)
    seen = {}
    for r in range(L.L):
        seen.setdefault(fingerprint(L.task_id[r], Xb[r], L.action[r]), r)
    rows, added_ids, added_ep, ep_records, duplicates = [], [], [], [], []
    offset = 0
    episodes = list(L.episodes)
    acquisition = []
    for qi, e in eps:
        rec = dict(e, source_episode_index=qi, policy_rows=e['end']-e['start'], admitted_rows=0, duplicate_rows=0)
        if e['success']:
            check(e['num_steps'] == e['end']-e['start'], 'source has missing recorded steps')
            check(np.array_equal(Q.step[e['start']:e['end']], np.arange(e['num_steps'])), 'nonconsecutive source steps')
            new_eid = len(episodes)
            begin = L.L + len(rows)
            for r in range(e['start'], e['end']):
                fp = fingerprint(e['task_id'], Xs[offset], Q.a_inf[r])
                offset += 1
                if fp in seen:
                    old = seen[fp]
                    # Confirm actual equality on a hash hit, not hash-only admission.
                    ox = Xb[old] if old < L.L else Xs[source_pos[old - L.L]]
                    oa = L.action[old] if old < L.L else Q.a_inf[rows[old - L.L]]
                    check(np.array_equal(ox, Xs[offset-1]) and np.array_equal(oa[:, :7], Q.a_inf[r, :, :7]), 'hash collision')
                    duplicates.append(dict(source_row=r, retained_row=old, uid=e['uid'], step=int(Q.step[r])))
                    rec['duplicate_rows'] += 1
                    continue
                seen[fp] = L.L + len(rows)
                if not rows:
                    source_pos = []
                source_pos.append(offset-1)
                rows.append(r)
                added_ep.append(new_eid)
                added_ids.append(f'growth:{key}:{e["uid"]}:{int(Q.step[r])}')
                rec['admitted_rows'] += 1
            end = L.L + len(rows)
            # Keep even an all-duplicate episode in provenance, never invent rows.
            if end > begin:
                episodes.append(dict(e, stem=f'growth:{e["uid"]}', start=begin, end=end,
                    n_rows=end-begin, contiguous=True, n_step_groups=e['num_steps'],
                    source_episode_index=qi, source_uid=e['uid'], source_start=e['start'], source_end=e['end'],
                    orig_init_state_idx=e['init'], source_cell=Q.cell))
        acquisition.append(rec)
    rows = np.array(rows, np.int64)
    total = L.L + len(rows)
    # Includes arrays, JSON, and generous headroom; evaluated immediately before writes.
    estimate = total * (2*32768*4 + 32*4 + 10*32*4 + 100) + 8*1024*1024
    free = shutil.disk_usage(SHM).free
    check(free >= estimate + 2*1024**3, f'insufficient shm space: {free} available, {estimate} estimate')
    print(json.dumps(dict(suite=suite,shm_free_before=free,estimated_bytes=estimate,base_rows=L.L,added_rows=len(rows))), flush=True)
    work = destinations[0].with_name(name + '.partial')
    work.mkdir()
    for field in ('key_v0', 'key_v1', 'rs', 'action'):
        src = Q.a_inf if field == 'action' else getattr(Q, field)
        base = getattr(L, field)
        a = np.lib.format.open_memmap(work / f'{field}.npy', mode='w+', dtype=base.dtype, shape=(total, *base.shape[1:]))
        a[:L.L] = base
        for lo in range(0, len(rows), 128):
            a[L.L+lo:L.L+lo+len(rows[lo:lo+128])] = src[rows[lo:lo+128]]
        a.flush()
        del a
    qep = np.asarray(Q.ep[rows], np.int64)
    lens = np.array([Q.episodes[i]['num_steps'] for i in qep], L.ep_len.dtype)
    steps = np.asarray(Q.step[rows], L.step.dtype)
    small = dict(task_id=np.array([Q.episodes[i]['task_id'] for i in qep], L.task_id.dtype),
                 episode=np.array(added_ep, L.episode.dtype), step=steps, ep_len=lens,
                 progress=(steps / np.maximum(lens.astype(np.float64)-1, 1)).astype(np.float32),
                 success=np.ones(len(rows), bool))
    a = {f: np.concatenate((getattr(L, f), v)) for f, v in small.items()}
    topo = topology(a['episode'], a['step'], a['ep_len'])
    # Existing metadata and edges retain their old semantics byte for byte.
    topo['prev'][:L.L] = L.prev
    topo['next'][:L.L] = L.next
    topo['next_valid'][:L.L] = L.next >= 0
    a.update(topo)
    a['source_query_row'] = np.r_[np.full(L.L, -1, np.int64), rows]
    a['source_query_episode'] = np.r_[np.full(L.L, -1, np.int32), qep.astype(np.int32)]
    a['is_growth'] = np.arange(total) >= L.L
    for f, v in a.items():
        np.save(work / f'{f}.npy', v)
    (work / 'ids.json').write_text(json.dumps(L.ids + added_ids))
    (work / 'episodes.json').write_text(json.dumps(episodes, indent=1))
    inputs = {str(p.relative_to(COLD)): dict(bytes=p.stat().st_size, sha256=sha256(p))
              for p in [*(L.dir / f'{f}.npy' for f in FIELDS), L.dir/'ids.json', L.dir/'episodes.json',
                        L.dir/'manifest.json', Q.dir/'episodes.json',
                        *(Q.dir / f'{f}.npy' for f in ('ep','step','key_v0','key_v1','rs','a_inf','a_exec'))]}
    prov = dict(label='50 + 250 policy episodes', nominal_base_episodes=50, actual_base_episodes=len(L.episodes),
                base_rows=L.L, acquisition_episodes=250, successful_acquisition_episodes=len(successful),
                paid_policy_calls=sum(r['policy_rows'] for r in acquisition), successful_source_rows=len(source_rows),
                admitted_rows=len(rows), exact_duplicates=len(duplicates), final_rows=total,
                final_episodes=len(episodes), acquisition_init_range=[0,24], evaluation_init_range=[25,49],
                admission='successful episodes only; exact task + frozen PCA64x2 + valid state8 + full Hx7 action equality',
                row_order='base unchanged; appended (init, task_id, actual step)',
                missing_edge='unknown continuation; never bridge gaps or infer terminal from next=-1',
                terminal_semantics='terminal_known from complete episode length; is_terminal iff step==ep_len-1',
                source_fit=dict(path=str(base_artifact(suite)),sha256=sha256(base_artifact(suite))),
                source_inputs=inputs, acquisition=acquisition, duplicates=duplicates,
                shm_free_before_bytes=free, estimated_bytes=estimate,
                tokens='not copied; AWM consumes pooled keys, state and full chunks',
                source_big_file_overlap=len({e['file'] for _,e in eps} & {e['file'] for e in store.LibraryView(COLD,key,'bpool_cs').episodes}))
    (work/'provenance.json').write_text(json.dumps(prov,indent=1))
    meta = dict(L.meta)
    meta.update(name=name, complete=True, tok_complete=False, tok_episodes=[],
                key_source='base rows copied unchanged; appended pooled keys/rs/a_inf from successful recorded policy episodes',
                counts=dict(rows=total, episodes=len(episodes), tok_rows=0, tok_episodes=0,
                            success_episodes=sum(e['success'] for e in episodes), tasks=10),
                sources=dict(base_library='current',query_cell=Q.cell,provenance='provenance.json'),
                id_scheme='base IDs unchanged; growth:<lib_key>:<source_uid>:<actual_step>',
                episode_order=prov['row_order'], batch=dict(label=prov['label']),
                verification=dict(base_prefix_unchanged=True, heldout_rows_admitted=0, exact_duplicates=len(duplicates)),
                build=dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_s=time.time()-start),
                arrays=arrays_info(work))
    # Include additive topology/provenance arrays in manifest too.
    for p in work.glob('*.npy'):
        v=np.load(p,mmap_mode='r')
        meta['arrays'][p.name]=dict(shape=list(v.shape),dtype=str(v.dtype),bytes=p.stat().st_size)
    meta['bytes_total']=sum(v['bytes'] for v in meta['arrays'].values())
    (work/'manifest.json').write_text(json.dumps(meta,indent=1))
    hashes={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(work.iterdir())}
    (work/'checksums.json').write_text(json.dumps(hashes,indent=1))
    work.rename(destinations[0])
    free_copy=shutil.disk_usage(SHM).free
    actual=sum(p.stat().st_size for p in destinations[0].iterdir())
    check(free_copy >= actual + 2*1024**3, 'shm free space changed before copy')
    tmp=destinations[1].with_name(name+'.partial')
    shutil.copytree(destinations[0],tmp)
    tmp.rename(destinations[1])
    result={k:v for k,v in prov.items() if k not in ('source_inputs','acquisition','duplicates')}
    result.update(suite=suite,library=name,bytes_per_copy=actual,shm_free_before_copy_bytes=free_copy,
                  shm_free_after_bytes=shutil.disk_usage(SHM).free,paths=[str(p) for p in destinations],wall_s=time.time()-start)
    (OUT/'results'/f'build_{suite}.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',choices=['l10','spatial'],required=True);p.add_argument('--name',default='grow250')
    a=p.parse_args();build(a.suite,a.name)
