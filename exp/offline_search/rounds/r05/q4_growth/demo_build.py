"""Immutable, nested balanced episode subsets of the already validated H5 store."""
import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
from exp.offline_search.harness import store
from exp.offline_search.rounds.r05.q4_growth.common import OUT,COLD,SHM,FIELDS,sha256
from exp.offline_search.rounds.r05.q4_growth.build import topology

RULE='q4_demo_nested_v1'
RESULTS=OUT/'results'/'demo'

def order_episodes(lib):
    per_task={}
    for t in range(10):
        eps=[(i,e) for i,e in enumerate(lib.episodes) if e['task_id']==t]
        assert len(eps)==50
        per_task[t]=sorted(eps,key=lambda ie: (hashlib.sha256(
            f'{RULE}|{lib.suite}|{t}|{ie[1]["stem"]}'.encode()).hexdigest(),ie[1]['stem']))
    return [per_task[t][rank] for rank in range(50) for t in range(10)]

def episode_digest(lib,e):
    h=hashlib.sha256(str(e['task_id']).encode())
    for f in ('key_v0','key_v1','rs','action','step'):
        a=getattr(lib,f)[e['start']:e['end']]
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()

def build(suite):
    RESULTS.mkdir(exist_ok=True)
    B=store.LibraryView(COLD,'pi05_'+suite,'bpool_cs');C=store.LibraryView(COLD,B.key)
    ordering=order_episodes(B)
    for root in (COLD,SHM):
        for n in (100,200,300):
            d=root/'library'/B.key/f'demo{n}'
            assert not d.exists() and not d.with_suffix('.partial').exists(),d
    estimate=sum(sum(e['n_rows'] for _,e in ordering[:n])*(2*32768*4+32*4+10*32*4+100)+4*1024**2 for n in (100,200,300))
    free=shutil.disk_usage(SHM).free
    assert free>estimate+2*1024**3,(free,estimate)
    print(json.dumps(dict(suite=suite,shm_free_before_bytes=free,estimated_all_sizes_bytes=estimate)),flush=True)
    bd={e['file']:episode_digest(B,e) for e in B.episodes}
    cd={e['file']:episode_digest(C,e) for e in C.episodes}
    overlap=dict(current_episodes=len(C.episodes),current_rows=C.L,bpool_episodes=len(B.episodes),bpool_rows=B.L,
                 shared_files=sorted(set(bd)&set(cd)),shared_ids=len(set(C.ids)&set(B.ids)),
                 exact_full_episode_payload_matches=len(set(cd.values())&set(bd.values())),
                 current_contained=False)
    assert not overlap['shared_files'] and overlap['shared_ids']==0 and overlap['exact_full_episode_payload_matches']==0
    sources={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in
             [*(B.dir/f'{f}.npy' for f in FIELDS),B.dir/'manifest.json',B.dir/'episodes.json',B.dir/'ids.json']}
    results=[]
    for n in (100,200,300):
        name=f'demo{n}';chosen=ordering[:n]
        rows=np.concatenate([np.arange(e['start'],e['end']) for _,e in chosen])
        eid=np.concatenate([np.full(e['n_rows'],j,np.int32) for j,(_,e) in enumerate(chosen)])
        work=COLD/'library'/B.key/(name+'.partial');work.mkdir()
        for f in FIELDS:
            if f in ('episode','next','prev'):continue
            src=getattr(B,f)
            out=np.lib.format.open_memmap(work/f'{f}.npy',mode='w+',dtype=src.dtype,shape=(len(rows),*src.shape[1:]))
            for lo in range(0,len(rows),128):out[lo:lo+len(rows[lo:lo+128])]=src[rows[lo:lo+128]]
            out.flush();del out
        top=topology(eid,np.asarray(B.step[rows]),np.asarray(B.ep_len[rows]))
        for f,a in dict(episode=eid,source_bpool_row=rows,source_bpool_episode=B.episode[rows],**top).items():
            np.save(work/f'{f}.npy',a)
        ids=[B.ids[r] for r in rows];eps=[];at=0
        for source_i,e in chosen:
            eps.append(dict(e,start=at,end=at+e['n_rows'],source_episode=source_i,source_start=e['start'],source_end=e['end']))
            at+=e['n_rows']
        (work/'ids.json').write_text(json.dumps(ids))
        (work/'episodes.json').write_text(json.dumps(eps,indent=1))
        prov=dict(label=f'{n} B-pool demonstration-collection episodes',source_library='bpool_cs',
                  source_dir=str(B.dir),source_H5_batch=B.meta['batch'],source_hashes=sources,
                  selection_rule=f'Within each task sort SHA256(UTF-8 "{RULE}|<suite>|<task_id>|<stem>") ascending, stem tie-break; select first N/10; row order (rank,task,step)',
                  outcome_filter='none; keep successful and failed selected episodes',sizes_nested=[100,200,300,500],
                  current_overlap=overlap,source_episode_digests={e['file']:bd[e['file']] for _,e in chosen},
                  selected_source_episodes=[i for i,_ in chosen],selected_episode_files=[e['file'] for _,e in chosen],
                  query_file_overlap={arm:len({e['file'] for _,e in chosen}&{e['file'] for e in store.QueryCell(COLD,B.key+'_'+arm).episodes}) for arm in ('inf','cache')},
                  no_query_arrays_used=True,initial_shm_free_bytes=free,estimated_all_sizes_bytes=estimate)
        assert all(v==0 for v in prov['query_file_overlap'].values())
        (work/'provenance.json').write_text(json.dumps(prov,indent=1))
        arrays={}
        for p in work.glob('*.npy'):
            a=np.load(p,mmap_mode='r');arrays[p.name]=dict(shape=list(a.shape),dtype=str(a.dtype),bytes=p.stat().st_size)
        meta=dict(B.meta)
        meta.update(name=name,complete=True,tok_complete=False,tok_episodes=[],
                    counts=dict(rows=len(rows),episodes=n,tasks=10,tok_rows=0,tok_episodes=0,success_episodes=sum(e['success'] for _,e in chosen)),
                    arrays=arrays,bytes_total=sum(a['bytes'] for a in arrays.values()),
                    sources=dict(library='bpool_cs',provenance='provenance.json'),
                    key_source='bit-exact selected full-episode rows from bpool_cs; no query data',
                    episode_order='SHA256 rank then task, then actual step; prefix-stable demo100->demo200->demo300',
                    verification=dict(source_rows_copied=True,current_overlap=overlap),
                    build=dict(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        (work/'manifest.json').write_text(json.dumps(meta,indent=1))
        checksums={p.name:dict(bytes=p.stat().st_size,sha256=sha256(p)) for p in sorted(work.iterdir())}
        (work/'checksums.json').write_text(json.dumps(checksums,indent=1))
        cold=work.with_name(name);work.rename(cold)
        actual=sum(p.stat().st_size for p in cold.iterdir());free_copy=shutil.disk_usage(SHM).free
        assert free_copy>actual+2*1024**3
        tmp=SHM/'library'/B.key/(name+'.partial');shutil.copytree(cold,tmp);tmp.rename(tmp.with_name(name))
        rec=dict(suite=suite,library=name,rows=len(rows),episodes=n,success_episodes=meta['counts']['success_episodes'],
                 episodes_per_task=n//10,bytes_per_copy=actual,shm_free_before_copy=free_copy,
                 shm_free_after_copy=shutil.disk_usage(SHM).free,manifest_sha256=sha256(cold/'manifest.json'),
                 checksums_sha256=sha256(cold/'checksums.json'))
        results.append(rec);print(json.dumps(rec),flush=True)
    (RESULTS/f'build_{suite}.json').write_text(json.dumps(dict(overlap=overlap,initial_shm_free_bytes=free,estimated_bytes=estimate,libraries=results),indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--suite',required=True,choices=['l10','spatial']);a=p.parse_args();build(a.suite)
