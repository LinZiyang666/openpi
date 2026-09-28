"""Existing loader/payload validators plus growth provenance/topology checks."""
import argparse
import json
import threading
from types import SimpleNamespace
import numpy as np
from exp.offline_search.harness import api, store, gates
from exp.offline_search.closed_loop import plugin
from exp.offline_search.rounds.r05.q4_growth.common import COLD, SHM, OUT, FIELDS, sha256
from exp.offline_search.rounds.r05.q4_growth.build import topology


def validate(root, suite):
    key=f'pi05_{suite}'
    L=store.LibraryView(root,key);G=store.LibraryView(root,key,'grow250');Q=store.QueryCell(root,key+'_inf')
    assert 'grow250' in store.library_names(root,key)
    assert G.ids[:L.L] == L.ids and len(set(G.ids)) == G.L
    hashes=json.loads((G.dir/'checksums.json').read_text())
    for name, h in hashes.items():
        assert (G.dir/name).stat().st_size == h['bytes']
        assert sha256(G.dir/name) == h['sha256'], name
    for f in FIELDS:
        a=getattr(G,f);b=getattr(L,f)
        assert a.dtype == b.dtype and a.shape == (G.L,*b.shape[1:])
        for lo in range(0,L.L,128):
            assert np.array_equal(a[lo:min(lo+128,L.L)],b[lo:lo+128]), f
        if a.dtype.kind == 'f':
            for lo in range(0,G.L,128):
                assert np.isfinite(a[lo:lo+128]).all(), f
    src=np.asarray(G.source_query_row[L.L:]);ep=np.asarray(G.source_query_episode[L.L:])
    assert np.array_equal(Q.ep[src],ep)
    assert all(Q.episodes[i]['success'] and 0 <= Q.episodes[i]['init'] <= 24 for i in ep)
    assert np.array_equal(G.step[L.L:],Q.step[src])
    for f in ('key_v0','key_v1','rs','action'):
        source=Q.a_inf if f=='action' else getattr(Q,f)
        for lo in range(0,len(src),128):
            rr=src[lo:lo+128]
            assert np.array_equal(getattr(G,f)[L.L+lo:L.L+lo+len(rr)],source[rr]), f
    top=topology(G.episode,G.step,G.ep_len)
    for f in ('prev','next','next_valid','terminal_known','is_terminal'):
        assert np.array_equal(getattr(G,f)[L.L:],top[f][L.L:]), f
    assert np.array_equal(G.progress, (G.step/np.maximum(G.ep_len.astype(float)-1,1)).astype(np.float32))
    for i,e in enumerate(G.episodes):
        rr=np.flatnonzero(G.episode==i)
        assert len(rr)==e['n_rows'] and (G.task_id[rr]==e['task_id']).all()
        assert (G.ep_len[rr]==e['num_steps']).all() and (G.success[rr]==e['success']).all()
    # Reuse the existing closed-loop payload validator, with an independent
    # source adapter rather than merely comparing G.action to itself.
    ids={s:i for i,s in enumerate(G.ids)}
    class SourceStorage:
        def count(self):return G.L
        def fetch_payload(self,eid):
            r=ids[eid]
            a=L.action[r] if r<L.L else Q.a_inf[int(G.source_query_row[r])]
            return SimpleNamespace(action_chunk=a)
    events=[]
    rt=SimpleNamespace(_vlock=threading.Lock(),_validated={},ids=G.ids,L=G.L,cur_action=G.action,
                       lib_key=key,emit=events.append)
    plugin.PluginRuntime.validate_library(rt,SourceStorage())
    ctx=api.Context(root=root,cell=key+'_cache',seed=0,scratch=OUT/'results')
    ctx.register_library('q4_validation_only',G.action,progress=G.progress,task_id=G.task_id)
    baseline=gates.gate_library(root,[key+'_cache'])
    assert baseline['ok'] and baseline['libs'][key]['ok'] is True, baseline
    return dict(root=str(root),suite=suite,rows=G.L,episodes=len(G.episodes),checksummed_files=len(hashes),
                base_prefix_bit_exact=True,source_arrays_bit_exact=True,heldout_rows_admitted=0,
                library_payload_validator=events,library_registration_validator=True,gate_G5=baseline,
                unknown_continuations=int(((G.next<0)&~G.is_terminal).sum()),
                terminal_rows=int(G.is_terminal.sum()),next_edges=int(G.next_valid.sum()),pass_all=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT/'results'/'validation.json'));a=p.parse_args()
    result=[validate(root,suite) for root in (COLD,SHM) for suite in ('l10','spatial')]
    for suite in ('l10','spatial'):
        c=COLD/'library'/f'pi05_{suite}'/'grow250'/'checksums.json'
        s=SHM/'library'/f'pi05_{suite}'/'grow250'/'checksums.json'
        assert c.read_bytes()==s.read_bytes()
    from pathlib import Path
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
