"""Full source equality, payload-validator and nested-ID checks in both stores."""
import argparse
import json
import threading
from types import SimpleNamespace
from pathlib import Path
import numpy as np
from exp.offline_search.harness import api,store,gates
from exp.offline_search.closed_loop import plugin
from exp.offline_search.rounds.r05.q4_growth.common import OUT,COLD,SHM,FIELDS,sha256
from exp.offline_search.rounds.r05.q4_growth.demo_build import order_episodes

def validate(root,suite):
    B=store.LibraryView(root,'pi05_'+suite,'bpool_cs');previous=None;result=[]
    for n in (100,200,300):
        G=store.LibraryView(root,B.key,f'demo{n}');source=np.asarray(G.source_bpool_row)
        expected=order_episodes(B)[:n]
        erows=np.concatenate([np.arange(e['start'],e['end']) for _,e in expected])
        assert np.array_equal(source,erows)
        assert len(G.episodes)==n and [sum(e['task_id']==t for e in G.episodes) for t in range(10)]==[n//10]*10
        assert G.ids==[B.ids[r] for r in source] and len(set(G.ids))==G.L
        if previous is not None:
            assert G.ids[:previous.L]==previous.ids
            for f in FIELDS:assert np.array_equal(getattr(G,f)[:previous.L],getattr(previous,f)),f
        hashes=json.loads((G.dir/'checksums.json').read_text())
        for f,h in hashes.items():assert (G.dir/f).stat().st_size==h['bytes'] and sha256(G.dir/f)==h['sha256'],f
        for f in FIELDS:
            if f in ('episode','next','prev'):continue
            a=getattr(G,f);b=getattr(B,f)
            assert a.dtype==b.dtype and a.shape==(G.L,*b.shape[1:])
            for lo in range(0,G.L,128):
                rr=source[lo:lo+128]
                assert np.array_equal(a[lo:lo+len(rr)],b[rr]),f
                if a.dtype.kind=='f':assert np.isfinite(a[lo:lo+len(rr)]).all(),f
        back=np.full(B.L,-1,np.int32);back[source]=np.arange(G.L)
        for f in ('next','prev'):
            old=getattr(B,f)[source];mapped=np.where(old>=0,back[np.maximum(old,0)],-1)
            assert np.array_equal(getattr(G,f),mapped)
        for i,e in enumerate(G.episodes):
            rr=np.arange(e['start'],e['end'])
            assert (G.episode[rr]==i).all() and np.array_equal(G.step[rr],np.arange(e['num_steps']))
            assert e['source_episode']==expected[i][0]
        assert np.array_equal(G.is_terminal,G.step==G.ep_len-1)
        assert np.array_equal(G.next_valid,G.next>=0) and G.terminal_known.all()
        lookup={eid:i for i,eid in enumerate(B.ids)}
        class Source:
            def count(self):return G.L
            def fetch_payload(self,eid):return SimpleNamespace(action_chunk=B.action[lookup[eid]])
        events=[];rt=SimpleNamespace(_vlock=threading.Lock(),_validated={},ids=G.ids,L=G.L,cur_action=G.action,lib_key=B.key,emit=events.append)
        plugin.PluginRuntime.validate_library(rt,Source())
        ctx=api.Context(root=root,cell=B.key+'_cache',seed=0,scratch=OUT/'results'/'demo')
        ctx.register_library('demo_validation_only',G.action,progress=G.progress,task_id=G.task_id)
        assert G.name in store.library_names(root,B.key)
        result.append(dict(root=str(root),suite=suite,library=G.name,rows=G.L,episodes=n,
                           checksummed_files=len(hashes),full_source_equality=True,prefix_nested=True,
                           payload_validator=events,register_validator=True,unknown_continuations=int(((G.next<0)&~G.is_terminal).sum()),pass_all=True))
        previous=G
    gate=gates.gate_library(root,[B.key+'_cache']);assert gate['ok'] and gate['libs'][B.key]['ok'] is True
    return dict(libraries=result,current_G5=gate)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',default=str(OUT/'results'/'demo'/'validation.json'));a=p.parse_args()
    result=[validate(root,s) for root in (COLD,SHM) for s in ('l10','spatial')]
    for s in ('l10','spatial'):
        for n in (100,200,300):
            rel=Path('library')/f'pi05_{s}'/f'demo{n}'/'checksums.json'
            assert (COLD/rel).read_bytes()==(SHM/rel).read_bytes()
    Path(a.out).write_text(json.dumps(result,indent=2));print(json.dumps(result))
