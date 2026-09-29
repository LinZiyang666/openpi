"""CPU replay of actual key-builder slices, trace serializers and plugin.tokens on recorded tokens."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import h5py
import numpy as np
import torch

from exp.offline_search.closed_loop import plugin
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE, STORE
from exp.offline_search.rounds.r06.p2_ablations.token_pca import pool_tokens
from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
from openpi.cache.groot.key_builder import GrootLiberoCP1SpatialPool16KeyBuilder
from openpi.cache.trace import pi05 as pi_trace, groot as groot_trace
from openpi.cache.types import CheckpointID


def check(model, t0, t1, state):
    if model == 'pi05':
        prefix = torch.zeros((1,771,2048), dtype=torch.float32)
        prefix[0,:256] = torch.from_numpy(t0.astype(np.float32))
        prefix[0,256:512] = torch.from_numpy(t1.astype(np.float32))
        s = NS(prefix_embs=prefix, state=torch.from_numpy(state.astype(np.float32))[None])
        kb = CP1SpatialPool16KeyBuilder()
        recorded, _, _ = pi_trace.slice_prefix_tokens(s)
    else:
        prefix = torch.zeros((1,550,2048), dtype=torch.float32)
        mask = torch.zeros((1,550), dtype=torch.bool)
        for t, start in [(t0,20),(t1,283)]:
            prefix[0,start:start+256] = torch.from_numpy(t.astype(np.float32))
            mask[0,start:start+256] = True
        s = NS(input_embeds=prefix, image_token_mask=mask,
               state=torch.from_numpy(state[:8].astype(np.float32))[None,None],
               state_mask=torch.ones((1,1,8),dtype=torch.bool))
        kb = GrootLiberoCP1SpatialPool16KeyBuilder()
        recorded, _, _ = groot_trace.slice_prefix_tokens(s, vision_fields=('vision_0','vision_1'), expected_state_index=None)
    kb.collect(CheckpointID.CP1, stage1=s)
    session = NS(rt=NS(api=api, opts=NS(os_tokens='on')), kb=kb, has_vision=[True], _tok_cache={})
    for f, want, trace in [('vision_0',t0,recorded[0]),('vision_1',t1,recorded[1])]:
        got = plugin.PluginSession.tokens(session, f)
        assert got.dtype == np.float16 and got.shape == (256,2048) and not got.flags.writeable
        assert got.tobytes() == want.tobytes() == trace.tobytes()
    return max(float(np.max(np.abs(kb.build(CheckpointID.CP1)[f].numpy()-pool_tokens(t,4))))
               for f,t in [('vision_0',t0),('vision_1',t1)])


def main():
    reports=[]
    for model in ('pi05','groot'):
        for suite in ('l10','spatial'):
            for stream in ('cache','inf'):
                qc=store.QueryCell(STORE,f'{model}_{suite}_{stream}')
                indexes=np.asarray(qc.tok_rows)
                samples=indexes[np.linspace(0,len(indexes)-1,8,dtype=int)]
                max_pool_error=max_builder_error=max_bf16_pool_error=0.
                bf16_exact=True
                for r in samples:
                    r=int(r); j=int(qc.tok_index[r]); e=qc.episodes[int(qc.ep[r])]
                    ts=[np.array(qc.tok(v)[j]) for v in ('v0','v1')]
                    with h5py.File(e['file'],'r') as h:
                        g=h[f'step_{int(qc.step[r]):04d}']
                        for k,t in enumerate(ts):
                            assert np.array_equal(g[f'vision_{k}'][()],t)
                    max_builder_error=max(max_builder_error,check(model,*ts,np.asarray(qc.rs[r])))
                    for k,t in enumerate(ts):
                        max_pool_error=max(max_pool_error,float(np.max(np.abs(pool_tokens(t,4)-getattr(qc,f'key_v{k}')[r]))))
                        tt=torch.from_numpy(t.copy()).to(torch.bfloat16)
                        bf16_exact=bf16_exact and np.array_equal(tt.to(torch.float16).numpy(),t)
                        from openpi.cache.components.key_builder import _spatial_pool_tokens
                        pooled=_spatial_pool_tokens(tt,16,4).float().numpy()
                        max_bf16_pool_error=max(max_bf16_pool_error,float(np.max(np.abs(pooled-getattr(qc,f'key_v{k}')[r]))))
                reports.append(dict(cell=f'{model}_{suite}_{stream}', samples=len(samples),
                                    exact_h5_store_serializer_plugin_tokens=True,
                                    pooled_tokens_vs_recorded_key_max_abs=max_pool_error,
                                    torch_f32_pool_vs_numpy_pool_max_abs=max_builder_error,
                                    stored_tokens_exact_bf16_roundtrip=bf16_exact,
                                    torch_bf16_pool_vs_recorded_key_max_abs=max_bf16_pool_error))
    library_reports=[]
    for model in ('pi05','groot'):
        for suite in ('l10','spatial'):
            for name in ('current','bpool_cs' if model=='pi05' else 'bpool_all'):
                lib=store.LibraryView(STORE,f'{model}_{suite}',name)
                samples=np.linspace(0,lib.L-1,8,dtype=int)
                for r in samples:
                    e=lib.episodes[int(lib.episode[r])]
                    ts=[np.array(lib.tok(v)[r]) for v in ('v0','v1')]
                    with h5py.File(e['file'],'r') as h:
                        g=h[f'step_{int(lib.step[r]):04d}']
                        for k,t in enumerate(ts):
                            assert t.dtype==np.float16 and t.shape==(256,2048)
                            assert g[f'vision_{k}'][()].tobytes()==t.tobytes()
                    check(model,*ts,np.asarray(lib.rs[r]))
                files={str(Path(e['file']).resolve()) for e in lib.episodes}
                for stream in ('cache','inf'):
                    qc=store.QueryCell(STORE,f'{model}_{suite}_{stream}')
                    assert files.isdisjoint(str(Path(e['file']).resolve()) for e in qc.episodes)
                library_reports.append(dict(library=f'{model}_{suite}/{name}',samples=len(samples),
                    exact_build_h5_library_serializer_plugin_tokens=True,query_source_files_disjoint=True))
    out=dict(PASS=True, reports=reports, cases=sum(r['samples'] for r in reports),
             libraries=library_reports,library_cases=sum(r['samples'] for r in library_reports),
             scope='Actual CPU slicer/serializer/plugin code on reconstructed stage-1 containers holding recorded tokens; no new model forward or GPU measurement')
    (HERE/'results/pca/token_equality.json').write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps(out),flush=True)


if __name__=='__main__': main()
