"""Fresh deployment fits; measured protocol-4 plugin artifacts live outside the repository."""
import gc, json, pathlib, pickle, time, sys
from exp.offline_search.harness import api, store
from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.rounds.r04.k1_blind.checks import ROOT,OUT

ART=pathlib.Path('/tmp/k1_blind_fits');ART.mkdir(exist_ok=True)
only_control = '--only-control' in sys.argv
reports = ([r for r in json.loads((OUT/'fits.json').read_text()) if not r['tag'].startswith('csl')]
           if only_control else [])
for scale in (50,500):
    for key in ('pi05_l10','pi05_spatial','groot_l10','groot_spatial'):
        kw=dict(lib='current' if scale==50 else 'big',kref=5 if scale==50 else 8)
        jobs=[('blind','blind_awm:BlindAWM',kw)]
        jobs += [('csl'+ab,'control_step:ControlStepLibrary',dict(kw,ablation=ab)) for ab in ('G','GS')]
        if key.startswith('pi05'):
            jobs += [('mixed','judge:BlindMixedJudge',dict(base_kwargs=kw)),
                     ('wrist','wrist:BlindWristAWM',kw),('wrist_mixed','wrist:BlindWristMixedJudge',dict(base_kwargs=kw))]
        for tag,suffix,kwargs in jobs:
            if only_control and not tag.startswith('csl'):
                continue
            cell=key+'_cache';spec='exp.offline_search.rounds.r04.k1_blind.'+suffix
            cls,_=load_method_class(spec)
            m=cls(**kwargs);m.prof=api.NULL_PROFILER
            ctx=api.Context(root=ROOT,cell=cell,seed=0,scratch=OUT/'fit_scratch'/f'{tag}_{key}_{scale}')
            ctx.scratch.mkdir(parents=True,exist_ok=True)
            lib=store.LibraryView(ROOT,key,'current')
            t0=time.perf_counter();m.fit(lib,ctx);elapsed=time.perf_counter()-t0
            artifact=ART/f'{tag}_{key}_{scale}.pkl'
            blob=dict(method=m,registered=ctx.registered,spec=spec,kwargs=kwargs,cell=cell,fit_s=elapsed)
            with artifact.open('wb') as f:pickle.dump(blob,f,protocol=4)
            # Real deserialization validates class provenance and custom MixedJudge reduce.
            with artifact.open('rb') as f:assert pickle.load(f)['method'].name==m.name
            deployed=store.LibraryView(ROOT,key,m.base.cand_name if hasattr(m,'base') else m.cand_name)
            report=dict(tag=tag,key=key,scale=scale,rows=deployed.L,episodes=len(deployed.episodes),
                        bytes_per_entry=m.bytes_per_entry(),fit_s=elapsed,pickle_bytes=artifact.stat().st_size,
                        artifact=str(artifact),spec=spec,kwargs=kwargs)
            reports.append(report);(OUT/'fits.json').write_text(json.dumps(reports,indent=2));print(json.dumps(report),flush=True)
            del m,blob,lib,ctx;gc.collect()
