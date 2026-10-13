"""Bounded G3 subset: actual captures/native suffix → P6 store/fit/replay/audit.

Unsupported shared advanced executors remain explicit PARTIAL gate coverage.
Fixed observations are labelled synthetic inputs; this is never teacher quality.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from dataclasses import asdict
import json
from pathlib import Path
import time

from .assets import write_json


def equal_native(one,two):
    import torch
    import numpy as np
    if isinstance(one,torch.Tensor) and isinstance(two,torch.Tensor):
        return one.dtype==two.dtype and torch.equal(one,two)
    a,b=np.asarray(one),np.asarray(two)
    return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--job',required=True)
    p.add_argument('--run',required=True)
    a=p.parse_args()
    job,run=json.loads(Path(a.job).read_text()),Path(a.run)
    report=dict(status='PARTIAL',checks=[],quality_claim=False,synthetic_inputs=job['domain']!='libero',
        recorded_inputs=job['domain']=='libero',actual_environment_controls=False,
        pending=['stock native warm/reset dispatch','certified CP2/twins/routing combinations',
                 'dev/test cohorts and semantic physical restore','replica/split qualification',
                 'pricing with >=50 samples'])
    write_json(run/'G3_RECEIPT.json',report)
    begin=time.monotonic()
    capture=None
    def remaining():
        if time.monotonic()-begin>760:
            raise TimeoutError('G3 phase deadline')
    try:
        import numpy as np
        import torch
        import yaml
        from .model_runtime import load,wire_observations,make_capture_factory
        from exp.r17_integration.common.contracts import RequestContext,CaptureSpec
        from exp.r17_integration.common.request_rng import RequestRNG
        from exp.r17_integration.common.serving import full_call,make_connection_factory
        from exp.r17_integration.common.config import load_cache_config,build_shared_storage
        from exp.r17_integration.debug.validate import model_binding,digest
        from exp.r17_integration.debug.reader_profiles import CaptureReader,read_h5
        from exp.r17_integration.debug.capture_bridge import export_h5
        from exp.r17_integration.search.build_library import build_library,save_stock_artifact
        from exp.r17_integration.search.fit import fit_recipe,save_fitted_recipe,load_fitted_recipe
        from exp.r17_integration.common.query_view import QueryView
        from exp.r17_integration.search.freeze import freeze
        from exp.r17_integration.search.audit import audit_freeze,audit_campaign
        from exp.r17_integration.search.analyze import summarize
        from exp.r17_integration.debug.analysis_profiles import decision_profile,write_report
        adapter,provider=load(job)
        remaining()
        spec=adapter.metadata()
        protocol=spec.allowed_benchmark_protocols[0]
        binding=model_binding(spec,protocol)
        document=yaml.safe_load((Path(job['assets'])/f'k{spec.default_sampler.num_steps}.yaml').read_text())
        document['checkpoints']['cp1']['gate']['type']='always_skip'
        (run/'capture.yaml').write_text(yaml.safe_dump(document))
        cfg=load_cache_config(run/'capture.yaml')
        storage=build_shared_storage(cfg)
        factory,capture=make_capture_factory(provider,cfg,storage,run/'capture')
        core=factory()
        observations=wire_observations(job)
        task_name='reach-v3' if job['domain']=='metaworld' else job['mechanics_task_name']
        task_id=43 if job['domain']=='metaworld' else 0
        episodes=[]
        core.on_task_begin()
        for ep in range(3):
            metadata=dict(task_uid='Q-G3-fixed-'+str(ep),task_data_hash='synthetic-fixed-'+str(ep),
                campaign='Q-G3-fixed',arm='fixed-observation',bundle_generation='capture-v1',benchmark_protocol=protocol,
                task_name=task_name,orig_init_idx=10+ep,original_init_idx=10+ep,attempt=1,
                n_decisions=3,split='library',closed_ok=True,success=False,synthetic=True,
                real_teacher=False,actual_environment_controls=False,task_id=task_id)
            episodes.append(metadata)
            core.on_episode_start(task_id,10+ep,ep,task=task_name,extra_metadata=metadata)
            for seq,obs in enumerate(observations):
                remaining()
                req=RequestContext('Q-G3-fixed',protocol,task_name,10+ep,seq,task_id=task_id,attempt=1,
                    arm='fixed-observation',bundle_generation='capture-v1',explicit_seed=18+ep*3+seq)
                reply=core.infer(dict(obs,__request__=asdict(req)))
                plain,wire=full_call(adapter,obs,req)
                assert reply['actions'].dtype==wire.actions.dtype
                assert reply['actions'].tobytes()==wire.actions[:spec.layout.replan_steps].tobytes()
                assert tuple(reply['__hit_meta__']['stage_calls'])==(1,1,spec.default_sampler.num_steps)
                # Synthetic execution acknowledgment; do not invent applied controls.
                core.acknowledge_execution(dict(decision_seq=seq,executed_count=0,terminal=True))
                if ep==0 and seq==0:
                    prepared=adapter.prepare(obs,req)
                    s1=adapter.run_stage1(prepared)
                    s2=adapter.run_stage2(s1,capture=True)
                    sampled=adapter.sample(s2,spec.default_sampler,RequestRNG(req),CaptureSpec(True,True))
                    if sampled.snapshots:
                        suffix=adapter.resume(s2,sampled.snapshots[-1],sampled.sampler,RequestRNG(req),CaptureSpec())
                        assert equal_native(suffix.normalized_chunk,sampled.normalized_chunk)
                        report['checks'].append(dict(name='native_suffix',status='PASS',extra_nfe=suffix.actual_nfe))
                    extra=[]
                    if job['model']!='groot':
                        for k in ((2,) if job['model']=='dp' else (2,1)):
                            variant=adapter.sample(s2,provider.sampler_for_steps(k),RequestRNG(req),CaptureSpec(True,True))
                            assert variant.actual_nfe==k
                            extra.append(dict(k=k,extra_nfe=variant.actual_nfe,native_grid=list(variant.sampler.native_timesteps)))
                    report['checks'].append(dict(name='explicit_native_augmentation',status='PASS' if extra else 'pending_separate_K2_K1_endpoints',variants=extra))
                    if hasattr(adapter,'warm_reset'):
                        reset=adapter.warm_reset(s2,sampled.normalized_chunk,loop_index=1,sampler=sampled.sampler,
                                                rng=RequestRNG(req),capture=CaptureSpec())
                        assert reset.actual_nfe==sampled.sampler.num_steps-1
                        report['checks'].append(dict(name='native_reset_hook',status='PASS',extra_nfe=reset.actual_nfe))
                    elif hasattr(adapter,'reset_snapshot'):
                        snapshot=adapter.reset_snapshot(s2,sampled.normalized_chunk,torch.zeros_like(sampled.normalized_chunk),sampled.sampler,1)
                        reset=adapter.resume(s2,snapshot,sampled.sampler,RequestRNG(req),CaptureSpec())
                        assert reset.actual_nfe==sampled.sampler.num_steps-1
                        report['checks'].append(dict(name='native_reset_hook',status='PASS',extra_nfe=reset.actual_nfe))
            core.on_episode_end(success=False)
        core.on_task_end()
        assert all(row['drained'] for row in capture.close())
        directories=[bridge.directory for bridge in capture.bridges.values()]
        assert len(directories)==1
        reader=CaptureReader(directories[0],expected_binding=binding)
        source_rows=list(reader.decisions())
        assert len(source_rows)==9
        h5=export_h5(directories[0],run/'captured.h5')
        _,h5_rows=read_h5(h5,expected_binding=binding)
        for native,saved in zip(source_rows,h5_rows):
            for group in (1,2):
                assert set(native[group])==set(saved[group])
                assert all(native[group][key].dtype==saved[group][key].dtype and
                    native[group][key].tobytes()==saved[group][key].tobytes() for key in native[group])
        store=build_library([h5],run/'fixed_input_library',binding,episodes,key_fields=spec.feature_spec.fields,
            split_policy='mt50_init_v1' if job['domain']=='metaworld' else 'declared_partition_v1')
        save_stock_artifact(store,run/'library.pkl')
        from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
        from .benchmark_binding import recipe_table
        names=recipe_table(protocol)['ordered_tasks']
        recipe,fit_receipt=fit_recipe(store,task_names=names,corrector=True,periodic_gap=2,
                                    early=False,commit_horizon=spec.layout.replan_steps,k=2,kref=2)
        save_fitted_recipe(recipe,fit_receipt,run/'fit')
        restored=load_fitted_recipe(run/'fit',expected_binding=binding)
        for i in range(3):
            req=RequestContext('Q-replay',protocol,task_name,30,i,task_id=task_id,attempt=1)
            query=QueryView(req,{name:store.array('key_'+name)[i] for name in spec.feature_spec.fields},
                            store.array('state')[i],hist_has_vision=(True,)*i,hist_hit=(0,)*i)
            one,two=recipe.query(query),restored.query(query)
            assert one.action.tobytes()==two.action.tobytes()
            assert one.rows.tobytes()==two.rows.tobytes() and one.weights.tobytes()==two.weights.tobytes()
            assert one.force_miss==two.force_miss and one.extras==two.extras
        document['checkpoints']['cp1']['gate']['type']='always_search'
        document['backend']['in_memory']['preload_path']=str(run/'library.pkl')
        (run/'hit.yaml').write_text(yaml.safe_dump(document))
        hit_cfg=load_cache_config(run/'hit.yaml')
        hit=make_connection_factory(provider,hit_cfg,build_shared_storage(hit_cfg))()
        hit.on_task_begin()
        req=RequestContext('Q-G3-artifact',protocol,task_name,10,0,task_id=task_id,attempt=1,explicit_seed=18)
        hit.on_episode_start(task_id,10,0,task=task_name)
        got=hit.infer(dict(observations[0],__request__=asdict(req)))
        assert tuple(got['__hit_meta__']['stage_calls'])==(1,0,0)
        hit.acknowledge_execution(dict(decision_seq=0,executed_count=0,terminal=True))
        hit.on_task_end()
        decisions=[r[0] for r in source_rows]
        journal=[dict(ep,accepted=True,error=None,status='failed') for ep in episodes]
        audit=audit_campaign([e['task_uid'] for e in episodes],journal,decisions)
        deps=dict(yaml=digest(document),model=spec.checkpoint_identity,norm=spec.norm_identity,
            layout=spec.layout.identity,sampler=spec.default_sampler.identity,fit=fit_receipt['digest'],
            library=store.manifest['content_digest'],protocol=protocol,split='fixed-input-library10-12',code=spec.code_digest)
        seal=freeze(deps,run/'freeze.json')
        assert audit_freeze(seal,deps)
        analysis=summarize([e['task_uid'] for e in episodes],journal,decisions,task_groups={task_name:'mechanics'},binding=binding)
        analysis['quality_claim']='fixed recorded LIBERO or synthetic MW observations; no real teacher/episode SR'
        write_report(analysis,run/'analysis',decision_cards=[decision_profile(binding,source_rows[0][1])])
        report['checks'] += [dict(name=n,status='PASS') for n in ('debug_byte_parity_9','closed_NPZ_H5_exact',
            'complete_fixed_input_library_3x3','library_only_LOEO_fit','fit_reload_action_donor_weight_exact',
            'stock_artifact_forced_hit','freeze_identity_error_audit','HTML_PNG_CSV_JSON')]
        report.update(subset_status='PASS',audit=audit,complete_G3=False)
        remaining()
    except BaseException as exc:
        report.update(status='PARTIAL' if isinstance(exc,TimeoutError) else 'FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if capture is not None:
            capture.close()
        report['elapsed_seconds']=time.monotonic()-begin
        write_json(run/'G3_RECEIPT.json',report)
    return 2  # Shared advanced executors remain unqualified, even when this subset passes.


if __name__=='__main__':
    raise SystemExit(main())
