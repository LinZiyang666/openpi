"""Review A6: real P1/P2 execution path, CPU fake weights/native environment."""
from dataclasses import replace
import time
import uuid
import pytest

from exp.r17_integration.common.fake_adapter import FakeAdapter, fake_sampler, fake_spec
from exp.r17_integration.common.provider import AdapterProvider, binding_for, register_provider
from exp.r17_integration.common.serving import make_connection_factory
from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.debug.capture_bridge import make_capture_sink_factory
from exp.r17_integration.debug.reader_profiles import CaptureReader
from exp.r17_integration.debug.validate import model_binding
from exp.r17_integration.search.analyze import summarize
from exp.r17_integration.search.audit import audit_campaign
from exp.r17_integration.tests.debug.test_framework_bridge import config


@pytest.mark.parametrize('capture',[False,True])
def test_p1_p2_actual_cost_ledger_and_terminal_ack_reach_analysis(tmp_path,capture):
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC, R17MetaWorldEpisodeRunner
    from exp.r17_integration.tests.benchmarks.test_metaworld import Env, task
    spec=replace(fake_spec(),allowed_benchmark_protocols=(FORMAL_SPEC.identity,))
    class KernelAdapter(FakeAdapter):
        def extract_state_only(self,obs):
            return super().extract_state_only({"state":obs.get("state",obs.get("observation/state"))})
    provider=AdapterProvider("p6.runtime."+uuid.uuid4().hex,spec,lambda:KernelAdapter(spec),fake_sampler)
    register_provider(provider)
    cfg=config(provider,k=2,commit=10)
    cfg.diagnostic_context=False
    cfg.adapter_binding=binding_for(provider,FORMAL_SPEC.identity,num_steps=2,commit_horizon=10)
    captures=make_capture_sink_factory(provider,tmp_path/"server",enabled=capture)
    core=make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    from exp.r17_integration.common.contracts import CaptureSpec
    core.capture_sink=captures(core)
    core.capture=CaptureSpec(enabled=capture,snapshots=capture)
    class Client:
        def select_bundle(self,bundle):
            pass
        def episode_start(self,**kwargs):
            core.on_episode_start(**kwargs)
        def infer(self,obs):
            started=time.perf_counter()
            result=core.infer(obs)
            result['server_timing']={'infer_ms':(time.perf_counter()-started)*1000}
            return result
        def episode_end(self,**kwargs):
            core.on_episode_end(**kwargs)
        def close(self):
            pass
    environment=Env(success=7)
    environment._r17_task_data_hash='cpu-native-taskdata'
    t=task()
    t=replace(t,extra=dict(t.extra,task_data_hash=environment._r17_task_data_hash,
                          bundle_generation='deliberately-not-bundle-id',bundle_identity=core.bundle_identity,
                          recipe_identity=core.recipe_identity,num_steps=2,sampler=core.sampler.__dict__))
    runner=R17MetaWorldEpisodeRunner(env_factory=lambda *a:environment,client_factory=lambda _:Client(),framework_context=True)
    try:
        result=runner.run(t,lambda *a:None)
    finally:
        captures.close(); runner.close()
    from exp.r17_integration.debug.reader_profiles import merge_decision_ledgers
    if capture:
        reader=CaptureReader(next((tmp_path/'server').iterdir()))
        rows=[row for row,_,_ in reader.decisions()]
        bound=reader.manifest['binding']
    else:
        assert not (tmp_path/'server').exists()
        raw=[r for r in result.per_step_rows if r.get('_kind') not in ('episode_summary','execution_ack')]
        acks=[r for r in result.per_step_rows if r.get('_kind')=='execution_ack']
        rows=merge_decision_ledgers(raw,execution_rows=acks)
        bound=model_binding(spec,FORMAL_SPEC.identity,sampler=core.sampler)
    assert [r['executed_count'] for r in rows]==[5,2]
    assert rows[-1]['terminal'] is True
    assert all(r['cost_ms']>=0 and r['stage_ms'] for r in rows)
    assert rows[-1]['stage_calls']==[0,0,0]
    assert all(r['bundle_generation']==t.extra['bundle_generation'] for r in rows)
    ep=dict(rows[0],accepted=True,status='done',success=result.success,n_decisions=2)
    report=summarize([t.task_uid],[ep],rows,task_groups={'reach-v3':'easy'},binding=bound,cost_rows=result.per_step_rows)
    assert report['cost']['status']=='available' and report['cost']['executed_controls']==7
    assert report['cost']['absolute_ms']>=0
    assert report['cost']['pricing_status']==('excluded_capture' if capture else 'available')
    assert (report['cost']['pricing_absolute_ms'] is None)==capture
    assert audit_campaign([t.task_uid],[ep],rows,cost_rows=result.per_step_rows)['cost_status']=='available'


def test_actual_groot_serve_assembly_publishes_worker_capture_metadata(tmp_path,monkeypatch):
    """Review A5 repro transformed into CPU production-assembly capture test."""
    import numpy as np
    import exp.r17_integration.tests.models.groot.conftest as fixture
    from exp.r17_integration.models.groot import serve as groot_serve
    from exp.r17_integration.models.groot.provider import make_provider, register_provider as publish
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from exp.r17_integration.debug.capture_bridge import PhysicalCaptureFactory
    from exp.r17_integration.tests.debug.fixtures import record
    backend=fixture.guarded_stub.__wrapped__(monkeypatch)()
    manifest=fixture.manifest.__wrapped__()
    provider=publish(replace(make_provider(backend,manifest),provider_id='p6.groot.'+uuid.uuid4().hex))
    cfg=groot_serve.default_config(provider,commit_horizon=5)
    physical=PhysicalCaptureFactory(tmp_path/'worker',{'r17_model_spec':provider.model_spec},sampler_resolver=provider.sampler_for_steps)
    def run_cpu(self,**kwargs):
        assert 'r17_model_spec' in self._metadata
        # Use the metadata supplied by the deployed server, not a parallel fixture manifest.
        physical.spec=self._metadata['r17_model_spec']
        core=self._connection_policy_factory(None,'default')
        from exp.metaworld.tasks import task_id_of
        # Canonical MT50 identity: reach-v3's task_id (the default config now
        # carries the canonical table and rejects inconsistent name/index pairs).
        ep=dict(record(init=0,n_decisions=1),benchmark_protocol=manifest.benchmark_protocol,
                original_init_idx=0,task_id=task_id_of('reach-v3'),seed=7,explicit_seed=42,bundle_identity=core.bundle_identity)
        core.on_episode_start(task_id=ep['task_id'],init_state_id=0,extra_metadata=ep)
        obs=fixture.obs.__wrapped__()
        obs['__request__']={'bundle_generation':ep['bundle_generation']}
        core.infer(obs)
        physical(dict(ep,event='control',control_idx=0,num_steps=4,
                      physical={'arrays':{'qpos':np.arange(3,dtype=np.float64)}}))
        core.on_episode_end(success=True,execution={'decision_seq':0,'executed_count':2})
    monkeypatch.setattr(WebsocketPolicyServer,'serve_forever',run_cpu)
    try:
        groot_serve.serve(backend,manifest,port=0,default_config=cfg,default_storage=build_shared_storage(cfg),
                          provider=provider,debug_directory=tmp_path/'server')
    finally:
        physical.close()
    server=CaptureReader(next((tmp_path/'server').iterdir()))
    worker=CaptureReader(next((tmp_path/'worker').iterdir()))
    assert server.manifest['binding']==worker.manifest['binding']
    model_row=next(server.decisions())[0]; physical_row,_=next(worker.physical())
    from exp.r17_integration.debug.validate import episode_identity
    assert episode_identity(model_row)==episode_identity(physical_row)
    assert model_row['executed_count']==2
