"""Review A5/A10: run deployed worker main with CPU renderer/socket boundaries."""
from dataclasses import asdict, replace
import sys
from types import ModuleType
import uuid

from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.common.fake_adapter import FakeAdapter, fake_sampler, fake_spec
from exp.r17_integration.common.provider import AdapterProvider, binding_for, register_provider
from exp.r17_integration.common.serving import make_connection_factory
from exp.r17_integration.debug.capture_bridge import make_capture_sink_factory
from exp.r17_integration.debug.reader_profiles import CaptureReader
from exp.r17_integration.debug.validate import episode_identity
from exp.r17_integration.tests.debug.test_framework_bridge import config


def test_production_worker_debug_uses_attested_sampler_and_server_episode(tmp_path,monkeypatch):
    from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
    from exp.r17_integration.conductor import worker_entry
    from exp.r17_integration.tests.benchmarks.test_metaworld import Env, task
    import openpi.conductor.worker
    import openpi_client.websocket_client_policy
    spec=replace(fake_spec(),allowed_benchmark_protocols=(FORMAL_SPEC.identity,))
    class KernelAdapter(FakeAdapter):
        def extract_state_only(self,obs):
            return super().extract_state_only({'state':obs.get('state',obs.get('observation/state'))})
    provider=AdapterProvider('p6.worker.cli.'+uuid.uuid4().hex,spec,lambda:KernelAdapter(spec),fake_sampler)
    register_provider(provider)
    cfg=config(provider,k=2,commit=10); cfg.diagnostic_context=False
    cfg.adapter_binding=binding_for(provider,FORMAL_SPEC.identity,num_steps=2,commit_horizon=10)
    captures=make_capture_sink_factory(provider,tmp_path/'server')
    core=make_connection_factory(provider,cfg,build_shared_storage(cfg),capture_sink_factory=captures)()
    environment=Env(success=7); environment._r17_task_data_hash='worker-cpu-data'
    rendering=ModuleType('exp.mw_finetune.rendering')
    rendering.STAMP={'status':'mock_cpu_renderer'}
    rendering.configure=lambda *args:None
    rendering.make_env=lambda *args:environment
    monkeypatch.setitem(sys.modules,'exp.mw_finetune.rendering',rendering)
    monkeypatch.setattr(worker_entry,'attest_worker',lambda *args:{'status':'mock_cpu_attestation'})
    metadata={'r17_model_spec':asdict(spec),'r17_samplers':{'2':asdict(fake_sampler(2))}}
    class Client:
        def __init__(self,*args,**kwargs):
            pass
        def get_server_metadata(self):
            return metadata
        def select_bundle(self,bundle):
            pass
        def episode_start(self,**kwargs):
            core.on_episode_start(**kwargs)
        def episode_end(self,**kwargs):
            core.on_episode_end(**kwargs)
        def infer(self,obs):
            return core.infer(obs)
        def close(self):
            pass
    monkeypatch.setattr(openpi_client.websocket_client_policy,'WebsocketClientPolicy',Client)
    t=task(); t=replace(t,extra=dict(t.extra,bundle_identity=core.bundle_identity,
        bundle_generation='worker-generation-not-id',num_steps=2,task_data_hash=environment._r17_task_data_hash,
        model_identity=spec.checkpoint_identity,norm_identity=spec.norm_identity,layout_identity=spec.layout.identity))
    results=[]
    class Loop:
        def __init__(self,worker_id,server_key,runner,**kwargs):
            self.runner=runner
        def run_forever(self):
            results.append(self.runner.run(t,lambda *args:None))
    monkeypatch.setattr(openpi.conductor.worker,'WorkerLoop',Loop)
    try:
        worker_entry.main(['--worker-id','cpu','--server-key','local:1','--driver-host','local','--driver-port','1',
                           '--debug-directory',str(tmp_path/'worker')])
    finally:
        captures.close()
    model=CaptureReader(next((tmp_path/'server').iterdir()))
    physical=CaptureReader(next((tmp_path/'worker').iterdir()))
    assert model.manifest['binding']==physical.manifest['binding']
    assert physical.manifest['binding']['sampler']['num_steps']==2
    rows=[row for row,_,_ in model.decisions()]
    controls=[row for row,_ in physical.physical() if row['event']=='control']
    assert [row['executed_count'] for row in rows]==[5,2] and len(controls)==7
    assert episode_identity(rows[0])==episode_identity(controls[0])
    assert rows[0]['bundle_generation']=='worker-generation-not-id' and results[0].success
