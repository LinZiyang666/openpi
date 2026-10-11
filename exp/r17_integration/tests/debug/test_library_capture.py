from dataclasses import asdict
import json

import numpy as np
import pytest

from exp.r17_integration.debug.library_capture import LibraryCaptureBridge,LibraryDecisionSink
from exp.r17_integration.debug.reader_profiles import CaptureReader
from exp.r17_integration.tests.debug.fixtures import binding, record


@pytest.mark.parametrize('bad',('dtype','missing','one_channel'))
def test_images_reject_missing_camera_and_non_uint8(bad,tmp_path):
    bridge = LibraryCaptureBridge(tmp_path/'capture',enabled=True,binding=binding('dp','libero'))
    sink = LibraryDecisionSink(bridge,record())
    obs = {'observation/image':np.ones((3,3,3),np.uint8),'observation/wrist_image':np.ones((3,3,3),np.uint8)}
    if bad=='dtype': obs['observation/wrist_image']=obs['observation/wrist_image'].astype(np.float32)
    elif bad=='missing': obs.pop('observation/wrist_image')
    else: obs['observation/wrist_image']=np.ones((3,3,1),np.uint8)
    try:
        with pytest.raises((KeyError,ValueError)): sink.bind_observation(obs)
    finally: bridge.close()


def test_disabled_allowlist_does_not_touch_arrays(tmp_path):
    bridge = LibraryCaptureBridge(None,enabled=False)
    assert bridge.capture({},None,rawkeys=None) is None
    LibraryDecisionSink(bridge,{})(None)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('enabled',(False,True))
@pytest.mark.parametrize('server_node',('local','h100'))
def test_formal_resident_entry_uses_shared_factory_preserves_action_bytes_and_closes_writer(tmp_path,monkeypatch,enabled,server_node):
    """Actual formal serve composition; CUDA resource calls replaced by CPU stubs."""
    from exp.r17_integration.tests.ops.test_library_campaign import assets
    from exp.r17_integration.ops import formal_server
    from exp.r17_integration.qualification import model_runtime
    from exp.r17_integration.common import websocket_policy_server
    from exp.r17_integration.common.config import load_cache_config,build_shared_storage
    from exp.r17_integration.common.serving import make_connection_factory
    import torch
    from types import SimpleNamespace
    campaign,provider = assets(tmp_path,smoke=True,library=enabled)
    campaign.update(device='cuda:0',memory_budget_gib=14,server_node=server_node,server_run=str(tmp_path/'servers'))
    if server_node=='h100':
        from exp.r17_integration.ops import library_staging
        campaign['h100_staging'] = dict(library_staging.DEFAULT_POLICY)
        monkeypatch.setattr(library_staging.shutil,'disk_usage',lambda path:SimpleNamespace(free=100*library_staging.GIB))
    adapter = provider.load_adapter()
    monkeypatch.setattr(model_runtime,'load',lambda *a:(adapter,provider))
    monkeypatch.setattr(torch.cuda,'get_device_properties',lambda *a:SimpleNamespace(total_memory=24*2**30))
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',lambda *a:None)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda:0)
    monkeypatch.setattr(formal_server,'enable_dp_graph',lambda *a,**kw:{'passed':True,'synthetic':True})
    monkeypatch.setattr(formal_server.signal,'signal',lambda *a:None)
    spec = provider.model_spec
    name = __import__('exp.r17_integration.ops.formal_eval',fromlist=['benchmark']).benchmark().ordered_tasks[30]
    obs = dict(state=np.arange(7,dtype=np.float32),**{'observation/image':np.arange(48,dtype=np.uint8).reshape(4,4,3),
        'observation/wrist_image':np.full((4,4,3),9,np.uint8)})
    request = dict(campaign='CPU',benchmark_protocol=spec.allowed_benchmark_protocols[0],task_name=name,
        original_init_idx=0,decision_seq=0,task_id=30,seed=7,explicit_seed=123,arm='full',attempt=1,bundle_generation='formal_full_v1')
    ep = dict(record(0,task=name),benchmark_protocol=request['benchmark_protocol'],campaign='CPU',
        bundle_generation='formal_full_v1',init_pool='B',split='library',library_episode_id=0,
        init_state_sha256='synthetic',suite_task_id=0)
    cfg = load_cache_config(tmp_path/'assets/full.yaml')
    plain = make_connection_factory(provider,cfg,build_shared_storage(cfg))()
    direct = plain.infer(dict(obs,__request__=request))
    cores = []
    class CpuServer:
        def __init__(self,core,**kwargs):
            self.core=core; self.kwargs=kwargs
        def serve_forever(self,**kwargs):
            core = self.kwargs['connection_policy_factory']()
            cores.append(core)
            core.on_episode_start(task_id=30,episode_id=1500,extra_metadata=ep)
            result = core.infer(dict(obs,__request__=request))
            assert result['actions'].tobytes()==direct['actions'].tobytes()
            assert core.adapter.counts==plain.adapter.counts
            assert result['__hit_meta__']['capture_enabled'] is enabled
            assert self.kwargs['metadata']['debug_capture'] is enabled
            assert self.kwargs['metadata']['formal_pure_inference'] is (not enabled)
            core.on_episode_end(__execution__=dict(decision_seq=0,executed_count=5,cursor=5,terminal=True),__success__=False)
        def request_stop(self): pass
    monkeypatch.setattr(websocket_policy_server,'WebsocketPolicyServer',CpuServer)
    formal_server.serve(campaign,23321,tmp_path/'servers/replica_0',cuda_graph=True)
    captures = list((tmp_path/'servers').rglob('r17_capture.json'))
    assert len(captures)==int(enabled)
    if enabled:
        row,values,keys = next(CaptureReader(captures[0].parent).decisions())
        assert row['executed_count']==5 and row['terminal'] is True
        assert row['episode_id']==row['library_episode_id']==0
        execution = json.loads((captures[0].parent/'execution.jsonl').read_text())
        assert execution['episode_id']==execution['library_episode_id']==0
        assert values['image_0'].tobytes()==obs['observation/image'].tobytes()
        assert values['image_1'].tobytes()==obs['observation/wrist_image'].tobytes()
        assert set(values)=={'normalized_actions','state','image_0','image_1'}
        assert row['image_sources']['0']['cutpoint']=='received_wire_uint8'
