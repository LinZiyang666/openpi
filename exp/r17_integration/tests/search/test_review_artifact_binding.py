"""Review A4/P1 migration: teacher provenance survives stock YAML loading."""
import uuid

from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.common.fake_adapter import fake_provider
from exp.r17_integration.common.provider import binding_for, validate_artifact
from exp.r17_integration.common.recipe import retrieval_model_identity
from exp.r17_integration.debug.validate import digest, model_binding
from exp.r17_integration.search.build_library import build_library, save_stock_artifact
from exp.r17_integration.tests.debug.fixtures import captured
from exp.r17_integration.tests.debug.test_framework_bridge import config


def test_stock_yaml_preload_preserves_teacher_k_and_retrieval_identity(tmp_path):
    provider=fake_provider('p6.teacher.'+uuid.uuid4().hex)
    b=model_binding(provider.model_spec,'fake_mw.v1')
    capture,episodes=captured(tmp_path,b,episodes=1)
    store=build_library([capture],tmp_path/'store',b,episodes,key_fields=b['model']['feature_spec']['fields'])
    path=tmp_path/'stock.pkl'; save_stock_artifact(store,path)
    cfg=config(provider,k=2)
    cfg.adapter_binding=binding_for(provider,'fake_mw.v1',num_steps=2,commit_horizon=10,
                                    library_sampler_identity=digest(b['sampler']))
    cfg.backend.in_memory.preload_path=str(path)
    storage=build_shared_storage(cfg)
    assert validate_artifact(cfg,storage) is None
    actual=storage.artifact_meta['model']['r17_binding']
    assert actual['sampler_identity']==digest(b['sampler'])
    assert actual['sampler_identity']!=cfg.adapter_binding['sampler_identity']
    assert actual['retrieval_model_identity']==retrieval_model_identity(provider.model_spec)
