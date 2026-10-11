"""Published-content admission and native B16 cache/stall serving regressions."""
import copy
from dataclasses import asdict, replace
import hashlib
from pathlib import Path
import tarfile
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from exp.r17_integration.common.contracts import digest
from exp.r17_integration.common.provider import AdapterProvider
from exp.r17_integration.models.fastwam.provider import configure_library_provider, register_provider
from exp.r17_integration.models.fastwam.ops.library_compatibility import producer_ast
from exp.r17_integration.models.fastwam.ops.library_evidence import EVIDENCE
from exp.r17_integration.models.fastwam.ops.cache_preflight import current_manifest


def published():
    from exp.r17_integration.ops.formal_eval import benchmark
    from exp.r17_integration.common.benchmarks import register_benchmark
    source, serving = benchmark(), benchmark(True)
    register_benchmark(serving)
    manifest, spec = current_manifest(serving)
    provider = AdapterProvider('p3.published.cpu', spec, lambda: None, lambda k: spec.default_sampler)
    binding = dict(model=copy.deepcopy(EVIDENCE['teacher_model_spec']),
        sampler=copy.deepcopy(EVIDENCE['teacher_model_spec']['default_sampler']), benchmark_protocol=source.identity)
    config = SimpleNamespace(adapter_binding=dict(num_steps=10, benchmark_protocol=serving.identity),
        recipe=SimpleNamespace(library_benchmark=dict(source=source.manifest(), serving=serving.manifest())))
    return provider, manifest, config, binding


def test_cpu_hook_returns_frozen_same_provider_and_live_receipt():
    from exp.r17_integration.common.recipe import retrieval_model_identity
    p, manifest, config, binding = published()
    cpu = configure_library_provider(p)
    live = configure_library_provider(p, manifest=manifest)
    assert p.library_validator is None
    assert cpu is not p and cpu.provider_id == p.provider_id and cpu.model_spec is p.model_spec
    a, b = cpu.library_validator(config, binding), live.library_validator(config, binding)
    assert a == b and a['status'] == 'PASS'
    assert a['teacher_identity'] == retrieval_model_identity(binding['model'])
    assert a['resident_identity'] == retrieval_model_identity(p.model_spec)
    assert a['evidence']['weights_norm_vendor_unchanged']
    assert len(a['evidence']['native_sources']) == len(manifest['content_digests'])


def test_published_accepted_hs6_control_and_fresh_runtime_are_both_exactly_attested():
    from exp.r17_integration.ops.cache_study import model_spec
    p, _, cfg, binding=published()
    path=Path('/data/mw_finetune/eval/hs6fw_ctl_143332/assets/model_manifest.json')
    import json
    control=json.loads(path.read_text())
    old=replace(p,model_spec=model_spec(control['spec']))
    result=configure_library_provider(old,manifest=control).library_validator(cfg,binding)
    assert result['evidence']['resident_checkpoint']==digest(EVIDENCE['resident_content_digests'][0])
    current=configure_library_provider(p).library_validator(cfg,binding)
    assert result['evidence']['native_producer_ast_sha256']==current['evidence']['native_producer_ast_sha256']
    assert old.model_spec.norm_identity==p.model_spec.norm_identity


@pytest.mark.parametrize('field', ['norm', 'preprocess', 'pooling', 'action_transform', 'geometry', 'teacher_code', 'teacher_policy', 'teacher_sampler', 'steps', 'source_profile'])
def test_published_library_admission_fails_closed_on_every_semantic_change(field):
    p, manifest, cfg, binding = published()
    if field == 'norm': p = replace(p, model_spec=replace(p.model_spec, norm_identity='changed'))
    elif field == 'preprocess': p = replace(p, model_spec=replace(p.model_spec,
        feature_spec=replace(p.model_spec.feature_spec, preprocess_digest='changed')))
    elif field == 'pooling': p = replace(p, model_spec=replace(p.model_spec,
        feature_spec=replace(p.model_spec.feature_spec, pooling='changed')))
    elif field == 'action_transform': p = replace(p, model_spec=replace(p.model_spec,
        layout=replace(p.model_spec.layout, output_transform_id='changed')))
    elif field == 'geometry': p = replace(p, model_spec=replace(p.model_spec,
        feature_spec=replace(p.model_spec.feature_spec, token_grid=(1,14,7))))
    elif field == 'teacher_code': binding['model']['code_digest'] = 'changed'
    elif field == 'teacher_policy': binding['model']['feature_spec']['policy_digest'] = 'changed'
    elif field == 'teacher_sampler': binding['sampler']['num_steps'] = 1
    elif field == 'steps': cfg.adapter_binding['num_steps'] = 2
    elif field == 'source_profile': cfg.recipe.library_benchmark['source']['seed'] += 1
    with pytest.raises(ValueError): configure_library_provider(p).library_validator(cfg, binding)


@pytest.mark.parametrize('key', ['base_weights', 'wan_vae', 'code:r17:key_builder.py', 'code:r17:staged.py'])
def test_rehashed_changed_weights_or_producer_cannot_be_admitted_by_width(key):
    p, manifest, cfg, binding = published()
    pins = manifest['content_digests']; pins[key] = '0'*64
    identity = digest(pins)
    spec = replace(p.model_spec, checkpoint_identity=identity,
        code_digest=digest({k:v for k,v in pins.items() if k.startswith('code:')}),
        feature_spec=replace(p.model_spec.feature_spec, policy_digest=identity))
    p = replace(p, model_spec=spec); manifest['spec'] = asdict(spec)
    with pytest.raises(ValueError, match='generation is not audited'):
        configure_library_provider(p, manifest=manifest).library_validator(cfg, binding)


def test_live_manifest_and_executing_source_are_fenced(monkeypatch):
    p, manifest, cfg, binding = published()
    manifest['content_digests']['wan_vae'] = 'changed'
    with pytest.raises(ValueError, match='live manifest differs'):
        configure_library_provider(p, manifest=manifest).library_validator(cfg, binding)
    original = Path.read_bytes
    def changed(path):
        return original(path)+b'\n# unsealed edit' if path.name == 'key_builder.py' else original(path)
    monkeypatch.setattr(Path, 'read_bytes', changed)
    with pytest.raises(ValueError, match='executing native source differs'):
        configure_library_provider(p).library_validator(cfg, binding)


def test_no_library_io_or_math_change_on_default_off_provider(monkeypatch, tmp_path):
    from exp.r17_integration.tests.models.fastwam.conftest import make_adapter, observation, request
    from exp.r17_integration.common.serving import full_call
    adapter = make_adapter('libero', allow_unqualified_batch=False)
    before, _ = full_call(adapter, observation('libero'), request())
    metadata = asdict(adapter.manifest)
    def forbidden(*args, **kwargs): raise AssertionError('default-off library read')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    p = register_provider(adapter, provider_id='p3.default.cache.'+uuid.uuid4().hex)
    after, _ = full_call(p.load_adapter(), observation('libero'), request())
    assert before.normalized_chunk.numpy().tobytes() == after.normalized_chunk.numpy().tobytes()
    assert asdict(adapter.manifest) == metadata and not list(tmp_path.iterdir())


def test_published_teacher_archive_has_identical_native_producer_ast():
    campaign = Path('/data/r17_library/fastwam_libero_libero_spatial/campaigns')/EVIDENCE['provenance']['teacher_campaign']
    with tarfile.open(campaign/'runtime.tar') as archive:
        source = archive.extractfile('exp/r17_integration/models/fastwam/staged.py').read()
    assert hashlib.sha256(source).hexdigest() == EVIDENCE['teacher_content_digests']['code:r17:staged.py']
    assert producer_ast(source.decode()) == EVIDENCE['native_producer_ast_sha256']
    assert hashlib.file_digest((campaign/'runtime.tar').open('rb'), 'sha256').hexdigest() == EVIDENCE['provenance']['teacher_archive_sha256']


def test_producer_attestation_is_stable_across_python311_and312(monkeypatch):
    import ast
    from exp.r17_integration.models.fastwam.ops import library_compatibility as compatibility
    source='class Example:\n    def forward(self, x):\n        return x * 2\n'
    expected=producer_ast(source); original=ast.parse
    def python312(text):
        tree=original(text)
        for node in ast.walk(tree):
            if isinstance(node,(ast.ClassDef,ast.FunctionDef)):
                if 'type_params' not in node._fields: node._fields=(*node._fields,'type_params')
                node.type_params=[]
        return tree
    monkeypatch.setattr(compatibility.ast,'parse',python312)
    assert producer_ast(source)==expected
    def changed_generic(text):
        tree=python312(text); tree.body[0].type_params=[ast.Name(id='T',ctx=ast.Load())]; return tree
    monkeypatch.setattr(compatibility.ast,'parse',changed_generic)
    assert producer_ast(source)!=expected


@pytest.mark.parametrize('group', ['c','cr','ch','chr','cg','crg','chg','chrg'])
def test_native_b16_yaml_hit_blind_tail_and_only_confirmed_stall_k10(group, tmp_path):
    from exp.r17_integration.tests.models.fastwam.conftest import make_adapter, observation
    from exp.r17_integration.tests.ops.test_cache_study import freeze
    from exp.r17_integration.common.benchmarks import register_benchmark
    from exp.r17_integration.common.contracts import BenchmarkSpec
    from exp.r17_integration.common.config import build_shared_storage
    from exp.r17_integration.common.recipe import fit_identity
    from exp.r17_integration.common.serving import make_connection_factory
    from exp.r17_integration.models.fastwam.batch_runtime import create_runtime
    register_benchmark(BenchmarkSpec('fake','r17.fixture.v1','fake_mw.v1',('reach',),('reach',),5,0,160),
        protocol_identity='fake_mw.v1')
    adapter = make_adapter('libero', allow_unqualified_batch=False)
    adapter.manifest = replace(adapter.manifest, spec=replace(adapter.metadata(), allowed_benchmark_protocols=('fake_mw.v1',)))
    runtime = create_runtime(adapter, 16, 5, tmp_path/'batches.jsonl')
    try:
        p = register_provider(adapter, provider_id='p3.cache.b16.'+uuid.uuid4().hex, coordinator=runtime.core)
        cfg, _ = freeze(p, tmp_path/'fit', 10 if 'h' in group else 5, gated='g' in group)
        cfg.recipe.corrector = 'r' in group; cfg.recipe.fit_identity = fit_identity(cfg)
        factory = runtime.wrap_connection_factory(make_connection_factory(p, cfg, build_shared_storage(cfg)))
        policy = factory(); policy.on_episode_start(extra_metadata={'task_name':'reach','task_id':0})
        class Tracker:
            def __init__(self, *args): self.count = 0
            def observe(self, *args): self.count += 1
            def status(self): return dict(state='ok' if self.count == 1 else 'slow_confirmed', W=2)
        policy.recipe.tracker_class = Tracker
        first = policy.infer(observation('libero'))
        assert first['__hit_meta__']['stage_calls'] == (1,0,0)
        assert first['__hit_meta__']['source'] == 'library' and first['__hit_meta__']['actual_nfe'] == 0
        assert set(first['__hit_meta__']['batching']['stages']) == {'stage1'}
        seq = 0
        def next_call():
            nonlocal seq
            out = policy.infer(dict(observation('libero'), __execution__={'decision_seq':seq,'executed_count':5}))
            seq += 1
            return out
        if 'h' in group:
            before = copy.deepcopy(adapter.model.calls)
            tail = next_call()
            assert tail['__hit_meta__']['stage_calls'] == (0,0,0) and tail['__hit_meta__']['actual_nfe'] == 0
            assert tail['__hit_meta__']['source'] == 'library_tail'
            assert tail['__hit_meta__']['batching']['stages'] == {}
            assert tail['actions'].tobytes() == first['actions'][5:].tobytes()
            assert adapter.model.calls == before
        second = next_call()
        meta = second['__hit_meta__']
        if 'g' in group:
            assert meta['stall_call'] and meta['source'] == 'policy'
            assert meta['stage_calls'] == (1,1,10) and meta['actual_nfe'] == 10
            assert set(meta['batching']['stages']) == {'stage1','stage2','stage3'}
            assert len(adapter.model.calls['head']) == 10
            if 'h' in group: next_call()  # Consume the native C10 tail.
            cooled = next_call()['__hit_meta__']
            assert cooled['stall_cooldown'] and not cooled['stall_call']
            assert cooled['actual_nfe'] == 0 and cooled['stage_calls'] == (1,0,0)
            assert len(adapter.model.calls['head']) == 10
        else:
            assert meta['actual_nfe'] == 0 and meta['stage_calls'] == (1,0,0)
            assert not adapter.model.calls['head'] and not adapter.model.calls['prefill'] and not adapter.model.calls['context']
    finally:
        runtime.close()


def test_all_eight_socket_hot_switches_share_one_b16_resident_and_batch_hits(tmp_path, monkeypatch):
    import asyncio
    import json
    import yaml
    import websockets.asyncio.client
    import websockets.asyncio.server
    from openpi_client import msgpack_numpy
    from openpi.serving import websocket_policy_server as stock
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from exp.r17_integration.common.benchmarks import register_benchmark
    from exp.r17_integration.common.contracts import BenchmarkSpec
    from exp.r17_integration.common.config import RecipeConfig, build_shared_storage
    from exp.r17_integration.common.recipe import fit_identity
    from exp.r17_integration.common.serving import make_connection_factory, make_bundle_validator
    from exp.r17_integration.models.fastwam.batch_runtime import create_runtime
    from exp.r17_integration.tests.models.fastwam.conftest import make_adapter, observation
    from exp.r17_integration.tests.ops.test_cache_study import freeze
    register_benchmark(BenchmarkSpec('fake','r17.fixture.v1','fake_mw.v1',('reach',),('reach',),5,0,160),
        protocol_identity='fake_mw.v1')
    adapter = make_adapter('libero', allow_unqualified_batch=False)
    adapter.manifest = replace(adapter.manifest, spec=replace(adapter.metadata(), allowed_benchmark_protocols=('fake_mw.v1',)))
    runtime = create_runtime(adapter,16,5,tmp_path/'batch_ledger.jsonl')
    try:
        p = register_provider(adapter,provider_id='p3.cache.socket.'+uuid.uuid4().hex,coordinator=runtime.core)
        short, _ = freeze(p,tmp_path/'short',5,gated=True)
        long, _ = freeze(p,tmp_path/'long',10,gated=True)
        plain = copy.deepcopy(short); plain.enabled=False; plain.recipe=RecipeConfig(); plain.checkpoints={}
        factory = runtime.wrap_connection_factory(make_connection_factory(p,plain,build_shared_storage(plain)))
        policies = []
        def tracked(*args, **kwargs):
            policy=factory(*args,**kwargs); policies.append(policy); return policy
        server = WebsocketPolicyServer(tracked(),concurrent=True,connection_policy_factory=tracked,
            bundle_validator=make_bundle_validator(p),metadata={'fixture':'p3_native_B16'})
        for name, value in (('_bundles',{}),('_current_bundle',None),('_bundle_version',0)):
            monkeypatch.setattr(stock,name,value)
        paths = []
        for group in ('c','cr','ch','chr','cg','crg','chg','chrg'):
            cfg=copy.deepcopy(long if 'h' in group else short); cfg.recipe.corrector='r' in group
            if 'g' not in group:
                cfg.recipe.miss_mode='never'; cfg.recipe.stall_path=None; cfg.recipe.stall_digest=None
            cfg.recipe.fit_identity=fit_identity(cfg)
            path=tmp_path/(group+'.yaml'); path.write_text(yaml.safe_dump(asdict(cfg))); paths.append((group,path))
        packer=msgpack_numpy.Packer()
        async def run():
            async with websockets.asyncio.server.serve(server._handler,'127.0.0.1',0) as listener:
                port=listener.sockets[0].getsockname()[1]
                from contextlib import AsyncExitStack
                async with AsyncExitStack() as stack:
                    clients=[await stack.enter_async_context(websockets.asyncio.client.connect(f'ws://127.0.0.1:{port}')) for _ in range(16)]
                    await asyncio.gather(*(client.recv() for client in clients))
                    async def rpc(client, payload):
                        await client.send(packer.pack(payload))
                        return msgpack_numpy.unpackb(await asyncio.wait_for(client.recv(),20))
                    for group,path in paths:
                        ack=await rpc(clients[0],{'__ctrl__':'load_cache_config','yaml_path':str(path),'bundle_id':group})
                        assert ack['__ack__']=='load_cache_config'
                        for client in clients:
                            assert (await rpc(client,{'__ctrl__':'select_bundle','bundle_id':group}))['__ack__']=='select_bundle'
                        await asyncio.gather(*(rpc(client,{'__ctrl__':'episode_start','__task__':'reach','__extra__':{'task_id':0}}) for client in clients))
                        outputs=await asyncio.gather(*(rpc(client,observation('libero')) for client in clients))
                        for out in outputs:
                            meta=out['__hit_meta__']
                            assert tuple(meta['stage_calls'])==(1,0,0) and meta['actual_nfe']==0
                            assert set(meta['batching']['stages'])=={'stage1'}
                        if 'h' in group:
                            tails=await asyncio.gather(*(rpc(client,dict(observation('libero'),__execution__={'decision_seq':0,'executed_count':5})) for client in clients))
                            for first,tail in zip(outputs,tails,strict=True):
                                assert tuple(tail['__hit_meta__']['stage_calls'])==(0,0,0)
                                assert tail['__hit_meta__']['batching']['stages']=={}
                                assert tail['actions'].tobytes()==first['actions'][5:].tobytes()
                        await asyncio.gather(*(rpc(client,{'__ctrl__':'episode_end',
                            '__execution__':{'decision_seq':1 if 'h' in group else 0,'executed_count':5,'terminal':True}}) for client in clients))
        asyncio.run(run())
        assert all(policy.adapter.model is adapter.model for policy in policies)
        assert len(stock._bundles)==8 and p.model_spec==adapter.metadata()
        assert not adapter.model.calls['head'] and not adapter.model.calls['prefill'] and not adapter.model.calls['context']
        rows=[json.loads(line) for line in (tmp_path/'batch_ledger.jsonl').read_text().splitlines()]
        stages=[r for r in rows if r['schema']=='r17.fastwam.stage_batch.v1']
        assert all(r['stage']=='stage1' and 1<=r['size']<=16 for r in stages)
        assert max(r['size'] for r in stages)>1
        # Hot-switch creates connection-local wrappers for the new bundle;
        # all128 wrappers retain the same weights and same S07 resident core.
        assert len({m['connection_id'] for r in stages for m in r['members']})==128
        for group in ('c','cr','ch','chr','cg','crg','chg','chrg'):
            members=[m for r in stages for m in r['members'] if m['request']['bundle_generation']==group]
            assert len({m['connection_id'] for m in members})==16
        assert len([r for r in rows if r['schema']=='r17.fastwam.decision_batch.v1'])==192
    finally:
        runtime.close()
