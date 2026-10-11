from contextlib import contextmanager
from dataclasses import replace
import json
import socket
import threading

import numpy as np
import pytest
import yaml

from exp.r17_integration.tests.models.pi05.test_pi05 import fixture_adapter, observation, request
from exp.r17_integration.tests.models.pi05.test_pi05_e2e import tiny_store
from exp.r17_integration.models.pi05.provider import register_provider, config_document, make_connection_factory
from exp.r17_integration.common.config import load_cache_config, build_shared_storage
from exp.r17_integration.common.recipe import bundle_identity, recipe_identity, fit_identity
from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC


@contextmanager
def server_for(factory, provider):
    from exp.r17_integration.common.serving import make_bundle_validator
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from openpi.serving import websocket_policy_server as registry
    sock = socket.socket()
    sock.bind(('127.0.0.1', 0))
    port = sock.getsockname()[1]
    sock.close()
    ready = threading.Event()
    saved = (dict(registry._bundles), registry._current_bundle, registry._bundle_version)
    server = WebsocketPolicyServer(factory(None, 'default'), host='127.0.0.1', port=port, concurrent=True,
                                  connection_policy_factory=factory, ready_callback=ready.set,
                                  bundle_validator=make_bundle_validator(provider))
    thread = threading.Thread(target=lambda: server.serve_forever(stop_on_request=True), daemon=True)
    thread.start()
    try:
        assert ready.wait(10)
        yield port, server
    finally:
        server.request_stop()
        thread.join(10)
        assert not thread.is_alive()
        with registry._bundle_lock:
            registry._bundles.clear()
            registry._bundles.update(saved[0])
            registry._current_bundle, registry._bundle_version = saved[1:]


def test_state_only_tail_matches_native_normalization_without_image_or_tokenizer():
    adapter, model = fixture_adapter()
    obs = observation()
    reference = adapter.prepare(obs, request()).state[:4].copy()
    # The old extract_state_only invoked this whole transform, requiring image.
    state = adapter.extract_state_only({'observation/state': obs['observation/state']}).state
    assert state.tobytes() == reference.tobytes()
    assert model.counts == [0, 0, 0]
    class Poison:
        transforms = adapter.policy._input_transform.transforms
        def __call__(self, data):
            raise AssertionError('vision/tokenizer ran during state-only tail')
    adapter.policy._input_transform = Poison()
    assert adapter.extract_state_only(obs).state.tobytes() == reference.tobytes()


def test_yaml_five_layers_independent_of_miss_k_and_content_bound_hot_switch(tmp_path):
    from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
    adapter, model = fixture_adapter()
    provider = register_provider(adapter, provider_id='p2_bundle_' + tmp_path.name)
    store, recipe = tiny_store(adapter, tmp_path/'teacher')
    documents = {
        'plain2': config_document(provider, k=2, evidence_dir=tmp_path/'miss'),
        'plain1': config_document(provider, k=1, evidence_dir=tmp_path/'miss'),
        'recipe10': config_document(provider, k=10, evidence_dir=tmp_path/'miss',
                                    recipe={**recipe, 'guard': False, 'periodic_gap': None}),
        'force10': config_document(provider, k=10, evidence_dir=tmp_path/'miss',
                                   recipe={**recipe, 'guard': True, 'corrector': False, 'periodic_gap': 0}),
        'gap2': config_document(provider, k=2, evidence_dir=tmp_path/'miss',
                               recipe={**recipe, 'periodic_gap': 3}),
    }
    configs, paths = {}, {}
    for name, doc in documents.items():
        path = paths[name] = tmp_path/(name+'.yaml')
        path.write_text(yaml.safe_dump(doc))
        configs[name] = load_cache_config(path)
    assert len({bundle_identity(c) for c in configs.values()}) == 5
    assert configs['recipe10'].recipe.enabled and not configs['plain2'].recipe.enabled
    assert recipe_identity(configs['recipe10']) != recipe_identity(configs['force10'])
    assert fit_identity(configs['recipe10']) != fit_identity(configs['force10'])
    factory = make_connection_factory(provider, configs['plain2'], build_shared_storage(configs['plain2']))
    with server_for(factory, provider) as (port, server):
        client = WebsocketClientPolicy('127.0.0.1', port)
        other = WebsocketClientPolicy('127.0.0.1', port)
        try:
            for name in documents:
                ack = client.load_cache_config(str(paths[name]), bundle_id=name, yaml_id=name)
                assert ack['bundle_identity'] == bundle_identity(configs[name])
            for c, name, source, nfe in [(client, 'recipe10', 'library', 0), (other, 'plain2', 'policy', 2),
                                         (client, 'plain1', 'policy', 1), (other, 'force10', 'policy', 10)]:
                c.select_bundle(name)
                c.episode_start(experiment='metaworld_mt50', task='reach-v3', episode_id=7,
                                extra_metadata={'campaign': 'round1', 'arm': name, 'bundle_generation': name,
                                                'task_id': 43, 'task_name': 'reach-v3', 'original_init_idx': 7,
                                                'seed': 7, 'task_uid': name+'_episode', 'attempt': 1,
                                                'task_data_hash': 'fixture', 'ack_protocol': 'r17.execution.v1',
                                                'bundle_identity': bundle_identity(configs[name])})
                out = c.infer({**observation(), '__request__': {'explicit_seed': 123}})
                hit = out['__hit_meta__']
                assert hit['source'] == source and hit['actual_nfe'] == nfe
                assert hit['bundle_identity'] == bundle_identity(configs[name])
                assert hit['cost_ms'] >= 0 and hit['stage_ms'] and out['server_timing']['infer_ms'] >= 0
                assert not hit['capture_enabled'] and hit['pricing_eligible']
                c.episode_end(success=False, execution={'decision_seq': 0, 'executed_count': 5, 'terminal': True})
            core = factory(None, 'gap2')
            assert core.recipe.periodic_gap == 3
            assert core.recipe.use_guard and core.recipe.use_corrector
            # A changed store digest cannot replace an already loaded generation.
            bad = tmp_path/'bad.yaml'
            bad.write_text(yaml.safe_dump({**documents['recipe10'], 'recipe': {**documents['recipe10']['recipe'], 'store_digest': 'bad'}}))
            with pytest.raises(RuntimeError):
                client.load_cache_config(str(bad), bundle_id='bad')
            assert factory(None, 'plain2').recipe is None
        finally:
            client.close()
            other.close()
