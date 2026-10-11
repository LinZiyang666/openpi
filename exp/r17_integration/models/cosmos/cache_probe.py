"""CPU preflight of real B500 fits, native wire transform, and resident YAML binds.

VAE/denoiser forwards are mocked explicitly. Library math, Q3 controller,
provider cloning, bundle publication and normalized-to-native output are real.
This is not a GPU qualification or an episode-success measurement.
"""
from __future__ import annotations

import argparse
import ast
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import numpy as np


def native_inverse(vendor):
    source = Path(vendor)/'cosmos_policy/experiments/robot/cosmos_utils.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'unnormalize_actions')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['unnormalize_actions']


def current_provider(checkpoint, vendor, *, register=True):
    from .loader import LoadSpec, cpu_manifest
    from .manifest import make_sampler
    from .provider import configure_library_provider
    from exp.r17_integration.common.provider import AdapterProvider, register_provider
    from exp.r17_integration.ops.cache_study import profiles, read
    parent = read('/data/r17_library/cosmos_libero_libero_spatial/bpool/READY.json')
    _, serving = profiles('cosmos_libero', parent)
    manifest = cpu_manifest(LoadSpec(str(checkpoint), vendor_root=str(vendor), benchmark_protocols=(serving.identity,)))
    provider = configure_library_provider(AdapterProvider(manifest.spec.model_id, manifest.spec, lambda: None, make_sampler))
    if register:
        register_provider(provider)
    return manifest, provider


def run_probe(output, *, checkpoint='/data/mw_finetune/models/cosmos_libero',
              vendor='/home/weiland/projects/openpi_ext/lines/cosmos-policy',
              study='exp/r17_integration/ops/cache_study/yaml/STUDY.json'):
    from .loader import read_stats, LoadSpec
    from .manifest import make_sampler
    from .provider import register_provider
    from .staged import CosmosAdapter
    from .protocols import PROPRIO_KEY
    from exp.r17_integration.common.config import config_from_dict, build_shared_storage
    from exp.r17_integration.common.contracts import Stage1, Stage2, SampleResult, digest
    from exp.r17_integration.common.qualification import config_for_probe
    from exp.r17_integration.common.serving import make_connection_factory, make_bundle_validator
    from exp.r17_integration.common.websocket_policy_server import WebsocketPolicyServer
    from exp.r17_integration.ops import cache_study as cache
    from exp.r17_integration.search.store import atomic_json
    from exp.r17_integration.common.recipe import bundle_identity
    import yaml

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest, _ = current_provider(checkpoint, vendor, register=False)
    model = SimpleNamespace(fields=None, stage1=0, stage2=0, nfe=0)
    stats = {k:np.asarray(v) for k,v in read_stats(LoadSpec(str(checkpoint))).items()}
    inverse = native_inverse(vendor)
    adapter = CosmosAdapter(model, manifest, SimpleNamespace(), stats,
                            {'cpu prompt': None}, SimpleNamespace(unnormalize_actions=inverse))
    provider = register_provider(adapter)
    default = config_for_probe(provider, 5, 16)
    default.enabled = False
    default.checkpoints = {}
    default.recipe.enabled = False
    prepared = cache.validate_manifest(study)
    report = dict(schema='r17.cosmos.cache_cpu_preflight.v1', status='PASS', cpu_only=True,
        gpu_executed=False, model=asdict(manifest.spec), study_digest=prepared['digest'], groups=[],
        forward_scope='VAE/denoiser mocked; recorded native VAE keys, real B500 fits/Q3/native inverse',
        server=dict(replicas=2, batch=1, mps=False, no_replica_restart=True), alias_pairs=cache.ALIASES)

    def stage1(self, prepared):
        model.stage1 += 1
        fields = {k:v.copy() for k,v in model.fields.items()}
        fields['robot_state'] = prepared.state.copy()
        return Stage1({}, fields, prepared.state.copy())
    def stage2(self, first, *, capture=False):
        assert capture is False
        model.stage2 += 1
        return Stage2({}, first)
    def sample(self, second, sampler, rng, capture):
        assert sampler == make_sampler(5) and rng.seed_for() == 195 and not capture.enabled
        model.nfe += 5
        return SampleResult(np.zeros((16,7), np.float32), 5, sampler)
    class Tracker:
        """Controller input fixture, not an estimate of real stall frequency."""
        def __init__(self, *args): self.step = -1
        def observe(self, code, control): self.step += 1
        def status(self):
            return dict(state=['ok', 'slow_ambiguous', 'slow_confirmed', 'slow_confirmed', 'ok'][self.step], W=2)

    with patch.object(CosmosAdapter, 'run_stage1', stage1), patch.object(CosmosAdapter, 'run_stage2', stage2), \
            patch.object(CosmosAdapter, 'sample', sample):
        factory = make_connection_factory(provider, default, build_shared_storage(default))
        server = WebsocketPolicyServer(factory(), host='127.0.0.1', port=0, concurrent=True,
            connection_policy_factory=factory, bundle_validator=make_bundle_validator(provider))
        previous = None
        for row in prepared['models']['cosmos_libero']:
            parent = cache.read(Path(row['library'])/'READY.json')
            source, serving = cache.profiles('cosmos_libero', parent)
            view = Path(row['library'])/'subsets/size_500'
            document = cache.config_document(provider, serving, source, parent, cache.read(view/'VIEW.json'),
                store_path=view, fit_path=row['fit'], fit_digest=row['fit_digest'],
                stall_path=row['stall'], stall_digest=row['stall_digest'], settings=row['settings'])
            config = config_from_dict(document)
            name = row['suite']+'_'+row['group']
            path = output/(name+'.yaml')
            path.write_text(yaml.safe_dump(document, sort_keys=False))
            bundle = 'cpu_cosmos_'+uuid.uuid4().hex
            ack = asyncio.run(server._load_r17_bundle(dict(yaml_path=str(path), yaml_id=name,
                bundle_id=bundle, bundle_identity=bundle_identity(config))))
            core = factory(bundle_id=bundle)
            other = factory(bundle_id=bundle)
            assert (core.adapter.model is adapter.model and core.adapter.lock is adapter.lock and
                    other.adapter.model is adapter.model and other.adapter.keys is not core.adapter.keys and
                    core.recipe is not other.recipe and core.recipe.base.action is other.recipe.base.action)
            if previous is not None:
                assert previous.adapter.model is core.adapter.model
            previous = core
            if config.recipe.miss_mode == 'stall_only': core.recipe.tracker_class = Tracker
            fitted = core.recipe.base
            i = next(i for i,t in enumerate(fitted.task) if int(t) in (0,30))
            index = np.load(view/'row_indices.npy', allow_pickle=False)[i]
            model.fields = {field:np.load(Path(row['library'])/parent['arrays']['key_'+field]['file'],
                mmap_mode='r', allow_pickle=False)[index].astype(np.float32)
                for field in manifest.spec.feature_spec.fields if field != 'robot_state'}
            tid = int(fitted.task[i])
            core.on_episode_start(extra_metadata=dict(task_name=serving.ordered_tasks[tid], task_id=tid,
                campaign='cosmos_cache_cpu_preflight', original_init_idx=0, seed=195, explicit_seed=195))
            obs = {'observation/image':np.zeros((256,256,3),np.uint8),
                   'observation/wrist_image':np.ones((256,256,3),np.uint8),
                   PROPRIO_KEY:fitted.state[i].copy(), 'prompt':'cpu prompt'}
            traces = []
            start = model.stage1, model.stage2, model.nfe
            for seq in range(5):
                request = dict(obs)
                if seq: request['__execution__'] = dict(decision_seq=seq-1, executed_count=16)
                result = core.infer(request)
                meta = result['__hit_meta__']
                assert result['actions'].shape == (16,7)
                assert result['actions'].tobytes() == inverse(core._chunk[None], stats)[0].tobytes()
                assert meta['actual_nfe'] == (5 if meta['source'] == 'policy' else 0)
                assert meta['source'] == 'library' or meta.get('stall_call') is True
                traces.append({k:meta.get(k) for k in ('source','hit_type','actual_nfe','stage_calls',
                    'stall_call','stall_status','stall_extra_look','stall_cooldown','corrector_enabled',
                    'corrector_applied','resident_pid','resident_generation')})
            core.on_episode_end(execution=dict(decision_seq=4, executed_count=16, terminal=True))
            delta = model.stage1-start[0], model.stage2-start[1], model.nfe-start[2]
            expected = (5, 1, 5) if row['settings']['gated'] else (5, 0, 0)
            assert delta == expected
            assert core._awaiting is None and other.recipe.controls == 0
            core.on_episode_start(extra_metadata=dict(task_name=serving.ordered_tasks[tid], task_id=tid,
                campaign='cosmos_cache_cpu_reset', original_init_idx=1, seed=195, explicit_seed=195))
            assert core.recipe.controls == 0 and core.recipe.tracker is None
            report['groups'].append(dict(suite=row['suite'], group=row['group'], status='PASS',
                C=16, yaml_sha256=cache.file_sha(path), bundle_identity=ack['bundle_identity'],
                teacher_compatibility=other.recipe.teacher_compatibility,
                stage_totals=list(delta), traces=traces, action_transform_byte_exact=True,
                fresh_wrappers_shared_resident=True))
    generations = {(t['resident_pid'],t['resident_generation']) for g in report['groups'] for t in g['traces']}
    assert len(generations) == 1
    identities = {(r['suite'],r['group']):r['bundle_identity'] for r in report['groups']}
    for suite in cache.SUITES:
        for alias, original in cache.ALIASES.items():
            assert identities[suite,alias] == identities[suite,original]
    report['resident_generations'] = len(generations)
    report['digest'] = digest(report)
    atomic_json(output/'PREFLIGHT.json', report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preflight', action='store_true', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--checkpoint', default='/data/mw_finetune/models/cosmos_libero')
    p.add_argument('--vendor', default='/home/weiland/projects/openpi_ext/lines/cosmos-policy')
    p.add_argument('--study', default='exp/r17_integration/ops/cache_study/yaml/STUDY.json')
    a = p.parse_args()
    result = run_probe(a.output, checkpoint=a.checkpoint, vendor=a.vendor, study=a.study)
    print(json.dumps(dict(status=result['status'], groups=len(result['groups']), cpu_only=True, output=a.output)))


if __name__ == '__main__':
    main()
