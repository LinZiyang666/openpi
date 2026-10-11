"""CPU-only real B500 admission/query check through P6's actual sealing seam."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import os
from pathlib import Path
import uuid

from exp.r17_integration.common.contracts import digest


def current_manifest(serving):
    """Metadata reconstruction only; physical weights are sealed by P6 prepare.

    No weight load or synthetic teacher data. The published content pins must
    match a later load_adapter/seal_descriptor receipt before GPU serving.
    """
    from .library_evidence import EVIDENCE
    from exp.r17_integration.ops.cache_study import model_spec
    content = EVIDENCE['resident_content_digests'][-1]
    value = copy.deepcopy(EVIDENCE['teacher_model_spec'])
    value['checkpoint_identity'] = digest(content)
    value['code_digest'] = digest({k:v for k,v in content.items() if k.startswith('code:')})
    value['feature_spec']['policy_digest'] = value['checkpoint_identity']
    value['allowed_benchmark_protocols'] = [serving.identity]
    spec = model_spec(value)
    return dict(schema='r17.model_manifest.v1', spec=asdict(spec), content_digests=copy.deepcopy(content)), spec


def preflight(study_path, output):
    # Match P6 server_env's one-thread BLAS. Q3 hashes the deployed float32
    # early metric bytes; recomputing them with another BLAS thread count can
    # differ by a few ulps and must not weaken the strict artifact fence.
    if any(os.environ.get(name) != '1' for name in
           ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')):
        raise ValueError('Run CPU preflight with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1, matching P6 server_env')
    return _preflight(study_path, output)


def _preflight(study_path, output):
    import numpy as np
    from exp.r17_integration.ops import cache_study as study
    from exp.r17_integration.common.provider import AdapterProvider, register_provider
    from exp.r17_integration.common.config import load_cache_config
    from exp.r17_integration.common.recipe import make_recipe_factory
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.common.query_view import QueryView
    from exp.r17_integration.models.fastwam.key_builder import FastWAMKeyBuilder
    from exp.r17_integration.models.fastwam.provider import configure_library_provider
    from exp.r17_integration.search.store import atomic_json
    from exp.r17_integration.ops.formal_eval import benchmark
    from exp.r17_integration.common.benchmarks import register_benchmark
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    assets = output/'assets'; assets.mkdir()
    serving = benchmark(True); register_benchmark(serving)
    manifest, spec = current_manifest(serving)
    provider = AdapterProvider('p3_cache_cpu_'+uuid.uuid4().hex, spec, lambda: None,
        lambda k: spec.default_sampler if k == 10 else (_ for _ in ()).throw(ValueError('K10 only')),
        key_builder_factory=lambda _: FastWAMKeyBuilder(spec.feature_spec))
    original = provider
    provider = configure_library_provider(provider, manifest=manifest)
    assert provider.provider_id == original.provider_id and provider.model_spec == original.model_spec
    assert original.library_validator is None
    register_provider(provider)
    atomic_json(assets/'model_manifest.json', manifest)
    atomic_json(assets/'model_spec.json', asdict(spec))
    atomic_json(assets/'benchmark.json', serving.manifest())
    campaign = dict(model='fastwam', cache_study=str(Path(study_path).resolve()),
        server_node='local', assets=str(assets.resolve()), local_assets=str(assets.resolve()),
        run_name=output.name, smoke=True, replicas=2, workers=48,
        endpoint_host='127.0.0.1', public_port=23420)
    study.stage_inputs(campaign, output, None)  # Local CPU metadata/fit copy only.
    study.seal(provider, campaign, serving)
    sealed = study.read(assets/'CACHE_STUDY.json')
    entries = []
    factory = make_recipe_factory(provider)
    for row in sealed['groups']:
        config = load_cache_config(assets/row['file'])
        recipe = factory(config, None)
        base = recipe.base
        # Query a recorded B key for each task: actual fitted B500 math, no RGB
        # or 49152D feature matrix copied. B donors are never evaluation tasks.
        index = np.load(Path(config.recipe.store_path)/'row_indices.npy', allow_pickle=False)
        parent = Path(row['library'])
        metadata = study.read(parent/'READY.json')
        task_rows = [next(i for i,t in enumerate(base.task) if int(t) == tid)
                     for tid in sorted(set(map(int, base.task)))]
        checks = []
        for i in task_rows:
            recipe.reset(None)
            fields = {name: np.load(parent/metadata['arrays']['key_'+name]['file'],
                mmap_mode='r', allow_pickle=False)[index[i]].copy()
                for name in spec.feature_spec.fields if name != 'robot_state'}
            fields['robot_state'] = base.state[i].copy()
            tid = int(base.task[i]); states = []
            for seq in range(4):
                if seq: recipe.on_execution(spec.layout.replan_steps)
                query = QueryView(RequestContext('p3_CPU_B_readonly', serving.identity,
                    serving.ordered_tasks[tid], 0, seq, task_id=tid), fields,
                    base.state[i].copy(), [True]*seq, [1]*seq,
                    previous_action=base.action[i] if seq else None)
                result = recipe.query(query)
                assert result.action.shape == (32,7) and np.isfinite(result.action).all()
                assert not result.force_miss or result.extras['stall_call']
                if config.recipe.miss_mode == 'never': assert not result.force_miss
                states.append(dict(stall=result.extras['stall_status']['state'],
                    stall_call=bool(result.force_miss), corrector=result.extras['corrector_applied']))
            checks.append(dict(task_id=tid, queries=states))
        entries.append(dict(suite=row['suite'], group=row['group'], status='PASS',
            compatibility=recipe.teacher_compatibility, checks=checks))
    _, _, tasks, _ = study.plan(campaign)
    assert len(tasks) == 32 and all(t.extra['init_pool'] == 'A' for t in tasks)
    report = dict(schema='r17.fastwam.cache_cpu_preflight.v1', status='PASS', cpu_only=True,
        gpu_executed=False, physical_weight_seal='pending P6 prepare-only',
        study_digest=sealed['source_study_digest'], resident_identity=spec.identity,
        groups=entries, query_count=sum(len(e['checks'])*4 for e in entries),
        smoke_episodes=len(tasks), seal=str(assets/'CACHE_STUDY.json'),
        horizon_selection=study.read(study_path)['horizon_selection'])
    report['digest'] = digest(report)
    atomic_json(output/'PREFLIGHT.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, default=Path('exp/r17_integration/ops/cache_study/yaml/STUDY.json'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = preflight(args.study, args.output)
    print({key: result[key] for key in ('status', 'query_count', 'smoke_episodes', 'seal')})


if __name__ == '__main__':
    main()
