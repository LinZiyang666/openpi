"""Write a code/payload hash inventory without deployment or data-corpus reads."""
import difflib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import platform
import sys
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[5]
SHARED = (
    'exp/offline_search/closed_loop/plugin.py',
    'exp/offline_search/closed_loop/ops/h100/assets.py',
    'exp/offline_search/rounds/r04/k4_eval/cost_ledger.py',
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while b := stream.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def inventory():
    # Import only serving/collection dependencies; no fit, servers, GPU or A.
    from exp.offline_search.closed_loop import plugin, stage_overrides
    from exp.offline_search.rounds.r04.k4_eval import cost_ledger
    from . import runtime, recording, accounting
    from .encoder import FrozenEncoder
    safety = json.loads((HERE/'descent_safety.json').read_text())
    assert safety['PASS']
    shared_server = {r['path']:r for r in safety['shared_server_files']}
    cells = {}
    for size in (50, 200):
        manifest = json.loads((HERE/f'artifacts/pi05_l10_{size}/manifest.json').read_text())
        for record in manifest['records'].values():
            assert digest(record['path']) == record['sha256']
            blob = pickle.loads(Path(record['path']).read_bytes())
            assert blob['method'].light_key_spec == record['kwargs']['encoder_spec']
            pickle.loads(Path(record['baseline']).read_bytes())
        spec = manifest['records']['off']['kwargs']['encoder_spec']
        encoder = FrozenEncoder(spec)
        output = encoder.encode(SimpleNamespace(img0=np.zeros((224,224,3),np.uint8),
            img1=np.zeros((224,224,3),np.uint8),rs=np.zeros(8,np.float32),task_id=0))
        assert output.shape == (2,64) and np.isfinite(output).all()
        cells[str(size)] = dict(manifest=str(HERE/f'artifacts/pi05_l10_{size}/manifest.json'),
            manifest_sha256=digest(HERE/f'artifacts/pi05_l10_{size}/manifest.json'),
            encoder=spec['kwargs'],fits={k:dict(path=r['path'],sha256=r['sha256'])
                                       for k,r in manifest['records'].items()},
            representation_sha256=spec['representation_sha256'])
    assert not torch.cuda.is_initialized()
    # Conservative code snapshot: the same project source roots used by the
    # ordinary fleet deployer. Only .py files are hashed here; no A init files,
    # identity data, evaluation arrays or model corpora are inspected.
    roots = ('src','scripts','packages/openpi-client/src','exp/offline_search',
             'exp/libero_groot','exp/gate_threshold_pareto',
             'exp/ablation_study/cache_size','examples/libero')
    sources = set()
    for root in roots:
        sources.update(p for p in (REPO/root).rglob('*.py')
                       if '__pycache__' not in p.parts and p.is_file())
    observed = []
    for module in tuple(sys.modules.values()):
        path = getattr(module,'__file__',None)
        if path:
            path = Path(path).resolve()
            if (path.suffix == '.py' and path.is_file() and path.is_relative_to(REPO)
                    and '.venv' not in path.parts):
                sources.add(path)
                observed.append(str(path.relative_to(REPO)))
    for path in tuple(sources):
        for parent in path.parents:
            if parent == REPO:
                break
            init = parent/'__init__.py'
            if init.is_file():
                sources.add(init)
    sources.update(REPO/name for name in shared_server)
    code = [dict(path=str(p.relative_to(REPO)),sha256=digest(p),bytes=p.stat().st_size)
            for p in sorted(sources)]
    qk_names = ['exp/offline_search/rounds/r15/qk/impl/'+name for name in
                 ('__init__.py','encoder.py','recipe.py','runtime.py','recording.py','accounting.py')]
    for entry in code:
        entry['shared_descent_server_source'] = entry['path'] in shared_server
        entry['QK_only_runtime'] = entry['path'] in qk_names
    plans = []
    payloads = {}
    for host in ('local','h100'):
        for size in (50,200):
            path = HERE/f'plans/{host}/r15_qk_pilot_pi05_l10_{size}/plan.json'
            plan = json.loads(path.read_text())
            plans.append(dict(host=host,path=str(path),sha256=digest(path),
                              files=len(plan['files']),bytes=plan['bytes']))
            for f in plan['files']:
                assert '/queries/' not in f['original']
                payloads[(host,f['original'])] = dict(host=host,original=f['original'],
                    staged=f['source'],target=f['rel'],source_sha256=f['source_sha256'],
                    served_sha256=f['sha256'],bytes=f['size'],reasons=f['reasons'])
    changes, patch = [], []
    for name in SHARED:
        old = HERE/'before'/Path(name).name
        new = REPO/name
        changes.append(dict(path=name,before_sha256=digest(old),after_sha256=digest(new)))
        patch.extend(difflib.unified_diff(old.read_text().splitlines(True),
            new.read_text().splitlines(True),fromfile='before/'+name,tofile=name))
    (HERE/'shared_changes.diff').write_text(''.join(patch))
    write(HERE/'shared_changes.json',changes)
    versions = {}
    for name in ('torch','numpy','scipy','scikit-learn','websockets','msgpack','PyYAML'):
        try:
            versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name]=None
    code_by_name = {r['path']:r for r in code}
    delta_code = [dict(code_by_name[name],role='QK only; not imported by disabled descent servers') for name in qk_names]
    delta_code += [dict(code_by_name[r['path']],role=r['role'],default_off_AST_equal=r['default_off_AST_equal'])
                   for r in safety['shared_changed_files']]
    definitions = []
    for size in (50,200):
        root = Path('/home/weiland/trace_runs/os_closed_loop')/f'r15_qk_pilot_pi05_l10_{size}'
        seal = json.loads((root/'ROOT.json').read_text())
        files = [dict(path=str(root/name),sha256=checksum) for name,checksum in seal['definitions_sha256'].items()]
        files.append(dict(path=str(root/'ROOT.json'),sha256=digest(root/'ROOT.json')))
        definitions.append(dict(root=str(root),seal_sha256=digest(root/'ROOT.json'),files=files))
    # Ordinary fleet planners defer B-worker pool staging to execution. List
    # those existing B-only inputs explicitly, without opening any A pool.
    from exp.offline_search.closed_loop.devset import pool_record
    from exp.offline_search.closed_loop.ops.h100.control import island
    import yaml
    pool_path, pool = pool_record('l10',pool='B')
    pool_dir = Path(pool['apool_dir'])
    assert pool_dir == REPO/'exp/common/data/db_init/libero/libero_10'
    worker_pool = [dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size,
                        target=str(p.relative_to(REPO)),role='existing B worker init; reuse if hash matches')
                   for p in sorted(pool_dir.glob('*.init'))]
    assert len(worker_pool) == 10
    for entry in worker_pool:
        assert entry['sha256'] == pool['per_task_digests'][Path(entry['path']).stem]
    worker_configs = []
    for host in ('local','h100'):
        plan=json.loads((HERE/f'plans/{host}/r15_qk_pilot_pi05_l10_50/plan.json').read_text())
        config=dict(pool,apool_dir=str(island(plan['worker_host'])/pool_dir.relative_to(REPO)))
        directory=HERE/'deployment_worker'/host
        directory.mkdir(parents=True,exist_ok=True)
        staged=directory/pool_path.name
        staged.write_text(yaml.safe_dump(config,sort_keys=False))
        worker_configs.append(dict(host=host,worker_host=plan['worker_host'],source=str(pool_path),
            source_sha256=digest(pool_path),staged=str(staged),served_sha256=digest(staged),
            target=str(pool_path.relative_to(REPO))))
    report = dict(PASS=True,cells=cells,shared_changes=changes,
        required_QK_modules=qk_names,
        required_code_updates=delta_code,
        shared_descent_server_files=safety['shared_server_files'],
        shared_descent_proof=dict(path=str(HERE/'descent_safety.json'),sha256=digest(HERE/'descent_safety.json'),
            changed_shared_server_files=safety['changed_shared_server_files'],
            no_QK_runtime_imports_when_disabled=True,assumptions=safety['assumptions'],
            deployment_notes=safety['deployment_notes']),
        serving_files_unchanged_during_final_swap=safety['serving_files_unchanged_during_final_swap'],
        code_scope='conservative fleet Python source snapshot; includes unchanged existing dependencies',
        code_files=code,observed_serving_modules=sorted(set(observed)),
        payload_files=list(payloads.values()),plans=plans,
        definitions=definitions,
        worker_B_inputs=worker_pool,worker_B_pool_configs=worker_configs,
        coordinator_queue=dict(path=str(HERE/'queue_pilot.txt'),sha256=digest(HERE/'queue_pilot.txt')),
        handoff_documents=[dict(path=str(HERE/name),sha256=digest(HERE/name))
                           for name in ('HANDBACK.md','DEPLOYMENT_SAFETY.md')],
        completion_audit=dict(path=str(HERE/'completion_audit.json'),sha256=digest(HERE/'completion_audit.json')),
        environment=dict(python=platform.python_version(),packages=versions,torch_runtime=torch.__version__,
            threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
            cpu_affinity=sorted(os.sched_getaffinity(0)),CUDA_initialized=False,
            new_packages_expected=False,
            h100_checked=False,
            h100_requirements='Existing full pi05 environment plus CPU TorchScript support. Check export load/forward '
                f'with compatible PyTorch (local {torch.__version__}), NumPy and existing plugin dependencies; Linux affinity '
                'must intersect CPUs 2-9,46-53. One Torch/interop thread and OMP/BLAS thread. Frozen TS requires '
                'no astra training code/checkpoint, library RGB, or derived evaluation arrays.'),
        operations=dict(deployments=0,remote=0,servers=0,GPU=0,test_A_reads=0))
    write(HERE/'deployment_manifest.json',report)
    print('DEPLOYMENT_MANIFEST_PASS',len(code),'code files',len(payloads),'host payload entries')
    return report


if __name__=='__main__':
    inventory()
