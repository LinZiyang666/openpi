"""Build an isolated worker transfer archive and exact local/worker SHA lists.

Definitions only. No sync, SSH, server, worker, or model launch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tarfile

import yaml

from exp.offline_search.closed_loop.ops.h100.control import source_files
from .prepare import HERE, REPO, atomic, write_json
from .conductor import sha, read_group


def build(island):
    if not island.startswith('/scratch/zixuans8/openpi_r16_conductor_'):
        raise ValueError('Worker island must be isolated from the running line')
    # The worker needs executable modules and frozen init bytes, not research
    # outcome/config trees from the general serving sync closure. Filter names
    # before any hash/read; the LIBERO config is generated below.
    files={str(rel):path for path,rel in source_files() if path.suffix in ('.py','.init')}
    # Regular package markers must accompany the static closure. A namespace
    # package in the island can otherwise lose to a regular package from an
    # editable checkout later on sys.path (notably exp/__init__.py).
    for path in list(files.values()):
        for parent in path.parents:
            if parent==REPO or not parent.is_relative_to(REPO):break
            marker=parent/'__init__.py'
            if marker.is_file():files[str(marker.relative_to(REPO))]=marker
    client='packages/openpi-client/src/'
    files.update({'src/'+name[len(client):]:path for name,path in list(files.items()) if name.startswith(client)})
    destination=HERE/'conductor_deployment';destination.mkdir(parents=True,exist_ok=True)
    marker=destination/'module_root.txt'
    atomic(marker,b'R16 worker module root sentinel; arm emission is coordinator-only.\n')
    # emit_arms discovers the project root by this directory at import time;
    # no template or experiment outcomes are needed by the worker agent.
    files['exp/trace_dual/config/.r16_worker_root']=marker
    config=destination/'libero_config.yaml'
    config.write_text(yaml.safe_dump(dict(assets='/home/zixuans8/.cache/libero/assets',
        bddl_files='/scratch/zixuans8/libero_sim/lib/python3.8/site-packages/libero/libero/bddl_files',
        benchmark_root='/scratch/zixuans8/libero_sim/lib/python3.8/site-packages/libero/libero',
        datasets='/scratch/zixuans8/libero_sim/lib/python3.8/site-packages/libero/datasets',
        init_states='/scratch/zixuans8/libero_sim/lib/python3.8/site-packages/libero/libero/init_files')))
    files['os_cl/libero/config.yaml']=config
    hashes={name:sha(path) for name,path in sorted(files.items())}
    checks=destination/'worker.sha256';atomic(checks,''.join(f'{h}  {n}\n' for n,h in hashes.items()).encode())
    archive=destination/'worker.tar'
    with tarfile.open(archive,'w') as tf:
        for name,path in sorted(files.items()):
            if path.is_symlink():raise ValueError('Symlink deployment source refused')
            tf.add(path,arcname=name,recursive=False)
        tf.add(checks,arcname='worker.sha256',recursive=False)
    local={str(path):sha(path) for path in sorted(HERE.glob('*.py'))}
    local.update({str(path):hashes[name] for name,path in files.items()
                  if not name.startswith('src/openpi_client/') and path!=config})
    for name in ('CONDUCTOR_LOCK.json','CONDUCTOR_AMENDMENT.json','DESIGN_LOCK.json','design.json','PROTOCOL.md',
                 'protocol.json','COORDINATOR_DECISIONS.md','conductor_commands.sh','RUNBOOK_CONDUCTOR.md',
                 'HANDBACK_CONDUCTOR.md','conductor_runtime_proof.json','conductor_tests.xml'):
        local[str(HERE/name)]=sha(HERE/name)
    for name in ('conductor_validation.json','conductor_memory.json'):
        if (HERE/name).exists():local[str(HERE/name)]=sha(HERE/name)
    groups=[]
    for name in ('conductor_dev_groups.json','conductor_smoke_groups.json','conductor_test_groups.json'):
        if not (HERE/name).exists():continue
        local[str(HERE/name)]=sha(HERE/name)
        for p in json.loads((HERE/name).read_bytes())['groups']:
            group=read_group(p);groups.append(p);local[p]=sha(p)
            local[str(Path(group['root'])/'ROOT.json')]=sha(Path(group['root'])/'ROOT.json')
            local.update(group['definitions_sha256'])
            for e in group['entries']:
                fit=e['row']['plugin_args'][e['row']['plugin_args'].index('--os-fit-artifact')+1]
                local[fit]=e['fit_sha256']
    for path in sorted(HERE.glob('queue_conductor_*.txt')):
        local[str(path)]=sha(path)
    atomic(destination/'local.sha256',''.join(f'{h}  {n}\n' for n,h in sorted(local.items())).encode())
    write_json(destination/'manifest.json',dict(schema='r16.conductor.deployment.v1',worker_host='timan107',
        worker_island=island,worker_files={n:dict(source=str(files[n]),sha256=h,destination=island+'/'+n) for n,h in hashes.items()},
        worker_archive=dict(path=str(archive),sha256=sha(archive),bytes=archive.stat().st_size),
        local_server_files=local,groups=groups,shared_runtime_edits=[],
        worker_client_mirror='src/openpi_client is a regular copied tree for the existing conda worker PYTHONPATH',
        worker_source_rule='Static Python source closure plus frozen init pools; research JSON/YAML excluded before reading',
        generated_import_root_sentinel='exp/trace_dual/config/.r16_worker_root; no arm templates copied',
        test_a_outcome_reads=0,remote_operations=0,launches=0))
    print(f'BUILT {len(hashes)} worker files, archive bytes={archive.stat().st_size}; no deployment')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--island',default='/scratch/zixuans8/openpi_r16_conductor_v1')
    build(ap.parse_args().island)


if __name__=='__main__':main()
