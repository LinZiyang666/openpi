"""Copy a 4090 long-segment job (built, or already run) to a new job served on a jupyter pod.

Only where the model is served changes: server_node 'h100' (bound to the pod by the
R17_H100_NODE alias when filter_failed.stage runs), one replica without MPS, a 14 GiB
budget. The scientific definition is the source's: assets and YAML bytes (linked),
STUDY.json, EPISODE_PLAN.json (task UIDs), FFK_DEFINITION, the frozen runtime and
worker archives of its filter_failed_source. `assets` keeps the source address because
the frozen YAMLs name it (stall_path); stage installs the same snapshot at that address
and at `local_assets` on the pod. The source directory is only read.

    python -m exp.offline_search.rounds.r17.longseg_1017.pod_copy \
        --source fa_ls_dp_r_f_m --name fa_ls_dp_r_f_m_j --pod jupyter-ziyang10-02 --driver-port 23192
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

from .audit import EVAL, HERE
from . import io_budget as budget
from exp.offline_search.rounds.r17.filter_failed_knob.common import read, REPO

H100_IP = '149.165.153.233'
# Fields a prepare/launch adds or rewrites; a finished source drops them (stage recomputes).
PREPARED = ('coordinator_runtime', 'local_worker_exception', 'orchestration', 'runtime_source_identity',
            'runtime_source_pins', 'worker_max_seconds', 'worker_node_failover', 'server_campaign',
            'cache_worker_orchestration', 'timan1_worker_lease')
DEFINITION = ('EPISODE_PLAN.json', 'STUDY.json', 'OFFLINE_READY.json', 'RECALIBRATION_APPLIED.json',
              'WORKER_REDISTRIBUTION.json')


def environment(pod, driver_port):
    return f'R17_H100_NODE={pod} R17_H100_HOST={H100_IP} R17_DRIVER_PORT={driver_port}'


def commands(name, pod, driver_port):
    """prepare_/launch_<name>.sh beside the other long-segment commands, with the pod alias."""
    paths = {}
    for verb, module in (('prepare', 'exp.offline_search.rounds.r17.filter_failed.stage'),
                         ('launch', 'exp.offline_search.rounds.r17.filter_failed_knob.launch')):
        path = HERE/'commands'/f'{verb}_{name}.sh'
        tmp = path.with_name(path.name+'.tmp')
        tmp.write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(REPO))+'\n'
                       'export '+environment(pod, driver_port)+'\n'+shlex.quote(str(REPO/'.venv/bin/python'))+
                       ' -m '+module+' --run '+shlex.quote(str(EVAL/name))+'\n')
        tmp.chmod(0o755)
        tmp.replace(path)
        paths[verb] = str(path)
    return paths


def copy_job(source, name, pod, driver_port, *, replicas=1, memory_gib=14):
    from exp.r17_integration.ops.formal_eval import POD_PROFILES
    if pod not in POD_PROFILES:
        raise ValueError('unknown pod '+pod)
    src, root = EVAL/source, EVAL/name
    if root.exists():
        raise ValueError(str(root)+' exists')
    c = read(src/'campaign.json')
    if c['server_node'] != 'local' or c['model'] != 'dp':
        raise ValueError('expects a 4090 DP job')
    ran = any((src/n).exists() for n in ('PREPARED.json', 'LAUNCH.json', 'RECEIPT.json'))
    origin = Path(c['filter_failed_source'])
    release = read(origin/'RUNTIME_RELEASE.json')
    if ran:
        for key in [k for k in c if k.startswith(('autoscale', 'scheduler_')) or k in PREPARED]:
            del c[key]
        if c['domain'] == 'metaworld':
            c['worker_max_episodes'] = 0  # as built; scheduler.defaults.apply sets MT50 recycling (32 / 1800 s)
        # Unfixed source runtime, as built; stage applies the long-segment fix and installs it
        # (stage's driver_release.prepare rebuilds driver_runtime).
        c.update(runtime=release['root'], runtime_identity=release['identity'],
                 driver_runtime='/data/mw_finetune/qualification/formal_driver_'+release['identity'])
    resources = dict(server_node='h100', replicas=replicas, replica_startup_width=1,
                     memory_budget_gib=float(memory_gib), mps=None)
    before = {k: c.get(k) for k in resources}
    c.update(run_name=name, local_assets=str(root/'assets'), server_run=str(root/'servers'),
             cache_study=str(root/'STUDY.json'), prepare_only=True, **resources)
    c['pod_copy'] = dict(source=str(src), source_ran=ran, pod=pod, driver_port=driver_port,
                         resources_before=before, resources_after=resources,
                         scientific_definition_unchanged=True,
                         reason='serve on a jupyter pod MIG slice (R1 without MPS: the MPS guard cannot read MIG via nvidia-smi)')
    root.mkdir()
    budget.tree(src/'assets', root/'assets')
    for name_ in DEFINITION:
        if (src/name_).exists():
            budget.copy(src/name_, root/name_)
    for name_ in ('runtime.tar', 'workers.tar', 'RUNTIME_RELEASE.json'):
        budget.copy(origin/name_, root/name_)
    job = read(src/'FFK_DEFINITION.json')
    job.update(name=name, run=str(root), gpu='H200 MIG 1g.18gb ('+pod+')', replicas=replicas,
               pod_copy_of=job.get('name', source))
    budget.dump(root/'FFK_DEFINITION.json', job)
    budget.dump(root/'campaign.json', c)
    paths = commands(name, pod, driver_port)
    print(json.dumps(dict(name=name, source=source, pod=pod, driver_port=driver_port, replicas=replicas,
                          memory_budget_gib=memory_gib, episodes=job.get('episodes'), **paths)), flush=True)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--pod', required=True)
    p.add_argument('--driver-port', type=int, required=True)
    p.add_argument('--replicas', type=int, default=1)
    p.add_argument('--memory-gib', type=float, default=14)
    a = p.parse_args()
    copy_job(a.source, a.name, a.pod, a.driver_port, replicas=a.replicas, memory_gib=a.memory_gib)


if __name__ == '__main__':
    main()
