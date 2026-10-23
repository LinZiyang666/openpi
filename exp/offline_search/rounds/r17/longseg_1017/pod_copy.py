"""Copy a 4090 DP or H100 GR00T/Fast-WAM long-segment job (built, or already run) to a new job served on a jupyter pod.

Only where the model is served changes: server_node 'h100' (bound to the pod by the
R17_H100_NODE alias when filter_failed.stage runs), R replicas without MPS (DP 1, GR00T 2),
a 14 GiB budget. The scientific definition is the source's: assets and YAML bytes (linked),
STUDY.json, EPISODE_PLAN.json (task UIDs), FFK_DEFINITION, the frozen runtime and
worker archives of its filter_failed_source. `assets` keeps the source address because
the frozen YAMLs name it (stall_path); stage installs the same snapshot at that address
and at `local_assets` on the pod. The source directory is only read. The scheduler fields
are dropped: stage places fleets afresh, and a pod server gets only the shared pool.
GR00T serves from the H100 env, vendor and checkpoint that ops/pod/setup.sh groot installs
on the pod and r17x binds at their H100 paths; Fast-WAM likewise (setup.sh fastwam_nfs), R1 and built
without umT5-xxl: prompt encodings come from the H100 table (fastwam_text_table, POD_TEXT_TABLE),
which fits the 16 GiB slice (~14.5 GiB reserved at B16). The pod needs tunnels for its public port
and R backend ports (formal_eval.POD_PROFILES).

    python -m exp.offline_search.rounds.r17.longseg_1017.pod_copy \
        --source fa_ls_dp_r_f_m --name fa_ls_dp_r_f_m_j --pod jupyter-ziyang10-02 --driver-port 23192
    python -m exp.offline_search.rounds.r17.longseg_1017.pod_copy \
        --source fa_ls_gr_f_u_m --name fa_ls_gr_f_u_m_j --pod jupyter-ziyang10-02 --driver-port 23192
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shlex
import time

from .audit import EVAL, HERE
from . import io_budget as budget
from exp.offline_search.rounds.r17.filter_failed_knob.common import read, REPO, sha

H100_IP = '149.165.153.233'
POD_TEXT_TABLE = '/data/mw_finetune/fastwam_text_tables/table_h100.pt'  # setup.sh fastwam_nfs binds it
# Fields a prepare/launch adds or rewrites; a finished source drops them (stage recomputes).
PREPARED = ('coordinator_runtime', 'local_worker_exception', 'orchestration', 'runtime_source_identity',
            'runtime_source_pins', 'worker_max_seconds', 'worker_node_failover', 'server_campaign',
            'cache_worker_orchestration', 'timan1_worker_lease')
DEFINITION = ('EPISODE_PLAN.json', 'STUDY.json', 'OFFLINE_READY.json', 'RECALIBRATION_APPLIED.json',
              'WORKER_REDISTRIBUTION.json')


def environment(pod, driver_port):
    return f'R17_H100_NODE={pod} R17_H100_HOST={H100_IP} R17_DRIVER_PORT={driver_port}'


def commands(name, pod, driver_port):
    """prepare_/launch_<name>.sh beside the other long-segment commands, with the pod alias,
    and the filter_failed_knob/commands wrappers that chain_ffk*.run.sh runs (when absent)."""
    from exp.offline_search.rounds.r17.filter_failed_knob.common import HERE as KNOB
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
        wrapper = KNOB/'commands'/path.name
        if not wrapper.exists():
            wrapper.write_text('#!/usr/bin/env bash\nset -euo pipefail\nexec bash '+shlex.quote(str(path))+'\n')
            wrapper.chmod(0o755)
    return paths


def pin_bridge(root, c, job):
    """Embed the current filter_failed/server_cli.py in this copy's own assets/ffk_bootstrap.py.

    The source froze the R4 admission bridge, which presents only replicas=2 to the DP cache-study
    admission; a pod copy also needs its source's server node and MPS presented (fd9b5be0). Only that
    module body changes. The old file is renamed aside (its inode may be shared); the bootstrap pins in
    campaign.json and FFK_DEFINITION.json follow. Refused once the copy has been launched."""
    if any((root/n).exists() for n in ('LAUNCH.json', 'RECEIPT.json')):
        raise ValueError('never re-pin a launched copy: '+str(root))
    boot = root/'assets/ffk_bootstrap.py'
    if sha(boot) != c['ffk_bootstrap_sha256'] or c['longseg_1017'].get('bootstrap_sha256') != c['ffk_bootstrap_sha256']:
        raise ValueError('pinned bootstrap bytes differ from campaign.json: '+str(root))
    from exp.offline_search.rounds.r17.filter_failed import server_cli
    module, path = server_cli.__name__, Path(server_cli.__file__)
    head = f'm=types.ModuleType({module!r});sys.modules[{module!r}]=m;exec(compile('
    tail = ','+repr(str(path))+',"exec"),m.__dict__)'
    lines = boot.read_text().split('\n')
    hits = [i for i, line in enumerate(lines) if line.startswith(head) and line.endswith(tail)]
    if len(hits) != 1:
        raise ValueError('expected one embedded server_cli module in '+str(boot))
    old_body = ast.literal_eval(lines[hits[0]][len(head):-len(tail)])
    new_body = path.read_text()
    if 'resources_before' in old_body:
        return dict(status='ALREADY_PINNED', bootstrap_sha256=c['ffk_bootstrap_sha256'])
    lines[hits[0]] = head+repr(new_body)+tail
    previous, stamp = c['ffk_bootstrap_sha256'], str(time.time_ns())
    boot.rename(boot.with_name(boot.name+'.pre_pod_bridge_'+stamp))
    boot.write_text('\n'.join(lines))
    c['ffk_bootstrap_sha256'] = c['longseg_1017']['bootstrap_sha256'] = job['bootstrap_sha256'] = sha(boot)
    receipt = dict(status='REPINNED', previous_bootstrap_sha256=previous, bootstrap_sha256=c['ffk_bootstrap_sha256'],
                   changed='embedded '+module+' only', commit='fd9b5be0')
    c['pod_copy']['bridge_repin'] = receipt
    return receipt


def repin(name):
    """pin_bridge for a copy made before it existed (prepared or not, never launched)."""
    root = EVAL/name
    c, job = read(root/'campaign.json'), read(root/'FFK_DEFINITION.json')
    if 'pod_copy' not in c:
        raise ValueError('not a pod copy: '+name)
    receipt = pin_bridge(root, c, job)
    if receipt['status'] == 'REPINNED':
        budget.dump(root/'FFK_DEFINITION.json', job)
        budget.dump(root/'campaign.json', c)
    print(json.dumps(dict(name=name, **receipt)), flush=True)
    return receipt


def copy_job(source, name, pod, driver_port, *, replicas=None, memory_gib=14, startup_width=1, max_batch=None):
    from exp.r17_integration.ops.formal_eval import POD_PROFILES
    if pod not in POD_PROFILES:
        raise ValueError('unknown pod '+pod)
    src, root = EVAL/source, EVAL/name
    if root.exists():
        raise ValueError(str(root)+' exists')
    c = read(src/'campaign.json')
    dp = c['model'] == 'dp'  # a 4090 or an H100 DP job (longseg_off_1022 builds DP MT50 on the H100)
    fw = (c['server_node'], c['model']) == ('h100', 'fastwam')
    if not dp and not fw and (c['server_node'], c['model']) != ('h100', 'groot'):
        raise ValueError('expects a DP job or an H100 GR00T/Fast-WAM job')
    if replicas is None:
        replicas = 1 if dp or fw else 2
    ran = any((src/n).exists() for n in ('PREPARED.json', 'LAUNCH.json', 'RECEIPT.json'))
    origin = Path(c['filter_failed_source'])
    release = read(origin/'RUNTIME_RELEASE.json')
    # Stage re-derives the fleet table (scheduler.defaults.apply): shared pool only for a pod server.
    for key in [k for k in c if k.startswith(('autoscale', 'scheduler_')) or (ran and k in PREPARED)]:
        del c[key]
    if ran:
        if c['domain'] == 'metaworld':
            c['worker_max_episodes'] = 0  # as built; scheduler.defaults.apply sets MT50 recycling (32 / 1800 s)
        # Unfixed source runtime, as built; stage applies the long-segment fix and installs it
        # (stage's driver_release.prepare rebuilds driver_runtime).
        c.update(runtime=release['root'], runtime_identity=release['identity'],
                 driver_runtime='/data/mw_finetune/qualification/formal_driver_'+release['identity'])
    resources = dict(server_node='h100', replicas=replicas, replica_startup_width=min(startup_width, replicas),
                     memory_budget_gib=float(memory_gib), mps=None)
    before = {k: c.get(k) for k in resources}
    c.update(run_name=name, local_assets=str(root/'assets'), server_run=str(root/'servers'),
             cache_study=str(root/'STUDY.json'), prepare_only=True, **resources)
    if fw:
        c['fastwam_text_table'] = POD_TEXT_TABLE
        if max_batch:  # B8 keeps the slice under ~14 GiB with a cache library (B16 reserves 14.5 GiB before it)
            for doc in (c['fastwam_batching'], c.get('fastwam_batch_treatment', {}).get('batching', {})):
                doc['max_batch'] = int(max_batch)
            c['pod_copy_max_batch'] = int(max_batch)
    c['pod_copy'] = dict(source=str(src), source_ran=ran, pod=pod, driver_port=driver_port,
                         resources_before=before, resources_after=resources,
                         scientific_definition_unchanged=True,
                         reason='serve on a jupyter pod MIG slice (R'+str(replicas)+' without MPS: the MPS guard cannot read MIG via nvidia-smi)')
    from exp.r17_integration.ops.immutable_release import nvme_dir
    nvme_dir(root)  # real dir on /nvme, /data path a symlink
    budget.tree(src/'assets', root/'assets')
    for name_ in DEFINITION:
        if (src/name_).exists():
            budget.copy(src/name_, root/name_)
    for name_ in ('runtime.tar', 'workers.tar', 'RUNTIME_RELEASE.json'):
        budget.copy(origin/name_, root/name_)
    job = read(src/'FFK_DEFINITION.json')
    job.update(name=name, run=str(root), gpu='H200 MIG 1g.18gb ('+pod+')', replicas=replicas,
               pod_copy_of=job.get('name', source))
    if dp:
        # The bridge presents the source's 4090 placement to the DP cache-study admission; a pod copy needs it
        # even when the source itself (R2) did not (fa_ls_v04_0_j2 was refused at 19:39).
        # GR00T has no cache-study resource admission (its server only caps VRAM at budget/replicas).
        c['ffk_dp_capacity_bridge'] = True
        pin_bridge(root, c, job)
    elif 'cache_worker_orchestration' in c:
        c['cache_worker_orchestration']['replicas'] = replicas  # record only (no check reads it for GR00T)
    budget.dump(root/'FFK_DEFINITION.json', job)
    budget.dump(root/'campaign.json', c)
    paths = commands(name, pod, driver_port)
    public = POD_PROFILES[pod]['public_port']
    print(json.dumps(dict(name=name, source=source, pod=pod, driver_port=driver_port, replicas=replicas,
                          memory_budget_gib=memory_gib, episodes=job.get('episodes'),
                          pod_ports=list(range(public, public+replicas+1)), **paths)), flush=True)
    return paths


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source')
    p.add_argument('--name', required=True)
    p.add_argument('--pod')
    p.add_argument('--driver-port', type=int)
    p.add_argument('--repin-bridge', action='store_true', help='only re-pin an existing copy (pin_bridge)')
    p.add_argument('--replicas', type=int, help='default: DP 1, GR00T 2')
    p.add_argument('--memory-gib', type=float, default=14)
    p.add_argument('--startup-width', type=int, default=1, help='replicas loading at once')
    p.add_argument('--max-batch', type=int, help='Fast-WAM only: stage batch cap on the slice')
    a = p.parse_args()
    if a.repin_bridge:
        return repin(a.name)
    copy_job(a.source, a.name, a.pod, a.driver_port, replicas=a.replicas, memory_gib=a.memory_gib,
             startup_width=a.startup_width, max_batch=a.max_batch)


if __name__ == '__main__':
    main()
