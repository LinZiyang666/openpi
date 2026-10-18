"""Owner-selected formal campaigns, private dynamic orchestration; global OFF."""
from contextlib import contextmanager
from dataclasses import asdict
import argparse
import builtins
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import time
from unittest.mock import patch

from . import formal_trial
from .node import atomic, read
from .policy import GIB

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
EVAL = Path('/data/mw_finetune/eval')
NAMES = ('fa_ls_fw_f_p_m', 'fa_pi_f_u_l', 'fa_pi_f_f_l')
NODE_CONFIG = {'timan107': dict(floor=96*GIB, critical=60*GIB, hysteresis=8*GIB,
                              spawn_interval_s=10, external_peak_floor=4*GIB)}


@contextmanager
def private_on():
    """The same four-byte private read overlay used by the reviewed trial path."""
    switch = HERE/'ENABLED'; original = switch.read_bytes()
    if original.strip() != b'off':raise ValueError('formal acceptance requires global OFF')
    def wrap(native):
        def opened(file, mode='r', *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and Path(os.fsdecode(file)).absolute() == switch:
                if any(k in mode for k in 'wax+'):raise PermissionError('never write global ENABLED')
                return io.BytesIO(b'on\n\n') if 'b' in mode else io.StringIO('on\n\n')
            return native(file, mode, *args, **kwargs)
        return opened
    with patch.object(builtins, 'open', wrap(builtins.open)), patch.object(io, 'open', wrap(io.open)):
        yield
    if switch.read_bytes() != original:raise RuntimeError('global switch changed')


def jobs(name):
    if name not in NAMES:raise ValueError('explicit owner-selected formal campaign required')
    if name == NAMES[0]:return [dict(name=name, run=str(EVAL/name), pipeline='R17')]
    from exp.offline_search.rounds.r17.filter_failed_knob.formal_a_queue import INDEX
    return next(u['jobs'] for u in read(INDEX)['units'] if u['name'] == name)


def desired(campaign):
    from .defaults import desired_maxima
    return desired_maxima(campaign['domain'], campaign.get('execution_worker_nodes', campaign['worker_nodes']),
                          campaign.get('worker_hard_limits'))


def dynamic_campaign(campaign):
    c = formal_trial.configure(campaign, desired(campaign))
    c.update(autoscale_node_config=copy.deepcopy(NODE_CONFIG), autoscale_external_legacy_nodes=['timan107'])
    return c


def legacy_plan(root, rows, done=()):
    from exp.gate_threshold_pareto.run_gtp import SweepStrategy
    from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import selected_tasks, load_manifest
    from openpi.conductor.task import ServerEndpoint
    from exp.offline_search.closed_loop.devset import pool_record
    servers = [ServerEndpoint('149.165.153.233', p) for p in (23230,23231,23232)]
    result=[]
    for row in rows:
        _, pool = pool_record(row['suite'], 'A')
        strategy = SweepStrategy(row['suite'], {row['arm']: row['yaml']}, 50, servers=servers,
                                 done_uids=set(done), parent_pool_sha256=pool['rollup_sha256'])
        # The native shard plan depends on the server list, not worker counts.
        graph = strategy.plan([row['arm']], {row['arm']: servers[0]})
        manifest = load_manifest(row['manifest'])
        for stage in graph.stages.values():result += [asdict(t) for t in selected_tasks(stage.episodes, manifest)]
    return result


def proof(name, output):
    from .compare import wait_io
    from .defaults import physical_maximum
    from exp.offline_search.rounds.r17.filter_failed_knob.build import cpu_plan
    rows=[]
    for j in jobs(name):
        wait_io();root=Path(j['run'])
        if j['pipeline']=='R17':
            off=read(root/'campaign.json');dynamic=dynamic_campaign(off)
            before=[asdict(t) for t in cpu_plan(off)[2]]
            after=[asdict(t) for t in cpu_plan(dynamic)[2]]
            assert before == after == read(root/'EPISODE_PLAN.json')['tasks']
            allowed=lambda k:k.startswith(('autoscale','worker_','execution_')) or k in ('workers','prepare_only')
            assert {k:v for k,v in off.items() if not allowed(k)} == {k:v for k,v in dynamic.items() if not allowed(k)}
            maxima=dynamic['autoscale_node_maxima']
        else:
            arms=read(root/'arms.json');before=legacy_plan(root, arms);after=legacy_plan(root, copy.deepcopy(arms))
            assert before == after
            maxima={'timan107':physical_maximum('timan107','libero')}
        assert len(before)==len({r['task_uid'] for r in before})
        rows.append(dict(name=root.name,episodes=len(before),all_task_fields_equal=True,
            task_sha256=hashlib.sha256(json.dumps(before,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            desired_maxima=maxima,serving_and_scientific_fields_unchanged=True))
    result=dict(name=name,status='PASS',rows=rows,episodes=sum(r['episodes'] for r in rows),
                global_switch='off',node_config=NODE_CONFIG)
    atomic(output,result);return result


def record_r17(root, before):
    after=read(root/'campaign.json')
    atomic(root/'FORMAL_DYNAMIC.json',dict(schema='autoscale.formal_acceptance.v1',
        desired_maxima=after['autoscale_node_maxima'],node_config=NODE_CONFIG,
        plan_sha256=hashlib.sha256((root/'EPISODE_PLAN.json').read_bytes()).hexdigest(),
        original_execution_workers=before.get('execution_worker_nodes',before['worker_nodes']),global_switch='off'))


def legacy_job(action, job):
    from .defaults import physical_maximum
    from . import legacy_release
    from .trial_lane import require_legacy_idle
    from exp.offline_search.closed_loop.ops.h100 import control
    from exp.offline_search.rounds.r17.filter_failed.legacy import rsync_transport
    from exp.offline_search.rounds.r17.filter_failed_knob import vla_shared
    from exp.r17_integration.ops.formal_eval import transfer
    root=Path(job['run']);names=[r['arm'] for r in read(root/'arms.json')]
    maximum=physical_maximum('timan107','libero')
    env=dict(job['topology']['env'],WORKER_HOST='timan107',FFK_WORKER_FLEET_ROOT=str(root),
        OPENPI_DESIRED_MAXIMUM=str(maximum),OSCL_ISOLATE_GENERATED_ASSETS='1',
        OPENPI_AUTOSCALE_NODE_CONFIG=json.dumps(NODE_CONFIG))
    with patch.dict(os.environ,env),patch.object(control,'run',rsync_transport(control.run,transfer)),vla_shared.r15(root):
        if action=='prepare':
            done=any((root/'state').glob('*.DONE'))
            # Three closed filtered arms must retain their exact recorded sync
            # plan and asset addresses. Only install new worker orchestration.
            if done:
                synced=read(root/'h100_sync/synced.json')
                if synced['worker_host']!='timan107':raise ValueError('native resume requires its original host')
                legacy_release.stage(root,'timan107',pipeline='R15',maximum=maximum,base=str(control.island()))
            else:control.sync(root,names,concurrent=True)
            for p in root.glob('runs/*/h100_launch.json'):
                backup=p.with_name('h100_launch.before_dynamic_'+hashlib.sha256(p.read_bytes()).hexdigest()+'.json')
                if not backup.exists():shutil.copy2(p,backup)
            atomic(root/'FORMAL_DYNAMIC.json',dict(schema='autoscale.formal_acceptance.v1',
                desired_maxima={'timan107':maximum},node_config=NODE_CONFIG,native_resume=True,
                retained_done_arms=[p.name for p in (root/'state').glob('*.DONE')],global_switch='off'))
        else:
            require_legacy_idle()
            control.chain(root,names)  # native DONE skipping and journal replay


def dispatch(action, name):
    with private_on():
        for j in jobs(name):
            root=Path(j['run'])
            if j['pipeline']=='R15':legacy_job(action,j);continue
            c=read(root/'campaign.json')
            if action=='prepare':
                formal_trial.prepare(root,desired(c),node_config=NODE_CONFIG,preserve_science=True,
                                     reprepare=(root/'PREPARED.json').exists())
                record_r17(root,c)
            else:
                if not read(root/'FORMAL_DYNAMIC.json'):raise ValueError('use the reviewed dynamic prepare first')
                from exp.offline_search.rounds.r17.filter_failed_knob.launch import main
                main(root)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=('prepare','launch','audit','dry-run'))
    p.add_argument('--name',required=True,choices=NAMES);p.add_argument('--output',type=Path)
    a=p.parse_args(argv)
    if a.action=='dry-run':
        result=proof(a.name,a.output or HERE/'formal_acceptance'/a.name/'PLAN.json')
    elif a.action=='audit':
        from .formal_dynamic_audit import audit
        result=audit(a.name,a.output or HERE/'formal_acceptance'/a.name)
    else:return dispatch(a.action,a.name)
    print(json.dumps(result,indent=2));return int(result['status']!='PASS')


if __name__=='__main__':raise SystemExit(main())
