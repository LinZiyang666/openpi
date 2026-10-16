"""Coordinator-only entry into the standard prepared campaign launcher.

Preserves the baseline worker archive instead of automatically refreshing its
recycle helpers. All normal admission, resource ownership and cleanup remain
in formal_prestage.launch_prepared.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from .audit_counts import HERE, read


def server_command(argv,needs_bridge):
    target='exp.r17_integration.ops.formal_server'
    if needs_bridge and target in argv:
        pos=argv.index(target)-1
        return [*argv[:pos],'-c',
                'import runpy;runpy.run_path('+repr(str(HERE/'server_cli.py'))+',run_name="__main__")',
                *argv[pos+2:]]
    return argv


WORKER='exp.r17_integration.ops.formal_workers'
# Frozen worker archives predate three owner decisions that the current code
# already follows; apply them in the worker command instead of changing the archive:
# - another user's process on a worker GPU no longer makes the whole card
#   unusable (its memory still counts; local d2019b8c);
# (A frozen per-worker VRAM refusal is kept: 2026-10-16 06:17 fastwam_li_18 lost
#  timan1 GPU3 EGL framebuffers with 40 LIBERO renderers beside another user's 15 GB.)
# - timan1 GPU3 lease allows up to 40 LIBERO cache workers (the source runs got
#   this from cache_capacity_authorization, which the filtered campaigns drop).
# The module name stays in argv so the timan1 GPU3 guard and fleet close checks
# still recognise the fleet; the code removes it before running the module.
SHARE=('import sys,runpy\n'
       'import exp.r17_integration.ops.worker_gpu as g\n'
       'native=g.inventory\n'
       'def inventory(*a,**k):\n'
       '    rows=native(*a,**k)\n'
       '    for r in rows:\n'
       '        if r.get("reason")=="other user GPU allocation":r["usable"]=True;r.pop("reason")\n'
       '    return rows\n'
       'g.inventory=inventory\n'
       'import exp.r17_integration.ops.timan1_workers as t\n'
       'lease=t.authorize\n'
       'def authorize(c):\n'
       '    nodes=c.get("execution_worker_nodes",c.get("worker_nodes",{}));n=nodes.get("timan1",0)\n'
       '    if 16<n<=40 and c.get("domain")=="libero" and not (c.get("cache_capacity_authorization") or c.get("live_fleet_authorization")):\n'
       '        key="execution_worker_nodes" if "execution_worker_nodes" in c else "worker_nodes"\n'
       '        r=lease(dict(c,**{key:dict(nodes,timan1=16)}))\n'
       '        if r:r["workers"]=n\n'
       '        return r\n'
       '    return lease(c)\n'
       't.authorize=authorize\n'
       'assert sys.argv[1]=='+repr(WORKER)+'\n'
       'del sys.argv[1]\n'
       'runpy.run_module('+repr(WORKER)+',run_name="__main__",alter_sys=True)\n')


def worker_command(value):
    if isinstance(value,list):
        if WORKER in value:
            i=value.index(WORKER)
            if i and value[i-1]=='-m':return value[:i-1]+['-c',SHARE,WORKER]+value[i+1:]
        return [worker_command(v) for v in value]
    if isinstance(value,dict):return {k:worker_command(v) for k,v in value.items()}
    if isinstance(value,str) and WORKER in value and value.startswith(('{','[')):
        return json.dumps(worker_command(json.loads(value)))
    return value


def main(root):
    from exp.r17_integration.ops import formal_eval
    from exp.r17_integration.ops.formal_prestage import launch_prepared
    campaign=read(Path(root)/'campaign.json')
    source=Path(campaign['filter_failed_source'])
    needs_bridge=(campaign['model']=='dp' and campaign['domain']=='libero' and campaign['replicas'] in (3,4)
                  and (source/'servers/replica_0/CACHE_CAPACITY_ADMISSION.json').exists())
    native_popen,native_remote=formal_eval.subprocess.Popen,formal_eval.remote
    def popen(argv,*args,**kwargs):
        return native_popen(server_command(argv,needs_bridge),*args,**kwargs)
    def remote(node,argv,*args,**kwargs):
        return native_remote(node,worker_command(argv),*args,**kwargs)
    with patch.object(formal_eval,'remote',remote),patch('exp.r17_integration.ops.cache_control.refresh_prepared_workers',
               side_effect=lambda c,d,workers:(c,workers)),patch.object(formal_eval.subprocess,'Popen',popen):
        return launch_prepared(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();raise SystemExit(main(args.run))
