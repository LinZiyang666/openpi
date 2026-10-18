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
from . import timan1_gpu2, worker_resilience


def server_command(argv,needs_bridge):
    target='exp.r17_integration.ops.formal_server'
    if needs_bridge and target in argv:
        pos=argv.index(target)-1
        return [*argv[:pos],'-c',
                'import runpy;runpy.run_path('+repr(str(HERE/'server_cli.py'))+',run_name="__main__")',
                *argv[pos+2:]]
    return argv


WORKER='exp.r17_integration.ops.formal_workers'
# Rebind ops only in this process. Keep the module marker in OS argv for
# receipt-bound fleet closure; timan1 uses its separate UUID-verified namespace.
SHARE=('import sys,runpy\n'+worker_resilience.worker_patch()+
       'assert sys.argv[1]=='+repr(WORKER)+'\n'
       'del sys.argv[1]\n'
       'runpy.run_module('+repr(WORKER)+',run_name="__main__",alter_sys=True)\n')


def worker_command(value,node=None):
    if isinstance(value,list):
        if WORKER in value:
            i=value.index(WORKER)
            if i and value[i-1]=='-m':
                if node=='timan1':return timan1_gpu2.bootstrap(value[0],[WORKER,*value[i+1:]],'fleet')
                return value[:i-1]+['-c',SHARE,WORKER]+value[i+1:]
        return [worker_command(v,node) for v in value]
    if isinstance(value,dict):return {k:worker_command(v,node) for k,v in value.items()}
    if isinstance(value,str) and WORKER in value and value.startswith(('{','[')):
        return json.dumps(worker_command(json.loads(value),node))
    return value


@timan1_gpu2.coordinator()
@worker_resilience.coordinator()
def main(root):
    from exp.r17_integration.ops import formal_eval
    from exp.r17_integration.ops.formal_prestage import launch_prepared
    campaign=read(Path(root)/'campaign.json')
    source=Path(campaign['filter_failed_source'])
    needs_bridge=(campaign['model']=='dp' and campaign['domain']=='libero' and campaign['replicas'] in (3,4)
                  and (source/'servers/replica_0/CACHE_CAPACITY_ADMISSION.json').exists())
    native_popen,native_remote=formal_eval.subprocess.Popen,formal_eval.remote
    def popen(argv,*args,**kwargs):
        return native_popen(worker_resilience.driver_command(server_command(argv,needs_bridge)),*args,**kwargs)
    def remote(node,argv,*args,**kwargs):
        return native_remote(node,worker_command(argv,node),*args,**kwargs)
    with patch.object(formal_eval,'remote',remote),patch('exp.r17_integration.ops.cache_control.refresh_prepared_workers',
               side_effect=lambda c,d,workers:(c,workers)),patch.object(formal_eval.subprocess,'Popen',popen):
        return launch_prepared(root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    args=p.parse_args();raise SystemExit(main(args.run))
