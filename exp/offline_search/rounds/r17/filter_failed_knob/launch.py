"""Coordinator-only entry to the unchanged prepared launcher."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from .common import HERE, read
from exp.offline_search.rounds.r17.filter_failed import timan1_gpu2, worker_resilience


def bootstrap(*,serve=True):
    from exp.offline_search.rounds.r17.filter_failed import row_filter, server_cli
    # Transmit these small Python bodies in the resident command. No frozen
    # runtime/worker file is installed or replaced on either server host.
    code='import sys,types\n'
    for name,path in ((row_filter.__name__,Path(row_filter.__file__)),(server_cli.__name__,Path(server_cli.__file__))):
        code+=f'm=types.ModuleType({name!r});sys.modules[{name!r}]=m;exec(compile({path.read_text()!r},{str(path)!r},"exec"),m.__dict__)\n'
    name='__main__' if serve else 'ffk1015_cpu_runtime'
    code+=f'ffk_runtime=dict(__name__={name!r});exec(compile({(HERE/"runtime.py").read_text()!r},"ffk1015_runtime","exec"),ffk_runtime)\n'
    return code


def rewrite(value,code,evidence=''):
    target='exp.r17_integration.ops.formal_server'
    if isinstance(value,list):
        if target in value:
            i=value.index(target)
            if i and value[i-1]=='-m':return value[:i-1]+['-c',code]+value[i+1:]
        return [rewrite(v,code,evidence) for v in value]
    if isinstance(value,dict):return {k:rewrite(v,code,evidence) for k,v in value.items()}
    if isinstance(value,str):
        finalizer='from exp.r17_integration.ops.formal_finalize import background'
        if evidence and finalizer in value and not value.startswith(('{','[')):
            return evidence+'\n'+value
        if (target in value or finalizer in value) and value.startswith(('{','[')):
            return json.dumps(rewrite(json.loads(value),code,evidence))
    return value


@timan1_gpu2.coordinator()
@worker_resilience.coordinator()
def main(root):
    from exp.r17_integration.ops import formal_eval as f
    from exp.r17_integration.ops.formal_prestage import launch_prepared
    popen,remote=f.subprocess.Popen,f.remote
    campaign=read(Path(root)/'campaign.json')
    code=(Path(campaign['local_assets'])/'ffk_bootstrap.py').read_text()
    body=(Path(campaign['local_assets'])/'ffk_evidence.py').read_text()
    evidence='ffk_cpu_evidence={};exec(compile('+repr(body)+',"ffk_cpu_evidence","exec"),ffk_cpu_evidence);ffk_cpu_evidence["install"]()'
    def new_popen(argv,*a,**kw):return popen(worker_resilience.driver_command(rewrite(argv,code,evidence)),*a,**kw)
    from exp.offline_search.rounds.r17.filter_failed.launch import worker_command
    # Also lift the frozen workers' other-user GPU gate (see filter_failed.launch).
    def new_remote(node,argv,*a,**kw):return remote(node,worker_command(rewrite(argv,code,evidence),node),*a,**kw)
    from exp.r17_integration.ops import cache_study
    with patch.object(f.subprocess,'Popen',new_popen),patch.object(f,'remote',new_remote),patch(
            'exp.r17_integration.ops.cache_control.refresh_prepared_workers',
            side_effect=lambda c,d,workers:(c,workers)),patch.object(cache_study,'aggregate',cache_study.aggregate):
        exec(compile(body,'ffk_cpu_evidence','exec'),scope:={})
        scope['install']()
        return launch_prepared(root)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    raise SystemExit(main(p.parse_args().run))
