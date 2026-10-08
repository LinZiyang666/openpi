"""Both ordinary fleet planners, with all remote/process/server operations blocked."""
import contextlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
from unittest.mock import patch
from exp.offline_search.closed_loop.ops.h100 import control
from exp.offline_search.closed_loop.ops.h100.assets import flag
from exp.offline_search.closed_loop import devset
from .build import HERE, write
from .encoder import sha


def blocked(*args, **kwargs):
    raise AssertionError('remote/process/server operations forbidden in a CPU fleet plan')


def plans():
    roots=json.loads((HERE/'pilot_roots.json').read_text())
    evidence=[]
    exists=Path.exists
    with contextlib.ExitStack() as stack:
        for name in ('run','remote','push','rpc','bundle','daemon','setup','chain','collect','fleet_lock','chain_lock'):
            stack.enter_context(patch.object(control,name,blocked))
        stack.enter_context(patch.object(subprocess,'Popen',blocked))
        stack.enter_context(patch.object(socket,'create_connection',blocked))
        # Teacher baseline's historical query identity map is cosmetic. Omit
        # it before any open: A query data is forbidden for this B pilot task.
        stack.enter_context(patch.object(Path,'exists',lambda p: False if 'queries' in p.parts else exists(p)))
        for host in ('local','h100'):
            for rec in roots:
                root=Path(rec['root'])
                work=HERE/'plans'/host/root.name
                with patch.dict(os.environ,dict(SERVER_HOST=host,WORKER_HOST='timan107' if host=='local' else 'timan108')):
                    for key in ('OSCL_MANIFEST','OSCL_EPISODES','OSCL_TASKS','OSCL_INIT_POOL','OSCL_RECORD_LOOKS','OSCL_EXECUTION_AUDIT'):
                        os.environ.pop(key,None)
                    with contextlib.redirect_stdout(io.StringIO()):
                        result=control.sync(root,rec['arms'],plan_only=True,workdir=work)
                assert len(result['arms'])==rec['arm_count']
                assert len(result['dev_contracts'])==rec['arm_count']
                assert not any('/queries/' in f['original'] for f in result['files'])
                for row in json.loads((root/'arms.json').read_text()):
                    artifact=flag(row['plugin_args'],'--os-fit-artifact')
                    ent=next(f for f in result['files'] if f['original']==artifact)
                    assert ent['source_sha256']==rec['fit_sha256'][artifact]
                    if '--os-light-key' in row['plugin_args']:
                        model=row['kwargs']['encoder_spec']['kwargs']['model_path']
                        ent=next(f for f in result['files'] if f['original']==model)
                        assert ent['sha256']==sha(model)
                write(work/'plan.json',result)
                evidence.append(dict(host=host,root=str(root),arms=rec['arm_count'],
                    plan=str(work/'plan.json'),plan_sha256=sha(work/'plan.json'),remote_operations=0,
                    omitted_A_identity_map=True,student_exports_present=True))
                print('PLAN_PASS',host,root.name,flush=True)
    write(HERE/'plans/verification.json',dict(PASS=True,records=evidence,remote_operations=0,servers=0))
    return evidence


if __name__=='__main__':
    plans()
