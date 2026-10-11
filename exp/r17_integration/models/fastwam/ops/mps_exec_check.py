"""Bounded real guard + exec-wrapper qualification (tiny CUDA allocation)."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .mps import account,replica_argv
from exp.r17_integration.ops.formal_process import ticks,stop_group,write_json


def main():
    out=Path('/data/mw_finetune/eval/p3mps_exec_1010')
    out.mkdir(exist_ok=False)
    ready=out/'MPS_READY.json'
    started=time.monotonic()
    guard=subprocess.Popen([sys.executable,'-u','-m','exp.r17_integration.models.fastwam.ops.mps','guard',
        '--campaign-directory',str(out),'--ready',str(ready)],start_new_session=True)
    ownership=dict(pid=guard.pid,pgid=guard.pid,start_ticks=ticks(guard.pid))
    child=None
    write_json(out/'ownership.json',ownership)
    receipt={}
    try:
        while not ready.exists():
            if guard.poll() is not None or time.monotonic()-started>10:
                raise RuntimeError('private guard failed to publish ready')
            time.sleep(.05)
        state=json.loads(ready.read_text())
        code=('import torch,time; from pathlib import Path; x=torch.ones(4*1024*1024,device="cuda:0"); '
              'torch.cuda.synchronize(); Path('+repr(str(out/'CLIENT_READY'))+').write_text(str(x.sum().item())); time.sleep(1)')
        child=subprocess.Popen(replica_argv(['/data/mw_finetune/dp/env/bin/python','-c',code],ready),
            env={**os.environ,'CUDA_MPS_PIPE_DIRECTORY':'/unrelated','CUDA_MPS_PINNED_DEVICE_MEM_LIMIT':'0=1M'},
            start_new_session=True)
        write_json(out/'child_ownership.json',dict(pid=child.pid,start_ticks=ticks(child.pid),pgid=child.pid))
        while not (out/'CLIENT_READY').exists():
            if child.poll() is not None or time.monotonic()-started>15:
                raise RuntimeError('exec replica failed')
            time.sleep(.02)
        receipt['attachment']=account(state,[child.pid])
        child.wait(timeout=5)
        receipt.update(status='PASS',child_exit=child.returncode,allocated_bytes=16*1024**2,
            inherited_1MiB_limit_removed=True,result=float((out/'CLIENT_READY').read_text()))
    finally:
        if child and child.poll() is None:
            stop_group(dict(pid=child.pid,start_ticks=ticks(child.pid),pgid=child.pid));child.wait(timeout=5)
        receipt['guard_cleanup']=stop_group(ownership,grace=30)
        guard.wait(timeout=5)
        if ready.exists():
            initial=json.loads(ready.read_text());receipt['mps']=json.loads((Path(initial['directory'])/'state.json').read_text())
        receipt['elapsed_seconds']=time.monotonic()-started
        write_json(out/'receipt.json',receipt)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
