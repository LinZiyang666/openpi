"""Tiny H100 CUDA attachment/context-isolation check; no model replicas."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    OUT=Path('/data/mw_finetune/eval/p3mps_hcheck_1010')
    spec=importlib.util.spec_from_file_location('p3_private_mps',OUT/'mps.py')
    mps=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mps)
    session=mps.Session(OUT)
    started=time.monotonic()
    try:
        state=session.start()
        subprocess_result=subprocess.run(['/data/mw_finetune/envs/fastwam/bin/python','-c',
            'import torch,time; x=torch.ones(8,device="cuda:0"); torch.cuda.synchronize(); print(x.sum().item()); time.sleep(1)'],
            env={**os.environ,**mps.client_env(state)},capture_output=True,text=True,timeout=15)
        receipt=dict(code=subprocess_result.returncode,stdout=subprocess_result.stdout,
                     stderr=subprocess_result.stderr,elapsed_seconds=time.monotonic()-started)
    finally:
        closure=session.close()
        receipt['mps']=closure
        receipt['elapsed_seconds']=time.monotonic()-started
        (OUT/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
