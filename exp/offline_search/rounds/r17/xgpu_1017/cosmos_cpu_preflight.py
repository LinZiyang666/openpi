"""CPU-default preflight through Cosmos's dynamic experiment registry.

Never calls instantiate(model), loads checkpoint tensors or creates a simulator.
An explicit dedicated-A100-only option lets native availability guards run,
still without instantiating a model. Use the frozen serving island/server_env.
"""
import argparse
import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import sys
import traceback
import subprocess


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--known-list',action='append',default=[])
    p.add_argument('--allow-a100-config-probe',action='store_true')
    a=p.parse_args()
    if a.allow_a100_config_probe:
        names=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True)
        if 'A100' not in names or 'H100' in names:raise ValueError('device-aware probe only on dedicated A100')
    elif os.environ.get('CUDA_VISIBLE_DEVICES')!='':raise ValueError('explicitly mask CUDA for this CPU tool')
    from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
    from cosmos_policy._src.predict2.utils import model_loader as ml
    cfg=PolicyEvalConfig()
    config_module=ml.get_config_module(cfg.config_file)
    error=None
    try:
        config=importlib.import_module(config_module).make_config()
        config=ml.override(config,['--','experiment=cosmos_predict2_2b_480p_libero__inference_only'])
        config.validate();config.freeze()
    except BaseException:
        # Some unrelated experiment definitions assert CUDA availability at
        # import time. Preserve that boundary; never fake an available GPU.
        error=traceback.format_exc()
    import torch
    if torch.cuda.is_initialized() and not a.allow_a100_config_probe:raise RuntimeError('CPU preflight unexpectedly initialized CUDA')
    base=Path(md.distribution('torch').locate_file('')).resolve();known=set()
    for path in a.known_list:known.update(Path(path).read_text().splitlines())
    missing=set()
    for module in list(sys.modules.values()):
        filename=getattr(module,'__file__',None)
        if filename:
            f=Path(filename).resolve()
            if f.is_relative_to(base) and f.is_file() and str(f) not in known:missing.add(str(f))
    roots={Path(f).relative_to(base).parts[0] for f in missing};distributions=set()
    for dist in md.distributions():
        if any(Path(str(f)).parts[0] in roots for f in dist.files or []):distributions.add(dist.metadata['Name'])
    report=dict(status=('A100_CONFIG_' if a.allow_a100_config_probe else 'CPU_CONFIG_')+('PARTIAL' if error else 'PASS'),error=error,
        cuda_initialized=torch.cuda.is_initialized(),device_availability_unmasked=a.allow_a100_config_probe,config_module=config_module,
        model_instantiated=False,checkpoint_loaded=False,simulator_created=False,
        missing_imported_files=sorted(missing),module_roots=sorted(roots),distribution_roots=sorted(distributions))
    Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='missing_imported_files'}))
    if error:raise SystemExit(2)
