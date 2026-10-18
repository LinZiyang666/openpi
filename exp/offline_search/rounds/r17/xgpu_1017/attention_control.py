"""Model-specific backend ablation, only in explicitly requested new residents."""
from contextlib import contextmanager
import importlib
import json
from pathlib import Path


def cosmos_math(directory=None,module=None):
    import torch
    if module is None:
        module=importlib.import_module('cosmos_policy._src.predict2.networks.attention')
    if getattr(module,'_xgpu_math_override',False):raise RuntimeError('Cosmos override installed twice')
    module._xgpu_math_override=True
    module.FLASH_ATTN_3_AVAILABLE=False
    native=torch.nn.attention.sdpa_kernel
    observed=False
    @contextmanager
    def fixed(*args,**kwargs):
        nonlocal observed
        with native(backends=[torch.nn.attention.SDPBackend.MATH]):
            if not observed and directory is not None:
                Path(directory).mkdir(parents=True,exist_ok=True)
                (Path(directory)/'ATTENTION_OVERRIDE_VISITED.json').write_text(json.dumps(dict(
                    model='cosmos',scope='predict2.networks.attention.sdpa_kernel context entered',
                    backend='MATH',flash3_branch_disabled=True,
                    warning='TE has separate NVTE controls; this receipt is not a profiler trace'),indent=2)+'\n')
                observed=True
            yield
    # Retain native attention() casts, scales, transposes and SDPA call. Its
    # local selector otherwise overrides global flags depending on GPU SM.
    module.sdpa_kernel=fixed
    return module
