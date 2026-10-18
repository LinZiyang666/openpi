"""Explicit ablation overlay in NEW residents; native launch never imports this.

Each option is independent. CUDA graph/compile ablations are labelled different
execution paths. SDPA math is not an implementation of FA2 or TE math attention.
"""
import json
import os
from pathlib import Path
import runpy
import sys

OPTIONS={'tf32_off','highest','sdpa_math','te_unfused','cosmos_math','compile_off','graphs_off','deterministic','full_reduction'}


def process_environment(options,model):
    """Set backend controls before importing TE/torch; fail unsupported labels."""
    options=set(options)
    if options-OPTIONS:raise ValueError('unknown numerics option')
    if 'sdpa_math' in options and model in ('groot','cosmos'):
        raise ValueError('direct FA2/TE backend has no qualified SDPA-math replacement; refuse false fixed-backend label')
    if options & {'te_unfused','cosmos_math'} and model!='cosmos':
        raise ValueError('Cosmos backend options are specific to the Cosmos serving island')
    values={}
    if options & {'te_unfused','cosmos_math'}:
        values.update(NVTE_FLASH_ATTN='0',NVTE_FUSED_ATTN='0')
    if 'deterministic' in options:values['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    if 'compile_off' in options:values['TORCHDYNAMO_DISABLE']='1'
    return values


def configure(options):
    import torch
    if 'tf32_off' in options:
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    if 'highest' in options:torch.set_float32_matmul_precision('highest')
    if options & {'sdpa_math','cosmos_math'}:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda,'enable_cudnn_sdp'):
            torch.backends.cuda.enable_cudnn_sdp(False)
        torch.backends.mha.set_fastpath_enabled(False)
    if 'deterministic' in options:
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
    if 'full_reduction' in options:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction=False


def run(bootstrap,options,model):
    options=set(options)
    os.environ.update(process_environment(options,model))
    configure(options)
    if 'cosmos_math' in options:
        from attention_control import cosmos_math
        cosmos_math(sys.argv[sys.argv.index('--directory')+1])
    from exp.r17_integration.qualification import model_runtime
    original=model_runtime.load
    def load(*args,**kwargs):
        result=original(*args,**kwargs)
        configure(options)  # pi/Cosmos loaders overwrite process flags.
        from inventory import collect
        directory=Path(sys.argv[sys.argv.index('--directory')+1])
        directory.mkdir(parents=True,exist_ok=True)
        inventory=collect(with_torch=True)
        inventory['scope']='live ablation resident immediately after model load and knob reapplication'
        receipt=dict(options=sorted(options),after_load=inventory,
            formal_native_execution=False,attention_kernel_identity='unprofiled; flags alone do not prove dispatch')
        (directory/'NUMERICS.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
        return result
    model_runtime.load=load
    if bootstrap:exec(compile(Path(bootstrap).read_text(),bootstrap,'exec'),{'__name__':'__main__'})
    else:runpy.run_module('exp.r17_integration.ops.formal_server',run_name='__main__')
