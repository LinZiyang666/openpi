"""CPU-only package/flag inventory; never initializes CUDA or reads credentials."""
from __future__ import annotations
import importlib.metadata as md
import json
import os
from pathlib import Path
import platform
import sys

PACKAGES = ('torch', 'torchvision', 'triton', 'transformers', 'diffusers', 'numpy',
            'jax', 'jaxlib', 'flash-attn', 'transformer-engine', 'nvidia-cudnn-cu12',
            'nvidia-cublas-cu12', 'nvidia-cuda-runtime-cu12','scipy','scikit-learn',
            'tokenizers','safetensors','websockets','msgpack')


def collect(with_torch=False):
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = md.version(name)
        except md.PackageNotFoundError:
            versions[name] = None
    driver = Path('/proc/driver/nvidia/version')
    result = dict(python=sys.version.split()[0], executable=sys.executable,
                  host=platform.node(), packages=versions,
                  driver=driver.read_text() if driver.exists() else None,
                  environment={k: os.environ.get(k) for k in
                      ('CUBLAS_WORKSPACE_CONFIG', 'NVIDIA_TF32_OVERRIDE',
                       'JAX_DEFAULT_MATMUL_PRECISION', 'DETERMINISTIC',
                       'TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','NVTE_FLASH_ATTN','NVTE_FUSED_ATTN',
                       'TORCHDYNAMO_DISABLE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
                  scope='fresh interpreter defaults; NOT live-process flags')
    cpu=Path('/proc/cpuinfo')
    result['cpu_architecture']=platform.machine()
    result['cpu_model']=next((line.split(':',1)[1].strip() for line in cpu.read_text().splitlines()
        if line.startswith('model name')),None) if cpu.exists() else None
    if with_torch:
        # Caller must mask CUDA, except when collecting inside an explicit replay.
        import torch
        result['torch_flags'] = dict(
            matmul_precision=torch.get_float32_matmul_precision(),
            matmul_tf32=torch.backends.cuda.matmul.allow_tf32,
            cudnn_tf32=torch.backends.cudnn.allow_tf32,
            cudnn_deterministic=torch.backends.cudnn.deterministic,
            cudnn_benchmark=torch.backends.cudnn.benchmark,
            deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            sdpa_flash=torch.backends.cuda.flash_sdp_enabled(),
            sdpa_mem_efficient=torch.backends.cuda.mem_efficient_sdp_enabled(),
            sdpa_math=torch.backends.cuda.math_sdp_enabled(),
            bf16_reduced_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
            fp16_reduced_reduction=torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            cuda_build=torch.version.cuda, cudnn=torch.backends.cudnn.version())
        result['cuda_initialized'] = torch.cuda.is_initialized()
        import contextlib,io
        try:
            import numpy as np
            buffer=io.StringIO()
            with contextlib.redirect_stdout(buffer):np.show_config()
            result['numpy_build']=buffer.getvalue()
        except ImportError:result['numpy_build']=None
        try:
            from threadpoolctl import threadpool_info
            result['cpu_threadpools']=threadpool_info()
        except ImportError:result['cpu_threadpools']=None
    return result


if __name__ == '__main__':
    print(json.dumps(collect('--torch' in sys.argv), indent=2))
