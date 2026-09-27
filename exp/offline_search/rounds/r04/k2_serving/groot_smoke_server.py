"""Smoke-only GR00T loader: discard stages 2/3 on CPU before any CUDA transfer.

The stock --stage1-only loader briefly places the full model on CUDA before
unloading, exceeding K2's 4 GB allowance. This wrapper preserves the same surviving
weights, while respecting the limit during startup too. No production src edit.
"""
import runpy
from pathlib import Path
import sys

from gr00t.model.policy import Gr00tPolicy
from exp.libero_groot.serve_groot_libero import _unload_stages_2_and_3

if '--stage1-only' not in sys.argv:
    raise SystemExit('this smoke loader requires --stage1-only')
original = Gr00tPolicy._load_model

def load_stage1_cpu_first(self, model_path):
    device = self.device
    self.device = 'cpu'
    original(self, model_path)
    _unload_stages_2_and_3(self.model)
    self.model._apply(lambda t: t if t.device.type == 'meta' else t.to(device))
    self.device = device
    import torch
    print(f'K2 CPU-first stage1 load peak CUDA allocated: {torch.cuda.max_memory_allocated() / 2**20:.1f} MiB', flush=True)

Gr00tPolicy._load_model = load_stage1_cpu_first
runpy.run_path(str(Path(__file__).resolve().parents[3] / 'closed_loop' / 'serve_groot.py'), run_name='__main__')
