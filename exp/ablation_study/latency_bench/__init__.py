"""Latency bench: per-executor batch=1 inference cost for the cache system.

Measures the wall-clock cost of every executor the cache system can dispatch to
— the pi0.5 teacher (staged: vision/token prep, LLM prefix, denoising) and the
two distilled students (ACT, SmolVLA) — under eager PyTorch and under
torch.compile, on a dedicated idle GPU and on CPU.
"""
