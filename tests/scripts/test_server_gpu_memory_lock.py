"""Server GPU memory lock policy.

The serving process should release tensors/wrapper state internally while
keeping PyTorch CUDA allocator blocks reserved by the process from the system's
perspective. Practically this means ``torch.cuda.empty_cache()`` is suppressed
by default in ``scripts.serve_policy``.
"""

from __future__ import annotations


def test_server_gpu_memory_lock_suppresses_empty_cache_by_default(monkeypatch):
    import torch

    from scripts.serve_policy import _configure_server_gpu_memory_lock

    calls = []

    def fake_empty_cache():
        calls.append("released")

    monkeypatch.delenv("OPENPI_SERVER_GPU_MEMORY_LOCK", raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

    _configure_server_gpu_memory_lock()
    torch.cuda.empty_cache()

    assert calls == []
    assert getattr(torch.cuda.empty_cache, "_openpi_gpu_memory_lock_noop", False)
    assert torch.cuda.empty_cache._openpi_original_empty_cache is fake_empty_cache


def test_server_gpu_memory_lock_can_be_disabled(monkeypatch):
    import torch

    from scripts.serve_policy import _configure_server_gpu_memory_lock

    calls = []

    def fake_empty_cache():
        calls.append("released")

    monkeypatch.setenv("OPENPI_SERVER_GPU_MEMORY_LOCK", "0")
    monkeypatch.setattr(torch.cuda, "empty_cache", fake_empty_cache)

    _configure_server_gpu_memory_lock()
    torch.cuda.empty_cache()

    assert calls == ["released"]
    assert torch.cuda.empty_cache is fake_empty_cache
