"""Tests for ``DumpConfig.deferred`` validator behavior (verdict_factor_judge B2).

`phase1_spec.py` emits sibling ``<eval_yaml_id>__warmup.yaml`` files with
``dump.deferred=true`` and an omitted ``dump.path`` so that
``validate_cache_config`` succeeds in CI / offline tools where the
server-owned ``/tmp/openpi_warmup`` root may not exist. The actual path is
filled later by the WebSocket ``load_cache_config`` ctrl handler. These
tests pin the validator's three relevant behaviours:

  1. ``deferred=True + path=""`` passes (the gate that lets warmup yamls
     load offline).
  2. ``deferred=False + path=""`` still fails (no regression on the legacy
     calibration path).
  3. ``deferred=True + path="/something"`` passes regardless of whether the
     parent exists — the deferred flag is the sole switch.
"""

from __future__ import annotations

from openpi.cache.config import (
    BackendConfig,
    CacheConfig,
    DumpConfig,
    _validate_dump_static,
)


def _backend_cfg(backend_type: str = "in_memory") -> CacheConfig:
    cfg = CacheConfig()
    cfg.backend.type = backend_type
    return cfg


def _run(dump: DumpConfig) -> list[str]:
    errors: list[str] = []
    _validate_dump_static("checkpoints.cp1", dump, _backend_cfg(), errors)
    return errors


def test_deferred_true_with_empty_path_passes() -> None:
    dump = DumpConfig(path="", config_id="cfg_warmup", factors=[], deferred=True)
    errors = _run(dump)
    # No path/parent errors — deferred bypasses both. config_id is set so the
    # remaining check (config_id non-empty) also passes.
    assert not any("dump.path" in e for e in errors), errors


def test_deferred_false_with_empty_path_still_rejects() -> None:
    """Regression: turning the new field off must restore the legacy error."""
    dump = DumpConfig(path="", config_id="cfg_a", factors=[], deferred=False)
    errors = _run(dump)
    assert any("dump.path is required" in e for e in errors), errors


def test_deferred_true_with_nonexistent_parent_passes() -> None:
    """Even when path is given, deferred bypasses the parent-dir check —
    the server overwrites path during ctrl-handler resolution anyway."""
    dump = DumpConfig(
        path="/no/such/dir/d.jsonl",
        config_id="cfg_warmup",
        factors=[],
        deferred=True,
    )
    errors = _run(dump)
    assert not any("parent directory does not exist" in e for e in errors), errors


def test_deferred_does_not_skip_config_id_check() -> None:
    """`deferred` is a path-resolution toggle only; identity invariants
    (config_id non-empty) still apply."""
    dump = DumpConfig(path="", config_id="", factors=[], deferred=True)
    errors = _run(dump)
    assert any("dump.config_id is required" in e for e in errors), errors


def test_dump_config_default_deferred_is_false() -> None:
    """Backward compat: ``deferred`` must default to False so existing yamls
    that never reference the field load with the legacy strict behaviour."""
    dump = DumpConfig()
    assert dump.deferred is False
