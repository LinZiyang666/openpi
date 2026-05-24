"""M4.5 Runtime write_policy enforcement (C2).

Validates that ``scripts.serve_policy._enforce_runtime_write_policy`` returns
a CacheConfig whose ``write_policy.type == "never"`` regardless of the
yaml-declared value, while leaving the original config object unchanged
(dataclass replace semantics) and emitting exactly one warning per override.

Auto-override (rather than fail-fast) is intentional: existing sweep yamls
ship with ``write_policy: on_any_miss`` and we keep them runnable without
manual edits while still honoring the C2 runtime contract at the backend
guard layer.
"""

from __future__ import annotations

import logging

import pytest

from openpi.cache.config import CacheConfig, WritePolicyConfig


def _make_cache_config(write_policy_type: str) -> CacheConfig:
    """Build a minimal CacheConfig with the requested write_policy type."""
    return CacheConfig(write_policy=WritePolicyConfig(type=write_policy_type))


def test_override_from_on_any_miss(caplog):
    from scripts.serve_policy import _enforce_runtime_write_policy

    original = _make_cache_config("on_any_miss")
    with caplog.at_level(logging.WARNING):
        overridden = _enforce_runtime_write_policy(original)

    assert overridden.write_policy.type == "never"
    # Source config must not be mutated (dataclasses.replace returns a new instance).
    assert original.write_policy.type == "on_any_miss"
    # Exactly one warning emitted; message references the original type.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "on_any_miss" in warnings[0].getMessage()
    assert "never" in warnings[0].getMessage()


def test_override_from_always(caplog):
    from scripts.serve_policy import _enforce_runtime_write_policy

    original = _make_cache_config("always")
    with caplog.at_level(logging.WARNING):
        overridden = _enforce_runtime_write_policy(original)

    assert overridden.write_policy.type == "never"
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "always" in warnings[0].getMessage()


def test_no_override_when_already_never(caplog):
    from scripts.serve_policy import _enforce_runtime_write_policy

    original = _make_cache_config("never")
    with caplog.at_level(logging.WARNING):
        overridden = _enforce_runtime_write_policy(original)

    # Returned config is equivalent (could be same object — protocol does
    # not require identity, only that no override happens).
    assert overridden.write_policy.type == "never"
    # No warning when already 'never'.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 0
