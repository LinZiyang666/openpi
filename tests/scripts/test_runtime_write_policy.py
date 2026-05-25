"""M4.5 Runtime write_policy enforcement (C2).

Validates that ``scripts.serve_policy._enforce_runtime_write_policy`` fails fast
(raises ``ConfigValidationError``) on any write-enabled policy, and returns the
config unchanged when it is already ``"never"``.

Fail-fast (rather than silent auto-override) is intentional: at server runtime
the backend is write-frozen (C2), so a config that declares an active
write_policy is a configuration error and must be surfaced loudly at server
start / ``load_cache_config`` instead of being neutralised behind the
operator's back.
"""

from __future__ import annotations

import pytest

from openpi.cache.config import CacheConfig, ConfigValidationError, WritePolicyConfig


def _make_cache_config(write_policy_type: str) -> CacheConfig:
    """Build a minimal CacheConfig with the requested write_policy type."""
    return CacheConfig(write_policy=WritePolicyConfig(type=write_policy_type))


def test_raises_on_on_any_miss():
    from scripts.serve_policy import _enforce_runtime_write_policy

    original = _make_cache_config("on_any_miss")
    with pytest.raises(ConfigValidationError, match="on_any_miss"):
        _enforce_runtime_write_policy(original)
    # Source config must not be mutated by the failed enforcement call.
    assert original.write_policy.type == "on_any_miss"


def test_raises_on_always():
    from scripts.serve_policy import _enforce_runtime_write_policy

    with pytest.raises(ConfigValidationError, match="always"):
        _enforce_runtime_write_policy(_make_cache_config("always"))


def test_passthrough_when_already_never():
    from scripts.serve_policy import _enforce_runtime_write_policy

    original = _make_cache_config("never")
    result = _enforce_runtime_write_policy(original)
    # 'never' is the only accepted runtime policy: returned unchanged, no raise.
    assert result.write_policy.type == "never"
