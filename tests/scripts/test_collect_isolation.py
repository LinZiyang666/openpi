"""--collect concurrency-isolation guard.

Embedding collection attaches module-global forward hooks to the shared base
model, so it is only correct on the single-connection (C1) path with one
replica. ``scripts.serve_policy._validate_collect_isolation`` rejects the
concurrent / multi-replica combinations up front rather than silently writing
cross-contaminated HDF5.
"""

from __future__ import annotations

import dataclasses

import pytest

from scripts.serve_policy import Args, _validate_collect_isolation


def _args(**overrides) -> Args:
    return dataclasses.replace(Args(), **overrides)


def test_no_collect_is_inert_under_any_concurrency():
    # The guard only fires for --collect; concurrent / multi-replica serving
    # without collection is unaffected.
    _validate_collect_isolation(_args(collect=False, concurrent=True, replicas=3))


def test_collect_with_default_concurrent_is_rejected():
    # concurrent defaults to True, so a bare --collect must fail fast.
    with pytest.raises(ValueError, match="non-concurrent"):
        _validate_collect_isolation(_args(collect=True))


def test_collect_non_concurrent_single_replica_is_allowed():
    _validate_collect_isolation(_args(collect=True, non_concurrent=True, replicas=1))


def test_collect_with_explicit_no_concurrent_is_allowed():
    # tyro's --no-concurrent sets concurrent=False directly (no --non-concurrent).
    _validate_collect_isolation(_args(collect=True, concurrent=False))


def test_collect_with_multiple_replicas_is_rejected():
    with pytest.raises(ValueError, match="single replica"):
        _validate_collect_isolation(_args(collect=True, non_concurrent=True, replicas=2))


def test_collect_multi_replica_message_takes_priority():
    # replicas>1 is checked before the concurrency check, so the operator sees
    # the replica-specific guidance even under the default concurrent mode.
    with pytest.raises(ValueError, match="single replica"):
        _validate_collect_isolation(_args(collect=True, replicas=2))
