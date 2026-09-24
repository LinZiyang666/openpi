"""``TraceConfig`` parsing and validation (plan §10, §12).

Covers: the ``trace:`` yaml block parses into the dataclass (registry), the
``shadow_teacher:`` block now parses too (HEAD defect fixed alongside), the
block-internal rules, the cross-section exclusions, the yaml + CLI three-state
merge, and the twin config stripping / read-only path field guard.
"""

from __future__ import annotations

import dataclasses
import re

import pytest

from openpi.cache import config as cfg
from openpi.cache.config import (
    CacheConfig,
    ConfigValidationError,
    ShadowTeacherConfig,
    TraceConfig,
    WritePolicyConfig,
    _dict_to_dataclass,
    _trace_errors,
    validate_effective_trace,
)
from openpi.cache.trace.runtime import (
    TWIN_READONLY_PATH_FIELDS,
    TWIN_STRIP_FIELDS,
    strip_twin_config,
)


def test_trace_block_parses_into_dataclass():
    raw = {
        "trace": {"enabled": True, "out_dir": "/tmp/t", "record_noise_actions": True,
                  "raw_image_keys": ["observation/image"]},
        "shadow_teacher": {"enabled": False, "path": "x.jsonl"},
        "write_policy": {"type": "never"},
    }
    c = _dict_to_dataclass(CacheConfig, raw)
    assert isinstance(c.trace, TraceConfig)
    assert c.trace.enabled and c.trace.record_noise_actions
    assert c.trace.raw_image_keys == ["observation/image"]
    # HEAD defect: ``shadow_teacher:`` used to stay a plain dict.
    assert isinstance(c.shadow_teacher, ShadowTeacherConfig)


def test_block_rules():
    c = CacheConfig(trace=TraceConfig(rng_isolation="bogus", queue_steps=0))
    errs = "\n".join(_trace_errors(c))
    assert "rng_isolation" in errs and "queue_steps" in errs
    c = CacheConfig(trace=TraceConfig(record_noise_actions=True, record_prefix_tokens=False))
    assert any("record_prefix_tokens" in e for e in _trace_errors(c))
    c = CacheConfig(trace=TraceConfig(enabled=True, out_dir=""))
    assert any("out_dir" in e for e in _trace_errors(c))


def test_cross_section_exclusions():
    base = CacheConfig(
        trace=TraceConfig(enabled=True, out_dir="/tmp/t"),
        write_policy=WritePolicyConfig(type="never"),
    )
    assert _trace_errors(base) == []
    bad = dataclasses.replace(base, write_policy=WritePolicyConfig(type="always"))
    assert any("write_policy" in e for e in _trace_errors(bad))
    bad = dataclasses.replace(base, collection=cfg.CollectionConfig(export_collect_meta=True))
    assert any("export_collect_meta" in e for e in _trace_errors(bad))
    bad = dataclasses.replace(base, shadow_teacher=ShadowTeacherConfig(enabled=True, path="x"))
    assert any("shadow_teacher" in e for e in _trace_errors(bad))
    bad = dataclasses.replace(base, routing=cfg.RoutingConfig(hit_to="h:1"))
    assert any("routing" in e for e in _trace_errors(bad))


def test_disabled_block_is_inert_for_any_config():
    c = CacheConfig()  # write_policy on_any_miss, trace disabled
    assert _trace_errors(c) == []
    assert validate_effective_trace(c, out_dir=None, build_cache=None) is None


def test_effective_merge_three_state():
    c = CacheConfig(write_policy=WritePolicyConfig(type="never"))
    eff = validate_effective_trace(c, out_dir="/tmp/x", build_cache=None)
    assert eff.enabled and eff.out_dir == "/tmp/x" and eff.record_noise_actions is False
    eff = validate_effective_trace(c, out_dir="/tmp/x", build_cache=True)
    assert eff.record_noise_actions is True
    c2 = dataclasses.replace(c, trace=TraceConfig(enabled=True, out_dir="/y", record_noise_actions=True))
    eff = validate_effective_trace(c2, out_dir=None, build_cache=None)
    assert eff.out_dir == "/y" and eff.record_noise_actions is True
    eff = validate_effective_trace(c2, out_dir=None, build_cache=False)
    assert eff.record_noise_actions is False
    # No cache config at all (build-cache form): only the block itself is checked.
    eff = validate_effective_trace(None, out_dir="/tmp/z", build_cache=True)
    assert eff.enabled and eff.record_noise_actions
    with pytest.raises(ConfigValidationError):
        validate_effective_trace(CacheConfig(), out_dir="/tmp/z", build_cache=None)


def test_strip_twin_config_clears_every_write_target():
    c = CacheConfig(write_policy=WritePolicyConfig(type="never"))
    judge = c.checkpoints["cp1"].judge
    judge = dataclasses.replace(judge, dump_dir="/d", state_log_dir="/s", snapshot_every=5)
    c = dataclasses.replace(
        c,
        checkpoints={"cp1": dataclasses.replace(c.checkpoints["cp1"], judge=judge)},
        timer=dataclasses.replace(c.timer, output_csv_dir="/csv"),
    )
    stripped = strip_twin_config(c)
    j = stripped.checkpoints["cp1"].judge
    assert j.dump is None and j.dump_dir is None and j.state_log_dir is None
    assert j.snapshot_every is None
    assert stripped.timer.output_csv_dir is None and stripped.timer.enabled is False
    # The original is untouched.
    assert c.checkpoints["cp1"].judge.dump_dir == "/d"


_PATH_LIKE = re.compile(r"(dir|path|file|dump|log)", re.IGNORECASE)


def _walk_dataclasses(root):
    seen = set()
    stack = [root]
    while stack:
        cls = stack.pop()
        if cls in seen or not dataclasses.is_dataclass(cls):
            continue
        seen.add(cls)
        yield cls
        for f in dataclasses.fields(cls):
            t = cfg._resolve_type(f.type) if isinstance(f.type, str) else f.type
            if dataclasses.is_dataclass(t):
                stack.append(t)
            # dict[str, CheckpointConfig] etc.: follow the registry names in the annotation
            if isinstance(f.type, str):
                for name in re.findall(r"[A-Z][A-Za-z0-9]*Config", f.type):
                    inner = cfg._CONFIG_TYPES.get(name)
                    if inner is not None:
                        stack.append(inner)


def test_every_path_like_config_field_is_classified():
    """Guard (plan §4.2): a new path-like field must be either stripped for the
    twin or declared read-only, never silently inherited."""
    unclassified = []
    for cls in _walk_dataclasses(CacheConfig):
        name = cls.__name__
        for f in dataclasses.fields(cls):
            if not _PATH_LIKE.search(f.name):
                continue
            stripped = f.name in TWIN_STRIP_FIELDS.get(name, ())
            readonly = f.name in TWIN_READONLY_PATH_FIELDS.get(name, ())
            if not (stripped or readonly):
                unclassified.append(f"{name}.{f.name}")
    assert not unclassified, unclassified
