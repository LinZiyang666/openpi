"""Tests for ``_fill_deferred_dump_paths`` (verdict_factor_judge B2).

`load_cache_config` ctrl handler invokes this between yaml parsing and
storing the bundle. It mutates the parsed ``CacheConfig`` in place so any
``dump.deferred=True`` checkpoint gets a concrete ``path`` derived from the
warmup dump root + yaml_id, then flips ``deferred=False`` so downstream
builders see a normal dump. Failure modes (missing yaml_id, unconfigured
root, path traversal) raise ``ValueError`` so the dispatcher can surface
them to the client as an error ack.
"""

from __future__ import annotations

import pytest

from openpi.cache.config import (
    CacheConfig,
    CheckpointConfig,
    DumpConfig,
    JudgeConfig,
)
from openpi.serving import websocket_policy_server as wps


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def warmup_root(tmp_path):
    root = tmp_path / "warmup"
    root.mkdir(mode=0o700)
    wps.set_warmup_dump_root(root)
    yield root.resolve()
    wps.set_warmup_dump_root(None)


def _config_with_deferred_dump(*, deferred: bool, path: str = "") -> CacheConfig:
    cfg = CacheConfig()
    cfg.checkpoints = {
        "cp1": CheckpointConfig(
            judge=JudgeConfig(
                type="always_hit",
                dump=DumpConfig(
                    path=path,
                    config_id="warmup_cfg",
                    factors=[],
                    deferred=deferred,
                ),
            ),
        ),
    }
    return cfg


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_fill_replaces_deferred_path_and_clears_flag(warmup_root) -> None:
    cfg = _config_with_deferred_dump(deferred=True)
    wps._fill_deferred_dump_paths(cfg, "phase1_idx0__warmup")
    dump = cfg.checkpoints["cp1"].judge.dump
    assert dump.path == str(warmup_root / "phase1_idx0__warmup.jsonl")
    # Flag flipped so downstream validators / builders see a regular dump.
    assert dump.deferred is False


def test_fill_is_noop_when_no_deferred_dump(warmup_root) -> None:
    cfg = _config_with_deferred_dump(deferred=False, path="/already/set.jsonl")
    wps._fill_deferred_dump_paths(cfg, None)  # yaml_id absence is OK
    # Untouched (legacy yaml unaffected by deferred resolution).
    dump = cfg.checkpoints["cp1"].judge.dump
    assert dump.path == "/already/set.jsonl"
    assert dump.deferred is False


def test_fill_skips_checkpoints_without_dump(warmup_root) -> None:
    cfg = CacheConfig()
    cfg.checkpoints = {"cp1": CheckpointConfig(judge=JudgeConfig(type="always_hit"))}
    # Should not raise even with yaml_id=None — no deferred dumps present.
    wps._fill_deferred_dump_paths(cfg, None)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_fill_raises_when_yaml_id_missing(warmup_root) -> None:
    cfg = _config_with_deferred_dump(deferred=True)
    with pytest.raises(ValueError, match="yaml_id is required"):
        wps._fill_deferred_dump_paths(cfg, None)
    with pytest.raises(ValueError, match="yaml_id is required"):
        wps._fill_deferred_dump_paths(cfg, "")


def test_fill_raises_when_warmup_root_unconfigured(monkeypatch) -> None:
    monkeypatch.setattr(wps, "_warmup_dump_root", None)
    cfg = _config_with_deferred_dump(deferred=True)
    with pytest.raises(ValueError, match="warmup-dump-root"):
        wps._fill_deferred_dump_paths(cfg, "phase1_idx0__warmup")


def test_fill_rejects_yaml_id_path_traversal(warmup_root) -> None:
    cfg = _config_with_deferred_dump(deferred=True)
    with pytest.raises(ValueError, match="allowlist"):
        wps._fill_deferred_dump_paths(cfg, "../etc/passwd")
    with pytest.raises(ValueError, match="allowlist"):
        wps._fill_deferred_dump_paths(cfg, "subdir/x")
