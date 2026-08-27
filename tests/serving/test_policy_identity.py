"""Tests for the policy attestation seam (content fingerprint + spec resolution)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))

from openpi.serving.policy_identity import compute_policy_fingerprint


def _make_ckpt(tmp_path, files: dict[str, bytes]):
    root = tmp_path / "ckpt"
    root.mkdir(exist_ok=True)
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(content)
    return root


def test_fingerprint_is_content_sensitive_at_equal_size(tmp_path):
    root = _make_ckpt(tmp_path, {"weights.bin": b"AAAA", "config.json": b"{}"})
    fp_a = compute_policy_fingerprint(str(root), "pi05_libero")
    compute_policy_fingerprint.cache_clear()
    (root / "weights.bin").write_bytes(b"BBBB")  # same size, different bytes
    fp_b = compute_policy_fingerprint(str(root), "pi05_libero")
    assert fp_a != fp_b


def test_fingerprint_depends_on_config_name(tmp_path):
    root = _make_ckpt(tmp_path, {"w.bin": b"x"})
    a = compute_policy_fingerprint(str(root), "cfg_a")
    b = compute_policy_fingerprint(str(root), "cfg_b")
    assert a != b


def test_fingerprint_rejects_empty_root(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    compute_policy_fingerprint.cache_clear()
    with pytest.raises(ValueError):
        compute_policy_fingerprint(str(root), "cfg")


def test_fingerprint_rejects_escaping_symlink(tmp_path):
    root = _make_ckpt(tmp_path, {"w.bin": b"x"})
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"secret")
    (root / "link.bin").symlink_to(outside)
    compute_policy_fingerprint.cache_clear()
    with pytest.raises(ValueError):
        compute_policy_fingerprint(str(root), "cfg")


def test_resolve_effective_policy_spec_branches():
    import serve_policy as sp

    ckpt = sp.Checkpoint(config="c", dir="/d")
    args = sp.Args(policy=ckpt)
    assert sp._resolve_effective_policy_spec(args) is ckpt

    args_default = sp.Args(env=sp.EnvMode.LIBERO, policy=sp.Default())
    resolved = sp._resolve_effective_policy_spec(args_default)
    assert resolved == sp.DEFAULT_CHECKPOINT[sp.EnvMode.LIBERO]
