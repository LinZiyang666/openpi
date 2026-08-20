"""Census and comparison behaviour, including the hard-link / symlink traps."""

from __future__ import annotations

import hashlib
import os

from exp.data_authority.verify import census, remote_command, sha256_file, verify_path


def _rec(sha: str, size: int, count: int) -> dict:
    return {
        "dataset_id": "demo/suite/thing",
        "authority": {
            "node": "weilandserver",
            "path": "/data/x.pkl",
            "access": "tether",
        },
        "integrity": {"sha256": sha, "size_bytes": size, "file_count": count},
    }


def test_sha256_file_matches_hashlib(tmp_path):
    blob = tmp_path / "a.bin"
    blob.write_bytes(b"payload" * 1000)
    assert sha256_file(blob) == hashlib.sha256(blob.read_bytes()).hexdigest()


def test_single_file_census(tmp_path):
    blob = tmp_path / "a.bin"
    blob.write_bytes(b"x" * 42)
    got = census(blob)
    assert got["file_count"] == 1
    assert got["size_bytes"] == 42


def test_tree_census_counts_every_path_of_a_hard_link_family(tmp_path):
    # du would report these two paths as one file's worth of bytes. The census
    # must not: "how many files does this tree present" is the question.
    root = tmp_path / "tree"
    (root / "sub").mkdir(parents=True)
    original = root / "a.bin"
    original.write_bytes(b"y" * 100)
    os.link(original, root / "sub" / "b.bin")

    got = census(root)
    assert got["file_count"] == 2
    assert got["size_bytes"] == 200


def test_tree_census_does_not_follow_symlinks(tmp_path):
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"z" * 999)
    root = tmp_path / "tree"
    root.mkdir()
    (root / "real.bin").write_bytes(b"w" * 7)
    (root / "link.bin").symlink_to(outside)

    got = census(root)
    assert got["file_count"] == 1
    assert got["size_bytes"] == 7


def test_tree_digest_is_order_independent_but_content_sensitive(tmp_path):
    def build(slug: str, writes: list[tuple[str, bytes]]) -> str:
        root = tmp_path / slug
        root.mkdir()
        for name, payload in writes:
            (root / name).write_bytes(payload)
        return census(root)["sha256"]

    forward = [("1.bin", b"aaa"), ("2.bin", b"bbb")]
    # Same members, created in the opposite order: the digest is over a sorted
    # listing, so creation order must not move it.
    assert build("fwd", forward) == build("rev", list(reversed(forward)))
    # A changed member must move it.
    assert build("fwd2", forward) != build(
        "alt", [("1.bin", b"aaa"), ("2.bin", b"ccc")]
    )


def test_verify_path_reports_each_axis(tmp_path):
    blob = tmp_path / "a.bin"
    blob.write_bytes(b"q" * 8)
    real = sha256_file(blob)

    ok = verify_path(_rec(real, 8, 1), blob)
    assert ok["ok"] is True
    assert all(c["ok"] for c in ok["checks"].values())

    # Right size, wrong bytes: the sha axis alone must fail, so the caller can
    # tell a swapped file from a truncated one.
    bad = verify_path(_rec("b" * 64, 8, 1), blob)
    assert bad["ok"] is False
    assert bad["checks"]["sha256"]["ok"] is False
    assert bad["checks"]["size_bytes"]["ok"] is True


def test_remote_command_targets_the_owning_node():
    cmd = remote_command(_rec("a" * 64, 1, 1))
    assert cmd.startswith("tether exec weilandserver -- ")
    assert "/data/x.pkl" in cmd
    assert "du" not in cmd
