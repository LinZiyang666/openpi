"""Check that a copy of a dataset still matches what the ledger claims.

Two shapes are supported: a single-file dataset (a cache artifact pkl) and a
tree dataset (a collected corpus of h5 files). Both are reduced to the same
three measured quantities -- ``file_count``, ``size_bytes``, ``sha256`` -- so a
record's ``integrity`` block means the same thing regardless of shape.

Why there is no ``du`` anywhere in this file: ``du`` de-duplicates hard links
and, on the symlinked corpora this project uses, silently under-reports. The
census here walks with ``os.walk(followlinks=False)`` and sums ``lstat`` sizes
of regular files only, so hard-linked families are counted once per path (which
is what "how many files does this tree present" means) and symlinks are never
followed into another tree's bytes.

For a remote dataset this module does not reach across the network itself; it
prints the exact ``tether exec`` command to run on the owning node, so the
measurement happens where the bytes are instead of pulling gigabytes to
measure them.

Public interface: ``sha256_file``, ``census``, ``verify_path``, ``remote_command``,
``verify_analysis_task``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from exp.data_authority.registry import (
    ANALYSIS_DIR,
    load_analysis_manifest,
    load_record,
    record_path_for,
)

_CHUNK = 1 << 20


# ------------------------------------------------------------------
# Measurement
# ------------------------------------------------------------------


def sha256_file(path: str | Path) -> str:
    """Streaming sha256 of one file, as 64 lowercase hex chars."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def census(root: str | Path) -> dict:
    """Measure a file or a tree into ``{file_count, size_bytes, sha256}``.

    For a tree the digest is over the sorted ``(relpath, sha256)`` listing, not
    over concatenated bytes: that makes it order-independent and lets a single
    mismatching member be located by re-running with ``--per-file``.
    """
    root = Path(root)
    if root.is_file() and not root.is_symlink():
        return {
            "file_count": 1,
            "size_bytes": root.stat().st_size,
            "sha256": sha256_file(root),
        }

    members: list[tuple[str, int, str]] = []
    for dirpath, _dirnames, filenames in os.walk(root, followlinks=False):
        for name in sorted(filenames):
            full = Path(dirpath) / name
            if not full.is_file() or full.is_symlink():
                continue
            rel = str(full.relative_to(root))
            members.append((rel, full.stat().st_size, sha256_file(full)))

    members.sort()
    listing = "\n".join(f"{rel} {sha}" for rel, _size, sha in members)
    return {
        "file_count": len(members),
        "size_bytes": sum(size for _rel, size, _sha in members),
        "sha256": hashlib.sha256(listing.encode("utf-8")).hexdigest(),
        "members": members,
    }


# ------------------------------------------------------------------
# Comparison against the ledger
# ------------------------------------------------------------------


def verify_path(rec: dict, path: str | Path) -> dict:
    """Compare a local copy against ``rec['integrity']``.

    Returns a report with a boolean ``ok`` plus every individual comparison, so
    a caller can tell "wrong file" (sha differs, size matches) from "truncated"
    (both differ) without a second run.
    """
    measured = census(path)
    claimed = rec["integrity"]
    checks = {
        key: {
            "claimed": claimed.get(key),
            "measured": measured[key],
            "ok": claimed.get(key) == measured[key],
        }
        for key in ("file_count", "size_bytes", "sha256")
    }
    return {
        "dataset_id": rec["dataset_id"],
        "path": str(path),
        "ok": all(c["ok"] for c in checks.values()),
        "checks": checks,
    }


def verify_analysis_task(task: str, *, analysis_dir: Path | None = None) -> dict:
    """Check every file a task manifest claims against the bytes on disk.

    Missing files are reported separately from mismatching ones: a missing figure
    means the collection was never completed, a mismatching one means somebody
    regenerated the plot without re-collecting it. Different fixes, so they must
    not be reported as the same failure. Files present but absent from the
    manifest are reported too -- an unregistered figure is exactly the orphan the
    manifest exists to prevent.
    """
    task_dir = (analysis_dir or ANALYSIS_DIR) / task
    manifest = load_analysis_manifest(task, analysis_dir)

    missing: list[str] = []
    mismatched: list[dict] = []
    for entry in manifest["files"]:
        full = task_dir / entry["name"]
        if not full.is_file():
            missing.append(entry["name"])
            continue
        size = full.stat().st_size
        sha = sha256_file(full)
        if size != entry["size_bytes"] or sha != entry["sha256"]:
            mismatched.append(
                {
                    "name": entry["name"],
                    "claimed": {
                        "sha256": entry["sha256"],
                        "size_bytes": entry["size_bytes"],
                    },
                    "measured": {"sha256": sha, "size_bytes": size},
                }
            )

    claimed = {entry["name"] for entry in manifest["files"]}
    present = {
        str(f.relative_to(task_dir))
        for f in task_dir.rglob("*")
        if f.is_file() and not f.is_symlink() and f.name != "MANIFEST.json"
    }
    unregistered = sorted(present - claimed)
    return {
        "task": task,
        "ok": not missing and not mismatched and not unregistered,
        "file_count": len(manifest["files"]),
        "missing": missing,
        "mismatched": mismatched,
        "unregistered": unregistered,
    }


def remote_command(rec: dict) -> str:
    """The command to measure a tether-hosted dataset on its owning node."""
    node = rec["authority"]["node"]
    path = rec["authority"]["path"]
    return (
        f"tether exec {node} -- bash -lc "
        f'\'find {path} -type f | wc -l; find {path} -type f -printf "%s\\n" | '
        f"paste -sd+ | bc; sha256sum {path}'"
    )


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Verify a dataset copy against the ledger")
    ap.add_argument(
        "dataset_id", help="dataset id, or an analysis task slug with --analysis"
    )
    ap.add_argument(
        "--path",
        help="local copy to check; omit for a remote dataset to print its command",
    )
    ap.add_argument(
        "--per-file", action="store_true", help="on mismatch, list per-member digests"
    )
    ap.add_argument(
        "--analysis",
        action="store_true",
        help="treat the argument as an analysis/<task> slug",
    )
    args = ap.parse_args(argv)

    if args.analysis:
        report = verify_analysis_task(args.dataset_id)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return 0 if report["ok"] else 2

    record_file = record_path_for(args.dataset_id)
    if not record_file.exists():
        print(f"no such dataset_id: {args.dataset_id}", file=sys.stderr)
        return 1
    rec = load_record(record_file)

    path = args.path
    if path is None:
        if rec["authority"]["access"] == "local":
            path = rec["authority"]["path"]
        else:
            print(
                f"# {args.dataset_id} lives on {rec['authority']['node']}; measure it there:"
            )
            print(remote_command(rec))
            return 0

    if not Path(path).exists():
        print(f"path does not exist: {path}", file=sys.stderr)
        return 1

    report = verify_path(rec, path)
    print(json.dumps({k: v for k, v in report.items()}, indent=2))
    if not report["ok"] and args.per_file:
        for rel, size, sha in census(path).get("members", []):
            print(f"  {sha}  {size:>12}  {rel}")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
