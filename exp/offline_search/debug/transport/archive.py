"""Coordinator remote staging: exact completed files, separate partial inventory.

This module has only standard-library imports so staging needs no NumPy or
simulator. No service or worker is launched here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

from .protocol import filename

SPILL_FILES = ("stream.osdebugspill", "stream.osdebugspill.json")
PARTIAL_FILES = ("stream.osdebugspill.part", "stream.osdebugspill.json.tmp", "stream.osdebugspill.json.part")


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build_archive(root, out, mode="stream"):
    root, out = Path(root), Path(out)
    if mode not in ("stream", "file"):
        raise ValueError("invalid collection mode")
    partials, selected = [], []
    for path in sorted(root.glob("*/*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if path.name.startswith("stream.osdebugspill") and path.name not in SPILL_FILES:
            partials.append(dict(path=relative, episode_key=path.parent.name, bytes=path.stat().st_size,
                                 reason="unfinished spill from an interrupted attempt"))
            continue
        if mode == "stream":
            if path.name not in SPILL_FILES:
                continue
        else:
            try:
                filename(path.name)
            except ValueError:
                continue
        if path.is_symlink():
            raise ValueError("symlink collection payload")
        selected.append((path, relative))
    part = out.with_name(out.name + ".part")
    with tarfile.open(str(part), "w") as tf:
        for path, relative in selected:
            tf.add(str(path), arcname=relative, recursive=False)
    part.replace(out)
    return dict(sha256=sha256(out), bytes=out.stat().st_size, files=len(selected), stale_partial_spills=partials)


def remove_verified_archive(path, expected_sha):
    path = Path(path)
    if not path.exists():
        return dict(removed=False)
    if path.is_symlink() or sha256(path) != expected_sha:
        raise ValueError("remote archive changed after verified collection")
    path.unlink()
    return dict(removed=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=("stream", "file"), default="stream")
    ap.add_argument("--remove", action="store_true")
    ap.add_argument("--sha256")
    args = ap.parse_args()
    if args.remove:
        if not args.sha256:
            ap.error("--sha256 required for removal")
        result = remove_verified_archive(args.out, args.sha256)
    else:
        if args.root is None:
            ap.error("--root required for staging")
        result = build_archive(args.root, args.out, args.mode)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
