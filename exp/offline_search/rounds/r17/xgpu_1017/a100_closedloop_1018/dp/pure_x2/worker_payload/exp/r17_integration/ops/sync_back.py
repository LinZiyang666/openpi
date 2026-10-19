"""Quiescent journal snapshot and verified, idempotent install (stdlib only)."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tarfile

from .build_client_bundle import add_bytes, json_bytes, safe_name, sha256


def atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError("symlink install target")
    part = path.with_name(path.name + ".part")
    with part.open("wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(part, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def snapshot_run(run, archive):
    run, archive = Path(run), Path(archive)
    if not (run / "manifest.json").is_file():
        raise ValueError("run manifest missing")
    if archive.resolve().is_relative_to(run.resolve()):
        raise ValueError("snapshot archive must be outside run")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with (run / ".eval.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("busy journal: driver must exit before snapshot") from exc
        sources, hashes, partials = {}, {}, []
        for path in sorted(run.rglob("*")):
            if path.is_symlink():
                raise ValueError("symlink snapshot source")
            if not path.is_file() or path.name == ".eval.lock":
                continue
            name = str(path.relative_to(run))
            if name == "TRANSFER_SHA256.json":
                continue
            with path.open("rb") as f:
                os.fsync(f.fileno())
            sources[name] = path
            hashes[name] = sha256(path)
            if path.name.endswith((".part", ".tmp", ".failed")):
                partials.append(name)
        manifest = dict(schema="r17.snapshot.v1", files=hashes, partials=partials)
        manifest["digest"] = hashlib.sha256(json_bytes(manifest)).hexdigest()
        part = archive.with_name(archive.name + ".part")
        with tarfile.open(part, "w") as tar:
            for name, source in sources.items():
                info = tarfile.TarInfo(name)
                info.size, info.mode, info.mtime = source.stat().st_size, 0o644, 0
                with source.open("rb") as f:
                    tar.addfile(info, f)
            add_bytes(tar, "TRANSFER_SHA256.json", json_bytes(manifest))
        # Catch an uncooperative writer rather than claim a consistent snapshot.
        now = {str(p.relative_to(run)) for p in run.rglob("*") if p.is_file() and p.name != ".eval.lock" and str(p.relative_to(run)) != "TRANSFER_SHA256.json"}
        if now != set(sources) or any(sha256(run / name) != manifest["files"][name] for name in sources):
            raise RuntimeError("run changed while snapshotting")
        with part.open("rb") as f:
            os.fsync(f.fileno())
        os.replace(part, archive)
    return dict(archive_sha256=sha256(archive), manifest=manifest)


def install_snapshot(archive, destination, *, expected_sha256):
    if sha256(archive) != expected_sha256:
        raise ValueError("transfer archive SHA mismatch")
    # Two passes keep large H5/NPZ snapshots bounded without writing unverified data.
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive members")
        for member in members:
            safe_name(member.name)
            if not member.isfile():
                raise ValueError("nonregular archive member")
        manifest = json.load(tar.extractfile("TRANSFER_SHA256.json"))
        sealed = {k: v for k, v in manifest.items() if k != "digest"}
        if hashlib.sha256(json_bytes(sealed)).hexdigest() != manifest["digest"]:
            raise ValueError("snapshot manifest digest mismatch")
        if set(names) - {"TRANSFER_SHA256.json"} != set(manifest["files"]):
            raise ValueError("archive missing/unlisted files")
        for name, expected in manifest["files"].items():
            h = hashlib.sha256()
            with tar.extractfile(name) as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            if h.hexdigest() != expected:
                raise ValueError("archive SHA mismatch: " + name)
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("symlink snapshot destination")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".eval.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("busy local journal") from exc
        for name in manifest["files"]:
            target = destination / safe_name(name)
            if not target.resolve().is_relative_to(destination.resolve()) or target.is_symlink():
                raise ValueError("escaped snapshot destination")
            # Existing conflicting data is never silently overwritten.
            if target.exists() and sha256(target) != manifest["files"][name]:
                raise ValueError("existing snapshot differs: " + name)
        with tarfile.open(archive) as tar:
            for name in manifest["files"]:
                target = destination / name
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                part = target.with_name(target.name + ".part")
                h = hashlib.sha256()
                with tar.extractfile(name) as source, part.open("wb") as out:
                    for chunk in iter(lambda: source.read(1024 * 1024), b""):
                        h.update(chunk)
                        out.write(chunk)
                    out.flush()
                    os.fsync(out.fileno())
                if h.hexdigest() != manifest["files"][name]:
                    raise ValueError("archive changed during installation")
                os.replace(part, target)
        if sha256(archive) != expected_sha256:
            raise ValueError("archive changed during installation")
        atomic_bytes(destination / "TRANSFER_SHA256.json", json_bytes(manifest))
    return dict(status="verified", files=len(manifest["files"]), digest=manifest["digest"], partials=manifest["partials"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    build = sub.add_parser("snapshot")
    build.add_argument("run", type=Path)
    build.add_argument("archive", type=Path)
    install = sub.add_parser("install")
    install.add_argument("archive", type=Path)
    install.add_argument("destination", type=Path)
    install.add_argument("--sha256", required=True)
    a = p.parse_args()
    value = snapshot_run(a.run, a.archive) if a.mode == "snapshot" else install_snapshot(a.archive, a.destination, expected_sha256=a.sha256)
    print(json.dumps(value))


if __name__ == "__main__":
    main()
