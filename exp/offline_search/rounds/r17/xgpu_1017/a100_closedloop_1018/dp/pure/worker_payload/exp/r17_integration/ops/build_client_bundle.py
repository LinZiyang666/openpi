"""Explicit client-only payload, deterministic tar and SHA attestation (stdlib)."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile

FORBIDDEN = {"models", "models_pytorch", "training", "vendor", "common", "cache", "serve.py"}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(name):
    p = Path(name)
    if not name or p.is_absolute() or ".." in p.parts or str(p) != name or name == ".eval.lock":
        raise ValueError("unsafe archive path: " + name)
    return p


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def add_bytes(tar, name, data):
    member = tarfile.TarInfo(name)
    member.size, member.mode, member.mtime = len(data), 0o644, 0
    tar.addfile(member, io.BytesIO(data))


def build_client_bundle(root, files, output):
    root, output = Path(root).resolve(), Path(output)
    names = sorted(set(files))
    if len(names) != len(files) or not names or "PAYLOAD.json" in names:
        raise ValueError("nonempty unique explicit client file list required")
    content = {}
    for name in names:
        relative = safe_name(name)
        contract_file = name in ("exp/r17_integration/common/__init__.py", "exp/r17_integration/common/contracts.py")
        if set(relative.parts) & FORBIDDEN and not contract_file:
            raise ValueError("model-island code cannot enter client payload: " + name)
        source = root / relative
        if source.is_symlink() or not source.resolve().is_relative_to(root):
            raise ValueError("symlink/escaped payload source")
        content[name] = source.read_bytes()
    manifest = dict(schema="r17.client_payload.v1", python="3.11", files={
        name: hashlib.sha256(data).hexdigest() for name, data in content.items()})
    manifest["digest"] = hashlib.sha256(json_bytes(manifest)).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    part = output.with_name(output.name + ".part")
    with tarfile.open(part, "w") as tar:
        for name, data in content.items():
            add_bytes(tar, name, data)
        add_bytes(tar, "PAYLOAD.json", json_bytes(manifest))
    with part.open("rb") as f:
        os.fsync(f.fileno())
    os.replace(part, output)
    return dict(archive_sha256=sha256(output), manifest=manifest)


def worker_payload_files(root, *, debug=False):
    """P2's declared profile plus transitive conductor package import closure."""
    from exp.r17_integration.conductor.worker_entry import CLIENT_ONLY_FILES
    root = Path(root)
    names = set(CLIENT_ONLY_FILES)
    # openpi.conductor.__init__ imports these core, stdlib-only modules.
    names.update(str(p.relative_to(root)) for p in (root / "src/openpi/conductor").glob("*.py"))
    names.update(str(p.relative_to(root)) for p in (root / "packages/openpi-client/src/openpi_client").glob("*.py")
                 if not p.name.endswith("_test.py"))
    if debug:
        names.update("exp/r17_integration/debug/" + name + ".py" for name in ("validate", "capture_bridge", "reader_profiles"))
        names.update("exp/offline_search/debug/" + name + ".py" for name in ("__init__", "schema"))
        names.update("exp/offline_search/debug/server/" + name + ".py" for name in (
            "__init__", "observer", "writer", "diagnostics", "manifest", "digests"))
    return sorted(names)


def verified_members(archive, manifest_name):
    """Validate every member before callers install anything."""
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(names) != len(set(names)):
            raise ValueError("duplicate archive members")
        for m in members:
            safe_name(m.name)
            if not m.isfile():
                raise ValueError("nonregular archive member")
        content = {m.name: tar.extractfile(m).read() for m in members}
    manifest = json.loads(content.pop(manifest_name))
    sealed = {k: v for k, v in manifest.items() if k != "digest"}
    if hashlib.sha256(json_bytes(sealed)).hexdigest() != manifest["digest"]:
        raise ValueError("manifest digest mismatch")
    if set(content) != set(manifest["files"]):
        raise ValueError("archive missing/unlisted files")
    for name, data in content.items():
        if hashlib.sha256(data).hexdigest() != manifest["files"][name]:
            raise ValueError("archive SHA mismatch: " + name)
    return manifest, content


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--file-list", type=Path, required=True, help="JSON list of client-only relative paths")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build_client_bundle(a.root, json.loads(a.file_list.read_text()), a.output)))


if __name__ == "__main__":
    main()
