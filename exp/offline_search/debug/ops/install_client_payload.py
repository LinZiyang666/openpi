"""Python 3.8, coordinator-only verified installation into the client tree.

Derived from P3 install_client_payload.py. Never replace shared launchers.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, partial = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            os.fchmod(handle.fileno(), 0o755 if path.suffix == ".sh" else 0o644)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, str(path))
    finally:
        if os.path.exists(partial):
            os.unlink(partial)


def install(bundle, root):
    bundle, root = Path(bundle), Path(root).resolve()
    manifest = json.loads((bundle / "client_manifest.json").read_text())
    if manifest["schema"] != "osdebug.client_bundle.v1":
        raise ValueError("wrong bundle schema")
    protected = {p: (root / p).read_bytes() for p in manifest["protected"] if (root / p).is_file()}
    staged, seen = [], set()
    for row in manifest["files"]:
        rel = PurePosixPath(row["relative_path"])
        allowed = (str(rel).startswith("exp/offline_search/debug/") or str(rel) in
                   ("exp/__init__.py", "exp/offline_search/__init__.py", "os_cl/run_arm_debug.sh") or
                   str(rel).startswith("os_cl/cfg/"))
        if rel.is_absolute() or ".." in rel.parts or not allowed or str(rel) in manifest["protected"] or str(rel) in seen:
            raise ValueError("unsafe/duplicate install target")
        seen.add(str(rel))
        src, target = bundle / "payload" / Path(*rel.parts), root / Path(*rel.parts)
        if any(p.is_symlink() for p in (src, *src.parents, target, *target.parents)):
            raise ValueError("symlink install target/source")
        data = src.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"] or len(data) != row["bytes"]:
            raise ValueError("payload checksum mismatch")
        if row["operation"] not in ("replace_owned", "marker_if_missing"):
            raise ValueError("invalid install operation")
        if src.suffix == ".py":
            compile(data, str(target), "exec", dont_inherit=True)
        staged.append((row, target, data))
    for row, target, data in staged:
        if row["operation"] != "marker_if_missing" or not target.exists():
            atomic(target, data)
    if any((root / name).read_bytes() != data for name, data in protected.items()):
        raise RuntimeError("shared launcher changed during install")
    return dict(files=len(staged), protected_unchanged=sorted(protected))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("/scratch/zixuans8/openpi_trace"))
    a = ap.parse_args()
    print(json.dumps(install(a.bundle, a.root)))


if __name__ == "__main__":
    main()
