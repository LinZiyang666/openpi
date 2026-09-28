"""Run by the coordinator inside the single remote deployment exec; Python 3.8."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import tempfile


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, data, mode=0o644):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name+".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            os.fchmod(f.fileno(), mode)
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def install(bundle, root):
    bundle, root = Path(bundle), Path(root)
    manifest = json.loads((bundle / "client_manifest.json").read_text())
    stock = root / "os_cl/run_arm.sh"
    before = sha(stock)
    backup = root / "os_cl/run_arm.stock.sh"
    if backup.exists() and sha(backup) != before:
        raise ValueError("existing run_arm.stock.sh differs; refusing to replace a potentially active stock copy")
    staged = []
    for row in manifest["files"]:
        rel = PurePosixPath(row["relative_path"])
        if rel.is_absolute() or ".." in rel.parts or str(rel) == "os_cl/run_arm.sh":
            raise ValueError("unsafe installation target")
        allowed = (str(rel).startswith("exp/offline_search/") or str(rel)=="exp/__init__.py"
                   or str(rel)=="os_cl/run_arm_v2.sh" or str(rel).startswith("os_cl/cfg/"))
        if not allowed:
            raise ValueError("target outside P3 bundle scope")
        src, target = bundle / "payload" / Path(*rel.parts), root / Path(*rel.parts)
        data = src.read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("payload hash mismatch: " + str(rel))
        if src.suffix == ".py":
            compile(data, str(target), "exec")
        if target.is_symlink() or any(p.is_symlink() for p in target.parents if p != root.parent):
            raise ValueError("refusing symlink installation target: " + str(target))
        staged.append((row, target, data))
    # All source hashes and Python syntax validated before installing any file.
    if not backup.exists():
        atomic(backup, stock.read_bytes(), 0o755)
    report = []
    for row, target, data in staged:
        keep = row["operation"] == "marker_if_missing" and target.exists()
        if not keep:
            atomic(target, data, 0o755 if target.suffix == ".sh" else 0o644)
        report.append(dict(path=str(target), sha256=sha(target), preserved_existing_marker=keep))
    if sha(stock) != before or sha(backup) != before:
        raise RuntimeError("stock launcher changed during deployment")
    report.extend([dict(path=str(stock), sha256=before, unchanged=True), dict(path=str(backup), sha256=sha(backup))])
    for row in report:
        print("remote", row["sha256"], row["path"], "preserved" if row.get("preserved_existing_marker") else "")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--root", type=Path, default=Path("/scratch/zixuans8/openpi_trace"))
    a = ap.parse_args()
    install(a.bundle, a.root)


if __name__ == "__main__":
    main()
