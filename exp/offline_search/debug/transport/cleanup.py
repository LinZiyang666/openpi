"""Remove only individually hash-certified spill files after local verification.

Coordinator only. No glob deletion, recursive removal, or dispatch-fence edits.
"""
import argparse
import json
from pathlib import Path
import re

from .receiver import digest, safe_path, sync_dir


def cleanup(root, proof):
    root = Path(root).resolve()
    staged = []
    for row in proof["files"]:
        relative = Path(row["path"])
        if relative.is_absolute() or len(relative.parts) != 2:
            raise ValueError("unsafe cleanup path")
        key, name = relative.parts
        if not re.fullmatch(r"[0-9a-f]{24}_a[1-9][0-9]*", key) or name not in ("stream.osdebugspill", "stream.osdebugspill.json"):
            raise ValueError("cleanup proof outside spill allowlist")
        path = safe_path(root, key, name)
        if not path.exists():
            continue
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError("remote spill changed after verified collection")
        staged.append(path)
    # Verify ALL proofs before removing any file. Keep unknown/new files/fences.
    for path in staged:
        path.unlink()
        sync_dir(path.parent)
    for directory in {p.parent for p in staged}:
        if not any(directory.iterdir()):
            directory.rmdir()
    if root.exists():
        sync_dir(root)
    return dict(removed=len(staged), proof_files=len(proof["files"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--proof", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(cleanup(a.root, json.loads(a.proof.read_text()))))


if __name__ == "__main__":
    main()
