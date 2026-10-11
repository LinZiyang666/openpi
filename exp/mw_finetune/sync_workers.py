"""Build and push a small worker-only snapshot; never sync weights or model islands."""
from __future__ import annotations

import argparse
import hashlib
import io
from pathlib import Path
import subprocess
import tarfile
from exp.mw_finetune.runtime import tether_transfer

WORKER_FILES = ("__init__.py", "evaluate.py", "aggregate.py", "adapters.py", "client.py",
                "journal.py", "runtime.py", "rendering.py", "camera_parity.py",
                "worker_launch.py", "setup_worker_node.sh", "CONTRACT.md")


def build_bundle(repo, archive):
    repo, archive = Path(repo), Path(archive)
    files = []
    for base in (repo / "exp/metaworld", repo / "packages/openpi-client"):
        files.extend(p for p in base.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                     and not any(x.startswith(".venv") for x in p.parts) and p.suffix != ".pyc")
    files.extend(repo / "exp/mw_finetune" / n for n in WORKER_FILES)
    # EpisodeRunner requires three small conductor modules. The bundle has an
    # empty namespace initializer, avoiding the full driver's eager imports.
    files.extend(repo / "src/openpi/conductor" / n for n in ("task.py", "protocol.py", "worker.py"))
    files.append(repo / "src/openpi/__init__.py")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "w:gz") as tar:
        for p in sorted(files):
            tar.add(p, arcname=str(p.relative_to(repo)), recursive=False)
        data = b'"""Minimal conductor namespace for the R17 worker snapshot."""\n'
        info = tarfile.TarInfo("src/openpi/conductor/__init__.py")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("nodes", nargs="+", choices=["timan107", "timan108"])
    ap.add_argument("--root", default="/scratch/zixuans8/mw_r17")
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--archive", type=Path, default=Path("/data/mw_finetune/reports/A_worker_code.tar.gz"))
    a = ap.parse_args()
    if not a.root.endswith("/mw_r17"):
        ap.error("root must be a new .../mw_r17 tree")
    repo = Path(__file__).resolve().parents[2]
    digest = build_bundle(repo, a.archive)
    for node in a.nodes:
        subprocess.run(["tether", "exec", node, "--", "mkdir", "-p", a.root + "/code"], check=True)
        # Tether file-transfer allow_roots are /home,/tmp,/srv. Stage the small
        # archive in our new /tmp tree, then unpack into the authorized scratch
        # destination through tether exec. Do not change agent configuration.
        stage = "/tmp/mw_r17_A"
        subprocess.run(["tether", "exec", node, "--", "mkdir", "-p", stage], check=True)
        dest = stage + "/worker_code.tar.gz"
        tether_transfer(["tether", "push", str(a.archive), f"{node}:{dest}", "--force"])
        # Arguments, including paths, never interpolate into shell code.
        code = 'set -eu; test "$(sha256sum "$1" | cut -d" " -f1)" = "$2"; tar -xzf "$1" -C "$3"'
        subprocess.run(["tether", "exec", node, "--", "bash", "-c", code, "r17-sync",
                        dest, digest, a.root + "/code"], check=True)
        if a.setup:
            subprocess.run(["tether", "exec", "--timeout", "10m", node, "--", "bash",
                            a.root + "/code/exp/mw_finetune/setup_worker_node.sh", a.root], check=True)
    print(f"worker code SHA256 {digest}")


if __name__ == "__main__":
    main()
