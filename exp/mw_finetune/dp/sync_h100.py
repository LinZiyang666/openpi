"""Copy only E's assets + read-only harness snapshots into E's H100 root."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import shlex


ROOT = Path("/data/mw_finetune/dp")
REPO = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--code-only", action="store_true")
    a = ap.parse_args()
    paths = [(REPO / "exp/mw_finetune/dp", "code/exp/mw_finetune/dp"),
        (REPO / "exp/mw_finetune/__init__.py", "code/exp/mw_finetune/__init__.py"),
        (REPO / "exp/metaworld/tasks.py", "code/exp/metaworld/tasks.py"),
        (REPO / "src/openpi/serving/websocket_policy_server.py", "code/src/openpi/serving/websocket_policy_server.py"),
        (REPO / "src/openpi/serving/monitor.py", "code/src/openpi/serving/monitor.py"),
        (REPO / "packages/openpi-client/src/openpi_client", "code/packages/openpi-client/src/openpi_client")]
    paths += [(REPO / "exp/mw_finetune" / (name + ".py"), "code/exp/mw_finetune/" + name + ".py")
              for name in ("serve", "adapters", "runtime")]
    if not a.code_only:
        paths += [(ROOT / "checkpoints", "checkpoints"), (ROOT / "clip_text", "clip_text")]
    if a.data:
        paths += [(ROOT / "mt50_96", "mt50_96")]
    archive = ROOT / ("sync_data.tar" if a.data else "sync_code.tar" if a.code_only else "sync_assets.tar")
    with tarfile.open(archive, "w") as tar:
        for source, target in paths:
            tar.add(source, arcname=target, filter=lambda info: None if "__pycache__" in info.name or "/reports/" in info.name else info)
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    size = archive.stat().st_size
    if size > 3.5e9:
        raise RuntimeError("Refuse archive>3.5GB; H100 E budget includes staging/extraction/checkpoint")
    subprocess.run(["tether", "exec", "--safe", "--timeout", "30s", "h100", "--", "mkdir", "-p", str(ROOT)], check=True)
    chunks = []
    if size > 2_000_000_000:
        with archive.open("rb") as stream:
            i = 0
            while True:
                blob = stream.read(1_400_000_000)
                if not blob:
                    break
                piece = archive.with_name(archive.name + f".part{i:02d}")
                piece.write_bytes(blob); chunks.append(piece); i += 1
    else:
        chunks = [archive]
    for piece in chunks:
        piece_hash = hashlib.sha256(piece.read_bytes()).hexdigest()
        subprocess.run(["tether", "push", str(piece), "h100:" + str(piece), "--force"], check=True)
        command = f"import hashlib,pathlib; p=pathlib.Path('{piece}'); assert hashlib.sha256(p.read_bytes()).hexdigest()=='{piece_hash}'"
        subprocess.run(["tether", "exec", "--safe", "--timeout", "90s", "h100", "--", "python3", "-c", command], check=True)
    # Stream verified pieces directly into tar; no duplicate concatenated archive on H100.
    unpack = "cat " + " ".join(shlex.quote(str(p)) for p in chunks) + " | tar -xf - -C " + shlex.quote(str(ROOT))
    subprocess.run(["tether", "exec", "--safe", "--timeout", "5m", "h100", "--", "bash", "-o", "pipefail", "-c", unpack], check=True)
    print(json.dumps(dict(archive=str(archive), sha256=sha, bytes=size)))


if __name__ == "__main__":
    main()
