"""Transfer prepared shards in bounded tar files using tether, without deletions."""
import argparse
from pathlib import Path
import subprocess
import tarfile


def sync(source, remote, node, staging):
    source, staging = Path(source), Path(staging)
    staging.mkdir(parents=True, exist_ok=True)
    files = sorted(source.glob("part-*.sqlite")) + [source / "index.npz", source / "manifest.json"]
    batches, batch, size = [], [], 0
    for file in files:
        n = file.stat().st_size
        if size + n > 900_000_000 and batch:
            batches.append(batch)
            batch, size = [], 0
        batch.append(file)
        size += n
    if batch:
        batches.append(batch)
    subprocess.run(["tether", "exec", "--safe", "--timeout", "60s", node, "/bin/mkdir", "-p", remote], check=True)
    for i, batch in enumerate(batches):
        archive = staging / "prepared-transfer.tar"
        with tarfile.open(archive, "w") as tar:
            for file in batch:
                tar.add(file, arcname=file.name)
        target = "/data/mw_finetune/fastwam/prepared-transfer.tar"
        subprocess.run(["tether", "push", str(archive), f"{node}:{target}", "--force"], check=True)
        subprocess.run(["tether", "exec", "--safe", "--timeout", "60s", node, "/bin/tar", "-xf", target, "-C", remote], check=True)
        print(f"Completed batch {i + 1}/{len(batches)} ({sum(f.stat().st_size for f in batch)/1e9:.3f} GB)", flush=True)
        # One retained transfer archive is reused; never accumulate duplicate data.


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="/data/mw_finetune/fastwam/mt50_224")
    p.add_argument("--remote", default="/data/mw_finetune/fastwam/mt50_224")
    p.add_argument("--node", default="h100")
    p.add_argument("--staging", default="/data/mw_finetune/fastwam/transfer")
    a = p.parse_args()
    sync(a.source, a.remote, a.node, a.staging)
