"""Coordinator-only telemetry pull. Local --archive mode never uses the network."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unpack(archive, target):
    target = Path(target)
    count = 0
    with tarfile.open(archive) as tf:
        members = tf.getmembers()
        for m in members:
            parts = PurePosixPath(m.name).parts
            if not parts or parts[0] != "p3_telemetry" or ".." in parts or m.issym() or m.islnk():
                raise ValueError("unsafe telemetry archive member: " + m.name)
            if not (m.isdir() or m.isfile()):
                raise ValueError("non-regular telemetry member")
        for m in members:
            if not m.isfile():
                continue
            dst = target.joinpath(*PurePosixPath(m.name).parts[1:])
            dst.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=dst.name+".", dir=dst.parent)
            try:
                with tf.extractfile(m) as src, os.fdopen(fd, "wb") as out:
                    while True:
                        data = src.read(1024*1024)
                        if not data:
                            break
                        out.write(data)
                    out.flush()
                    os.fsync(out.fileno())
                os.replace(tmp, dst)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--archive", type=Path, help="local fixture/already pulled tar; no remote access")
    a = ap.parse_args()
    for value in (a.run_root.name, a.arm):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("unsafe run/arm identifier")
    dest = a.run_root / "runs" / a.arm / "client_telemetry"
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    archive = a.archive
    remote_sha = None
    if archive is None:
        remote = "/scratch/zixuans8/openpi_trace/os_cl/p3_client/{}/{}".format(a.run_root.name, a.arm)
        tar_remote = "/tmp/p3_telemetry_{}_{}.tar".format(a.run_root.name, a.arm)
        command = "set -e; cd '{}' ; test -d p3_telemetry; tar -cf '{}' p3_telemetry; sha256sum '{}'".format(remote, tar_remote, tar_remote)
        result = subprocess.run(["tether", "exec", "timan107", "--", "bash", "-c", command],
                                check=True, capture_output=True, text=True, timeout=1800)
        remote_sha = result.stdout.strip().splitlines()[-1].split()[0]
        archive = parent / "client_telemetry.tar"
        # Fresh local destination, then atomically publish the validated archive.
        fd, temporary = tempfile.mkstemp(prefix="p3_pull_", dir=parent)
        os.close(fd)
        os.unlink(temporary)
        subprocess.run(["tether", "pull", "timan107:"+tar_remote, temporary], check=True, timeout=1800)
        if sha256_file(temporary) != remote_sha:
            raise ValueError("client telemetry SHA mismatch")
        os.replace(temporary, archive)
    digest = sha256_file(archive)
    count = unpack(archive, dest)
    if not count or not list(dest.rglob("controls.jsonl")):
        raise ValueError("telemetry archive has no control traces")
    report = dict(files=count, local_sha256=digest, remote_sha256=remote_sha,
                  directory=str(dest), network_used=a.archive is None)
    (parent / "client_telemetry_collect.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
