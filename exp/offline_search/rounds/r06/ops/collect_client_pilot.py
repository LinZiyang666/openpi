"""Coordinator copy of P3 collect_client.py for the pilot (file mode).

Differences from the P3 original: the remote archive is gzip-compressed, and when it is larger than the tether broker
payload ceiling (447,074,607 bytes observed) it is split into parts that are pulled one by one, concatenated locally
and verified against the remote SHA of the whole archive. Remote parts are removed after a verified pull. The remote
telemetry directory and the whole archive are removed later by the pilot lane after its own local==remote check.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile
import tempfile

PART = 400_000_000


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


def rexec(command, timeout=1800):
    return subprocess.run(["tether", "exec", "timan107", "--", "bash", "-c", command],
                          check=True, capture_output=True, text=True, timeout=timeout).stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    a = ap.parse_args()
    for value in (a.run_root.name, a.arm):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("unsafe run/arm identifier")
    dest = a.run_root / "runs" / a.arm / "client_telemetry"
    parent = dest.parent
    parent.mkdir(parents=True, exist_ok=True)
    remote = "/scratch/zixuans8/openpi_trace/os_cl/p3_client/{}/{}".format(a.run_root.name, a.arm)
    tar_remote = "/tmp/p3_telemetry_{}_{}.tar".format(a.run_root.name, a.arm)
    out = rexec("set -e; cd '{r}'; test -d p3_telemetry; tar -cf - p3_telemetry | gzip -1 > '{t}'; "
                "sha256sum '{t}'; stat -c %s '{t}'".format(r=remote, t=tar_remote)).strip().splitlines()
    remote_sha, size = out[-2].split()[0], int(out[-1])
    fd, temporary = tempfile.mkstemp(prefix="p3_pull_", dir=parent)
    os.close(fd)
    os.unlink(temporary)
    if size <= PART:
        subprocess.run(["tether", "pull", "timan107:"+tar_remote, temporary], check=True, timeout=1800)
        parts = []
    else:
        listing = rexec("set -e; rm -f '{t}'.part* 2>/dev/null || true; split -b {p} -d -a 3 '{t}' '{t}.part'; "
                        "ls '{t}'.part*".format(t=tar_remote, p=PART)).split()
        parts = [p for p in listing if p.startswith(tar_remote + ".part")]
        with open(temporary, "wb") as whole:
            for p in parts:
                fdp, piece = tempfile.mkstemp(prefix="p3_part_", dir=parent)
                os.close(fdp)
                os.unlink(piece)
                subprocess.run(["tether", "pull", "timan107:"+p, piece], check=True, timeout=1800)
                with open(piece, "rb") as src:
                    for block in iter(lambda: src.read(1024*1024), b""):
                        whole.write(block)
                os.unlink(piece)
            whole.flush()
            os.fsync(whole.fileno())
    if sha256_file(temporary) != remote_sha:
        raise ValueError("client telemetry SHA mismatch")
    if parts:
        rexec("set -e; " + "; ".join("rm '{}'".format(p) for p in parts))
    archive = parent / "client_telemetry.tar.gz"
    os.replace(temporary, archive)
    digest = sha256_file(archive)
    count = unpack(archive, dest)
    if not count or not list(dest.rglob("controls.jsonl")):
        raise ValueError("telemetry archive has no control traces")
    report = dict(files=count, local_sha256=digest, remote_sha256=remote_sha, archive_bytes=size,
                  parts=len(parts), compression="gzip-1", directory=str(dest), network_used=True,
                  collector="r06_p3_pilot/ops/collect_client_pilot.py")
    (parent / "client_telemetry_collect.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
