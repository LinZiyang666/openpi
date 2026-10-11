"""Pull a quiescent timan run and verify every file with SHA256; resume stays intact."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
from exp.mw_finetune.runtime import tether_transfer

SNAPSHOT = r'''
import fcntl,hashlib,io,json,pathlib,sys,tarfile
run,archive=map(pathlib.Path,sys.argv[1:])
if not (run/'manifest.json').is_file(): raise SystemExit('run manifest missing')
lock=(run/'.eval.lock').open('a')
try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError: raise SystemExit('driver is running; sync after it exits to get a consistent resumable snapshot')
files=sorted(p for p in run.rglob('*') if p.is_file() and p.name!='.eval.lock')
hashes={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
data=(json.dumps(hashes,sort_keys=True,indent=2)+'\n').encode()
archive.parent.mkdir(parents=True,exist_ok=True)
with tarfile.open(archive,'w:gz') as tar:
 for p in files: tar.add(p,arcname=str(p.relative_to(run)),recursive=False)
 info=tarfile.TarInfo('TRANSFER_SHA256.json');info.size=len(data);tar.addfile(info,io.BytesIO(data))
print(hashlib.sha256(archive.read_bytes()).hexdigest())
'''


def install_snapshot(archive, dest):
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / ".eval.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with tarfile.open(archive) as tar:
            hashes = json.load(tar.extractfile("TRANSFER_SHA256.json"))
            for member in tar.getmembers():
                name = Path(member.name)
                if not member.isfile() or name.is_absolute() or ".." in name.parts or member.name == ".eval.lock":
                    raise ValueError(f"unsafe snapshot entry {member.name}")
                data = tar.extractfile(member).read()
                if member.name != "TRANSFER_SHA256.json" and hashlib.sha256(data).hexdigest() != hashes[member.name]:
                    raise ValueError(f"SHA256 mismatch: {member.name}")
                path = dest / name
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.name + f".sync.{os.getpid()}")
                with temp.open("wb") as out:
                    out.write(data); out.flush(); os.fsync(out.fileno())
                os.replace(temp, path)
        for name, digest in hashes.items():
            if hashlib.sha256((dest / name).read_bytes()).hexdigest() != digest:
                raise ValueError(f"installed SHA256 mismatch: {name}")
    return hashes


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("node", choices=["timan107", "timan108"])
    ap.add_argument("run_name")
    ap.add_argument("--root", default="/scratch/zixuans8/mw_r17")
    ap.add_argument("--destination", type=Path, default=Path("/data/mw_finetune/eval"))
    a = ap.parse_args()
    if Path(a.run_name).name != a.run_name or a.run_name in (".", ".."):
        ap.error("run_name must be one directory name")
    remote = f"/tmp/mw_r17_A/{a.run_name}.tar.gz"
    result = subprocess.run(["tether", "exec", a.node, "--", a.root + "/envs/sim/bin/python", "-c", SNAPSHOT,
                             a.root + "/runs/" + a.run_name, remote], check=True, text=True, capture_output=True)
    digest = result.stdout.strip().splitlines()[-1]
    archive = Path("/data/mw_finetune/reports/A_transfers") / f"{a.node}_{a.run_name}.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    tether_transfer(["tether", "pull", f"{a.node}:{remote}", str(archive), "--force"])
    if hashlib.sha256(archive.read_bytes()).hexdigest() != digest:
        raise ValueError("archive transfer SHA256 mismatch")
    dest = a.destination / a.run_name
    hashes = install_snapshot(archive, dest)
    record = {"node": a.node, "source": a.root + "/runs/" + a.run_name, "archive_sha256": digest,
              "verified_files": len(hashes), "time": time.time(), "destination": str(dest)}
    (archive.with_suffix(".verification.json")).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
