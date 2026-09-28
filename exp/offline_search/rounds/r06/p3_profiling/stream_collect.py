"""Coordinator health, spill replay, journal verification and bounded cleanup.

Only main(--collect) uses tether. Tests use local archives/Store and never hosts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile

from .stream_protocol import attempt_key, identifier, receive, request
from .stream_receiver import Store, atomic_json, digest


def verify_archive(archive, extracted):
    """Verify extracted bytes, return the remote find/sort/sha256sum tree digest."""
    rows = {}
    with tarfile.open(archive) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            parts = Path(member.name).parts
            if not parts or parts[0] != "p3_telemetry" or any(p in (".", "..") for p in parts):
                raise ValueError("unsafe archive path")
            local = Path(extracted).joinpath(*parts[1:])
            h = hashlib.sha256()
            with tf.extractfile(member) as f:
                for block in iter(lambda: f.read(1024*1024), b""):
                    h.update(block)
            if not local.is_file() or local.is_symlink() or digest(local) != h.hexdigest() or local.stat().st_size != member.size:
                raise ValueError("local extracted SHA mismatch")
            if member.name in rows:
                raise ValueError("duplicate archive file")
            # Only canonical identifiers/filenames are used in this tree hash.
            if any(not re.fullmatch(r"[A-Za-z0-9_.-]+", p) for p in parts):
                raise ValueError("unsafe archive component")
            rows[member.name] = h.hexdigest()
    return hashlib.sha256("".join(rows[n]+"  "+n+"\n" for n in sorted(rows)).encode()).hexdigest()


def certify_tree(run, arm):
    """Add receipts to already collected file-mode data, without changing bytes."""
    store = Store(run)
    root = Path(run)/"runs"/arm/"client_telemetry"
    for path in sorted(root.glob("*/controls.jsonl")):
        with path.open() as f:
            first = json.loads(next(f))
        uid, attempt = first["task_uid"], int(first.get("attempt", 1))
        key = attempt_key(uid, attempt)
        if key != path.parent.name:
            raise ValueError("attempt directory mismatch")
        header = dict(run=Path(run).name, arm=arm, uid=uid, attempt=attempt, key=key)
        files = {}
        for f in sorted(path.parent.iterdir()):
            if f.name == "controls.jsonl" or re.fullmatch(r"step_[0-9]{1,12}\.npz", f.name):
                info = dict(bytes=f.stat().st_size, sha256=digest(f))
                store.apply(dict(**header, op="finish", file=f.name, **info))
                files[f.name] = info
        store.apply(dict(**header, op="attempt", files=files))


def verify_arm(run, arm):
    identifier(Path(run).name); identifier(arm, arm=True)
    root = Path(run)/"runs"/arm
    accepted = {}
    for line in (root/"client/journal.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row.get("accepted") and row.get("status") in ("done", "failed") and not row.get("error"):
            uid, attempt = row["task_uid"], int(row.get("attempt", 1))
            if uid in accepted and accepted[uid] != (attempt, bool(row["success"])):
                raise ValueError("conflicting accepted attempt/outcome")
            accepted[uid] = attempt, bool(row["success"])
    if not accepted:
        raise ValueError("no accepted journal outcomes to verify")
    files, total = 0, 0
    for uid, (attempt, success) in accepted.items():
        key = attempt_key(uid, attempt)
        record = json.loads((root/"p3_stream"/key/"complete.json").read_text())
        if (record["uid"], record["attempt"], record["status"], record["success"]) != (uid, attempt, "complete", success):
            raise ValueError("accepted attempt has no matching complete record")
        expected_sha = hashlib.sha256(json.dumps(record["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if record.get("bytes") != sum(x["bytes"] for x in record["files"].values()) or record.get("sha256") != expected_sha:
            raise ValueError("attempt aggregate SHA/size mismatch")
        for name, info in record["files"].items():
            from .stream_protocol import filename
            filename(name)
            path = root/"client_telemetry"/key/name
            if path.is_symlink() or path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
                raise ValueError("completion file SHA mismatch")
            files += 1; total += info["bytes"]
    result = dict(verified=True, accepted_attempts=len(accepted), files=files, bytes=total,
                  journal_sha256=digest(root/"client/journal.jsonl"))
    atomic_json(root/"p3_stream_verified.json", result)
    return result


def replay_spills(run, arm, directory):
    store = Store(run)
    count = 0
    for path in sorted(Path(directory).glob("*/stream.p3spill")):
        info = json.loads(path.with_name("stream.p3spill.json").read_text())
        if info["run"] != Path(run).name or info["arm"] != arm or path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError("spill checksum/identity mismatch")
        with path.open("rb") as f:
            while True:
                event = receive(f.read)
                if event is None:
                    break
                if event[0].get("run") != info["run"] or event[0].get("arm") != arm or event[0].get("uid") != info["uid"]:
                    raise ValueError("spill frame identity mismatch")
                store.apply(*event)
        count += 1
    return count


def remote_paths(run, arm):
    name = identifier(Path(run).name); identifier(arm, arm=True)
    parent = "/scratch/zixuans8/openpi_trace/os_cl/p3_client/{}/{}".format(name, arm)
    return parent, "/tmp/p3_telemetry_{}_{}.tar".format(name, arm)


def cleanup_command(run, arm, tree_sha, archive_sha):
    for value in (tree_sha, archive_sha):
        if value is not None and not re.fullmatch(r"[0-9a-f]{64}", value):
            raise ValueError("invalid cleanup proof")
    parent, archive = remote_paths(run, arm)
    # No glob, no -f, no parent deletion. Re-hash the *current* remote tree,
    # rejecting new files, mutation, or symlinks since the verified collect.
    return """set -euo pipefail
parent='{parent}'
root='{parent}/p3_telemetry'
archive='{archive}'
if [ -e "$root" ]; then
  test -d "$root" && test ! -L "$root"
  test -z "$(find "$root" -type l -print -quit)"
  actual=$(cd "$parent" && find p3_telemetry -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum | sha256sum | cut -d' ' -f1)
  test "$actual" = '{tree_sha}'
fi
if [ -e "$archive" ]; then
  test ! -L "$archive"
  test "$(sha256sum "$archive" | cut -d' ' -f1)" = '{archive_sha}'
fi
if [ -d "$root" ]; then rm -r -- "$root"; fi
if [ -f "$archive" ]; then rm -- "$archive"; fi
""".format(parent=parent, archive=archive, tree_sha=tree_sha, archive_sha=archive_sha or "NONE")


def cleanup_remote(run, arm, tree_sha, archive_sha):
    command = cleanup_command(run, arm, tree_sha, archive_sha)
    subprocess.run(["tether", "exec", "timan107", "--", "bash", "-c", command], check=True, timeout=1800)
    atomic_json(Path(run)/"runs"/arm/"p3_cleanup.json", dict(
        tree_sha256=tree_sha, archive_sha256=archive_sha, command=command, verified_before_delete=True))


def collect(run, arm, archive=None, no_remote=False):
    from .collect_client import unpack
    root = Path(run)/"runs"/arm
    root.mkdir(parents=True, exist_ok=True)
    archive_sha = None
    if archive is None and not no_remote:
        parent, remote_tar = remote_paths(run, arm)
        command = """set -euo pipefail
if [ -d '{parent}/p3_telemetry' ] && [ -n "$(find '{parent}/p3_telemetry' -type f -name 'stream.p3spill*' -print -quit)" ]; then
  cd '{parent}'
  find p3_telemetry -type f -name 'stream.p3spill*' -print0 | tar -cf '{remote_tar}' --null -T -
  sha256sum '{remote_tar}'
else echo NONE; fi
""".format(parent=parent, remote_tar=remote_tar)
        r = subprocess.run(["tether", "exec", "timan107", "--", "bash", "-c", command], check=True, capture_output=True, text=True, timeout=1800)
        last = r.stdout.strip().splitlines()[-1]
        if last != "NONE":
            archive_sha = last.split()[0]
            archive = root/"client_spills.tar"
            tmp = root/"client_spills.tar.pulling"
            subprocess.run(["tether", "pull", "timan107:"+remote_tar, str(tmp)], check=True, timeout=1800)
            if digest(tmp) != archive_sha:
                raise ValueError("spill archive SHA mismatch")
            tmp.replace(archive)
    count, proof = 0, hashlib.sha256(b"").hexdigest()
    if archive is not None:
        archive = Path(archive)
        archive_sha = digest(archive)
        dest = root/"client_spills"
        unpack(archive, dest)
        proof = verify_archive(archive, dest)
        count = replay_spills(run, arm, dest)
    verified = verify_arm(run, arm)
    if not no_remote:
        cleanup_remote(run, arm, proof, archive_sha)
    result = dict(**verified, spills_replayed=count, archive_sha256=archive_sha,
                  remote_tree_sha256=proof, cleaned_remote=not no_remote)
    atomic_json(root/"p3_stream_collect.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--arm")
    ap.add_argument("--health", metavar="HOST:PORT")
    ap.add_argument("--token-file", type=Path)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--no-remote", action="store_true")
    ap.add_argument("--archive", type=Path)
    a = ap.parse_args()
    if a.health:
        token = (a.token_file or a.run_root/"state/p3_stream.token").read_text().strip()
        result = request(a.health, dict(op="health", run=a.run_root.name, arm=a.arm, token=token))
    elif a.collect and a.arm:
        result = collect(a.run_root, a.arm, a.archive, a.no_remote)
    elif a.verify and a.arm:
        result = verify_arm(a.run_root, a.arm)
    else:
        ap.error("select --health, or --arm with --collect/--verify")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
