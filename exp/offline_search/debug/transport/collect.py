"""Offset-ACK spill replay and receipt/journal reconciliation, no silent cleanup."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
import tarfile

from exp.offline_search.debug.schema import episode_key, read_jsonl
from .protocol import MAX_BODY, filename, identifier, receive, request
from .receiver import Store, atomic_json, digest
from .validation import validate_episode
from .archive import PARTIAL_FILES


def header(run, arm, episode):
    return dict(run=Path(run).name, arm=arm, uid=episode["task_uid"], attempt=episode["attempt"], key=episode["episode_key"])


def certify_tree(run, arm):
    """File-mode receipts use exactly the same Store as streaming."""
    store = Store(run)
    for path in sorted((Path(run) / "runs" / arm / "debug/client").glob("*/episode.json")):
        ep = json.loads(path.read_text())
        if path.parent.name != episode_key(ep["task_uid"], ep["attempt"]):
            raise ValueError("episode directory mismatch")
        h, files = header(run, arm, ep), {}
        for payload in sorted(path.parent.iterdir()):
            try:
                filename(payload.name)
            except ValueError:
                continue
            info = dict(bytes=payload.stat().st_size, sha256=digest(payload))
            store.apply(dict(h, op="finish", file=payload.name, **info))
            files[payload.name] = info
        store.apply(dict(h, op="attempt", files=files))


def replay_spills(run, arm, directory, skipped=None):
    store = Store(run)
    count = 0
    for path in sorted(Path(directory).glob("*/stream.osdebugspill")):
        sidecar = path.with_name("stream.osdebugspill.json")
        if not sidecar.exists():
            # A kill can happen between final spill and metadata publication.
            # Admission checks the accepted attempt's receipts independently.
            if skipped is not None:
                skipped.append(dict(path=str(path.relative_to(directory)), episode_key=path.parent.name,
                                    bytes=path.stat().st_size, reason="unfinished spill: metadata was not published"))
            continue
        info = json.loads(sidecar.read_text())
        if info["run"] != Path(run).name or info["arm"] != arm or path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
            raise ValueError("spill checksum/identity mismatch")
        with path.open("rb") as handle:
            while True:
                event = receive(handle.read)
                if event is None:
                    break
                h, body = event
                if any(h.get(k) != info[k] for k in ("run", "arm", "uid", "attempt", "key")):
                    raise ValueError("spill frame identity mismatch")
                store.apply(h, body)
        count += 1
    return count


def journal_outcomes(path, issues=None):
    accepted = {}
    for row in read_jsonl(path, issues):
        if row.get("accepted") and row.get("status") in ("done", "failed") and not row.get("error"):
            uid, attempt, success = row["task_uid"], int(row.get("attempt", 1)), bool(row["success"])
            if uid in accepted and accepted[uid] != (attempt, success):
                raise ValueError("conflicting accepted attempts/outcomes")
            accepted[uid] = (attempt, success)
    return accepted


def verify_arm(run, arm, expected=None, require_server=True, write_report=True):
    identifier(Path(run).name)
    identifier(arm, arm=True)
    root = Path(run) / "runs" / arm
    journal = root / "client/journal.jsonl"
    skipped_journal, skipped_client = [], []
    accepted = journal_outcomes(journal, skipped_journal)
    if not accepted or (expected is not None and len(accepted) != expected):
        raise ValueError("accepted journal count mismatch: {} expected {}".format(len(accepted), expected))
    server = {}
    skipped = []
    accepted_keys = {episode_key(uid, attempt) for uid, (attempt, _) in accepted.items()}
    if require_server:
        for path in sorted((root / "debug").glob("server_*/decisions*.jsonl")):
            for r in read_jsonl(path, skipped):
                if r.get("episode_key") not in accepted_keys:
                    continue
                did = r["decision_id"]
                if did in server:
                    raise ValueError("duplicate server decision: " + did)
                server[did] = r
        if write_report:
            atomic_json(root / "debug/server_jsonl_skips.json", skipped)
    files = total = 0
    client_ids = set()
    for uid, (attempt, success) in accepted.items():
        key = episode_key(uid, attempt)
        receipts, directory = root / "debug/receipts" / key, root / "debug/client" / key
        record = json.loads((receipts / "complete.json").read_text())
        trace = validate_episode(directory, uid, attempt)
        skipped_client.extend(trace["skipped_event_lines"])
        if (record["uid"], record["attempt"], record["status"], record["success"]) != (uid, attempt, "complete", success):
            raise ValueError("accepted attempt has no matching complete receipt")
        if trace["status"] != "complete" or trace["success"] != success or record.get("dispatch_gen") != trace["dispatch_gen"]:
            raise ValueError("episode status/generation/outcome mismatch")
        expected_sha = hashlib.sha256(json.dumps(record["files"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if record.get("bytes") != sum(x["bytes"] for x in record["files"].values()) or record.get("sha256") != expected_sha:
            raise ValueError("aggregate receipt hash mismatch")
        for name, info in record["files"].items():
            filename(name)
            path = directory / name
            per_file = json.loads((receipts / (name + ".json")).read_text())
            if path.is_symlink() or path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
                raise ValueError("payload checksum mismatch")
            if any(per_file[k] != info[k] for k in ("bytes", "sha256")):
                raise ValueError("per-file receipt mismatch")
            files += 1
            total += info["bytes"]
        for row in read_jsonl(directory / "events.jsonl"):
            if row.get("ev") != "decision":
                continue
            did = row["decision_id"]
            client_ids.add(did)
            if require_server:
                other = server.get(did)
                if not other or row.get("server_echo") != did or row.get("server_join") != "verified":
                    raise ValueError("unverified/missing server decision: " + did)
                if (other["task_uid"], other["attempt"], other["dispatch_gen"]) != (uid, attempt, trace["dispatch_gen"]):
                    raise ValueError("server identity mismatch")
    if require_server and client_ids != set(server):
        raise ValueError("orphan server decisions: " + str(sorted(set(server) - client_ids)))
    result = dict(verified=True, accepted_attempts=len(accepted), files=files, bytes=total, decisions=len(client_ids),
                  server_join_verified=require_server, journal_sha256=digest(journal), skipped_server_lines=skipped)
    result.update(skipped_journal_lines=skipped_journal, skipped_client_lines=skipped_client)
    if write_report:
        atomic_json(root / "debug/verified.json", result)
    return result


def unpack(archive, destination):
    """No extractall: validate names and reject links/duplicate paths before IO."""
    destination = Path(destination)
    seen = set()
    partials = []
    with tarfile.open(str(archive)) as tf:
        members = tf.getmembers()
        for m in members:
            parts = Path(m.name).parts
            if not m.isfile() or len(parts) != 2 or any(x in (".", "..") for x in parts) or Path(m.name).is_absolute():
                raise ValueError("unsafe client archive")
            if m.name in seen:
                raise ValueError("duplicate archive path")
            seen.add(m.name)
            identifier(parts[0])
            if parts[1] in PARTIAL_FILES:
                partials.append(dict(path=m.name, episode_key=parts[0], bytes=m.size,
                                     reason="unfinished spill in older archive"))
                continue
            if parts[1] not in ("stream.osdebugspill", "stream.osdebugspill.json"):
                filename(parts[1])
        for m in members:
            if Path(m.name).name in PARTIAL_FILES:
                continue
            path = destination / m.name
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise ValueError("symlink extraction path")
            data = tf.extractfile(m).read()
            if path.exists():
                if path.read_bytes() != data:
                    raise ValueError("archive conflicts with existing data")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as handle:
                handle.write(data)
    return partials


def stale_spill_report(run, arm, partials):
    accepted = journal_outcomes(Path(run) / "runs" / arm / "client/journal.jsonl")
    keys = {episode_key(uid, attempt) for uid, (attempt, _) in accepted.items()}
    unique = {row["path"]: dict(row, accepted_attempt=row["episode_key"] in keys,
                                attempt_status="accepted" if row["episode_key"] in keys else "unaccepted")
              for row in partials}
    rows = [unique[path] for path in sorted(unique)]
    atomic_json(Path(run) / "runs" / arm / "debug/stale_partial_spills.json", rows)
    return rows


def remote_archive_path(run, arm):
    return "/tmp/osdebug_{}_{}.tar".format(identifier(Path(run).name), identifier(arm, arm=True))


def remote_archive_command(args):
    return "cd /scratch/zixuans8/openpi_trace && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /scratch/zixuans8/openpi/.venv/bin/python -m exp.offline_search.debug.transport.archive " + " ".join(shlex.quote(str(arg)) for arg in args)


def remote_collect(run, arm, mode):
    """Coordinator-only remote reads; retain remote spill until verification."""
    tag, arm = identifier(Path(run).name), identifier(arm, arm=True)
    remote = "/scratch/zixuans8/openpi_trace/os_cl/debug_client/{}/{}".format(tag, arm)
    tar = remote_archive_path(run, arm)
    command = remote_archive_command(["--root", remote, "--out", tar, "--mode", mode])
    r = subprocess.run(["tether", "exec", "timan107", "--", "bash", "-c", command], check=True, capture_output=True, text=True, timeout=600)
    inventory = json.loads(r.stdout.strip().splitlines()[-1])
    sha = inventory["sha256"]
    local = Path(run) / "runs" / arm / "debug/client_collect.tar"
    local.parent.mkdir(parents=True, exist_ok=True)
    # tether pull can fail transiently (e.g. too many transfers in flight); retry before failing the arm.
    for attempt in range(4):
        r = subprocess.run(["tether", "pull", "--force", "timan107:" + tar, str(local) + ".part"], timeout=600)
        if r.returncode == 0:
            break
        if attempt == 3:
            raise subprocess.CalledProcessError(r.returncode, r.args)
        time.sleep(20 * (attempt + 1))
    part = local.with_name(local.name + ".part")
    if digest(part) != sha:
        raise ValueError("remote archive checksum mismatch")
    part.replace(local)
    stale_spill_report(run, arm, inventory["stale_partial_spills"])
    return local


def cleanup_remote_archive(run, arm, archive):
    command = remote_archive_command(["--out", remote_archive_path(run, arm), "--remove", "--sha256", digest(archive)])
    subprocess.run(["tether", "exec", "timan107", "--", "bash", "-c", command], check=True, timeout=300)


def cleanup_remote_spills(run, arm, archive):
    """The home archive is retained; remote removals require current byte hashes."""
    records = []
    with tarfile.open(str(archive)) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                raise ValueError("unsafe cleanup archive member")
            if Path(member.name).name in PARTIAL_FILES:
                continue
            data = tf.extractfile(member).read()
            records.append(dict(path=member.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    proof_path = Path(run) / "runs" / arm / "debug/spill_cleanup_proof.json"
    atomic_json(proof_path, dict(files=records, verified_archive_sha256=digest(archive)))
    tag, arm = identifier(Path(run).name), identifier(arm, arm=True)
    stage = "/tmp/osdebug_cleanup_{}_{}.json".format(tag, arm)
    subprocess.run(["tether", "push", "--force", str(proof_path), "timan107:" + stage], check=True, timeout=300)
    remote = "/scratch/zixuans8/openpi_trace/os_cl/debug_client/{}/{}".format(tag, arm)
    command = "cd /scratch/zixuans8/openpi_trace && PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src /scratch/zixuans8/openpi/.venv/bin/python -m exp.offline_search.debug.transport.cleanup --root {} --proof {}".format(shlex.quote(remote), shlex.quote(stage))
    subprocess.run(["tether", "exec", "timan107", "--", "bash", "-c", command], check=True, timeout=300)


def collect(run, arm, archive=None, mode="stream", no_remote=False, expected=None, require_server=True, keep_remote=False):
    root = Path(run) / "runs" / arm / "debug"
    fetched_remote = archive is None and not no_remote
    archive = archive or (None if no_remote else remote_collect(run, arm, mode))
    count = 0
    stale_path = root / "stale_partial_spills.json"
    partials = json.loads(stale_path.read_text()) if fetched_remote and stale_path.exists() else []
    if archive:
        destination = root / ("client_spills" if mode == "stream" else "client")
        partials.extend(unpack(archive, destination))
        if mode == "stream":
            count = replay_spills(run, arm, destination, partials)
    if mode == "file":
        certify_tree(run, arm)
    partials = stale_spill_report(run, arm, partials)
    result = verify_arm(run, arm, expected, require_server)
    cleaned = False
    if not no_remote and not keep_remote and mode == "stream" and archive:
        cleanup_remote_spills(run, arm, archive)
        cleaned = True
    if fetched_remote:
        cleanup_remote_archive(run, arm, archive)
    result.update(spills_replayed=count, remote_files_retained=not cleaned, remote_archive_removed=fetched_remote,
                  stale_partial_spills=partials)
    atomic_json(root / "collect.json", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--arm")
    ap.add_argument("--health", metavar="HOST:PORT")
    ap.add_argument("--token-file", type=Path)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--mode", choices=("stream", "file"), default="stream")
    ap.add_argument("--no-remote", action="store_true")
    ap.add_argument("--archive", type=Path)
    ap.add_argument("--expected", type=int)
    ap.add_argument("--client-only", action="store_true", help="transport development only; cannot admit R8")
    ap.add_argument("--keep-remote", action="store_true", help="retain verified spills on scratch; default removes only individually certified spill bytes")
    a = ap.parse_args()
    if a.health:
        token = (a.token_file or a.run_root / "state/osdebug_stream.token").read_text().strip()
        result = request(a.health, dict(op="health", run=a.run_root.name, arm=a.arm, token=token))
    elif a.arm and a.collect:
        result = collect(a.run_root, a.arm, a.archive, a.mode, a.no_remote, a.expected, not a.client_only, a.keep_remote)
    elif a.arm and a.verify:
        result = verify_arm(a.run_root, a.arm, a.expected, not a.client_only)
    else:
        ap.error("select --health or --arm with --collect/--verify")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
