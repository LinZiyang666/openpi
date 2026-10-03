"""Fix round 1: partial spills, post-verification cleanup and process files."""
import io
import json
from pathlib import Path
import tarfile
from types import SimpleNamespace as NS

import pytest

from exp.offline_search.debug.schema import episode_key
from exp.offline_search.debug.transport import archive, collect as collector
from exp.offline_search.debug.tests.test_client_capture import make_episode
from exp.offline_search.debug.tests.test_transport_debug import run_root


def completed_arm(tmp_path):
    root = run_root(tmp_path)
    client, _, _, _ = make_episode(root / "runs/A/debug/client", n=7)
    collector.certify_tree(root, "A")
    return root, client


def test_archive_selects_exact_names_and_partial_attempts_never_block_collection(tmp_path):
    root, client = completed_arm(tmp_path)
    source = tmp_path / "remote"
    stale_key = episode_key("A:eval:0:0", 2)  # journal accepted attempt 1
    directory = source / stale_key
    directory.mkdir(parents=True)
    for name in archive.PARTIAL_FILES:
        (directory / name).write_bytes(b"killed halfway\xff")
    finalized = source / episode_key("A:eval:0:1", 2)
    finalized.mkdir()
    (finalized / "stream.osdebugspill").write_bytes(b"not published metadata")
    # This stale finalized prefix is skipped without a manifest; accepted
    # attempt receipt checks remain independent of all unaccepted fragments.
    path = tmp_path / "remote.tar"
    inventory = archive.build_archive(source, path)
    with tarfile.open(str(path)) as tf:
        assert tf.getnames() == [finalized.name + "/stream.osdebugspill"]
    assert len(inventory["stale_partial_spills"]) == 3
    collector.stale_spill_report(root, "A", inventory["stale_partial_spills"])
    # Pass the inventory into an older-archive collection too; those partial
    # files are reported without being extracted/replayed or losing receipts.
    with tarfile.open(str(path), "a") as tf:
        for name in archive.PARTIAL_FILES:
            entry = tarfile.TarInfo(stale_key + "/" + name)
            entry.size = 1
            tf.addfile(entry, io.BytesIO(b"x"))
    result = collector.collect(root, "A", path, no_remote=True, expected=1, require_server=False)
    assert result["verified"] and result["spills_replayed"] == 0
    assert len(result["stale_partial_spills"]) == 4
    assert all(row["attempt_status"] == "unaccepted" for row in result["stale_partial_spills"])
    assert not any((root / "runs/A/debug/client_spills").glob("*/stream.osdebugspill.part"))
    assert collector.verify_arm(root, "A", 1, False)["verified"]


def test_remote_collect_transfers_exact_archive_and_lists_partials_without_tether(tmp_path, monkeypatch):
    root, client = completed_arm(tmp_path)
    directory = tmp_path / "remote" / episode_key("A:eval:0:0", 2)
    directory.mkdir(parents=True)
    (directory / "stream.osdebugspill.part").write_bytes(b"partial")
    remote_tar = tmp_path / "staged.tar"
    metadata = archive.build_archive(directory.parent, remote_tar)
    calls = []
    def tether(argv, **kwargs):
        calls.append(argv)
        if argv[1] == "exec":
            return NS(stdout=json.dumps(metadata) + "\n")
        Path(argv[-1]).write_bytes(remote_tar.read_bytes())
        return NS(returncode=0)
    monkeypatch.setattr(collector.subprocess, "run", tether)
    local = collector.remote_collect(root, "A", "stream")
    assert local.read_bytes() == remote_tar.read_bytes()
    assert "debug.transport.archive" in calls[0][-1] and "--mode stream" in calls[0][-1]
    rows = json.loads((root / "runs/A/debug/stale_partial_spills.json").read_text())
    assert len(rows) == 1 and rows[0]["attempt_status"] == "unaccepted"


def test_archive_removal_checks_current_digest(tmp_path):
    path = tmp_path / "remote.tar"
    path.write_bytes(b"old")
    sha = archive.sha256(path)
    path.write_bytes(b"new")
    with pytest.raises(ValueError, match="changed"):
        archive.remove_verified_archive(path, sha)
    assert path.exists()
    assert archive.remove_verified_archive(path, archive.sha256(path))["removed"]
    assert not path.exists()


@pytest.mark.parametrize("verify_succeeds", [False, True])
def test_remote_tar_removal_follows_verification_even_when_retaining_spills(tmp_path, monkeypatch, verify_succeeds):
    root, client = completed_arm(tmp_path)
    archive_path = tmp_path / "empty.tar"
    archive.build_archive(tmp_path / "absent", archive_path)
    calls = []
    monkeypatch.setattr(collector, "remote_collect", lambda *a: archive_path)
    def verify(*args):
        calls.append("verify")
        if not verify_succeeds:
            raise ValueError("missing accepted server records")
        return {"verified": True}
    monkeypatch.setattr(collector, "verify_arm", verify)
    monkeypatch.setattr(collector, "cleanup_remote_archive", lambda *a: calls.append("remove_tar"))
    monkeypatch.setattr(collector, "cleanup_remote_spills", lambda *a: pytest.fail("keep_remote retains spills"))
    if verify_succeeds:
        result = collector.collect(root, "A", expected=1, keep_remote=True)
        assert result["remote_archive_removed"] and result["remote_files_retained"]
        assert calls == ["verify", "remove_tar"]
    else:
        with pytest.raises(ValueError, match="missing accepted"):
            collector.collect(root, "A", expected=1, keep_remote=True)
        assert calls == ["verify"]


@pytest.mark.parametrize("tail", [b'{"unfinished":', b'{"valid_but_unpublished":true}', b'\xf0\x9f'])
def test_verify_multiple_process_files_skips_and_reports_unterminated_final_line(tmp_path, tail):
    root, client = completed_arm(tmp_path)
    directory = root / "runs/A/debug/client" / client.identity["episode_key"]
    events = [json.loads(raw) for raw in (directory / "events.jsonl").read_text().splitlines()]
    rows = [dict(client.identity, decision_id=row["decision_id"]) for row in events if row["ev"] == "decision"]
    server = root / "runs/A/debug/server_24090"
    server.mkdir()
    for filename, record in zip(("decisions.jsonl", "decisions_42.jsonl"), rows):
        (server / filename).write_text(json.dumps(record) + "\n")
    with (server / "decisions_42.jsonl").open("ab") as stream:
        stream.write(tail)
    result = collector.verify_arm(root, "A", 1)
    assert result["decisions"] == 2 and result["verified"]
    assert len(result["skipped_server_lines"]) == 1 and result["skipped_server_lines"][0]["line"] == 2
    assert result["skipped_server_lines"][0]["path"].endswith("decisions_42.jsonl")
    assert json.loads((root / "runs/A/debug/server_jsonl_skips.json").read_text()) == result["skipped_server_lines"]
    # A read-only audit must not overwrite either persisted report.
    report = root / "runs/A/debug/verified.json"
    before = report.read_bytes()
    assert collector.verify_arm(root, "A", 1, write_report=False)["verified"]
    assert report.read_bytes() == before


def test_completed_malformed_server_line_is_still_rejected(tmp_path):
    root, _ = completed_arm(tmp_path)
    server = root / "runs/A/debug/server_24090"
    server.mkdir()
    (server / "decisions_42.jsonl").write_bytes(b"{bad JSON}\n")
    with pytest.raises(ValueError):
        collector.verify_arm(root, "A", 1)


def test_journal_and_client_final_fragments_are_skipped_and_reported(tmp_path):
    root = run_root(tmp_path)
    client, _, _, _ = make_episode(root / "runs/A/debug/client", n=7)
    directory = root / "runs/A/debug/client" / client.identity["episode_key"]
    for path in (root / "runs/A/client/journal.jsonl", directory / "events.jsonl"):
        with path.open("ab") as stream:
            stream.write(b'{"killed_after_complete":\xf0')
    collector.certify_tree(root, "A")
    result = collector.verify_arm(root, "A", 1, False)
    assert result["verified"] and len(result["skipped_journal_lines"]) == len(result["skipped_client_lines"]) == 1
    # Losing the actual terminal event remains incomplete, rather than silently
    # admitting an episode whose done lifecycle was never published.
    events = (directory / "events.jsonl").read_bytes().splitlines(keepends=True)
    (directory / "events.jsonl").write_bytes(b"".join(events[:-2]) + events[-2].rstrip(b"\n"))
    from exp.offline_search.debug.transport.validation import validate_episode
    with pytest.raises(ValueError, match="missing reset/done"):
        validate_episode(directory, "A:eval:0:0", 1)
