import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from exp.offline_search.debug.server import ServerObserver
from exp.offline_search.debug.server.digests import DigestCache, cache_path
from exp.offline_search.debug.tests.test_server_support import no_git, server_json


def test_cache_reuses_realpath_and_invalidates_size_and_mtime(tmp_path):
    source = tmp_path / "artifact.bin"
    source.write_bytes(b"abcd")
    alias = tmp_path / "alias.bin"
    alias.symlink_to(source)
    path = tmp_path / "catalog/digest_cache.json"
    first = DigestCache(path)
    original = first.file(source)
    assert first.misses == 1
    second = DigestCache(path)
    assert second.file(alias) == original and second.hits == 1 and second.bytes_hashed == 0
    stamp = source.stat()
    source.write_bytes(b"longer")
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))  # size alone invalidates
    assert second.file(source) == hashlib.sha256(b"longer").hexdigest()
    stamp = source.stat()
    source.write_bytes(b"change")
    os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))  # mtime alone invalidates
    changed = second.file(source)
    assert changed == hashlib.sha256(b"change").hexdigest() and second.misses == 2
    other = tmp_path / "other.bin"
    other.write_bytes(b"change")
    os.utime(other, ns=(stamp.st_atime_ns, stamp.st_mtime_ns + 1))
    assert second.file(other) == changed and second.misses == 3  # a different realpath is a miss
    data = json.loads(path.read_text())
    assert set(data["files"]) == {str(source.resolve()), str(other.resolve())}
    signature = data["files"][str(source.resolve())]["signature"]
    assert signature == dict(realpath=str(source.resolve()), size=6, mtime_ns=stamp.st_mtime_ns + 1)


def test_cache_distinguishes_file_hash_mapped_views_and_copied_payload(tmp_path):
    source = tmp_path / "action.npy"
    np.save(source, np.arange(32, dtype=np.float32).reshape(4, 8))
    payload = np.load(source, mmap_mode="r", allow_pickle=False)
    path = tmp_path / "catalog/digest_cache.json"
    first, second = DigestCache(path), DigestCache(path)
    expected = hashlib.sha256(payload.tobytes()).hexdigest()
    assert first.array(np.ascontiguousarray(payload)) == expected
    assert second.array(np.asarray(payload)) == expected and second.hits == 1
    assert first.file(source) != expected  # the .npy header is outside deployed payload bytes
    subset = payload[1:]
    assert second.array(subset) == hashlib.sha256(subset.tobytes()).hexdigest()
    copied = payload.copy()
    copied[0, 0] += 1
    assert second.array(copied) != expected  # an unmapped modified copy must never reuse the source hash
    assert second.file(source) == first.file(source)
    assert second.bytes_hashed == subset.nbytes + copied.nbytes
    private = np.load(source, mmap_mode="c", allow_pickle=False)
    private[0, 0] = -99  # copy-on-write bytes differ while source size/mtime stay identical
    assert second.array(private) == hashlib.sha256(private.tobytes()).hexdigest() != expected


def test_corrupt_or_stale_cache_recomputes(tmp_path):
    source = tmp_path / "artifact"
    source.write_bytes(b"bytes")
    path = tmp_path / "digest_cache.json"
    for bad in ("{", "[]", '{"schema":"old","files":{}}'):
        path.write_text(bad)
        cache = DigestCache(path)
        assert cache.file(source) == hashlib.sha256(b"bytes").hexdigest()
        assert cache.misses == 1
    data = json.loads(path.read_text())
    data["files"][str(source.resolve())]["digests"]["file_sha256"] = "not-a-digest"
    path.write_text(json.dumps(data))
    assert DigestCache(path).file(source) == hashlib.sha256(b"bytes").hexdigest()


def test_locked_cache_merges_concurrent_updates_and_hashes_once(tmp_path):
    sources = [tmp_path / "artifact0", tmp_path / "artifact1"]
    for i, source in enumerate(sources):
        source.write_bytes(bytes([i]) * (1024 * 1024))
    path = tmp_path / "catalog/digest_cache.json"
    def job(i):
        cache = DigestCache(path)
        digest = cache.file(sources[i % 2])
        return digest, cache.hits, cache.misses
    with ThreadPoolExecutor(max_workers=8) as workers:
        reports = list(workers.map(job, range(8)))
    assert sum(r[2] for r in reports) == 2 and sum(r[1] for r in reports) == 6
    assert len(json.loads(path.read_text())["files"]) == 2
    for i, (digest, _, _) in enumerate(reports):
        assert digest == hashlib.sha256(sources[i % 2].read_bytes()).hexdigest()


def test_startups_share_campaign_cache_without_reading_library_bytes_again(tmp_path):
    import types
    run = tmp_path / "campaign"
    libdir = tmp_path / "store/library/pi05_spatial/current"
    libdir.mkdir(parents=True)
    np.save(libdir / "action.npy", np.zeros((2, 10, 32), np.float32))
    np.save(libdir / "key_v0.npy", np.zeros((2, 128), np.float32))
    (libdir / "manifest.json").write_text(json.dumps(dict(sources=dict(pkl_sha256="source-sha"))))
    def runtime():
        mapped = np.load(libdir / "action.npy", mmap_mode="r", allow_pickle=False)
        return types.SimpleNamespace(model="pi05", suite="spatial", H=10, method=None,
            lib=types.SimpleNamespace(dir=libdir, meta=dict(sources=dict(pkl_sha256="source-sha"))),
            tables=dict(current=np.ascontiguousarray(mapped)),
            opts=types.SimpleNamespace(os_method="fake", kwargs={}, os_fit_artifact=""))
    first_dir = run / "runs/arm1/debug/server_23310"
    second_dir = run / "runs/arm2/debug/server_23311"
    first = ServerObserver(runtime(), first_dir, dict(campaign="campaign"))
    first.close()
    second = ServerObserver(runtime(), second_dir, dict(campaign="campaign"))
    second.close()
    a, b = server_json(first_dir, "meta"), server_json(second_dir, "meta")
    assert a["artifacts"] == b["artifacts"] and a["library_content_fingerprints"] == b["library_content_fingerprints"]
    assert b["library_shas"]["current"] == "source-sha"
    assert b["digest_cache"]["bytes_hashed"] == 0 and b["digest_cache"]["hits"] > 3
    assert Path(b["digest_cache"]["path"]) == run / "catalog/digest_cache.json"
    assert cache_path(first_dir) == cache_path(second_dir)
