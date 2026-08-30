"""G1 freeze record: which document bytes the confirmation plan froze
(confirmation plan section 10, G2R1-B1).

Two binding modes exist because the plan log is append-only by workflow
(Code / G2 records are appended to the same file after G1) while the frozen
G1 text must never change:

* ``documents_sha256``  -- whole-file SHA-256 for documents that are not
  appended to after the freeze (protocol draft, paper TODO);
* ``frozen_prefix``     -- for an append-only log: the SHA-256 of the
  canonical prefix that ends with the G1 boundary line. The prefix is every
  byte up to and including the single line that starts with ``end_marker``
  (plus its newline). Appending after that line never changes the digest;
  changing any byte inside the prefix, duplicating or removing the marker
  does.

Only this module derives the frozen bytes; the test and any future seal
call ``verify``.
"""

from __future__ import annotations

import hashlib
import json
import pathlib

RECORD_PATH = pathlib.Path("exp/dispatch_surface/config/confirmation_freeze_record.json")


def frozen_prefix_bytes(data: bytes, end_marker: str) -> bytes:
    """Bytes of ``data`` up to and including the unique line starting with ``end_marker``."""
    marker = end_marker.encode("utf-8")
    if not marker:
        raise SystemExit("freeze record: empty end_marker")
    offset = 0
    hits: list[int] = []
    for line in data.split(b"\n"):
        if line.startswith(marker):
            hits.append(offset + len(line) + 1)
        offset += len(line) + 1
    if len(hits) != 1:
        raise SystemExit(f"freeze record: end marker {end_marker!r} occurs {len(hits)} times, expected exactly once")
    end = min(hits[0], len(data))
    return data[:end]


def frozen_prefix_sha256(path: pathlib.Path, end_marker: str) -> str:
    return hashlib.sha256(frozen_prefix_bytes(path.read_bytes(), end_marker)).hexdigest()


def load_record(path: pathlib.Path = RECORD_PATH) -> dict:
    rec = json.loads(pathlib.Path(path).read_text())
    for key in ("documents_sha256", "frozen_prefix", "constants"):
        if key not in rec:
            raise SystemExit(f"freeze record lacks {key}")
    return rec


def verify(rec: dict, repo_root: pathlib.Path) -> dict[str, str]:
    """Recompute every frozen digest from the files under ``repo_root``.

    Returns ``{relative path: digest}``; raises SystemExit on the first
    drift so a caller can never mistake a partial check for a pass."""
    out: dict[str, str] = {}
    for rel, sha in rec["documents_sha256"].items():
        got = hashlib.sha256((repo_root / rel).read_bytes()).hexdigest()
        if got != sha:
            raise SystemExit(f"{rel} drifted since the G1 freeze (expected {sha[:12]}..., got {got[:12]}...)")
        out[rel] = got
    for rel, spec in rec["frozen_prefix"].items():
        if rel in out:
            raise SystemExit(f"{rel} is frozen twice (whole file and prefix)")
        got = frozen_prefix_sha256(repo_root / rel, spec["end_marker"])
        if got != spec["sha256"]:
            raise SystemExit(f"{rel}: the G1 frozen prefix drifted (expected {spec['sha256'][:12]}..., got {got[:12]}...)")
        out[rel] = got
    return out
