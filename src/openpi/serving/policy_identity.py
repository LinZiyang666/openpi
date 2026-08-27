"""Content-level policy identity for the dispatch-surface attestation seam.

A surface artifact certifies boundaries calibrated against ONE policy. The
binding therefore cannot be an operator-supplied string or a (name, size)
listing: replacing checkpoint bytes at equal size must change the identity.
This module computes a canonical content digest of the *resolved* checkpoint
root and is shared verbatim by the three parties that must agree:

  1. offline rebuild / calibration scripts (write the value into the surface
     artifact's retrieval_contract),
  2. ``scripts/serve_policy.py`` (computes it for the checkpoint it actually
     loads and reports it in the websocket metadata), and
  3. precheck / cost-bench runners (read the server-reported value and compare
     against the artifact contract, failing fast on mismatch).

The checkpoint URI is materialised through the same downloader the policy
loader itself uses (``openpi.shared.download.maybe_download``), so the digest
is computed over exactly the bytes the policy is created from.

Coupling map:
  DEPENDS ON:  openpi.shared.download (URI resolution only).
  CONSUMED BY: scripts/serve_policy.py (metadata field), exp/dispatch_surface
               pipeline scripts, precheck runners.
  IF CHANGED:  every stored surface artifact's policy binding is invalidated;
               bump the artifact schema when altering the digest recipe.
"""

from __future__ import annotations

import functools
import hashlib
import pathlib

from openpi.shared import download as _download

_CHUNK_BYTES = 4 * 1024 * 1024


def resolve_checkpoint_root(checkpoint_uri: str) -> pathlib.Path:
    """Materialise a checkpoint URI to a local directory.

    Delegates to the policy loader's own resolver so server-side attestation
    and policy creation see the same bytes. Raises if the result is not an
    existing directory.
    """
    root = pathlib.Path(_download.maybe_download(str(checkpoint_uri)))
    if not root.is_dir():
        raise ValueError(f"resolved checkpoint root is not a directory: {root}")
    return root


@functools.lru_cache(maxsize=8)
def compute_policy_fingerprint(resolved_checkpoint_root: str, config_name: str) -> str:
    """Canonical content digest of (config name, checkpoint file contents).

    Recipe: sha256 over the config name plus, for every regular file under the
    root (recursively, sorted by POSIX relative path), the tuple
    ``(relative_path, size, sha256(content))``. Symlinks that escape the root,
    special files and empty roots are rejected — a fingerprint over nothing or
    over ambiguous content would be an attestation in name only.

    Cached per process: the digest is computed once at startup / pipeline
    entry, never on the request path.
    """
    root = pathlib.Path(resolved_checkpoint_root)
    if not root.is_dir():
        raise ValueError(f"checkpoint root is not a directory: {root}")
    root_resolved = root.resolve()

    entries: list[tuple[str, int, str]] = []
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if path.is_dir() and not path.is_symlink():
            continue
        if path.is_symlink():
            target = path.resolve()
            if not target.is_relative_to(root_resolved):
                raise ValueError(f"symlink escapes checkpoint root: {path} -> {target}")
        if not path.is_file():
            raise ValueError(f"unsupported special file under checkpoint root: {path}")
        h = hashlib.sha256()
        size = 0
        with open(path, "rb") as f:
            while chunk := f.read(_CHUNK_BYTES):
                h.update(chunk)
                size += len(chunk)
        entries.append((path.relative_to(root).as_posix(), size, h.hexdigest()))

    if not entries:
        raise ValueError(f"checkpoint root contains no regular files: {root}")

    outer = hashlib.sha256()
    outer.update(config_name.encode("utf-8"))
    for rel, size, digest in entries:
        outer.update(f"\x00{rel}\x00{size}\x00{digest}".encode("utf-8"))
    return outer.hexdigest()
