"""Weights<->artifact<->YAML provenance binding for TRACER Phase 6 (plan §6.3d).

The projected library artifact and the serve YAML must reference the SAME trained
ProjectionParams file, or the backend stores keys projected by one head while queries are
projected by another -- the exact raw-vs-projected mixed-space footgun this whole phase
exists to prevent. This module computes a cryptographic digest of the weights file and
asserts the artifact's recorded ``projection_params`` and the YAML's ``weights_path`` both
resolve to a file with that digest.
"""

from __future__ import annotations

import hashlib
import pathlib


def weights_sha256(path) -> str:
    """SHA-256 hex digest of a weights (.pt) file."""
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(f"weights file missing: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def assert_binding(artifact_weights_path, yaml_weights_path) -> str:
    """Fail loud unless the artifact and YAML weights paths hash-match. Returns the digest."""
    a = weights_sha256(artifact_weights_path)
    y = weights_sha256(yaml_weights_path)
    if a != y:
        raise ValueError(
            "projected-space mismatch: artifact weights "
            f"({artifact_weights_path}, sha256={a[:12]}...) != YAML weights "
            f"({yaml_weights_path}, sha256={y[:12]}...)"
        )
    return a


def record_weights_digest(artifact: dict, weights_path) -> dict:
    """Stamp the artifact's ``projection_params`` with an IMMUTABLE recorded digest (§6.3d).

    The build chain calls this once so the artifact carries both the weights PATH and the
    SHA-256 of the bytes it was built from -- a path alone can be repointed at a different
    head after the fact; the recorded digest makes the projected space the artifact was keyed
    in cryptographically fixed. Fails loud if the artifact is not projected.
    """
    pp = artifact.get("projection_params")
    if not isinstance(pp, dict) or not pp.get("projection_weights_path"):
        raise ValueError("artifact records no projection_params.projection_weights_path (not projected)")
    pp["projection_weights_sha256"] = weights_sha256(weights_path)
    return artifact


def assert_recorded_digest(artifact: dict, weights_path) -> str:
    """Fail loud unless the live weights file hashes to the artifact's RECORDED digest.

    This is the immutability check: it does not trust the recorded path, it re-hashes the
    actual bytes at ``weights_path`` and compares to the frozen ``projection_weights_sha256``
    stamped at build time. A mismatch means the weights were swapped after the artifact was
    keyed -- the projected-space footgun this phase exists to prevent.
    """
    pp = artifact.get("projection_params") or {}
    recorded = pp.get("projection_weights_sha256")
    if not recorded:
        raise ValueError("artifact records no immutable projection_weights_sha256 (rebuild via record_weights_digest)")
    live = weights_sha256(weights_path)
    if live != recorded:
        raise ValueError(
            f"weights digest drift: recorded={recorded[:12]}... != live {weights_path} ={live[:12]}..."
        )
    return live


def assert_artifact_yaml_binding(artifact: dict, yaml_cfg: dict) -> str:
    """Serve/build-time check: the projected artifact's recorded weights == the YAML's.

    Reads the weights path the ARTIFACT recorded under ``projection_params`` and the path
    the serve YAML declares under ``key_builder.projection.weights_path``, and hash-binds
    them (§6.3d). Fails loud if either is absent (an unprojected artifact / raw YAML must
    not be paired with a projected lane) or if they differ.
    """
    art_wp = (artifact.get("projection_params") or {}).get("projection_weights_path")
    if not art_wp:
        raise ValueError("artifact records no projection_params.projection_weights_path (not projected)")
    kb = (yaml_cfg.get("key_builder") or {})
    if kb.get("type") != "projection":
        raise ValueError("YAML key_builder.type is not 'projection'")
    yaml_wp = (kb.get("projection") or {}).get("weights_path")
    if not yaml_wp or yaml_wp == "__FILL_AT_EXECUTION__":
        raise ValueError("YAML key_builder.projection.weights_path is unset/placeholder")
    return assert_binding(art_wp, yaml_wp)


def assert_serve_binding(artifact: dict, yaml_cfg: dict, weights_path) -> str:
    """The serve-init hook (§6.3d): enforce the full recorded-digest <-> live-bytes <-> YAML chain.

    Called at projected-lane serve initialization BEFORE any query is keyed. It (1) re-hashes
    the live weights and checks them against the artifact's immutable recorded digest, then
    (2) hash-binds the artifact-recorded path and the YAML-declared path. Any break aborts
    serving rather than silently keying queries in a different projected space than the store.
    Returns the enforced digest.
    """
    live = assert_recorded_digest(artifact, weights_path)
    bound = assert_artifact_yaml_binding(artifact, yaml_cfg)
    if bound != live:
        raise ValueError(
            f"serve binding split: recorded/live digest {live[:12]}... != path-bound {bound[:12]}..."
        )
    return live
