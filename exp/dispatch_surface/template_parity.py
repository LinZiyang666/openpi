"""Template parity between an exploratory artifact and its Rev 1 source (G2R1-B4).

One helper, used by the exporter (before writing), the emitter, the runner
validator and the Phase 0 discipline validator: everything except ``delta``
and the exported boundaries (``s_min_full`` / ``s_min_warm``) must be
identical, field by field and array by array. A second helper walks any
nested meta structure and refuses placeholder provenance values.
"""

from __future__ import annotations

import numpy as np

SCALAR_FIELDS = (
    "schema_version", "k", "h_exec", "start_t_ws", "quantile_alpha", "certification_mode",
    "uses_disagreement", "conformal_c", "n_calibration_episodes",
)
ARRAY_FIELDS = ("w", "active_mask", "v_bin_edges")
FORBIDDEN_TOKENS = ("pending", "placeholder", "tbd", "todo", "fixme")


def assert_template_parity(artifact, source, *, what: str) -> None:
    """Raise SystemExit unless ``artifact`` equals ``source`` off the delta/boundary axis."""
    for field in SCALAR_FIELDS:
        a, b = getattr(artifact, field), getattr(source, field)
        if a != b:
            raise SystemExit(f"{what}: field {field}={a!r} differs from the source template {b!r}")
    for field in ARRAY_FIELDS:
        a, b = np.asarray(getattr(artifact, field)), np.asarray(getattr(source, field))
        if a.shape != b.shape or not np.array_equal(a, b):
            raise SystemExit(f"{what}: array {field} differs from the source template")
    if artifact.retrieval_contract != source.retrieval_contract:
        raise SystemExit(f"{what}: retrieval contract differs from the source template")
    if not (np.isfinite(artifact.delta) and artifact.delta > 0):
        raise SystemExit(f"{what}: delta must be finite and positive")


def assert_no_placeholders(obj, *, what: str, path: str = "meta") -> None:
    """Recursively refuse placeholder strings anywhere in dicts / lists."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            assert_no_placeholders(value, what=what, path=f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            assert_no_placeholders(value, what=what, path=f"{path}[{idx}]")
    elif isinstance(obj, str):
        low = obj.lower()
        if any(tok in low for tok in FORBIDDEN_TOKENS):
            raise SystemExit(f"{what}: {path}={obj!r} is a placeholder, not provenance")


EXPORT_RECORD_KEYS = frozenset({
    "protocol", "posthoc_exploratory", "source_role", "family", "rev1_package_manifest_sha256",
    "source_artifact_sha256", "source_fit_record_sha256", "table_sha256", "d0_record_sha256",
    "dev_membership_sha256", "final_fit_digests", "quantile_method", "cost_model_digest",
    "git_commit", "python", "numpy", "artifacts",
})
EXPORT_ARTIFACT_KEYS = frozenset({"path", "quantile", "delta", "output_sha256"})


def assert_export_record_schema(rec: dict, *, what: str, cost_model_digest: str,
                                protocol: str) -> None:
    if set(rec) != EXPORT_RECORD_KEYS:
        raise SystemExit(f"{what}: export record keys {sorted(set(rec) ^ EXPORT_RECORD_KEYS)} differ from the frozen schema")
    if rec["protocol"] != protocol or rec["posthoc_exploratory"] is not True:
        raise SystemExit(f"{what}: export record is not a Phase 0 exploratory record")
    if rec["cost_model_digest"] != cost_model_digest:
        raise SystemExit(f"{what}: export record cost model digest != the cost authority")
    if rec["family"] not in ("sv", "s0") or rec["source_role"] != f"artifact.dsp_{rec['family']}":
        raise SystemExit(f"{what}: export record family/source_role inconsistent")
    if not isinstance(rec["artifacts"], dict) or not rec["artifacts"]:
        raise SystemExit(f"{what}: export record has no artifacts")
    for name, art in rec["artifacts"].items():
        if set(art) != EXPORT_ARTIFACT_KEYS:
            raise SystemExit(f"{what}: export artifact {name} keys differ from the frozen schema")
