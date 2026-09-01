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


# ------------------------------------------------------------------
# RIT-PL export / fit record schemas (plan logs/rit_pl_ir_ladder_plan.log.md)
# ------------------------------------------------------------------

#: The schema constants live here (not in export_rit_pl) so the emitter can
#: validate a record without importing the exporter, and the exporter imports
#: this module rather than the other way round.
RIT_PL_FAMILY = "s0_pl"
RIT_PL_SOURCE_ROLE = "artifact.dsp_s0"
RIT_PL_ADDRESSING = ("target_ir", "quantile")
RIT_PL_EXPORT_RECORD_KEYS = frozenset({
    "protocol", "posthoc_exploratory", "source_role", "family", "estimator", "addressing",
    "rev1_package_manifest_sha256", "source_artifact_sha256", "source_fit_record_sha256",
    "table_sha256", "d0_record_sha256", "dev_membership_sha256", "pl_fit_record_path",
    "pl_fit_record_sha256", "pl_fit_digests", "eps_total", "n_seg", "quantile_method",
    "cost_model_digest", "git_commit", "python", "numpy", "artifacts",
})
RIT_PL_EXPORT_ARTIFACT_KEYS = frozenset({
    "path", "addressing", "target_ir", "predicted_ir", "ir_gap", "quantile", "delta",
    "theta_full", "theta_warm", "floor_info", "output_sha256",
})
RIT_PL_FIT_RECORD_KEYS = frozenset({
    "rev1_package_manifest_sha256", "source_artifact_sha256", "source_fit_record_sha256",
    "d0_record_sha256", "table_sha256", "dev_membership_sha256", "cost_model_digest", "estimator",
    "alpha", "eps_total", "n_seg_req", "n_seg", "n_dev_rows", "knots", "q_warm", "q_full",
    "pl_fit_digests", "ir_curve", "s_range", "git_commit", "python", "numpy",
})
_RIT_PL_IDENTITY_KEYS = (
    "rev1_package_manifest_sha256", "source_artifact_sha256", "source_fit_record_sha256",
    "d0_record_sha256", "table_sha256", "dev_membership_sha256", "cost_model_digest", "estimator",
    "eps_total", "n_seg", "pl_fit_digests",
)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and np.isfinite(x)


def assert_rit_pl_export_record_schema(rec: dict, *, what: str, cost_model_digest: str) -> None:
    """Exact-key schema of a RIT-PL export record, both addressing modes."""
    from exp.dispatch_surface.rit_pl import ESTIMATOR, PROTOCOL_RIT_PL

    if set(rec) != RIT_PL_EXPORT_RECORD_KEYS:
        raise SystemExit(f"{what}: RIT-PL export record keys {sorted(set(rec) ^ RIT_PL_EXPORT_RECORD_KEYS)} differ from the schema")
    if rec["protocol"] != PROTOCOL_RIT_PL or rec["posthoc_exploratory"] is not True:
        raise SystemExit(f"{what}: export record is not a RIT-PL exploratory record")
    if rec["cost_model_digest"] != cost_model_digest:
        raise SystemExit(f"{what}: export record cost model digest != the cost authority")
    if rec["family"] != RIT_PL_FAMILY or rec["source_role"] != RIT_PL_SOURCE_ROLE or rec["estimator"] != ESTIMATOR:
        raise SystemExit(f"{what}: export record family / source_role / estimator inconsistent with RIT-PL")
    if rec["addressing"] not in RIT_PL_ADDRESSING:
        raise SystemExit(f"{what}: export record addressing {rec['addressing']!r} not in {RIT_PL_ADDRESSING}")
    if not (_is_number(rec["eps_total"]) and rec["eps_total"] > 0):
        raise SystemExit(f"{what}: RIT-PL export record needs eps_total > 0")
    if not isinstance(rec["artifacts"], dict) or not rec["artifacts"]:
        raise SystemExit(f"{what}: export record has no artifacts")
    for name, art in rec["artifacts"].items():
        if set(art) != RIT_PL_EXPORT_ARTIFACT_KEYS:
            raise SystemExit(f"{what}: export artifact {name} keys differ from the RIT-PL schema")
        if art["addressing"] != rec["addressing"]:
            raise SystemExit(f"{what}: artifact {name} addressing differs from the record")
        if rec["addressing"] == "target_ir":
            if not (_is_number(art["target_ir"]) and _is_number(art["ir_gap"])):
                raise SystemExit(f"{what}: artifact {name} lacks a numeric target_ir / ir_gap")
        elif art["target_ir"] is not None or art["ir_gap"] is not None:
            raise SystemExit(f"{what}: quantile-addressed artifact {name} must carry null target_ir / ir_gap")
        if not (_is_number(art["delta"]) and art["delta"] > 0):
            raise SystemExit(f"{what}: artifact {name} delta must be finite and positive")
        if not (_is_number(art["predicted_ir"]) and _is_number(art["quantile"])):
            raise SystemExit(f"{what}: artifact {name} lacks predicted_ir / quantile")
        full, warm = float(art["theta_full"]), float(art["theta_warm"])
        if warm > full:
            raise SystemExit(f"{what}: artifact {name} theta_warm exceeds theta_full")


def assert_rit_pl_fit_record(fit_rec: dict, export_rec: dict, *, what: str,
                             artifact_cuts: dict | None = None) -> None:
    """Semantic leg of the PL fit-record validation (after the byte SHA leg):
    exact keys, consistent knot counts, recomputed digests, identity fields
    equal to the export record and -- when ``artifact_cuts`` maps artifact
    names to their stored ``(s_min_full, s_min_warm)`` -- the cuts recomputed
    from the fit at each recorded delta equal the stored ones exactly."""
    from exp.dispatch_surface.rit_pl import ESTIMATOR, KNOT_LADDER, cuts, fit_from_record, pl_fit_digests

    if set(fit_rec) != RIT_PL_FIT_RECORD_KEYS:
        raise SystemExit(f"{what}: PL fit record keys {sorted(set(fit_rec) ^ RIT_PL_FIT_RECORD_KEYS)} differ from the schema")
    if fit_rec["estimator"] != ESTIMATOR:
        raise SystemExit(f"{what}: PL fit record estimator {fit_rec['estimator']!r} != {ESTIMATOR!r}")
    knots = np.asarray(fit_rec["knots"], dtype=np.float64)
    if fit_rec["n_seg_req"] not in KNOT_LADDER:
        raise SystemExit(f"{what}: PL fit record n_seg_req {fit_rec['n_seg_req']!r} is not a ladder rung")
    if knots.ndim != 1 or not np.isfinite(knots).all() or not (np.diff(knots) > 0).all():
        raise SystemExit(f"{what}: PL fit record knots are not finite and strictly increasing")
    if fit_rec["n_seg"] != len(knots) - 1 or not (2 <= fit_rec["n_seg"] <= fit_rec["n_seg_req"]):
        raise SystemExit(f"{what}: PL fit record n_seg / knots / n_seg_req are inconsistent")
    if len(fit_rec["q_warm"]) != len(knots) or len(fit_rec["q_full"]) != len(knots):
        raise SystemExit(f"{what}: PL fit record layer lengths differ from the knot count")
    if not (_is_number(fit_rec["eps_total"]) and fit_rec["eps_total"] > 0):
        raise SystemExit(f"{what}: PL fit record eps_total must be positive")
    fit = fit_from_record(fit_rec)
    if pl_fit_digests(fit) != fit_rec["pl_fit_digests"]:
        raise SystemExit(f"{what}: PL fit record pl_fit_digests do not match its own arrays")
    for key in _RIT_PL_IDENTITY_KEYS:
        if fit_rec.get(key) != export_rec.get(key):
            raise SystemExit(f"{what}: PL fit record {key} differs from the export record")
    for name, (full, warm) in (artifact_cuts or {}).items():
        art = export_rec["artifacts"].get(name)
        if art is None:
            raise SystemExit(f"{what}: artifact {name} is not in the export record")
        got_full, got_warm = cuts(fit, float(art["delta"]))
        if got_full != float(full) or got_warm != float(warm):
            raise SystemExit(f"{what}: artifact {name} cuts differ from the PL fit at its recorded delta")


def _same(a, b) -> bool:
    """Type-safe exact equality: numbers by value (bool never equals int),
    None only to None, dicts / lists element-wise, everything else by type and value."""
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(a) == float(b)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


#: Fields an artifact's meta must repeat exactly from its export-record entry.
_RIT_PL_META_VS_ARTIFACT = ("addressing", "target_ir", "predicted_ir", "ir_gap", "quantile", "floor_info")
#: Fields an artifact's meta must repeat exactly from the export record top level.
_RIT_PL_META_VS_RECORD = (
    "estimator", "family", "addressing", "eps_total", "n_seg", "pl_fit_digests", "pl_fit_record_sha256",
    "source_role", "source_artifact_sha256", "source_fit_record_sha256", "rev1_package_manifest_sha256",
)
#: Fields an artifact's meta must repeat exactly from the PL fit record.
_RIT_PL_META_VS_FIT = ("n_seg_req", "s_range")


def assert_rit_pl_artifact_coherence(artifact, art_rec: dict, export_rec: dict, fit_rec: dict, *, what: str) -> None:
    """Close the duplicated-field triangle of one RIT-PL arm: the export
    record's cuts / delta must equal the deployed artifact arrays, and every
    addressing / provenance field the artifact meta repeats must equal the
    export record (artifact entry and top level) and the PL fit record.
    Values are compared exactly (they were all generated by one export run)."""
    meta = artifact.meta
    if meta.get("posthoc_exploratory") is not True:
        raise SystemExit(f"{what}: artifact is not marked posthoc_exploratory")
    stored_full, stored_warm = float(artifact.s_min_full[0]), float(artifact.s_min_warm[0])
    if not _same(art_rec.get("theta_full"), stored_full) or not _same(art_rec.get("theta_warm"), stored_warm):
        raise SystemExit(f"{what}: export record theta_full/theta_warm differ from the deployed artifact cuts")
    if not _same(art_rec.get("delta"), float(artifact.delta)):
        raise SystemExit(f"{what}: export record delta differs from the deployed artifact")
    for key in _RIT_PL_META_VS_ARTIFACT:
        if key not in meta or not _same(meta[key], art_rec.get(key)):
            raise SystemExit(f"{what}: artifact meta {key} differs from the export record entry")
    for key in _RIT_PL_META_VS_RECORD:
        if key not in meta or not _same(meta[key], export_rec.get(key)):
            raise SystemExit(f"{what}: artifact meta {key} differs from the export record")
    for key in _RIT_PL_META_VS_FIT:
        if key not in meta or not _same(meta[key], fit_rec.get(key)):
            raise SystemExit(f"{what}: artifact meta {key} differs from the PL fit record")
