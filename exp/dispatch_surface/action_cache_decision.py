"""Action Cache decision record: frozen schema, validator and seal branch
(confirmation plan 3.8, G1R1-B3 / G1R2 non-blocking 2).

The schema and the seal branches are frozen with the plan; the owner signs
the ``inclusion`` value later. ``inclusion = "yes"`` makes the confirmation
seal refuse **unconditionally** until an independent Action Cache package
validator exists (G2R1-B9): a package must be verified by content (schema,
member SHAs, its own G1/G2 review verdicts, C-pool / roster / config / code /
cost / runner / discipline bindings), never accepted from a self-reported
digest object. ``YES_PACKAGE_FIELDS`` documents the digests such a validator
must derive. ``"no"`` and ``"post_confirmation_descriptive"`` require a
reason code and a machine-readable claim restriction and forbid any "vs
Action Cache" field in the confirmation output.
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import re

INCLUSION_VALUES = ("yes", "no", "post_confirmation_descriptive")
STATISTICAL_STATUS = ("descriptive", "secondary")
NULLABLE_WHEN_EXCLUDED = ("development_selection_protocol", "config_digest", "code_digest", "c_pool_binding")
REQUIRED_KEYS = ("inclusion", "reason_code", "statistical_status", "development_selection_protocol",
                 "config_digest", "code_digest", "cost_mapping", "c_pool_binding", "claim_restriction")
COST_MAPPING_FROZEN = {
    "axis": "total model-forward compute budget per family",
    "cp2_rule": "not mapped into CP1 three-tier unit table",
}
YES_PACKAGE_FIELDS = ("development_selection_artifact_sha256", "config_digest", "code_digest",
                      "cost_mapping_digest", "c_arm_roster_sha256", "fresh_pool_binding_sha256",
                      "runner_analyzer_support_sha256", "completeness_discipline_sha256",
                      "g1_review_sha256", "g2_review_sha256")
FORBIDDEN_OUTPUT_KEY_RE = re.compile(r"action[_ ]?cache", re.IGNORECASE)
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _is_sha(x) -> bool:
    return isinstance(x, str) and bool(_SHA_RE.match(x))


def validate_record(rec: dict) -> dict:
    """Return a normalised copy or raise SystemExit."""
    if not isinstance(rec, dict):
        raise SystemExit("Action Cache decision record must be an object")
    missing = [k for k in REQUIRED_KEYS if k not in rec]
    if missing:
        raise SystemExit(f"Action Cache decision record lacks {missing}")
    inc = rec["inclusion"]
    if inc not in INCLUSION_VALUES:
        raise SystemExit(f"inclusion must be one of {INCLUSION_VALUES}, got {inc!r}")
    if rec.get("cost_mapping") != COST_MAPPING_FROZEN:
        raise SystemExit("cost_mapping must equal the frozen mapping (total model-forward compute per family; CP2 not in the CP1 table)")
    if not isinstance(rec.get("reason_code"), str) or not rec["reason_code"]:
        raise SystemExit("reason_code is required")
    if not isinstance(rec.get("claim_restriction"), dict) or not rec["claim_restriction"]:
        raise SystemExit("claim_restriction must be a non-empty machine-readable object")
    if inc == "yes":
        if rec.get("statistical_status") not in STATISTICAL_STATUS:
            raise SystemExit("inclusion=yes requires statistical_status in {descriptive, secondary} (never primary)")
        for k in NULLABLE_WHEN_EXCLUDED:
            v = rec.get(k)
            if k in ("config_digest", "code_digest"):
                if not _is_sha(v):
                    raise SystemExit(f"inclusion=yes requires a sha256 {k}")
            elif not v:
                raise SystemExit(f"inclusion=yes requires a non-empty {k}")
    else:
        for k in NULLABLE_WHEN_EXCLUDED:
            if rec.get(k) is not None:
                raise SystemExit(f"inclusion={inc} requires {k} to be canonical null")
        if rec.get("statistical_status") is not None and rec["statistical_status"] not in STATISTICAL_STATUS:
            raise SystemExit("statistical_status must be null/descriptive/secondary")
    return json.loads(json.dumps(rec, sort_keys=True))


def record_sha256(rec: dict) -> str:
    return hashlib.sha256(json.dumps(validate_record(rec), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def seal_branch(rec: dict, action_cache_package: dict | None) -> dict:
    """Decide whether the confirmation seal may proceed under this record.

    ``inclusion = "yes"`` fails closed regardless of ``action_cache_package``:
    no validator that verifies an Action Cache package by content exists yet,
    and a dict of well-formed digest strings proves nothing. When such a
    validator lands it must take a package PATH and verify every binding
    before this branch may return ``ok``."""
    rec = validate_record(rec)
    if rec["inclusion"] == "yes":
        return {"ok": False, "reason": "action_cache_package_validator_not_implemented",
                "required_digests": list(YES_PACKAGE_FIELDS),
                "message": ("inclusion=yes: the seal refuses until an independently G1/G2-reviewed Action Cache "
                            "package is verified by content (schema, member SHAs, review verdicts, C-pool / roster / "
                            "config / code / cost / runner / discipline bindings); self-reported digests are never accepted")}
    return {"ok": True, "reason": f"action_cache_{rec['inclusion']}", "claim_restriction": rec["claim_restriction"]}


def assert_no_action_cache_fields(obj, *, what: str) -> None:
    """Recursively refuse any key naming Action Cache in a confirmation output."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if FORBIDDEN_OUTPUT_KEY_RE.search(str(k)):
                raise SystemExit(f"{what}: output carries an Action Cache comparison field {k!r}")
            assert_no_action_cache_fields(v, what=what)
    elif isinstance(obj, list):
        for v in obj:
            assert_no_action_cache_fields(v, what=what)


def load_record(path) -> dict:
    return validate_record(json.loads(pathlib.Path(path).read_text()))
