"""Bind materialized init pools to their parent identities and launch cohorts."""

from __future__ import annotations

from exp.online_rit.common import canonical_sha256


def validate_pool(manifest: dict, key: str, record: dict, trials: int) -> dict[int, list[int]]:
    """Return the original-index mapping only for the attested pool contents."""
    if manifest.get("suite") != record.get("suite") or not manifest.get("parent_pool_sha256"):
        raise SystemExit("init pool manifest has a different suite or no parent pool identity")
    expected = manifest.get("pool_records", {}).get(key)
    if not expected or any(expected.get(k) != record.get(k) for k in ("rollup_sha256", "per_task_digests", "total_inits")):
        raise SystemExit("actual init pool differs from the selected manifest pool")
    mapping = {int(t): list(v) for t, v in manifest.get(key, {}).items()}
    if set(mapping) != set(range(10)) or record.get("total_inits") != 10 * trials:
        raise SystemExit("init pool has a wrong task set or trial count")
    if canonical_sha256(manifest[key]) != expected.get("index_map_sha256"):
        raise SystemExit("init pool original-index mapping changed")
    for t, indices in mapping.items():
        if len(indices) != trials or indices != sorted(set(indices)) or any(type(i) is not int or not 0 <= i < 50 for i in indices):
            raise SystemExit(f"invalid original indices for task {t}")
    for t in range(10):
        if set(manifest.get("adapt", {}).get(str(t), [])) & set(manifest.get("terminal", {}).get(str(t), [])):
            raise SystemExit("adapt and terminal pools overlap")
    return mapping
