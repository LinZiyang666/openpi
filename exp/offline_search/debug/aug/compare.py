"""Read-only old/new augmentation comparison, joined by decision_id."""
import argparse
import json
from pathlib import Path

import numpy as np

from .. import reader


def _load(directory, kind):
    pieces, identities = {}, set()
    for path in sorted((Path(directory) / kind).glob("part_*.npz")):
        data = reader.read_npz(path)
        meta = json.loads(str(data.pop("_meta_json")))
        identities.add(tuple(meta.get(k) for k in ("model", "checkpoint_sha", "fit_config_sha", "dtype")))
        for key, value in data.items():
            if not value.ndim or len(value) != len(data["decision_id"]):
                raise ValueError("unaligned field %s in %s" % (key, path))
            pieces.setdefault(key, []).append(value)
    result = {k: np.concatenate(v) for k, v in pieces.items()}
    ids = result.get("decision_id", np.array([], str)).astype(str).tolist()
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate comparison decision IDs in " + kind)
    return result, {did: i for i, did in enumerate(ids)}, identities


def compare(reference, candidate, kinds=("policy_shadow", "policy_draws", "shadow_look"), atol=0.03, rtol=0.02, limit=None):
    if atol < 0 or rtol < 0:
        raise ValueError("nonnegative tolerances required")
    report = dict(status="PASS", atol=atol, rtol=rtol, kinds={},
                  row_identity_note="Neighbour row changes are reported; their weighted cache chunks, weights and scores must still pass float tolerances.")
    selected_ids = None
    if limit is not None:
        _, index, _ = _load(reference, "policy_shadow")
        selected_ids = set(list(index)[:limit])
    for kind in kinds:
        old, old_index, old_prov = _load(reference, kind)
        new, new_index, new_prov = _load(candidate, kind)
        ids = sorted(did for did in old_index if selected_ids is None or did in selected_ids)
        missing = sorted(set(ids) - set(new_index))
        extra = sorted(set(new_index) - set(old_index)) if limit is None else []
        result = dict(decisions=len(ids), missing=missing, extra=extra, fields={}, status="PASS")
        if (not old_index and kind != "policy_draws") or missing or extra or old_prov != new_prov:
            result["status"] = "FAIL"
            result["provenance_equal"] = old_prov == new_prov
        if set(old) != set(new):
            result.update(status="FAIL", reference_fields=sorted(old), candidate_fields=sorted(new))
        common = [did for did in ids if did in new_index]
        ix, jx = [old_index[x] for x in common], [new_index[x] for x in common]
        for field in sorted(set(old) & set(new) - {"decision_id"}):
            a, b = old[field][ix], new[field][jx]
            if a.shape != b.shape:
                result["fields"][field] = dict(pass_tolerance=False, reference_shape=list(a.shape), candidate_shape=list(b.shape))
                result["status"] = "FAIL"
                continue
            exact = np.array_equal(a, b, equal_nan=True) if a.dtype.kind == "f" else np.array_equal(a, b)
            if a.dtype.kind == "f" and b.dtype.kind == "f":
                passed = bool(np.allclose(a, b, atol=atol, rtol=rtol, equal_nan=True))
                finite = np.isfinite(a) & np.isfinite(b)
                delta = np.abs(a[finite].astype(float) - b[finite].astype(float))
                result["fields"][field] = dict(bitwise_equal=exact, pass_tolerance=passed,
                    max_abs=float(delta.max()) if len(delta) else 0., mean_abs=float(delta.mean()) if len(delta) else 0.)
            elif field == "rows" or field.endswith("_rows") or field.startswith("rows_"):
                passed = True
                result["fields"][field] = dict(bitwise_equal=exact, changed_decisions=int(np.any(a != b, axis=tuple(range(1, a.ndim))).sum()))
            else:
                passed = exact
                result["fields"][field] = dict(bitwise_equal=exact, pass_tolerance=passed)
            if not passed:
                result["status"] = "FAIL"
        report["kinds"][kind] = result
        if result["status"] != "PASS":
            report["status"] = "FAIL"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-aug", type=Path, required=True)
    parser.add_argument("--candidate-aug", type=Path, required=True)
    parser.add_argument("--kinds", nargs="+", default=["policy_shadow", "policy_draws", "shadow_look"])
    parser.add_argument("--atol", type=float, default=0.03)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.reference_aug, args.candidate_aug, args.kinds, args.atol, args.rtol, args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, sort_keys=True))
    raise SystemExit(0 if result["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
