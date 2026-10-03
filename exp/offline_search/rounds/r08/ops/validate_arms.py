"""Validate all emitted fit contracts and unchanged retrievals without serving."""
import argparse
import json
import pickle
from pathlib import Path

from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import fingerprint, sha
from .emit_arms import CELLS, HERE, make_specs, verify_source, writable_output
from .prefit import load_frozen


def validate(lock, fit_root="/tmp/r8_S5/fits"):
    rows = make_specs(lock, fit_root)
    records = []
    for cell in CELLS:
        model, suite, size = cell.split("_")
        api_cell = "{}_{}_cache".format(model, suite)
        frozen = lock["cells"][cell]
        base = load_frozen(frozen["A"], api_cell)["method"]
        base_fingerprint = fingerprint(base)
        selected = [r for r in rows if r["model"] == model and r["suite"] == suite and
                    (r["r8"]["library_size"] == int(size) or r["r8"]["variant"] == "P10" and size == "50")]
        wrist_fingerprint = None
        for row in selected:
            variant = row["r8"]["variant"]
            path = row["plugin_args"][row["plugin_args"].index("--os-fit-artifact") + 1]
            verify_source(row["r8"]["source"])
            # Deliberately use pickle.load: this is the plugin's deployed path.
            with open(path, "rb") as stream:
                blob = pickle.load(stream)
            expected = dict(spec=row["method"], kwargs=row["kwargs"], cell=api_cell)
            if {k: blob.get(k) for k in expected} != expected:
                raise ValueError("emitted spec and fit metadata differ: " + row["name"])
            method = blob["method"]
            cls, _ = load_method_class(row["method"])
            if not isinstance(method, cls):
                raise ValueError("emitted class and fit differ: " + row["name"])
            api.check_method_attrs(method)
            deployed_base = getattr(method, "base", method)
            if fingerprint(deployed_base) != base_fingerprint:
                raise ValueError("frozen retrieval changed: " + row["name"])
            if variant in ("W5", "W10", "SW"):
                if wrist_fingerprint is None:
                    wrist_fingerprint = fingerprint(load_frozen(frozen["SW"], api_cell)["method"].wrist)
                if fingerprint(method.wrist) != wrist_fingerprint:
                    raise ValueError("frozen wrist metric changed: " + row["name"])
                if method.base.budget != (0 if variant == "W5" else 1):
                    raise ValueError("wrong wrist cadence: " + row["name"])
            if variant in ("CU", "CT"):
                if method.random_seed != row["kwargs"]["random_seed"] or method.rho != row["r8"]["rho"]:
                    raise ValueError("wrong frozen call budget/seed: " + row["name"])
                if method.fit_info.get("c_calibration_sha256") != sha(row["kwargs"]["calibration_path"]):
                    raise ValueError("call calibration mismatch: " + row["name"])
            if variant not in ("A", "SF1", "SW"):
                if variant not in ("CU", "CT"):
                    json.dumps(method.debug_record(), allow_nan=False)
                if blob["provenance"]["code_sha256"] != sha(HERE.parent / "methods/methods.py"):
                    raise ValueError("new fit predates current method code: " + row["name"])
            records.append(dict(arm=row["name"], variant=variant, artifact=path, sha256=sha(path),
                                source_sha256=row["r8"]["source"]["sha256"], frozen_retrieval_identical=True,
                                metadata_identical=True, deployable_pickle_load=True))
    return dict(status="PASS", arms=len(records), records=records)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-lock", type=Path, default=HERE / "r7_sources.json")
    ap.add_argument("--fit-root", default="/tmp/r8_S5/fits")
    ap.add_argument("--out", type=Path, default=Path("/tmp/r8_S5/artifact_validation.json"))
    a = ap.parse_args(argv)
    report = validate(json.loads(a.source_lock.read_text()), a.fit_root)
    out = writable_output(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(status=report["status"], arms=report["arms"], report=str(out))))


if __name__ == "__main__":
    main()
