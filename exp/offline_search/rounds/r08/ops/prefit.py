"""Prepare R8 artifacts from SHA-verified frozen R7 library-only fits.

Retrieval, wrist PCA/LDA, stages and CU/CT calibration are never re-solved from
evaluation outcomes. Cadence/lottery/call variants change only their controller.
Serialized fitted objects use the established plugin pickle format; debug data
itself never contains pickle. All writes stay under /tmp/r8_S5.
"""
import argparse
import json
import os
import pickle
import time
from pathlib import Path

from exp.offline_search.closed_loop.plugin import clone_method, load_method_class
from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler, fingerprint, sha
from exp.offline_search.rounds.r07.c1_follow.methods import FollowExtension
from exp.offline_search.rounds.r08.methods.methods import FollowLottery, WristEveryLook
from .emit_arms import HERE, CELLS, COUNTS, make_specs, verify_source

ROOT = Path("/tmp/r8_S5")


def output_root(path):
    p = Path(path).resolve()
    if p != ROOT and ROOT not in p.parents:
        raise ValueError("R8 S5 prefits write only under /tmp/r8_S5")
    return p


def load_frozen(source, cell):
    verify_source(source)
    with open(source["artifact"], "rb") as f:
        blob = FitUnpickler(f).load()
    expected = dict(spec=source["method"], kwargs=source["kwargs"], cell=cell)
    if {k: blob.get(k) for k in expected} != expected:
        raise ValueError("frozen artifact spec/kwargs/cell mismatch: " + source["artifact"])
    return blob


def adapt_cache(base, cls, kwargs):
    obj = cls(**kwargs)
    config = {k: v for k, v in vars(obj).items() if k.startswith(("_r8", "follow_", "lottery_"))
              or k in ("name", "budget", "_follow_plan")}
    cloned, _ = clone_method(base, strict=True)
    vars(obj).update(vars(cloned))
    vars(obj).update(config)
    obj.prof = api.NULL_PROFILER
    obj.invalidate_anchor()
    return obj


def build_method(row, blobs):
    variant, kwargs = row["r8"]["variant"], row["kwargs"]
    cls, _ = load_method_class(row["method"])
    if variant in ("CU", "CT"):
        method, _ = clone_method(blobs[variant]["method"], strict=True)
        method.random_seed = kwargs["random_seed"]
        if method.rho != row["r8"]["rho"] or method.tilt != (variant == "CT"):
            raise ValueError("frozen call-controller budget/tilt mismatch")
        return method
    if variant in ("W5", "W10"):
        method, _ = clone_method(blobs["SW"]["method"], strict=True)
        method.__class__ = WristEveryLook
        config = WristEveryLook(**kwargs)
        method._r8_every_controls, method._r8_diag = config._r8_every_controls, {}
        method.name = config.name
        method.finish_cadence()
        return method
    base = blobs["A"]["method"]
    if variant in ("FL", "SHIFT", "A5"):
        method = adapt_cache(base, cls, kwargs)
        if variant == "FL":
            method.follow_table = blobs["SF1"]["method"].follow_table
            if method.follow_table.retrieval_fingerprint != fingerprint(base):
                raise ValueError("lottery stage/valve table belongs to another frozen retrieval")
            method.follow_component = FollowExtension(method.follow_table, extend_blocks=2,
                                                      stage_gate=False, state_valve=False)
        if fingerprint(method) != fingerprint(base):
            raise ValueError("R8 variant changed frozen retrieval")
        return method
    if variant in ("IP", "P10", "O5a", "O5b"):
        method = cls(**kwargs)
        method.base, _ = clone_method(base, strict=True)
        method.prof = method.base.prof = api.NULL_PROFILER
        method.fit_info = dict(base_sha256=row["r8"]["source"]["sha256"], library_only=True,
                               stall=False, cooldown=False, policy_commit_controls=10, policy_tail_blocks=1)
        return method
    raise ValueError("no new fit needed for " + variant)


def publish(path, artifact):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_suffix(path.suffix + ".part")
    with part.open("wb") as f:
        pickle.dump(artifact, f, protocol=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(part), str(path))


def prefit(lock, out=ROOT, cells=None, variants=None, reuse_existing=True):
    out = output_root(out)
    rows = make_specs(lock, str(out / "fits"))
    records = []
    for cell in CELLS:
        if cells and cell not in cells:
            continue
        model, suite, _size = cell.split("_")
        selected = [r for r in rows if r["model"] == model and r["suite"] == suite and
                    (r["r8"]["library_size"] == int(_size) or r["r8"]["variant"] == "P10" and _size == "50")
                    and (not variants or r["r8"]["variant"] in variants)]
        if not selected:
            continue
        blobs = {}
        for row in selected:
            variant = row["r8"]["variant"]
            source = row["r8"]["source"]
            if variant in ("A", "SF1", "SW"):
                verify_source(source)
                records.append(dict(arm=row["name"], status="REUSED_BY_SHA", path=source["artifact"],
                                    sha256=source["sha256"], source_sha256=source["sha256"]))
                continue
            path = out / "fits" / (row["name"] + ".pkl")
            start = time.perf_counter()
            provenance = dict(source_sha256=source["sha256"], source_artifact=source["artifact"],
                dependencies=source.get("dependencies", {}), library_only=True,
                retrieval_rule="frozen R7 A; wrist uses frozen R7 independent 72-D fit",
                code_sha256=sha(HERE.parent / "methods/methods.py"),
                random_seed=row["r8"]["new_random_seed"])
            expected = dict(spec=row["method"], kwargs=row["kwargs"], cell="{}_{}_cache".format(model, suite))
            if reuse_existing and path.is_file():
                with path.open("rb") as f:
                    artifact = pickle.load(f)
                if ({k: artifact.get(k) for k in expected} != expected or artifact.get("provenance") != provenance):
                    raise ValueError("existing R8 fit has different configuration/provenance: " + str(path))
                # Verify input dependencies on resumption too.
                verify_source(source)
                status = "EXISTING_VERIFIED"
            else:
                needed = {"A", row["r8"]["source_variant"]}
                for key in needed:
                    if key not in blobs:
                        blobs[key] = load_frozen(lock["cells"][cell][key], expected["cell"])
                method = build_method(row, blobs)
                api.check_method_attrs(method)
                artifact = dict(expected, method=method, registered=blobs["A"].get("registered", {}),
                                fit_s=time.perf_counter() - start, provenance=provenance)
                publish(path, artifact)
                # Round-trip metadata and fitted class, using the deployed loader.
                with path.open("rb") as f:
                    loaded = pickle.load(f)
                if {k: loaded[k] for k in expected} != expected or type(loaded["method"]) != type(method):
                    raise ValueError("prefit round-trip failed")
                status = "BUILT"
            rec = dict(arm=row["name"], status=status, path=str(path), sha256=sha(path),
                       bytes=path.stat().st_size, source_sha256=source["sha256"], seconds=time.perf_counter() - start)
            records.append(rec)
            print(json.dumps(rec), flush=True)
        del blobs
    out.mkdir(parents=True, exist_ok=True)
    report_path = out / "prefit_report.json"
    if report_path.is_file():
        old = {r["arm"]: r for r in json.loads(report_path.read_text())}
        old.update({r["arm"]: r for r in records})
        records = sorted(old.values(), key=lambda r: r["arm"])
    report_path.write_text(json.dumps(records, indent=2, allow_nan=False) + "\n")
    return records


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-lock", type=Path, default=HERE / "r7_sources.json")
    ap.add_argument("--out", type=Path, default=ROOT)
    ap.add_argument("--cells", nargs="+", choices=CELLS)
    ap.add_argument("--variants", nargs="+", choices=list(COUNTS))
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args(argv)
    records = prefit(json.loads(a.source_lock.read_text()), a.out, a.cells, a.variants, not a.rebuild)
    print(json.dumps(dict(artifacts=len(records), report=str(a.out / "prefit_report.json"))))


if __name__ == "__main__":
    main()
