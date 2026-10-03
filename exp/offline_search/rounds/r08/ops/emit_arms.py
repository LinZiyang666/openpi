"""Emit SELECTION's 70 arms and a content-hashed R7 source lock.

The ordinary closed-loop emitter is reused for YAMLs/matrices. The coordinator
must launch these through debug.ops.chain_debug, which supplies per-port capture
directories and verifies capture before DONE. This script starts no services.
"""
import argparse
import copy
import json
import os
from pathlib import Path

from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import sha, sources

HERE = Path(__file__).resolve().parent
R7 = Path("/home/weiland/trace_runs/os_closed_loop/r07_main")
STORE = "/home/weiland/trace_runs/offline_search_store"
SPEC = "exp.offline_search.rounds.r08.methods.methods:"
CALL_SEED = 26093005  # shared CU/CT coin stream; distinct from R7's 26092903/rep2
CELLS = ["{}_{}_{}".format(m, s, n) for m in ("groot", "pi05")
         for s in ("l10", "spatial") for n in (50, 500)]
COUNTS = dict(P10=4, A=8, CU=8, CT=8, SF1=8, SW=4, FL=8, W10=4,
              W5=4, IP=4, SHIFT=2, A5=2, O5a=2, O5b=4)


def writable_output(path):
    p = Path(path).resolve()
    roots = [Path("/tmp/r8_S5"), HERE]
    # Coordinator campaign roots are allowed only when named explicitly.
    extra = os.environ.get("R8_EMIT_ROOT")
    if extra:
        roots.append(Path(extra).resolve())
    if not any(p == r or r in p.parents for r in roots):
        raise ValueError("S5 output must stay in rounds/r08/ops, /tmp/r8_S5 or $R8_EMIT_ROOT")
    return p


def discover_sources(r7_root=R7):
    r7 = Path(r7_root)
    rows = {r["arm"]: r for r in json.loads((r7 / "arms.json").read_text())}
    bases = sources()
    cells = {}
    for cell in CELLS:
        model, suite, size = cell.split("_")
        base = bases[cell]
        source_rows = dict(A=dict(method=base["method"], kwargs=base["kwargs"],
                                  artifact=base["source_artifact"]))
        for variant in ("CU", "CT", "SF1") + (("SW",) if model == "pi05" else ()):
            name = ("r7_sw_pi05_{}_{}".format("sp" if suite == "spatial" else suite, size)
                    if variant == "SW" else "r7_{}_{}".format(cell, variant +
                        ("30" if size == "50" else "18") if variant in ("CU", "CT") else variant))
            row = rows[name]
            artifact = r7 / "fits" / (name + ".pkl")
            source_rows[variant] = dict(method=row["method"], kwargs=row["kwargs"], artifact=str(artifact))
        for variant, row in source_rows.items():
            row["sha256"] = sha(row["artifact"])
            # R7 sidecars pin the published fit, where one exists.
            sidecar = Path(row["artifact"]).with_suffix(".json")
            if sidecar.is_file():
                expected = json.loads(sidecar.read_text()).get("sha256")
                if expected and expected != row["sha256"]:
                    raise ValueError("R7 fit differs from published sha: " + str(row["artifact"]))
            dependencies = {}
            for key in ("base_fit", "wrist_fit", "stage_fit", "calibration_path", "stall_model_path"):
                if not row["kwargs"].get(key):
                    continue
                path = Path(row["kwargs"][key])
                candidates = sorted(path.rglob("*")) if path.is_dir() else [path]
                if key == "calibration_path":
                    candidates = sorted(path.parent.rglob("*"))
                for p in candidates:
                    if p.is_file():
                        dependencies[str(p)] = sha(p)
            row["dependencies"] = dependencies
        cells[cell] = source_rows
    manifests = {}
    for label in ("eval500", "groot_l10_bval20", "groot_spatial_bval20", "pi05_l10_bval20", "pi05_spatial_bval20"):
        path = r7 / "manifests" / (label + ".json")
        manifests[label] = dict(path=str(path), sha256=sha(path))
        if label == "eval500":
            pairs = json.loads(path.read_text())["selected"]
            if len(pairs) != 500 or {(int(r["task"]), int(r["init"])) for r in pairs} != {
                    (task, init) for task in range(10) for init in range(50)}:
                raise ValueError("R7 official manifest is not the selected 10 x 50 test pairs")
    return dict(schema="r8.s5.sources.v1", r7_root=str(r7), cells=cells, manifests=manifests)


def verify_source(source, dependencies=True):
    if sha(source["artifact"]) != source["sha256"]:
        raise ValueError("frozen source sha mismatch: " + source["artifact"])
    if dependencies:
        for path, digest in source.get("dependencies", {}).items():
            if sha(path) != digest:
                raise ValueError("frozen dependency sha mismatch: " + path)


def make_specs(lock, fit_root="<RUN>/fits", run_root="<RUN>"):
    result = []
    for cell in CELLS:
        model, suite, size = cell.split("_")
        frozen = lock["cells"][cell]
        base = frozen["A"]
        variants = ["A", "CU", "CT", "SF1", "FL"]
        if size == "50":
            variants.append("IP")
            if suite == "l10":
                variants += ["P10", "A5", "O5a"]
            else:
                variants.append("P10")
            variants.append("O5b")
        if model == "pi05":
            variants += ["SW", "W10", "W5"]
        if model == "groot" and suite == "l10":
            variants.append("SHIFT")
        for variant in variants:
            name = "r8_{}_{}".format(cell, variant) if variant != "P10" else "r8_{}_{}_P10".format(model, suite)
            source = frozen.get(variant, frozen["SW"] if variant in ("W5", "W10") else
                                frozen["SF1"] if variant == "FL" else base)
            kwargs = copy.deepcopy(source["kwargs"])
            method = source["method"]
            fitted_path = source["artifact"] if variant in ("A", "SF1", "SW") else str(fit_root) + "/" + name + ".pkl"
            if variant in ("CU", "CT"):
                kwargs["random_seed"] = CALL_SEED
            elif variant == "FL":
                kwargs = dict(base["kwargs"], random_seed=CALL_SEED, coin_domain="R8/follow/" + cell)
                method = SPEC + "FollowLottery"
            elif variant in ("W10", "W5"):
                kwargs["every_controls"] = int(variant[1:])
                method = SPEC + "WristEveryLook"
            elif variant in ("IP", "P10", "O5a", "O5b"):
                kwargs = dict(base_kwargs=base["kwargs"], base_fit=base["artifact"], random_seed=CALL_SEED,
                              coin_domain="R8/call/" + cell)
                method = SPEC + ("IdentificationProbe" if variant == "IP" else "PolicyEveryTen" if variant == "P10" else "OracleGraspCalls")
                if variant.startswith("O5"):
                    kwargs["tight"] = variant == "O5b"
            elif variant == "A5":
                kwargs = dict(base["kwargs"], budget=0)
                method = SPEC + "EveryFiveAWM"
            elif variant == "SHIFT":
                kwargs = dict(base["kwargs"])
                method = SPEC + "ShiftedAWM"
            mixed = variant in ("CU", "CT", "IP", "P10", "O5a", "O5b")
            camera = variant in ("SW", "W5", "W10")
            args = ["--os-root", STORE, "--os-no-shadow-native", "--os-blind"]
            if mixed:
                args += ["--os-policy-tail", "--os-policy-tail-blocks", "1", "--os-judge", "guard_only"]
            if camera:
                args += ["--os-tokens", "off", "--os-request-cameras"]
            if variant.startswith("O5"):
                args += ["--os-oracle"]
            args += ["--os-fit-artifact", fitted_path]
            priority = ("P2" if variant in ("SHIFT", "A5") else "P0" if variant in
                        ("P10", "A", "CU", "CT") or variant == "FL" and suite == "l10" else "P1")
            manifest = lock.get("manifests", {}).get("eval500", {}).get("path", str(run_root) + "/manifests/eval500.json")
            row = dict(name=name, model=model, suite=suite, mode="plugin", method=method, kwargs=kwargs,
                cost_ledger=True, manifest=manifest,
                client_overrides=dict(replan_steps=5, resize_size=224 if model == "pi05" else 256),
                plugin_args=args,
                r8=dict(variant=variant, priority=priority, library_size=int(size) if variant != "P10" else None,
                    library=("current" if size == "50" else "bpool_cs" if model == "pi05" else "bpool_all") if variant != "P10" else None,
                    rho=.30 if size == "50" else .18, debug_required=True, schema="osdebug.v1", env_seed=7,
                    diagnostic_oracle=variant.startswith("O5"), oracle_client=variant.startswith("O5"),
                    source=copy.deepcopy(source), source_variant=variant if variant in frozen else
                        "SW" if camera else "SF1" if variant == "FL" else "A",
                    new_random_seed=CALL_SEED if variant in ("CU", "CT", "FL", "IP") else None,
                    augmentation=["policy_shadow", "policy_draws", "shadow_look"] + (["camera_shadow"] if model == "pi05" else [])))
            if mixed or camera:
                row["full_model"] = True
            if camera:
                row["server_env"] = dict(BATCHING_MAX_BATCH_SIZE="1")
            result.append(row)
    result.sort(key=lambda r: (r["r8"]["priority"], r["model"] != "groot", r["suite"] != "l10", r["name"]))
    counts = {v: sum(r["r8"]["variant"] == v for r in result) for v in COUNTS}
    if counts != COUNTS or len({r["name"] for r in result}) != 70:
        raise AssertionError("R8 selection coverage mismatch: " + repr(counts))
    return result


def smoke_specs(rows, lock):
    """Use the frozen non-test B-val pairs and distinct arm/attempt identities."""
    result = copy.deepcopy(rows)
    for row in result:
        key = "{}_{}_bval20".format(row["model"], row["suite"])
        source = lock["manifests"][key]
        if sha(source["path"]) != source["sha256"]:
            raise ValueError("frozen non-test manifest sha mismatch: " + source["path"])
        manifest = json.loads(Path(source["path"]).read_text())
        if manifest.get("role") != "NONTEST_BVAL":
            raise ValueError("smoke requires non-test manifests")
        row["name"] += "_smoke"
        row["manifest"] = source["path"]
        row["r8"]["smoke"] = True
        row["r8"]["manifest_sha256"] = source["sha256"]
    return result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-lock", type=Path, default=HERE / "r7_sources.json")
    ap.add_argument("--refresh-source-lock", action="store_true")
    ap.add_argument("--r7-root", type=Path, default=R7)
    ap.add_argument("--fit-root", help="defaults to <run-root>/fits")
    ap.add_argument("--run-root", default="<RUN>")
    ap.add_argument("--out", type=Path, default=HERE / "arms_in.json")
    ap.add_argument("--emit-config", action="store_true", help="also write ordinary arm YAMLs/matrices to --run-root")
    ap.add_argument("--smoke", action="store_true", help="non-test frozen R7 B-val pairs, separate _smoke arm identities")
    ap.add_argument("--variants", nargs="+", choices=list(COUNTS))
    a = ap.parse_args(argv)
    if a.refresh_source_lock:
        lock = discover_sources(a.r7_root)
        p = writable_output(a.source_lock)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(lock, indent=2, allow_nan=False) + "\n")
    else:
        lock = json.loads(a.source_lock.read_text())
    rows = make_specs(lock, a.fit_root or str(a.run_root) + "/fits", a.run_root)
    if a.smoke:
        rows = smoke_specs(rows, lock)
    if a.variants:
        rows = [r for r in rows if r["r8"]["variant"] in a.variants]
    if lock.get("manifests") and not a.smoke:
        official = lock["manifests"]["eval500"]
        if sha(official["path"]) != official["sha256"]:
            raise ValueError("frozen official manifest sha mismatch")
    out = writable_output(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    if a.emit_config:
        run = writable_output(a.run_root)
        from exp.offline_search.closed_loop.ops.emit_arms import main as emit
        emit(["--run-root", str(run), "--spec", str(out)])
        path = run / "arms.json"
        metadata = {r["name"]: r["r8"] for r in rows}
        emitted = json.loads(path.read_text())
        for r in emitted:
            if r["arm"] in metadata:
                r["r8"] = metadata[r["arm"]]
        path.write_text(json.dumps(emitted, indent=2) + "\n")
    episodes = sum(len(json.loads(Path(r["manifest"]).read_text())["selected"]) for r in rows) if a.smoke else len(rows) * 500
    print(json.dumps(dict(arms=len(rows), episodes=episodes, priorities={p: sum(r["r8"]["priority"] == p for r in rows)
        for p in ("P0", "P1", "P2")}, call_seed=CALL_SEED, out=str(out))))


if __name__ == "__main__":
    main()
