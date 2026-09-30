"""Library-only frozen-A adapter fits and emit_arms input generation."""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import sources, load_base, sha, fingerprint
from exp.offline_search.rounds.r07.stages.stages import StageTable
from .methods import StageFollow, UniformFollow

HERE = Path(__file__).resolve().parent
OUT = Path("/tmp/r7_C1")
ROOT = Path("/home/weiland/trace_runs/offline_search_store")
VARIANTS = {"SF1": (StageFollow, 1, True, True), "SF2": (StageFollow, 2, True, True),
            "UF1": (UniformFollow, 1, False, False)}


def kwargs_for(source, variant, stages_path=None):
    cls, cap, stage, valve = VARIANTS[variant]
    kw = dict(source["kwargs"])
    if kw["lib"] == "big":
        kw["lib"] = "bpool_cs" if source["cell"].startswith("pi05_") else "bpool_all"
    result = dict(kw, extend_blocks=cap, stage_gate=stage, state_valve=valve)
    if stages_path is not None:
        result['stages_path'] = str(stages_path)
    return result


def adapt(base, kwargs, cls=StageFollow):
    obj = cls(**kwargs)
    own = {k: v for k, v in obj.__dict__.items() if k.startswith("follow_") or k in ("name", "_follow_plan")}
    obj.__dict__.update(base.__dict__)
    obj.__dict__.update(own)
    obj.prof = api.NULL_PROFILER
    obj.invalidate_anchor()
    return obj


def emit_specs():
    rows = []
    for cell, source in sources().items():
        model, suite, size = cell.split("_")
        for variant, (cls, *_rest) in VARIANTS.items():
            name = f"r7_{cell}_{variant}"
            rows.append(dict(name=name, model=model, suite=suite, mode="plugin", cost_ledger=True,
                method=f"exp.offline_search.rounds.r07.c1_follow.methods:{cls.__name__}",
                kwargs=kwargs_for(source, variant),
                client_overrides=dict(replan_steps=5, resize_size=224 if model == "pi05" else 256),
                plugin_args=["--os-root", str(ROOT), "--os-no-shadow-native", "--os-blind",
                             "--os-fit-artifact", f"<RUN>/fits/{name}.pkl"]))
    (HERE / "arms_eval500.json").write_text(json.dumps(rows, indent=2) + "\n")
    profile = [{**r, "name": r["name"] + "_profile", "plugin_args": r["plugin_args"] + ["--os-log-inputs"],
                "manifest": f"<RUN>/manifests/{r['model']}_{r['suite']}_bval20.json"} for r in rows]
    (HERE / "arms_profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps(dict(profile=len(profile), eval500=len(rows))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", default=["all"])
    ap.add_argument("--emit-specs", action="store_true")
    ap.add_argument("--reuse-stages", action="store_true", help="reuse verified frozen stage artifacts")
    ap.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    args = ap.parse_args()
    if args.emit_specs:
        emit_specs(); return
    OUT.mkdir(parents=True, exist_ok=True)
    for cell, source in sources().items():
        if args.cells != ["all"] and cell not in args.cells:
            continue
        start = time.perf_counter()
        base, blob = load_base(source)
        model, suite, _size = cell.split("_")
        lib = store.LibraryView(ROOT, f"{model}_{suite}", base.cand_name)
        ctx = api.Context(root=ROOT, cell=f"{model}_{suite}_cache", seed=0, scratch=OUT/"scratch"/cell)
        ctx.scratch.mkdir(parents=True, exist_ok=True)
        manifest = json.loads((lib.dir / "manifest.json").read_text())
        stage_path = OUT/"stages"/(cell + ".pkl")
        if args.reuse_stages:
            table = StageTable.load(stage_path, library=lib, manifest=manifest)
        else:
            table = StageTable.fit(lib, manifest={**manifest, "_retrieval": base})
            table.save(stage_path)
        records = []
        for variant, (cls, *_rest) in VARIANTS.items():
            if variant not in args.variants:
                continue
            kw = kwargs_for(source, variant, stage_path)
            method = adapt(base, kw, cls)
            method.finish_follow_fit(lib, ctx)
            assert fingerprint(method) == fingerprint(base)
            api.check_method_attrs(method)
            spec = f"exp.offline_search.rounds.r07.c1_follow.methods:{cls.__name__}"
            path = OUT/"fits"/f"r7_{cell}_{variant}.pkl"
            path.parent.mkdir(parents=True, exist_ok=True)
            artifact = dict(method=method, registered=blob.get("registered", {}), spec=spec,
                            kwargs=kwargs_for(source, variant), cell=ctx.cell,
                            fit_s=time.perf_counter()-start,
                            provenance=dict(source=source, source_sha256=sha(source["source_artifact"]),
                                            stage_sha256=sha(stage_path), library_fingerprint=table.fingerprint,
                                            retrieval_fingerprint=table.retrieval_fingerprint, library_only=True,
                                            method_source_sha256=sha(HERE/'methods.py'),
                                            stage_source_sha256=sha(HERE.parent/'stages/stages.py')))
            with path.open("wb") as f:
                pickle.dump(artifact, f, protocol=4)
            records.append(dict(variant=variant, path=str(path), sha256=sha(path), bytes=path.stat().st_size))
        note = dict(cell=cell, stage=str(stage_path), stage_sha256=sha(stage_path),
                    fit_seconds=time.perf_counter()-start, calibration=table.calibration,
                    successful_rows=int(table.success.sum()), rows=len(table.mode),
                    event_occupancy=table.event_occupancy, fits=records)
        note_path = OUT/(cell + "_fit.json")
        if note_path.exists() and set(args.variants) != set(VARIANTS):
            old = json.loads(note_path.read_text())
            if old['stage_sha256'] != note['stage_sha256']:
                raise ValueError('selective rebuild cannot mix different stage tables')
            merged = {r['variant']: r for r in old['fits']}
            merged.update({r['variant']: r for r in records})
            note['fits'] = [merged[v] for v in VARIANTS if v in merged]
        note_path.write_text(json.dumps(note, indent=2)+"\n")
        print(json.dumps(note), flush=True)


if __name__ == "__main__":
    main()
