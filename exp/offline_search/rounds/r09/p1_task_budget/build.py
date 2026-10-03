"""Freeze discovery ranking, emit 12 standard arms, and prefit task switches.

Run locally on CPU. This module never contacts h100 or launches experiments.
An existing freeze must match exactly; table/source drift is refused.
"""
from __future__ import annotations

import copy
import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml

from exp.offline_search.closed_loop.ops.emit_arms import main as emit
from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.harness import api
from exp.offline_search.rounds.r09.explore_fable.tools import paired, task_alloc

from .common import CELLS, DERIVED, FABLE, HERE, R8, RUN, SPEC, fit_path, output, r8_name, sha, source, write_json
from .methods import TaskBudgetController


def immutable_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("frozen artifact drift refused: " + str(path))
    if not path.exists():
        write_json(path, value)


def freeze_tables():
    table = dict(schema="r9.p1.task_budget.v1", signal="gap_all", discovery_inits=list(range(30)),
                 task_ids=list(range(10)), k=3, rho_cu=.3,
                 selection="P1 exact confirmation step 2: top-3 for both treatments in all three cells",
                 deviation="none; top-2 Spatial CU and top-4 pi05 L10 gripper P10 are other proposal operating points",
                 label_free=True, tie_break="descending score then ascending task ID (fable stable task order)",
                 provenance={str(p): sha(p) for p in (
                     FABLE / "PROPOSALS.md", FABLE / "DATA_ANALYSIS.md", FABLE / "tools/task_alloc.py",
                     FABLE / "out/task_alloc/rules.json")}, cells={})
    for cell in CELLS:
        path = FABLE / "out/shadow" / ("episodes_" + r8_name(cell, "A") + ".parquet")
        # Read only discovery rows and label-free columns. task_scores also
        # returns a_fail; supply a constant sentinel and discard that column.
        epi = pd.read_parquet(path, columns=["task_id", "init", "gap_all"],
                              filters=[("init", ">=", 0), ("init", "<", 30)])
        expected = {(t, i) for t in range(10) for i in range(30)}
        if len(epi) != 300 or set(zip(epi.task_id, epi.init)) != expected or not np.isfinite(epi.gap_all).all():
            raise ValueError("incomplete/nonfinite/duplicate discovery shadow scores: " + cell)
        epi["success"] = False  # never read actual labels for ranking
        scores = task_alloc.task_scores(epi, range(30), signals=("gap_all",))["gap_all"]
        order = scores.sort_values(ascending=False, kind="stable").index.astype(int).tolist()
        hard = order[:3]
        assignments = task_alloc.rank_allocation(scores, 3, "CU", "A")
        assert {t for t, v in assignments.items() if v == "CU"} == set(hard)
        table["cells"][cell] = dict(shadow_source=str(path), shadow_sha256=sha(path),
            counts={str(t): 30 for t in range(10)}, scores={str(t): float(scores[t]) for t in range(10)},
            order=order, hard_tasks=hard,
            topk_cu={str(t): .3 if t in hard else 0. for t in range(10)},
            topk_p10={str(t): 1. if t in hard else 0. for t in range(10)})
    immutable_json(HERE / "task_tables.json", table)
    return table


def freeze_sources():
    all_rows = {r["arm"]: r for r in json.loads((R8 / "arms.json").read_text())}
    lock = dict(schema="r9.p1.sources.v1", source_run=str(R8), arms_json_sha256=sha(R8 / "arms.json"), cells={})
    for cell in CELLS:
        lock["cells"][cell] = {}
        for variant in ("A", "CU", "P10"):
            row = all_rows[r8_name(cell, variant)]
            src = source(row)
            with fit_path(row).open("rb") as f:
                blob = pickle.load(f)
            if any(blob.get(k) != src[k] for k in ("spec", "kwargs", "cell")):
                raise ValueError("R8 fit metadata mismatch: " + row["arm"])
            for path, expected in row.get("r8", {}).get("source", {}).get("dependencies", {}).items():
                if sha(path) != expected:
                    raise ValueError("R8 frozen dependency SHA mismatch: " + path)
            if variant == "A" and src["sha256"] != row["r8"]["source"]["sha256"]:
                raise ValueError("R8 A frozen SHA mismatch")
            lock["cells"][cell][variant] = dict(source=src, row=copy.deepcopy(row),
                config_sha256=sha(row["yaml"]), matrix_sha256=sha(row["matrix"]))
    immutable_json(HERE / "source_lock.json", lock)
    return lock


def make_specs(table, lock):
    specs = []
    manifest = str(RUN / "manifests/discovery300.json")
    for cell in CELLS:
        frozen = lock["cells"][cell]
        for variant in ("A", "CU", "topk_cu", "topk_p10"):
            old = frozen["A" if variant == "A" else "CU"]["row"]
            row = {k: copy.deepcopy(old[k]) for k in (
                "model", "mode", "method", "kwargs", "plugin_args", "client_overrides", "cost_ledger",
                "yaml_patch", "full_model", "server_env") if k in old}
            row.update(name=f"r9p1_{cell}_{variant}", suite=old["suite_short"], manifest=manifest)
            if variant.startswith("topk"):
                high = "CU" if variant == "topk_cu" else "P10"
                key = "CU:0.3" if high == "CU" else "P10"
                row["method"] = SPEC
                row["kwargs"] = dict(task_rates=table["cells"][cell][variant],
                    sources={"A": frozen["A"]["source"], key: frozen[high]["source"]},
                    policy_tasks=table["cells"][cell]["hard_tasks"] if high == "P10" else [])
                idx = row["plugin_args"].index("--os-fit-artifact") + 1
                row["plugin_args"][idx] = str(RUN / "fits" / (row["name"] + ".pkl"))
            if any(x.startswith(("--os-debug", "--os-oracle", "--trace", "--os-log-inputs")) for x in row["plugin_args"]):
                raise ValueError("capture/oracle flags forbidden")
            specs.append(row)
    return specs


def prefit(rows, lock, rebuild=False):
    records = []
    for row in rows:
        path = fit_path(row)
        expected = dict(spec=row["method"], kwargs=row["kwargs"], cell=row["cell"])
        if row["method"] == SPEC:
            provenance = dict(task_tables_sha256=sha(HERE / "task_tables.json"),
                              controller_sha256=sha(HERE / "methods.py"),
                              sources={k: s["sha256"] for k, s in row["kwargs"]["sources"].items()},
                              no_refit=True)
            if path.exists():
                with path.open("rb") as f:
                    blob = pickle.load(f)
                if any(blob.get(k) != v for k, v in expected.items()):
                    raise ValueError("existing wrapper prefit configuration drift: " + str(path))
                if not rebuild and blob.get("provenance") != provenance:
                    raise ValueError("existing wrapper prefit drift: " + str(path))
                status = "EXISTING_VERIFIED"
            if not path.exists() or rebuild:
                method = TaskBudgetController(**row["kwargs"])
                method.prof = api.NULL_PROFILER
                method.fit(None, SimpleNamespace(cell=row["cell"]))
                api.check_method_attrs(method)
                blob = dict(expected, method=method, registered={}, fit_s=0., provenance=provenance)
                tmp = output(path.with_suffix(".pkl.part"))
                with tmp.open("wb") as f:
                    pickle.dump(blob, f, protocol=4)
                tmp.replace(output(path))
                status = "BUILT_FROM_FROZEN_FITS"
        else:
            status = "REUSED_BY_SHA"
            with path.open("rb") as f:
                blob = pickle.load(f)
        if any(blob.get(k) != v for k, v in expected.items()):
            raise ValueError("prefit metadata mismatch: " + row["arm"])
        api.check_method_attrs(blob["method"])
        _, clone_mode = clone_method(blob["method"], strict=True)
        records.append(dict(arm=row["arm"], artifact=str(path), sha256=sha(path), bytes=path.stat().st_size,
                            status=status, clone_mode=clone_mode))
    write_json(RUN / "prefit_report.json", records)
    return records


def predictions(table):
    rules = json.loads((FABLE / "out/task_alloc/rules.json").read_text())
    report = dict(schema="r9.p1.predictions.v1", n=300, owner_prices={
        "pi05": {"look": .152, "call": .848}, "groot": {"look": .148, "call": .852}},
        caveat="fixed-table simulation uses the same discovery observations as ranking; CV refits ranks on other folds",
        arms={})
    for cell in CELLS:
        frames = []
        sources = {}
        for v in ("A", "CU", "P10"):
            path = DERIVED / "episodes" / (r8_name(cell, v) + ".parquet")
            df = pd.read_parquet(path, filters=[("init", ">=", 0), ("init", "<", 30)])
            if len(df) != 300 or df[["task_id", "init"]].duplicated().any():
                raise ValueError("incomplete discovery dose data")
            frames.append(df)
            sources[str(path)] = sha(path)
        model, suite, size = cell.split("_")
        matrix = paired.cell_matrix(pd.concat(frames), model, suite, int(size))
        for v in ("A", "CU", "topk_cu", "topk_p10"):
            high = "CU" if v == "topk_cu" else "P10"
            assign = pd.Series([v if v in ("A", "CU") else high if t in table["cells"][cell]["hard_tasks"] else "A"
                                for t in matrix.index.get_level_values("task_id")], index=matrix.index)
            sim = paired.summarize(matrix, assign, n_boot=0)
            cv = None if v in ("A", "CU") else next(r for r in rules[cell]["rules"]
                if r["signal"] == "gap_all" and r["rule"] == dict(k=3, high=high, low="A"))
            report["arms"][f"r9p1_{cell}_{v}"] = dict(fixed_table_simulator=sim, fable_cross_validated=cv,
                                                      discovery_sources=sources)
    write_json(HERE / "predictions.json", report)
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rebuild-wrappers", action="store_true", help="rebuild only owned wrappers after controller source edits")
    args = ap.parse_args()
    table, lock = freeze_tables(), freeze_sources()
    manifest = [[t, i] for t in range(10) for i in range(30)]
    write_json(RUN / "manifests/discovery300.json", manifest)
    write_json(RUN / "discovery300.json", manifest)
    specs = make_specs(table, lock)
    write_json(HERE / "arms_in.json", specs)
    emit(["--run-root", str(RUN), "--spec", str(HERE / "arms_in.json")])
    rows = json.loads((RUN / "arms.json").read_text())
    if len(rows) != 12:
        raise ValueError("run root must contain exactly the 12 P1 arms")
    # Preserve the R8 serving config bytes, including historical retrieval paths.
    for row in rows:
        cell = row["arm"].removeprefix("r9p1_")
        cell = next(c for c in CELLS if cell.startswith(c + "_"))
        source_yaml = lock["cells"][cell]["A"]["row"]["yaml"]
        data = Path(source_yaml).read_bytes()
        config = yaml.safe_load(data)
        if config.get("trace") or config.get("debug"):
            raise ValueError("R8 config contains capture flags")
        output(row["yaml"]).write_bytes(data)
        row["yaml_sha256"] = sha(row["yaml"])
        row["p1"] = dict(cell=cell, variant=row["arm"][len("r9p1_" + cell + "_"):],
                         task_tables_sha256=sha(HERE / "task_tables.json"), k=3, debug_required=False)
    write_json(RUN / "arms.json", rows)
    records = prefit(rows, lock, rebuild=args.rebuild_wrappers)
    pred = predictions(table)
    print("P1_BUILD_OK arms=12 pairs=300 reused=6 wrappers=6 capture=off")
    for cell in CELLS:
        print(f"TASKS {cell} order={table['cells'][cell]['order']} hard={table['cells'][cell]['hard_tasks']}")
    for arm, p in pred["arms"].items():
        s = p["fixed_table_simulator"]
        cv = p["fable_cross_validated"]
        print(f"PREDICT {arm} frozen_sr={s['sr']:.6f} frozen_ir={s['ir']:.6f}" +
              (f" cv_sr={cv['sr']:.6f} cv_ir={cv['ir']:.6f}" if cv else ""))
    for r in records:
        print(f"FIT {r['arm']} {r['status']} sha256={r['sha256']}")


if __name__ == "__main__":
    main()
