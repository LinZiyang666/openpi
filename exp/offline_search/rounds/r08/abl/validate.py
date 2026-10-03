"""Validate frozen source specs, fresh fitted state, configs, and PAPER_AB pairing."""
import argparse
import copy
import json
import math
from pathlib import Path
import pickle
import re

import numpy as np
import yaml

from exp.offline_search.closed_loop import plugin
from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest
from exp.offline_search.rounds.r08.abl.make_arms import HERE, RUN, RUNS, artifact, sha, write_json
import openpi.cache.config as cc


def json_diff(a, b, path=""):
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for key in sorted(a.keys() | b.keys()):
            child = f"{path}/{key}"
            if key not in a or key not in b:
                out.append(dict(path=child, source=a.get(key), arm=b.get(key)))
            else:
                out.extend(json_diff(a[key], b[key], child))
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return sum((json_diff(x, y, f"{path}/{i}") for i, (x, y) in enumerate(zip(a, b))), [])
    return [] if a == b else [dict(path=path, source=a, arm=b)]


def assert_source_fields(row, prov):
    src = prov["source_row"]
    restored = copy.deepcopy(row)
    restored["name"], restored["method"] = src["name"], src["method"]
    assert restored["kwargs"].pop("disabled_guards") == prov["disabled_guards"]
    args = restored["plugin_args"]
    args[args.index("--os-fit-artifact") + 1] = artifact(src)
    # Includes absent keys, list order, and serialized field order.
    assert json.dumps(restored) == json.dumps(src), row["name"]
    return dict(arm=row["name"], differences=json_diff(src, row), non_guard_differences=[],
                restored_serialization_exact=True)


def state_differences(a, b, path="method", seen=None):
    """Compare all fitted arrays, thresholds and nested state; exclude root switch/name only."""
    if seen is None:
        seen = set()
    pair = (id(a), id(b))
    if pair in seen:
        return []
    seen.add(pair)
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        same = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
        return [] if same else [path]
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return [path + "/keys"]
        return sum((state_differences(a[k], b[k], f"{path}/{k}", seen) for k in a), [])
    if isinstance(a, (list, tuple)) and isinstance(b, type(a)):
        if len(a) != len(b):
            return [path + "/length"]
        return sum((state_differences(x, y, f"{path}/{i}", seen) for i, (x, y) in enumerate(zip(a, b))), [])
    if hasattr(a, "__dict__") and hasattr(b, "__dict__"):
        if path != "method" and type(a) is not type(b):
            return [path + "/type"]
        av, bv = dict(vars(a)), dict(vars(b))
        if path == "method":
            for key in ("name", "disabled_guard", "disabled_guards", "disabled_mask"):
                av.pop(key, None)
                bv.pop(key, None)
        return state_differences(av, bv, path + "/state", seen)
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return []
    return [] if type(a) is type(b) and a == b else [path]


def reference_pairs(spec):
    run, name = spec.split(":")
    path = RUNS / run / "runs" / name / "client/journal.jsonl"
    terminal = {}
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            match = re.fullmatch(re.escape(name) + r":eval:(\d+):(\d+)", row.get("task_uid", ""))
            if (match and "success" in row and not row.get("error")
                    and row.get("accepted") and row.get("status") in ("done", "failed")):
                terminal[tuple(map(int, match.groups()))] = bool(row["success"])
    expected = {(t, i) for t in range(10) for i in range(50)}
    assert set(terminal) == expected, (spec, len(terminal), expected - terminal.keys())
    return dict(reference=spec, journal=str(path), journal_sha256=sha(path), pairs=len(terminal),
                successes=sum(terminal.values()))


def write_arm_table(rows, provenance, fits):
    provs = {p["arm"]: p for p in provenance}
    fitmap = {f["arm"]: f for f in fits}
    text = ["# r08_abl supplementary guard ablations", "",
        "Standard mode; no debug capture. Every arm uses the official 500 test pairs (tasks 0–9 × inits 0–49).",
        "Each arm is paired separately against all three A and all three B references in its cell, as in R6 PAPER_AB.md.",
        "Only name, method, disabled_guards and the fit path change from the frozen deployed B input row.",
        "Fits are fresh through the same plugin prefit entry point as R6 P2. Fit SHAs identify the complete pickle;",
        "the source B SHAs and exact fitted-state comparisons are in the run's validation/validation.json.", "",
        "| Arm | B source row (replicate 1) | Disabled guards | Fit SHA256 | A×3 / B×3 references |",
        "|---|---|---|---|---|"]
    for row in rows:
        p, f = provs[row["name"]], fitmap[row["name"]]
        cell = f"{p['model']}_{p['suite']}_{p['scale']}"
        text.append(f"| `{row['name']}` | `{Path(p['source']).parent.name}:{p['source_name']}` | "
                    f"{', '.join(p['disabled_guards'])} | `{f['sha256']}` | `{cell}` (below) |")
    text += ["", "References are the exact run:arm names from r06/ops/paper_ab.py (the generator of PAPER_AB.md).", "",
             "| Cell | A replicates 1 / 2 / 3 | B replicates 1 / 2 / 3 |", "|---|---|---|"]
    seen = set()
    for p in provenance:
        cell = f"{p['model']}_{p['suite']}_{p['scale']}"
        if cell in seen:
            continue
        seen.add(cell)
        refs = p["references"]
        text.append(f"| `{cell}` | " + "<br>".join(f"`{r}`" for r in refs["A"]) + " | "
                    + "<br>".join(f"`{r}`" for r in refs["B"]) + " |")
    text += ["", "Source-row locations: π0.5 = `/home/weiland/trace_runs/os_closed_loop/r05_q1/arms_in.json`;",
             "GR00T = `/home/weiland/trace_runs/os_closed_loop/r06_paper/arms_in.json`.",
             "Run root: `/home/weiland/trace_runs/os_closed_loop/r08_abl`.",
             "The frozen source rows/deployed rows and hashes are in `provenance.json`; the permitted-field diff is",
             "in `validation/source_diff.json`. No manifest field was added to the B specs; the coordinator should",
             "use `manifests/eval500.json` for an explicit selection, or the harness's full official test default.", ""]
    (HERE / "ARMS.md").write_text("\n".join(text))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args(argv)
    rows = json.loads((args.run / "arms_in.json").read_text())
    provenance = json.loads((args.run / "provenance.json").read_text())
    provs = {p["arm"]: p for p in provenance}
    # Other supplementary tasks can append their own arms in the shared root.
    own_names = {r["name"] for r in rows}
    emitted = [r for r in json.loads((args.run / "arms.json").read_text()) if r["arm"] in own_names]
    emitted_by_name = {r["arm"]: r for r in emitted}
    assert len(rows) == len(provs) == len(emitted_by_name) == len(emitted) == 24
    assert sum(len(r["kwargs"]["disabled_guards"]) == 1 for r in rows) == 16
    assert sum(r["kwargs"]["disabled_guards"] == ["stuck", "terminal", "overtime"] for r in rows) == 8
    manifest = load_manifest(args.run / "manifests/eval500.json")
    assert set(manifest["selected"]) == {(t, i) for t in range(10) for i in range(50)}
    diffs, fits, refs, source_blobs = [], [], {}, {}
    for row in rows:
        name = row["name"]
        p, e = provs[name], emitted_by_name[name]
        assert sha(p["source"]) == p["source_file_sha256"]
        assert sha(p["source_artifact"]) == p["source_fit_sha256"]
        diffs.append(assert_source_fields(row, p))
        for field in ("method", "kwargs", "plugin_args", "full_model", "cost_ledger", "client_overrides"):
            assert e.get(field) == row.get(field), (name, field)
        opts, rest = plugin.parse_cli(["--os-method", e["method"], "--os-kwargs", json.dumps(e["kwargs"]),
            "--os-cell", e["cell"], "--os-log-dir", str(args.run / "validation/parser"), *e["plugin_args"]])
        assert not rest and opts.os_blind and opts.os_no_shadow_native and opts.os_policy_tail
        assert opts.judge.mode == "guard_only" and e["full_model"] and e["replan_steps"] == 5
        # pi05 B leaves this CLI option absent; PluginRuntime supplies one.
        assert opts.os_policy_tail_blocks in (None, 1)
        if row["model"] == "groot":
            assert opts.os_policy_tail_blocks == 1
            assert e["client_overrides"]["resize_size"] == 256
        cc.load_cache_config(e["yaml"])
        config = yaml.safe_load(Path(e["yaml"]).read_text())
        source_config = yaml.safe_load(Path(e["src_yaml"]).read_text())
        source_config.pop("trace", None)
        assert config == source_config and "trace" not in config
        assert not any("debug" in x for x in e["plugin_args"])
        assert sha(e["yaml"]) == e["yaml_sha256"] and sha(e["src_yaml"]) == e["src_sha256"]
        matrix = yaml.safe_load(Path(e["matrix"]).read_text())
        assert matrix == {"arms": [dict(arm=name, yaml=e["remote_yaml"], suite=e["suite"],
                                        client_overrides=e["client_overrides"])]}
        path = Path(opts.os_fit_artifact)
        with path.open("rb") as stream:
            blob = pickle.load(stream)
        assert {k: blob[k] for k in ("spec", "kwargs", "cell")} == dict(
            spec=opts.os_method, kwargs=opts.kwargs, cell=e["cell"])
        cls, _ = plugin.load_method_class(opts.os_method)
        m = blob["method"]
        assert type(m) is cls
        mask = sum({"stuck": 1, "terminal": 2, "overtime": 4, "no_progress": 8}[g] for g in p["disabled_guards"])
        assert m.disabled_mask == mask and list(m.disabled_guards) == p["disabled_guards"]
        assert m.base.k == 16 and m.base.budget == 1 and m.base.serving == "anchor_tail"
        source_path = p["source_artifact"]
        if source_path not in source_blobs:
            with Path(source_path).open("rb") as stream:
                source_blobs[source_path] = pickle.load(stream)
        source = source_blobs[source_path]
        mismatches = state_differences(source["method"], m)
        assert not mismatches, (name, mismatches)
        assert not state_differences(source["registered"], blob["registered"], "registered")
        digest = sha(path)
        fits.append(dict(arm=name, path=str(path), bytes=path.stat().st_size, sha256=digest,
            source_fit=source_path, source_fit_sha256=p["source_fit_sha256"],
            complete_pickle_sha_equal_to_B=digest == p["source_fit_sha256"],
            fitted_state_exact_to_B=True, fit_seconds=blob["fit_s"], library=m.base.cand_name,
            method=blob["spec"], mask=mask))
        for spec in p["references"]["A"] + p["references"]["B"]:
            if spec not in refs:
                refs[spec] = reference_pairs(spec)
    out = dict(PASS=True, arms=24, leave_one_out=16, only_no_progress=8, source_fields_exact=24,
               artifact_metadata_and_config_pass=24, fitted_state_exact_to_B=24,
               reference_replicates=len(refs), reference_pairs_each=500,
               official_manifest_pairs=500, official_manifest_sha256=manifest["sha256"],
               artifacts=fits, references=list(refs.values()))
    write_json(args.run / "validation/source_diff.json", diffs)
    write_json(args.run / "validation/validation.json", out)
    (args.run / "validation/fit_sha256.txt").write_text("".join(f"{f['sha256']}  {f['path']}\n" for f in fits))
    write_arm_table(rows, provenance, fits)
    print(json.dumps({k: v for k, v in out.items() if k not in ("artifacts", "references")}), flush=True)


if __name__ == "__main__":
    main()
