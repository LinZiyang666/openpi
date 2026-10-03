"""Shared read-only inputs, explicit coverage, and fixed-task cluster inference."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
from pathlib import Path
import time

import numpy as np
import pandas as pd


class Unavailable(ValueError):
    """Required scientific information is absent; never substitute a guess."""


def missing(value):
    return value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and np.isnan(value))


def field(row, *names, **kwargs):
    """Read canonical fields or explicitly named adapter/diagnostic aliases."""
    for name in names:
        if name in row and not missing(row[name]):
            return row[name]
        def search(nested):
            if not isinstance(nested, dict):
                return None
            if name in nested and not missing(nested[name]):
                return nested[name]
            for key, item in nested.items():
                if key.startswith("layer_") or key in ("_last_log", "_tilt_log", "_plan_extras", "last_blind_extras", "randomization", "assignment"):
                    result = search(item)
                    if result is not None:
                        return result
            return None
        for container in ("diag", "randomization", "retrieval", "assignment", "extras", "blind_extras"):
            result = search(row.get(container))
            if result is not None:
                return result
    return kwargs.get("default")


def boolean(value):
    if missing(value):
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value in (0, 1, "0", "1", "false", "true", "False", "True"):
        return value in (1, "1", "true", "True")
    return None


def number(value):
    try:
        result = float(value)
        return result if np.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def require(frame, columns):
    absent = [name for name in columns if name not in frame]
    if absent:
        raise Unavailable("missing columns: " + ", ".join(absent))


def metadata(arm):
    result = {}
    def visit(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if not isinstance(item, (dict, list)):
                    result[key] = item
            for key in ("model_manifest", "action_manifest", "manifest", "cost_weights", "prices", "arm_spec", "r8"):
                if isinstance(value.get(key), dict):
                    visit(value[key])
            for key, item in value.items():
                if key.startswith("server_") and isinstance(item, dict):
                    visit(item)
            for key, item in value.items():
                if isinstance(item, list):
                    result[key] = item
    visit(getattr(arm, "manifest", {}))
    meta = getattr(arm, "server_meta", {})
    if isinstance(meta, list):
        for entry in meta:
            visit(entry)
    else:
        visit(meta)
    manifest = getattr(arm, "manifest", {})
    round_spec = manifest.get("arm_spec", {}).get("r8", manifest.get("r8", {}))
    if isinstance(round_spec, dict) and "library_size" in round_spec:
        result["library_size"] = round_spec["library_size"]
    overrides = getattr(arm, "profile_metadata", {})
    result.update(overrides)
    if getattr(arm, "manifest", {}).get("adapter") == "r6p3.v2":
        # Early P3 reader versions default model to pi05 even for GR00T.
        # Pricing must be attested, never reconstructed from an arm-name guess.
        declared = overrides.get("model", getattr(arm, "manifest", {}).get("model"))
        result["model"] = declared
    return result


def catalog(arm):
    """Use the reader catalogue, or a separately built, attested scratch catalogue."""
    root = getattr(arm, "profile_catalog_root", None)
    if root is None:
        return arm.catalog()
    if hasattr(arm, "profile_catalog_table"):
        return arm.profile_catalog_table
    from exp.offline_search.debug.catalog import catalog_key
    meta = metadata(arm)
    key = catalog_key(meta.get("model"), meta.get("suite"), meta.get("lib", meta.get("library", "current")))
    directory = Path(root) / "catalog" / key
    path, manifest = directory / "rows.parquet", directory / "rows.json"
    if not path.is_file() or not manifest.is_file():
        raise Unavailable("scratch catalogue and its builder provenance unavailable: " + str(directory))
    info = json.loads(manifest.read_text())
    if catalog_key(info.get("model"), info.get("suite"), info.get("lib")) != key:
        raise ValueError("scratch catalogue identity differs from arm")
    if hashlib.sha256(path.read_bytes()).hexdigest() != info.get("sha256"):
        raise ValueError("scratch catalogue differs from builder sha256")
    artifact_key = "library_%s_action.npy" % info["lib"]
    captured = getattr(arm, "server_meta", {}).get("artifacts", {}).get(artifact_key, {})
    action = info.get("store_arrays", {}).get("action", {})
    if not captured.get("sha256") or captured["sha256"] != action.get("sha256"):
        raise Unavailable("scratch catalogue action fingerprint is not attested by the captured arm")
    verified = []
    for name, source in info.get("store_arrays", {}).items():
        item = getattr(arm, "server_meta", {}).get("artifacts", {}).get("library_%s_%s.npy" % (info["lib"], name), {})
        if item.get("sha256"):
            if item["sha256"] != source.get("sha256"):
                raise Unavailable("scratch catalogue %s fingerprint differs from captured arm" % name)
            verified.append(name)
    arm.profile_catalog_table = pd.read_parquet(path)
    arm.profile_catalog_provenance = dict(path=str(path), metadata=str(manifest), lib_sha=info.get("lib_sha"),
                                        stages=info.get("stages"), captured_action_sha256=captured["sha256"],
                                        verified_store_artifacts=verified)
    return arm.profile_catalog_table


def stages(arm, records):
    """Catalogue stages are descriptive proxies, not certified causal moderators."""
    from .kernels import catalog_map, kernel
    try:
        catalog, catalog_reason = catalog_map(arm), None
    except Unavailable as error:
        catalog, catalog_reason = None, str(error)
    output = []
    for row in records:
        pre = field(row, "stage_pre", "pre_stage")
        stage = field(row, "stage")
        result = dict(stage="unknown", stage_source="unavailable", stage_mass={},
                      stage_reason=catalog_reason or "retrieval rows/weights unavailable",
                      stage_pre_reason="assignment-time stage certification absent")
        if pre is not None:
            result.update(stage=str(pre), stage_source="pre_assignment", stage_reason=None, stage_pre_reason=None)
        elif stage is not None:
            result.update(stage=str(stage), stage_source="recorded_descriptive", stage_reason=None)
        elif catalog is not None and field(row, "rows") is not None and field(row, "weights") is not None:
            try:
                provenance = kernel(field(row, "rows"), field(row, "weights"), catalog, field(row, "lib"))
                mass = provenance["stage_mass"]
                result["stage_mass"] = mass
                if not mass or not np.isclose(provenance["unknown_stage_mass"], 0.):
                    result["stage_reason"] = "positive-weight neighbours have unavailable catalogue stage labels"
                else:
                    largest = max(mass.values())
                    winners = [key for key, value in mass.items() if np.isclose(value, largest)]
                    if len(winners) != 1:
                        result["stage_reason"] = "weighted catalogue stage mass is tied"
                    else:
                        result.update(stage="catalog:" + winners[0], stage_source="catalog_descriptive",
                                      stage_reason="dominant weighted neighbour stage; not a simulator-truth label")
            except Unavailable as error:
                result["stage_reason"] = str(error)
        output.append(result)
    arm.profile_stage_coverage = dict(denominator=len(output),
        available=sum(row["stage"] != "unknown" for row in output),
        pre_assignment_available=sum(row["stage_source"] == "pre_assignment" for row in output),
        sources=dict(Counter(row["stage_source"] for row in output)),
        reasons=dict(Counter(row["stage_reason"] for row in output if row["stage_reason"])),
        pre_assignment_reason="Only explicitly certified stage_pre/pre_stage moderates randomized assignment; catalogue labels remain descriptive.")
    return output


def inputs(arm):
    ds = arm.decisions(accepted_only=True).copy()
    eps = arm.episodes(accepted_only=True).copy()
    require(eps, ["episode_key", "task_id", "init"])
    if eps.episode_key.duplicated().any() or eps[["task_id", "init"]].duplicated().any():
        raise ValueError("multiple accepted attempts for an episode or task/init")
    if len(ds):
        require(ds, ["decision_id", "episode_key", "decision_seq"])
        if ds.decision_id.duplicated().any():
            raise ValueError("duplicate decision_id")
        if not ds.episode_key.isin(eps.episode_key).all():
            raise ValueError("decision outside accepted episode population")
        joined = ds.merge(eps, on="episode_key", how="left", suffixes=("", "_episode"), validate="many_to_one")
        for name in ("task_id", "init", "success"):
            other = name + "_episode"
            if other in joined:
                if name in ds and name != "success":
                    a, b = joined[name], joined[other]
                    if (a.notna() & b.notna() & (a != b)).any():
                        raise ValueError("conflicting decision/episode " + name)
                joined[name] = joined[other]
        ds = joined.sort_values(["episode_key", "decision_seq"], kind="stable").reset_index(drop=True)
    labels = stages(arm, ds.to_dict("records"))
    for key in ("stage", "stage_source", "stage_mass", "stage_reason", "stage_pre_reason"):
        ds[key] = [row[key] for row in labels]
    arm.profile_denominators = {"decision_denominator": len(ds), "episode_denominator": len(eps)}
    return ds, eps


def causal_readiness(decisions, episodes):
    """Do not select complete prefixes after observing treatment/outcomes."""
    problems = []
    if "server_join" in decisions:
        bad = ~decisions.server_join.isin(["verified", "legacy_unverified"])
        if bad.any():
            problems.append("%d decisions have missing server/client joins" % bad.sum())
    for ep in episodes.to_dict("records"):
        rows = decisions[decisions.episode_key == ep["episode_key"]]
        expected = number(ep.get("n_decisions"))
        if expected is not None and len(rows) != int(expected):
            problems.append("episode %s decision count differs from capture" % ep["episode_key"])
        if sorted(rows.decision_seq.astype(int)) != list(range(len(rows))):
            problems.append("episode %s decision sequence has gaps" % ep["episode_key"])
        if ep.get("termination_reason") == "exception":
            problems.append("episode %s exception is invalid/excluded from outcome estimates" % ep["episode_key"])
        elif boolean(ep.get("success")) is None or (not missing(ep.get("error")) and bool(ep.get("error"))):
            problems.append("episode %s accepted outcome unavailable or infrastructure error" % ep["episode_key"])
    return problems


def identity(row):
    return {key: row.get(key) for key in ("decision_id", "episode_key", "task_id", "init", "decision_seq",
                                         "stage", "stage_source", "stage_reason", "stage_pre_reason", "stage_mass")}


def arrays(arm, keys, ids):
    """The reader returns aligned arrays. Missing keys are explicit unavailable."""
    try:
        result = arm.decision_arrays(keys, list(ids))
    except KeyError:
        result = {}
        for key in keys:
            try:
                result.update(arm.decision_arrays([key], list(ids)))
            except KeyError:
                pass
    return {key: np.asarray(result[key]) for key in keys if key in result}


def augmentation(arm, kind, ids):
    try:
        data = arm.aug(kind, decision_ids=list(ids))
    except KeyError:
        # Sampled kinds legitimately lack most requested IDs. Read available
        # parts and align them with NaN holes; absence is not a zero noise floor.
        try:
            data = arm.aug(kind)
        except KeyError:
            return {}
    if not data:
        return {}
    data = {key: np.asarray(value) for key, value in data.items()}
    if "decision_id" in data:
        source = list(map(str, data["decision_id"]))
        if len(set(source)) != len(source):
            raise ValueError("duplicate augmentation decision_id")
        index = {key: i for i, key in enumerate(source)}
        aligned = {}
        for key, value in data.items():
            if value.ndim == 0 or len(value) != len(source):
                continue
            if key == "decision_id":
                continue
            shape = (len(ids),) + value.shape[1:]
            if value.dtype.kind in "biufc":
                target = np.full(shape, np.nan, dtype=np.float64)
            else:
                target = np.full(shape, "", dtype=value.dtype)
            for i, did in enumerate(ids):
                if str(did) in index:
                    target[i] = value[index[str(did)]]
            aligned[key] = target
        return aligned
    for key, value in data.items():
        if value.ndim and len(value) != len(ids):
            raise ValueError("unaligned augmentation " + kind + ":" + key)
    return data


def pick(data, *keys):
    for key in keys:
        if key in data:
            return data[key]
    return None


def dimensions(arm, width):
    meta = metadata(arm)
    value = meta.get("valid_action_dims", meta.get("action_valid_dims"))
    if value is None:
        raise Unavailable("manifest valid_action_dims absent; padded channels cannot be inferred")
    dims = list(range(value)) if isinstance(value, int) else list(value)
    if len(dims) != len(set(dims)) or not dims or min(dims) < 0 or max(dims) >= width:
        raise ValueError("invalid valid_action_dims")
    scale = meta.get("action_scale", meta.get("action_std"))
    if scale is None:
        sigma, units = np.ones(width), "normalized_action_units"
    else:
        sigma = np.asarray(scale, dtype=float)
        if sigma.shape == (len(dims),):
            expanded = np.ones(width)
            expanded[dims] = sigma
            sigma = expanded
        if sigma.shape != (width,) or not np.isfinite(sigma[dims]).all() or (sigma[dims] <= 0).any():
            raise ValueError("invalid manifest action_scale")
        units = "library_action_sigma"
    grip = meta.get("gripper_dim")
    if grip is not None and grip not in dims:
        raise ValueError("gripper_dim outside valid action dimensions")
    return dims, sigma, grip, meta.get("gripper_threshold"), units


def coverage(total, available, reason=None):
    return {"status": "available" if available else "unavailable", "denominator": int(total),
            "available": int(available), "unavailable": int(total - available), "reason": reason}


def ess(weights):
    value = np.asarray(weights, dtype=float)
    den = float(np.sum(value ** 2))
    return float(value.sum() ** 2 / den) if den else 0.0


def cluster_interval(frame, numerator="numerator", denominator="denominator", bootstraps=1000, seed=0, alpha=.05):
    """Resample init clusters within fixed tasks; repeats/arms of a pair stay together."""
    require(frame, ["task_id", "init", numerator, denominator])
    grouped = frame.groupby(["task_id", "init"], dropna=False)[[numerator, denominator]].sum()
    nclusters = len(grouped)
    den = float(grouped[denominator].sum())
    point = float(grouped[numerator].sum() / den) if den else None
    out = {"estimate": point, "denominator": den, "task_init_clusters": nclusters,
           "interval": None, "interval_status": "unavailable", "bootstrap": "task/init within fixed tasks",
           "exploratory": nclusters < 30}
    variable_task = any(len(group) > 1 for _, group in grouped.groupby(level=0))
    if bootstraps <= 0 or nclusters < 2 or point is None or not variable_task:
        out["interval_reason"] = "bootstrap disabled, insufficient within-task init clusters, or zero denominator"
        return out
    rng = np.random.default_rng(seed)
    total = np.zeros((bootstraps, 2))
    for _, task in grouped.groupby(level=0):
        values = task[[numerator, denominator]].to_numpy(float)
        # Bound temporary memory even with thousands of clusters/replicates.
        for start in range(0, bootstraps, 128):
            count = min(128, bootstraps - start)
            ix = rng.integers(0, len(values), size=(count, len(values)))
            total[start:start + count] += values[ix].sum(axis=1)
    good = total[:, 1] > 0
    samples = total[good, 0] / total[good, 1]
    if len(samples):
        out.update(interval=np.quantile(samples, [alpha / 2, 1 - alpha / 2]).tolist(),
                   interval_status="available", valid_bootstraps=int(good.sum()), alpha=alpha)
    return out


def summarize(records, groups, metrics):
    if not records:
        return []
    frame = pd.DataFrame(records)
    existing = [key for key in groups if key in frame]
    iterator = frame.groupby(existing, dropna=False, sort=False) if existing else [((), frame)]
    output = []
    for keys, group in iterator:
        keys = keys if isinstance(keys, tuple) else (keys,)
        entry = dict(zip(existing, keys))
        entry["decision_denominator"] = int(group.decision_id.nunique()) if "decision_id" in group else len(group)
        entry["row_denominator"] = len(group)
        for metric in metrics:
            if metric not in group:
                continue
            values = pd.to_numeric(group[metric], errors="coerce").dropna().to_numpy(float)
            entry[metric + "_n"] = len(values)
            entry[metric + "_mean"] = float(values.mean()) if len(values) else None
            entry[metric + "_p50"] = float(np.median(values)) if len(values) else None
            entry[metric + "_p90"] = float(np.quantile(values, .9)) if len(values) else None
        output.append(entry)
    return output


def clean(value):
    if value is pd.NA:
        return None
    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean(item) for item in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_report(tool, out, reports, seconds):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    payload = clean({"tool": tool, "schema": "osdebug.profile.v1", "runtime_seconds": seconds, "arms": reports})
    (out / (tool + ".json")).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    tables = {}
    summary = ["# " + tool, "", "Accepted attempts only. Missing information is unavailable; each table retains its denominator.", ""]
    for arm, report in payload["arms"].items():
        summary.append("- %s: %s; %s" % (arm, report.get("status", "available"),
                       report.get("reason", "coverage " + json.dumps(report.get("coverage", {}), sort_keys=True))))
        if tool == "stage_ledger" and report.get("tables", {}).get("stages"):
            total = report["tables"]["stages"][0]
            summary.append("  Measured request IR=%s; control IR=%s; N=%s, V=%s, M=%s." %
                           tuple(total.get(key) for key in ("measured_IR_requests", "measured_IR_controls", "N", "V", "M")))
        elif tool == "churn" and report.get("tables", {}).get("decomposition"):
            total = report["tables"]["decomposition"][0]
            summary.append("  Paired outcomes=%s; flips=%s; symmetric churn=%s; net loss bias=%s." %
                (total.get("paired_episode_denominator"), total.get("gain", 0) + total.get("loss", 0), total.get("symmetric_churn"), total.get("net_loss_bias")))
        elif tool in ("call_value", "exposure_hazard"):
            supported = [r for r in report.get("tables", {}).get("contrasts", []) if r.get("status") == "available"]
            summary.append("  %d supported endpoint/stratum contrasts. Actual propensities and task/init cluster intervals are in the tables." % len(supported))
        for name, rows in report.get("tables", {}).items():
            for row in rows:
                tables.setdefault(name, []).append(dict(arm=arm, **row))
    for name, rows in tables.items():
        frame = pd.DataFrame(rows)
        for column in frame:
            frame[column] = frame[column].map(lambda value: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value)
        frame.to_csv(out / (tool + "_" + name + ".csv"), index=False)
    # An empty/unsupported input still produces the promised CSV artifact.
    if not tables:
        pd.DataFrame([{"arm": arm, "status": item.get("status"), "reason": item.get("reason")}
                      for arm, item in payload["arms"].items()]).to_csv(out / (tool + "_coverage.csv"), index=False)
    (out / (tool + ".md")).write_text("\n".join(summary) + "\n")
    return payload


def cli(tool, analyze):
    parser = argparse.ArgumentParser(description=analyze.__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--procs", type=int, default=4, help="bounded arm I/O threads; no child Python processes")
    parser.add_argument("--adapter", choices=("debug", "p3v2"), default="debug")
    parser.add_argument("--metadata", type=Path, help="explicit model/action metadata overrides JSON; recorded in report")
    parser.add_argument("--catalog-root", type=Path, help="scratch run root populated by debug.catalog; capture files remain read-only")
    parser.add_argument("--bootstraps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--onsets", type=Path, help="S6 onset labels CSV/JSON/JSONL")
    parser.add_argument("--triggers", nargs="+", help="explicit numeric/boolean online diagnostic field names")
    parser.add_argument("--reference", help="accepted A reference arm for churn; defaults to first --arms entry")
    parser.add_argument("--replicates", nargs="*", default=[])
    parser.add_argument("--placebos", nargs="*", default=[])
    args = parser.parse_args()
    if args.procs < 1 or args.bootstraps < 0:
        parser.error("--procs must be positive and --bootstraps nonnegative")
    from exp.offline_search.debug import reader
    opened = {}
    overrides = json.loads(args.metadata.read_text()) if args.metadata else {}
    if not isinstance(overrides, dict):
        parser.error("--metadata must contain a JSON object")
    start = time.monotonic()
    def one(name):
        arm = opened.get(name)
        if arm is None:
            arm = reader.p3v2_adapter(args.run_root, name) if args.adapter == "p3v2" else reader.open_arm(args.run_root, name)
        opened[name] = arm
        arm.profile_metadata = overrides
        arm.profile_catalog_root = args.catalog_root
        try:
            report = analyze(arm, args)
            if overrides:
                report["metadata_overrides"] = overrides
        except Unavailable as error:
            denominators = getattr(arm, "profile_denominators", {"decision_denominator": None, "episode_denominator": None})
            report = {"status": "unavailable", "reason": str(error), **denominators, "tables": {}}
        if hasattr(arm, "profile_stage_coverage"):
            report["stage_coverage"] = arm.profile_stage_coverage
        if hasattr(arm, "profile_catalog_provenance"):
            report["catalog_provenance"] = arm.profile_catalog_provenance
        report["input_diagnostics"] = getattr(arm, "read_issues", [])
        return name, report
    if tool == "churn":
        names = list(dict.fromkeys(args.arms + args.replicates + args.placebos + ([args.reference] if args.reference else [])))
        for name in names:
            opened[name] = reader.p3v2_adapter(args.run_root, name) if args.adapter == "p3v2" else reader.open_arm(args.run_root, name)
            opened[name].profile_metadata = overrides
        args.opened = opened
        reports = dict(one(name) for name in args.arms if name != (args.reference or args.arms[0]))
        if not reports:
            name = args.reference or args.arms[0]
            arm = opened[name]
            reports[name] = dict(status="unavailable", reason="churn requires a distinct candidate/reference pair; only the reference arm was supplied",
                                 episode_denominator=len(arm.episodes()), tables={},
                                 input_diagnostics=getattr(arm, "read_issues", []))
    else:
        # Threads bound I/O concurrency; affinity is inherited from the required taskset command.
        with ThreadPoolExecutor(max_workers=min(args.procs, len(args.arms))) as executor:
            reports = dict(executor.map(one, args.arms))
    result = write_report(tool, args.out, reports, time.monotonic() - start)
    print(json.dumps({"tool": tool, "arms": {name: item.get("status", "available") for name, item in result["arms"].items()},
                      "runtime_seconds": result["runtime_seconds"]}, sort_keys=True))
