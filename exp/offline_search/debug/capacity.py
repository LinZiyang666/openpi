"""Storage forecasts and deployment budgets from authoritative live counts.

Run with --budget for N/V/M and declared/measured/R4 ledgers. All storage
numbers are bytes; shared blocks are apportioned by their decision membership.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import zipfile

import numpy as np

from . import reader


def _stats(values):
    values = np.asarray(list(values), float)
    return dict(n=len(values), mean=float(values.mean()), p50=float(np.percentile(values, 50)),
                p95=float(np.percentile(values, 95)), max=float(values.max()), total=float(values.sum())) if len(values) else dict(n=0, status="unavailable", reason="no measured episodes")


def price_tables(arm):
    model = arm.server_meta.get("model", arm.manifest.get("model"))
    if model not in ("pi05", "groot"):
        raise ValueError("price table unavailable for model %r" % model)
    measured = dict(full_look=.152 if model == "pi05" else .148,
                    stage23=.848 if model == "pi05" else .852,
                    wrist_look=.0646 if model == "pi05" else None, completion=.0502 if model == "pi05" else None)
    assumed = dict(measured, wrist_look=.055198 if model == "pi05" else None, completion=.049890 if model == "pi05" else None)
    declared = dict(measured)
    weights = arm.manifest.get("price_table", arm.server_meta.get("price_table", arm.server_meta.get("cost_weights", {})))
    aliases = dict(stage1="full_look", full="full_look", vision="full_look", c1="full_look", policy="stage23", miss="stage23", c23="stage23",
                   wrist="wrist_look", wrist_only="wrist_look", camera_completion="completion")
    for key, value in weights.items():
        mapped = aliases.get(key, key)
        if mapped in declared:
            declared[mapped] = float(value)
    return dict(declared=declared, measured=measured, r4_assumed=assumed)


def budget(arm):
    decisions = arm.decisions()
    required = ("stage1_calls", "camera_completions", "policy_calls", "stage23_calls", "camera_mode")
    absent = [c for c in required if c not in decisions or decisions[c].isna().any()]
    if absent:
        return dict(arm=arm.arm_name, status="unavailable", reason="missing live dispatch fields: " + ", ".join(absent))
    if (decisions[list(required[:-1])] < 0).any().any():
        raise ValueError("negative live invocation count")
    prices = price_tables(arm)
    rows, warnings = [], []
    for episode in arm.episodes().to_dict("records"):
        ds = decisions[decisions.episode_key.eq(episode["episode_key"])]
        if ds.empty:
            warnings.append("missing decisions for " + episode["episode_key"])
            continue
        wrist = float(ds.loc[ds.camera_mode.eq("wrist_only"), "stage1_calls"].sum())
        third = float(ds.loc[ds.camera_mode.eq("third_only"), "stage1_calls"].sum())
        if third:
            warnings.append("third-only live price unavailable for " + episode["episode_key"])
        vision = float(ds.stage1_calls.sum())
        calls = float(ds.policy_calls.sum())
        heads = float(ds.stage23_calls.sum())
        completion = float(ds.camera_completions.sum())
        if calls != heads:
            warnings.append("policy_calls/stage23_calls differ for " + episode["episode_key"])
        try:
            controls = arm.controls(episode["episode_key"], ["is_settle"])
            active = int(np.count_nonzero(~controls["is_settle"]))
        except (KeyError, ValueError, OSError):
            active = None
        row = dict(episode_key=episode["episode_key"], task_id=episode["task_id"], init=episode["init"],
                   N=len(ds), V=vision, M=calls, full_looks=vision - wrist - third,
                   wrist_looks=wrist, third_looks=third, camera_completions=completion, stage23_calls=heads,
                   active_controls=active)
        for ledger, table in prices.items():
            charges = {"full_look": row["full_looks"], "wrist_look": wrist, "completion": completion, "stage23": heads}
            supported = not third and all(table[k] is not None or not v for k, v in charges.items())
            work = sum(table[k] * v for k, v in charges.items() if v) if supported else None
            row[ledger + "_work"] = work
            row[ledger + "_IR"] = work / len(ds) if supported else None
            row[ledger + "_control_IR"] = 5 * work / active if active and supported else None
        if "owner_cost" in ds and ds.owner_cost.notna().all() and row["declared_work"] is not None:
            row["logged_owner_cost"] = float(ds.owner_cost.sum())
            if not np.isclose(row["logged_owner_cost"], row["declared_work"], atol=1e-7, rtol=1e-6):
                warnings.append("logged owner_cost disagrees with recomputed declared work: " + episode["episode_key"])
        rows.append(row)
    counts = {k: sum(r[k] for r in rows) for k in ("N", "V", "M", "full_looks", "wrist_looks", "third_looks", "camera_completions", "stage23_calls")}
    active = sum(r["active_controls"] for r in rows) if rows and all(r["active_controls"] is not None for r in rows) else None
    ledgers = {}
    for name, table in prices.items():
        supported = all(r[name + "_work"] is not None for r in rows)
        work = sum(r[name + "_work"] for r in rows) if supported else None
        ledgers[name] = dict(prices=table, work=work,
                             owner_IR=work / counts["N"] if counts["N"] and supported else None,
                             control_IR=5 * work / active if active and supported else None,
                             equal_episode_IR=float(np.mean([r[name + "_IR"] for r in rows])) if rows and supported else None)
    return dict(arm=arm.arm_name, status="available" if rows else "unavailable", counts=counts,
                active_controls=active, accepted_episodes=len(rows), ledgers=ledgers, episodes=rows,
                warnings=warnings, denominator="accepted decisions; control_IR excludes settling")


def capacity(arm, remaining_arms=0, episodes_per_arm=500, remaining_episodes=None):
    journal = arm.journal()
    accepted = set(journal.loc[journal.accepted.eq(True), "episode_key"])
    bytes_ep = defaultdict(float, {k: 0. for k in accepted})
    fields_ep, fields, categories = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: dict(compressed=0, uncompressed_npy=0)), defaultdict(int)
    attribution, unaccepted_bytes = 0., 0.
    source_bytes = 0
    for path in sorted(arm.debug_dir.rglob("*")):
        if not path.is_file() or "derived" in path.relative_to(arm.debug_dir).parts or path.name.endswith(".part"):
            continue
        rel = path.relative_to(arm.debug_dir)
        size = path.stat().st_size
        source_bytes += size
        category = "server" if rel.parts[0].startswith("server_") else rel.parts[0]
        categories[category] += size
        ep_weights = {}
        if category in ("client", "receipts") and len(rel.parts) > 1:
            ep_weights = {rel.parts[1]: 1.}
        elif path.suffix == ".npz" and category in ("server", "aug"):
            with np.load(path, allow_pickle=False) as z:
                if "decision_id" in z:
                    ids = z["decision_id"].astype(str)
                    for did in ids:
                        ek = did.split(":")[0]
                        ep_weights[ek] = ep_weights.get(ek, 0.) + 1. / max(len(ids), 1)
        elif category == "server" and path.match("decisions*.jsonl"):
            # Assign actual JSON line bytes, avoiding episode-length bias.
            valid_bytes = 0
            for row, line_bytes in reader.iter_jsonl(path, arm.read_issues):
                valid_bytes += line_bytes
                ek = row.get("episode_key")
                if ek in accepted:
                    bytes_ep[ek] += line_bytes
                    fields_ep["server/decisions.jsonl"][ek] += line_bytes
                else:
                    unaccepted_bytes += line_bytes
            fields["server/decisions.jsonl"]["compressed"] += size
            fields["server/decisions.jsonl"]["uncompressed_npy"] += size
            attribution += valid_bytes
            continue
        for ek, weight in ep_weights.items():
            if ek in accepted:
                bytes_ep[ek] += size * weight
            else:
                unaccepted_bytes += size * weight
        attribution += size * sum(ep_weights.values())
        if path.suffix == ".npz":
            with zipfile.ZipFile(path) as archive:
                for info in archive.infolist():
                    field = category + "/" + Path(info.filename).stem
                    fields[field]["compressed"] += info.compress_size
                    fields[field]["uncompressed_npy"] += info.file_size
                    for ek, weight in ep_weights.items():
                        if ek in accepted:
                            fields_ep[field][ek] += info.compress_size * weight
        else:
            field = category + "/" + path.name
            fields[field]["compressed"] += size
            fields[field]["uncompressed_npy"] += size
            for ek, weight in ep_weights.items():
                if ek in accepted:
                    fields_ep[field][ek] += size * weight
    per_episode = _stats(bytes_ep.values())
    n_remaining = remaining_episodes if remaining_episodes is not None else int(remaining_arms) * int(episodes_per_arm)
    forecast = dict(remaining_episodes=n_remaining,
                    estimated_remaining_bytes=per_episode.get("mean", 0.) * n_remaining if accepted else None,
                    p95_scenario_bytes=per_episode.get("p95", 0.) * n_remaining if accepted else None,
                    basis="measured mean/p95 of this arm; specify representative model/suite arms separately")
    stats_paths = list(arm.debug_dir.glob("server_*/writer_stats*.json")) + list(arm.debug_dir.rglob("*receiver*stats*.json"))
    sender_stats = {}
    for path in (arm.debug_dir / "receipts").glob("*/complete.json"):
        completed = reader.read_json(path)
        if completed.get("sender_stats"):
            sender_stats[path.parent.name] = completed["sender_stats"]
    return dict(arm=arm.arm_name, model=arm.server_meta.get("model") or arm.manifest.get("model"),
                suite=arm.server_meta.get("suite") or arm.manifest.get("suite"),
                source_bytes=source_bytes, shared_metadata_bytes=source_bytes - attribution,
                unaccepted_bytes=unaccepted_bytes, categories=dict(categories),
                episode_bytes=per_episode, episodes=[dict(episode_key=k, bytes=v) for k, v in bytes_ep.items()],
                fields={k: dict(v, episode_bytes=_stats(fields_ep[k].get(ek, 0) for ek in accepted)) for k, v in fields.items()},
                forecast=forecast, io_stats={str(p.relative_to(arm.debug_dir)): reader.read_json(p) for p in stats_paths},
                sender_stats=sender_stats,
                read_issues=list(arm.read_issues),
                field_bytes_note="ZIP member compressed bytes and NPY bytes including headers; container overhead in episode/source totals")


def forecast_campaign(arms, planned_specs, episodes_per_arm=500):
    """Forecast by measured model/suite cells; missing cells stay unavailable."""
    reports = [capacity(arm) for arm in arms]
    cells, completed = defaultdict(list), {}
    for arm, report in zip(arms, reports):
        cells[report["model"], report["suite"]].extend(r["bytes"] for r in report["episodes"])
        completed[arm.arm_name] = len(arm.journal().loc[lambda f: f.accepted.eq(True)])
    rows = []
    for spec in planned_specs:
        name = spec.get("arm", spec.get("name"))
        cell = spec.get("model"), spec.get("suite")
        remaining = max(0, int(spec.get("expected_episodes", episodes_per_arm)) - completed.get(name, 0))
        measured = _stats(cells[cell])
        available = bool(measured["n"]) or not remaining
        rows.append(dict(arm=name, model=cell[0], suite=cell[1], remaining_episodes=remaining,
                         status="available" if available else "unavailable",
                         reason="measured cell mean/p95; methods within a cell can differ in episode duration" if available else "no measured episodes in this model/suite cell",
                         estimated_bytes=remaining * measured.get("mean", 0) if available else None,
                         p95_scenario_bytes=remaining * measured.get("p95", 0) if available else None))
    missing = [r["arm"] for r in rows if r["status"] == "unavailable"]
    return dict(status="available" if not missing else "unavailable", arms=rows, missing_cells_arms=missing,
                remaining_episodes=sum(r["remaining_episodes"] for r in rows),
                estimated_remaining_bytes=sum(r["estimated_bytes"] or 0 for r in rows) if not missing else None,
                p95_scenario_bytes=sum(r["p95_scenario_bytes"] or 0 for r in rows) if not missing else None,
                current_source_bytes=sum(r["source_bytes"] for r in reports))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--remaining-arms", type=int, default=0)
    parser.add_argument("--episodes-per-arm", type=int, default=500)
    parser.add_argument("--remaining-episodes", type=int)
    parser.add_argument("--budget", action="store_true")
    parser.add_argument("--forecast-specs", type=Path, help="emitted arms JSON; forecast remaining episodes by measured model/suite")
    args = parser.parse_args()
    data = forecast_campaign([reader.open_arm(args.run_root, name) for name in args.arms], json.loads(args.forecast_specs.read_text()), args.episodes_per_arm) if args.forecast_specs else [budget(reader.open_arm(args.run_root, name)) if args.budget else
            capacity(reader.open_arm(args.run_root, name), args.remaining_arms, args.episodes_per_arm, args.remaining_episodes)
            for name in args.arms]
    output = json.dumps(data, indent=2, allow_nan=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n")
    print(output)


if __name__ == "__main__":
    main()
