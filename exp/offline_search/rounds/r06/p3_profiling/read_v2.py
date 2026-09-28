"""Loss-audited V2 tables, with accepted-attempt and authoritative control joins.

Server audit works on CPU replay too. --client-root is required for factual
transitions; no-controls server smoke is explicitly an incomplete data product.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from .assignment import assign
from .design import Design, uniform
from .read_logs import flatten, write_csv, identical
from exp.offline_search.rounds.r04.k5_rand.estimate import jsonl


def server_rows(directory):
    groups = defaultdict(lambda: defaultdict(dict))
    configs, ends = {}, {}
    for path in sorted(Path(directory).glob("decisions_*.jsonl")):
        for r in jsonl(path):
            if r["ev"] == "p3_v2_startup":
                config_key = r["tag"], r["conn"]
                old = configs.get(config_key)
                if old is not None and any(old[k] != r[k] for k in ("p", "seed", "replicate", "strata", "design", "catalog_sha256")):
                    raise ValueError("conflicting profile configuration")
                configs[config_key] = r
            if r["ev"] == "episode" and r.get("reason") == "episode_end":
                end_key = r["uid"], int(r.get("attempt", 1) or 1)
                old = ends.get(end_key)
                if old is not None and any(old[k] != r[k] for k in ("n_decisions", "n_exec", "n_miss", "success")):
                    raise ValueError("conflicting server terminal outcome")
                ends[end_key] = r
            if r["ev"] in ("dec", "p3_anchor", "p3_decision"):
                key = r["uid"], int(r.get("attempt", 1) or 1)
                old = groups[key][r["ev"]].get(r["step"])
                if old is not None and not identical(old, r):
                    raise ValueError("contradictory decision duplicate")
                groups[key][r["ev"]][r["step"]] = r
    return groups, configs, ends


def audit_group(directory, records, config, end, require_inputs=True):
    ds, anchors, decisions = (records[k] for k in ("dec", "p3_anchor", "p3_decision"))
    if sorted(ds) != list(range(len(ds))) or set(ds) != set(decisions):
        raise ValueError("missing/shifted decision skeleton")
    if set(anchors) != {i for i, d in ds.items() if d["vision"]}:
        raise ValueError("p3_anchor coverage differs from vision")
    if end is None or end["n_decisions"] != len(ds) or end["n_exec"] != len(ds):
        raise ValueError("missing/mismatching server outcome")
    design = Design(config["design"])
    counts = dict(decisions=len(ds), anchors=len(anchors), misses=sum(not d["hit"] for d in ds.values()),
                  policy_evaluations=sum(d["full_policy_forwards"] for d in decisions.values()),
                  extra_stage3=0, controls_verified=False)
    counts["full_policy_forwards"] = counts["policy_evaluations"]
    for i, r in sorted(anchors.items()):
        if r["schema"] != "r6p3.anchor.v2" or not r["ok"] or r.get("error"):
            raise ValueError("failed/unknown anchor")
        a = r["assignment"]
        p = config["p"]
        doses = config["design"]["episode_doses"]
        if doses is not None:
            u = uniform(config["seed"], r["task_id"], r["init"], config["replicate"], 0, "episode_dose")
            p = doses[min(int(u*len(doses)), len(doses)-1)]
        stratum = 0
        if config["strata"] is not None:
            stratum = int(np.searchsorted(config["strata"]["edges"], r["retrieval"]["d1_loeo_quantile"], side="right"))
            p = config["strata"]["p"][stratum]
        if (a["propensity"], a["stratum"], a["seed"], a["replicate"]) != (p, stratum, config["seed"], config["replicate"]):
            raise ValueError("assignment does not match configured episode/state propensity")
        initial = assign(a["seed"], r["task_id"], r["init"], a["replicate"], i, a["propensity"])
        initial["stratum"] = a["stratum"]
        expected = design.resolve(initial, a["baseline_hit"])
        for k, v in expected.items():
            if a.get(k) != v:
                raise ValueError(f"inconsistent sequential assignment: {i}:{k}")
        call = a["executed_policy"]
        if bool(ds[i]["hit"]) == call:
            raise ValueError("treatment noncompliance")
        chunks = {k: np.asarray(r[k], np.float32) for k in ("cache_chunk", "policy_chunk", "executed_chunk")}
        chosen = chunks["policy_chunk" if call else "cache_chunk"]
        if not np.isfinite(chosen).all() or chunks["executed_chunk"].tobytes() != chosen.tobytes():
            raise ValueError("shadow/cache selection byte mismatch")
        if not np.array_equal(chosen[:5, :7], np.asarray(ds[i]["served_head"], np.float32)):
            raise ValueError("server head mismatch")
        if i + 1 in ds:
            tail = ds[i + 1]
            if a["commit_controls"] == 10:
                baseline_early_look = (not call and tail["vision"] and not config["design"]["pre_guard"]
                    and "CommitJudge" in config["catalog"]["base_spec"])
                if not baseline_early_look and (tail["vision"] or tail["src"] != ("policy_tail" if call else "cache_blind")):
                    raise ValueError("missing ten-control tail")
                if not baseline_early_look and not np.array_equal(chosen[5:10, :7], np.asarray(tail["served_head"], np.float32)):
                    raise ValueError("tail action mismatch")
            elif not tail["vision"]:
                raise ValueError("duration5 did not reobserve")
        extra = len(r["resampling"]["extra_chunks"])
        if r["cost"]["additional_forward_for_injection"] != 0 or r["cost"]["extra_stage2_forwards"] != 0:
            raise ValueError("unexpected additional forward")
        counts["extra_stage3"] += extra
    counts["policy_evaluations"] += counts["extra_stage3"]
    if counts["misses"] != end["n_miss"]:
        raise ValueError("MISS total mismatch")
    if require_inputs:
        for i, r in decisions.items():
            with np.load(Path(directory) / r["input_archive"], allow_pickle=False) as data:
                if r["shadow_available"]:
                    if any(k not in data for k in ("vision_0", "vision_1", "robot_state", "policy_chunk")):
                        raise ValueError("missing observational keys/shadow")
                    if not np.isfinite(data["vision_0"]).all() or not np.isfinite(data["policy_chunk"]).all():
                        raise ValueError("nonfinite shadow input")
                if not np.array_equal(data["executed_chunk"][:5, :7], np.asarray(ds[i]["served_head"], np.float32)):
                    raise ValueError("archive action mismatch")
    return counts


def audit_server(directory):
    groups, configs, ends = server_rows(directory)
    total = defaultdict(int)
    for key, records in groups.items():
        first = records["p3_decision"][0]
        config = configs[first["tag"], first["conn"]]
        for k, v in audit_group(directory, records, config, ends.get(key)).items():
            total[k] += v
    return dict(total, episodes=len(groups))


def control_join(path, key, records, journal):
    rows = list(jsonl(path))
    if any((r["task_uid"], int(r.get("attempt", 1))) != key for r in rows):
        raise ValueError("client identity mismatch")
    controls = [r for r in rows if r["ev"] == "control"]
    decision_rows = [r for r in rows if r["ev"] == "decision"]
    decisions = {r["decision_step"]: r for r in decision_rows}
    if len(decisions) != len(decision_rows):
        raise ValueError("duplicate client decision")
    ends = [r for r in rows if r["ev"] == "rollout_end"]
    if len(ends) != 1 or any(r["ev"].endswith("error") for r in rows):
        raise ValueError("missing/error client completion")
    end = ends[0]
    if bool(end["success"]) != bool(journal["success"]) or end["controls"] != len(controls):
        raise ValueError("client/outcome mismatch")
    if not controls or bool(controls[-1]["done"]) != bool(end["success"]) or any(r["done"] for r in controls[:-1]):
        raise ValueError("terminal control differs from episode outcome")
    if [r["control"] for r in controls] != list(range(len(controls))) or set(decisions) != set(records["dec"]):
        raise ValueError("missing control/decision skeleton")
    by_step = defaultdict(list)
    for r in controls:
        i = r["decision_step"]
        if i is None:
            continue
        by_step[i].append(r)
        wire = decisions[i]["wire_chunk"]
        offset = r["chunk_offset"]
        if offset != len(by_step[i]) - 1 or offset >= 5 or not np.array_equal(
                np.asarray(r["action_issued"], np.float32), np.asarray(wire[offset], np.float32)):
            raise ValueError("application action/offset mismatch")
    for i in decisions:
        n = len(by_step[i])
        if not 1 <= n <= 5 or (i < max(decisions) and n != 5):
            raise ValueError("missing or partial nonterminal chunk")
    return controls, decisions, end, by_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--client-root", type=Path, help="tree containing controls.jsonl for these arms")
    mode.add_argument("--server-only", action="store_true", help="explicit partial product: no application/physics verification")
    ap.add_argument("--require-stage-counts", action="store_true", help="require independent measured dispatch counts, not CPU mocks")
    ap.add_argument("--require-snapshots", action="store_true", help="check every selected client snapshot and its identity")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.server_only and a.require_snapshots:
        ap.error("--require-snapshots requires --client-root")
    a.out.mkdir(parents=True, exist_ok=False)
    paths = {}
    for path in a.client_root.rglob("controls.jsonl") if a.client_root is not None else []:
        first = next(jsonl(path))
        key = first["task_uid"], int(first.get("attempt", 1))
        if key in paths:
            raise ValueError("duplicate client attempt trace")
        paths[key] = path
    tables = defaultdict(list)
    for arm in a.arms:
        directory = a.run_root / "runs" / arm
        accepted = {}
        accepted_uids = {}
        for r in jsonl(directory / "client/journal.jsonl"):
            tables["attempts"].append(flatten(dict(arm=arm, **r)))
            if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
                key = r["task_uid"], int(r.get("attempt", 1))
                if key[0] in accepted_uids and accepted_uids[key[0]] != key[1]:
                    raise ValueError("multiple accepted attempts for one uid")
                accepted_uids[key[0]] = key[1]
                if key in accepted and accepted[key] != r:
                    raise ValueError("conflicting accepted outcomes")
                accepted[key] = r
        found = set()
        for server in directory.glob("server_*"):
            groups, configs, ends = server_rows(server)
            for key, records in groups.items():
                if key not in accepted:
                    tables["unaccepted"].append(dict(arm=arm, uid=key[0], attempt=key[1], decisions=len(records["dec"])))
                    continue
                if key in found:
                    raise ValueError("accepted attempt appears on multiple servers")
                found.add(key)
                first = records["p3_decision"][0]
                counts = audit_group(server, records, configs[first["tag"], first["conn"]], ends.get(key))
                if bool(ends[key]["success"]) != bool(accepted[key]["success"]):
                    raise ValueError("server/journal outcome mismatch")
                if a.require_stage_counts:
                    for i, r in records["p3_decision"].items():
                        extra = len(records["p3_anchor"][i]["resampling"]["extra_chunks"]) if r["vision"] else 0
                        expected = dict(stage1=1, stage2=1, stage3=1+extra)
                        if r.get("stage_invocations") != expected or r["full_policy_forwards"] != 1:
                            raise ValueError("missing/mismatching measured stage counts")
                if a.server_only:
                    controls, decisions, end, by_step = [], {}, {"unavailable": "server-only collection"}, {}
                else:
                    if key not in paths:
                        raise ValueError("accepted attempt missing client telemetry")
                    controls, decisions, end, by_step = control_join(paths[key], key, records, accepted[key])
                    counts["controls_verified"] = True
                    if a.require_snapshots:
                        for i in records["p3_anchor"]:
                            sample = decisions[i].get("snapshot")
                            if sample is None:
                                raise ValueError("missing snapshot sampling record")
                            if sample["selected"]:
                                snapshot = paths[key].parent / sample["path"]
                                with np.load(snapshot, allow_pickle=False) as data:
                                    meta = json.loads(str(data["metadata_json"]))
                                    if (meta["task_uid"], int(meta.get("attempt", 1)), meta["decision_step"]) != (*key, i):
                                        raise ValueError("snapshot identity mismatch")
                                    if any(k not in data for k in ("sim_state", "rng_json", "controller_skipped_json")):
                                        raise ValueError("incomplete snapshot payload")
                ident = dict(arm=arm, uid=key[0], attempt=key[1], Y=int(accepted[key]["success"]),
                             task_id=first["task_id"], init=first["init"],
                             collection_mode="server_only" if a.server_only else "client_verified",
                             physical_transitions_available=not a.server_only,
                             client_environment_seed=None if a.server_only else end.get("environment_seed"),
                             environment_seed_verified=not a.server_only and end.get("environment_seed") is not None)
                anchor0 = records["p3_anchor"][0]
                ident.update(model=anchor0["model"], suite=anchor0["suite"], lib=anchor0["lib"])
                c1 = .152 if ident["model"] == "pi05" else .148
                actual = None if a.server_only else sum(len(v) for v in by_step.values())
                tables["episodes"].append(flatten({**ident, **counts, "controls": None if a.server_only else len(controls),
                    "active_controls": actual, "client": end,
                    "mean_cache_policy_rms": float(np.mean([v["distance"]["rms"] for v in records["p3_anchor"].values()])),
                    "mean_d1_loeo_quantile": float(np.mean([v["retrieval"]["d1_loeo_quantile"] for v in records["p3_anchor"].values()])),
                    "deployment_IR_per_actual_5_controls": None if a.server_only else (c1*counts["anchors"]+(1-c1)*counts["misses"])/(actual/5),
                    "deployment_IR_per_request": (c1*counts["anchors"]+(1-c1)*counts["misses"])/counts["decisions"],
                    "episode_assignment": anchor0["episode_assignment"], "provenance": anchor0["provenance"]}))
                for c in controls:
                    tables["controls"].append(flatten({**ident, **c}))
                for i, r in records["p3_decision"].items():
                    tables["decisions"].append(flatten({**ident, **r, "actual_controls": None if a.server_only else len(by_step[i]),
                        "client": decisions.get(i), "absolute_input_archive": str(server / r["input_archive"])}))
                for i, r in records["p3_anchor"].items():
                    continued = i + 1 in records["dec"] and not records["dec"][i + 1]["vision"]
                    c = by_step.get(i, []) + (by_step.get(i + 1, []) if r["commit_controls"] == 10 and continued else [])
                    row = {**ident, **r, "actual_commit_controls": None if a.server_only else len(c),
                        "commit_truncated": None if a.server_only else len(c) < r["commit_controls"],
                        "next_anchor_step": next((j for j in sorted(records["p3_anchor"]) if j > i), None),
                        "successor_after_head": None if a.server_only else by_step[i][-1]["after"],
                        "successor_after_commit": None if a.server_only else c[-1]["after"],
                        "future_controls": None if a.server_only else sum(len(v) for j, v in by_step.items() if j >= i),
                        "future_misses": sum(not v["hit"] for j, v in records["dec"].items() if j >= i)}
                    for big in ("cache_chunk", "policy_chunk", "executed_chunk", "served_wire_chunk", "candidate_wire"):
                        row.pop(big, None)
                    tables["anchors"].append(flatten(row))
                    aid = dict(ident, step=i)
                    ret = r["retrieval"]
                    for rank in range(len(ret["rows"])):
                        tables["neighbours"].append({**aid, "rank": rank+1, **{k: ret[k][rank] for k in (
                            "rows", "distances", "ranked_distances", "weights", "episodes", "progress", "steps", "phase_rows", "phase_steps")}})
                    for offset in range(len(r["policy_chunk"])):
                        item = dict(aid, chunk_step=offset, intended_commit=offset < r["commit_controls"],
                            actual_commit=None if a.server_only else offset < len(c), rms=r["distance"]["per_step_rms"][offset])
                        for name in ("cache", "policy", "executed"):
                            item.update({f"{name}.{j}": r[name+"_chunk"][offset][j] for j in range(7)})
                        tables["action_steps"].append(item)
        if found != set(accepted):
            raise ValueError("missing accepted server attempt(s)")
    for name, rows in tables.items():
        if rows:
            write_csv(a.out / (name + ".csv"), rows)
    summary = {k: len(v) for k, v in tables.items()}
    (a.out / "audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
