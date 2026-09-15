"""Select an O-cold arm's terminal state and emit its frozen evaluation arm.

The server writes a full snapshot every ``snapshot_every`` committed batches,
on every connection close and at exit. When the adaptation stage has ended
the last snapshot is the terminal state, provided it accounts for every
learned batch in the feedback log: the two counts must agree, otherwise the
snapshot is stale and the script refuses.

Public interface: ``pick_terminal``, ``frozen_arm_from``.
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib

import yaml

from exp.online_rit.common import load_jsonl, sha256_file, write_json
from openpi.cache.config import load_cache_config
from openpi.cache.components.online_rit import OnlineRiskCurves, verify_state_sha


def pick_terminal(state_dir: pathlib.Path) -> tuple[pathlib.Path, dict]:
    """Select the latest verified event, rejecting any invalid event in the stream."""
    feedback = state_dir / "feedback.jsonl"
    rows = []
    n_learned = 0
    if feedback.exists():
        rows = load_jsonl(feedback)
        if any(row.get("diag", {}).get("flow_invalid") for row in rows):
            raise SystemExit("feedback marks the stream invalid")
        n_learned = sum(1 for row in rows if row.get("diag", {}).get("learned"))
    snaps = sorted(state_dir.glob("state_*.json"))
    if not snaps:
        raise SystemExit(f"no state snapshots under {state_dir}")
    docs = [(p, json.loads(p.read_text(encoding="utf-8"))) for p in snaps]
    if any(d.get("flow_invalid") for _, d in docs):
        raise SystemExit("snapshot marks the stream flow_invalid; earlier valid states cannot be used")
    identities = set()
    for _, doc in docs:
        verify_state_sha(doc)
        OnlineRiskCurves.from_snapshot(doc, update_enabled=False)
        identity = tuple(doc.get(k) for k in ("yaml_id", "library_sha256", "server_instance_id", "fingerprint"))
        if any(v is None for v in identity) or not isinstance(doc.get("snapshot_seq"), int):
            raise SystemExit("snapshot lacks stream identity or event sequence")
        identities.add(identity)
    if len(identities) != 1:
        raise SystemExit("snapshots belong to different streams")
    best, state = max(docs, key=lambda item: item[1]["snapshot_seq"])
    latest = state_dir / "state_latest.json"
    if not latest.is_file() or json.loads(latest.read_text()) != state:
        raise SystemExit("latest snapshot is missing or stale; stage did not end cleanly")
    for _, doc in docs:
        if doc["snapshot_seq"] == state["snapshot_seq"] and doc != state:
            raise SystemExit("conflicting snapshots for the latest event")
    if int(state["n_updates"]) != n_learned:
        raise SystemExit(
            f"terminal snapshot {best.name} has n_updates={state['n_updates']} but the feedback log "
            f"holds {n_learned} learned batches; the stage did not end cleanly"
        )
    if state.get("update_seq") != len(rows) or [r.get("diag", {}).get("update_seq") for r in rows] != list(range(1, len(rows) + 1)):
        raise SystemExit("feedback sequence is missing, duplicated or newer than the snapshot")
    return best, state


def require_completed_stage(data_dir: pathlib.Path, yaml_id: str, manifest: dict, state: dict) -> None:
    """Check existing accepted journal/per-step evidence before terminal export."""
    from exp.online_rit.aggregate_online import aggregate

    result = aggregate(data_dir, None, pool_manifest=manifest, pool_key="adapt").get(yaml_id)
    if result is None or result["incomplete"]:
        raise SystemExit("adaptation stage is missing or incomplete")
    expected = {(int(t), int(i)) for t, indices in manifest["adapt"].items() for i in indices}
    episodes = list(result["episodes"].values())
    actual = {(e.get("task_id"), e.get("orig_init_state_idx")) for e in episodes}
    if len(episodes) != len(expected) or actual != expected:
        raise SystemExit("adaptation episode set does not equal the manifest")
    if result["learned_batches"] != state["n_updates"]:
        raise SystemExit("adaptation per-step learned batches differ from terminal state")
    instances = {r["server_instance_id"] for r in result["dynamics"] if r.get("server_instance_id")}
    if instances != {state["server_instance_id"]}:
        raise SystemExit("adaptation evidence belongs to a different server stream")


def frozen_arm_from(source_yaml: pathlib.Path, terminal_state: pathlib.Path, *, new_stem: str) -> dict:
    doc = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    judge = doc["checkpoints"]["cp1"]["judge"]
    if judge.get("type") != "online_rit":
        raise SystemExit(f"{source_yaml}: not an online_rit arm")
    judge = copy.deepcopy(judge)
    judge["init_state_path"] = str(terminal_state)
    judge["update_enabled"] = False
    doc["checkpoints"]["cp1"]["judge"] = judge
    return doc


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--state-dir", required=True, help="<state_log_dir>/<yaml_id>__<sha12>/<server_instance_id>")
    ap.add_argument("--source-yaml", required=True, help="the O-cold arm yaml")
    ap.add_argument("--out-yaml", required=True, help="frozen arm yaml (new stem = new yaml_id)")
    ap.add_argument("--terminal-copy", required=True, help="where to copy the chosen snapshot")
    ap.add_argument("--data-dir", required=True, help="completed adaptation journal and per_step directory")
    ap.add_argument("--pool-manifest", required=True)
    args = ap.parse_args()
    best, state = pick_terminal(pathlib.Path(args.state_dir))
    source = pathlib.Path(args.source_yaml)
    if source.stem != state["yaml_id"] or pathlib.Path(args.out_yaml).stem == source.stem:
        raise SystemExit("source yaml must match the stream and frozen yaml must have a new identity")
    manifest = json.loads(pathlib.Path(args.pool_manifest).read_text())
    from exp.online_rit.cohorts import validate_pool

    validate_pool(manifest, "adapt", {"suite": manifest["suite"], **manifest["pool_records"]["adapt"]}, 25)
    require_completed_stage(pathlib.Path(args.data_dir), source.stem, manifest, state)
    launch = json.loads((pathlib.Path(args.data_dir) / "per_step.jsonl.launch.json").read_text())
    if launch.get("yaml_sha256", {}).get(source.stem) != sha256_file(source):
        raise SystemExit("source yaml changed since the adaptation launch")
    if launch.get("init_map_sha256") != sha256_file(args.pool_manifest) or launch.get("init_map_key") != "adapt":
        raise SystemExit("init pool manifest differs from the adaptation launch")
    source_doc = yaml.safe_load(source.read_text())
    judge = source_doc["checkpoints"]["cp1"]["judge"]
    fixed = state.get("fixed_params", {})
    if (state["library_sha256"] != sha256_file(source_doc["backend"]["in_memory"]["preload_path"])
            or fixed.get("scales_sha256") != sha256_file(judge["update_scales_path"])
            or fixed.get("h_exec") != judge["h_exec"]):
        raise SystemExit("terminal state library/scales/h_exec differ from the source arm")
    terminal = pathlib.Path(args.terminal_copy)
    terminal.parent.mkdir(parents=True, exist_ok=True)
    terminal.write_text(best.read_text(encoding="utf-8"), encoding="utf-8")
    out = pathlib.Path(args.out_yaml)
    doc = frozen_arm_from(pathlib.Path(args.source_yaml), terminal, new_stem=out.stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    load_cache_config(str(out))
    from exp.online_rit.emit_online_arms import MATRIX_COHORT

    matrix = out.with_suffix(".matrix.yaml")
    matrix.write_text(yaml.safe_dump({"cohort": MATRIX_COHORT["terminal"],
                                     "arms": [{"arm": out.stem, "yaml": str(out), "suite": manifest["suite"]}]}, sort_keys=False))
    write_json(
        out.with_suffix(".record.json"),
        {
            "source_yaml": str(args.source_yaml),
            "source_yaml_sha256": sha256_file(source),
            "matrix": str(matrix), "pool_manifest_sha256": sha256_file(args.pool_manifest),
            "terminal_snapshot": str(best),
            "terminal_sha256": sha256_file(terminal),
            "learning_state_sha256": state["learning_state_sha256"],
            "n_updates": state["n_updates"],
            "revision": state["revision"],
            "out_yaml": str(out),
            "out_sha256": sha256_file(out),
        },
    )
    print(f"terminal {best.name} (n_updates={state['n_updates']}) -> {out}")


if __name__ == "__main__":
    main()
