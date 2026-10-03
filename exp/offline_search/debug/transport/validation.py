"""Client payload validation, shared by receiver and journal reconciliation."""
import json
from pathlib import Path

import numpy as np

from exp.offline_search.debug.schema import SCHEMA_VERSION, decision_id, episode_key, snapshot_sampled, read_jsonl


def validate_episode(directory, uid, attempt):
    directory = Path(directory)
    episode = json.loads((directory / "episode.json").read_text())
    key = episode_key(uid, attempt)
    if (episode.get("schema"), episode.get("task_uid"), episode.get("attempt"),
            episode.get("episode_key")) != (SCHEMA_VERSION, uid, attempt, key):
        raise ValueError("episode identity/schema mismatch")
    events = []
    lifecycle = []
    skipped = []
    for row in read_jsonl(directory / "events.jsonl", skipped):
        if (row.get("task_uid"), row.get("attempt"), row.get("dispatch_gen")) != (
                uid, attempt, episode["dispatch_gen"]):
            raise ValueError("event identity mismatch")
        if row.get("ev") == "decision":
            seq = len(events)
            if row["decision_seq"] != seq or row["decision_id"] != decision_id(key, episode["dispatch_gen"], seq):
                raise ValueError("decision gap/identity mismatch")
            events.append(row)
        else:
            lifecycle.append(row)
    if len(events) != episode["n_decisions"]:
        raise ValueError("event count mismatch")
    for row in events:
        selected = snapshot_sampled(episode["campaign"], uid, row["decision_seq"])
        expected = "snap_{:06d}.npz".format(row["decision_seq"]) if selected else None
        if row.get("snapshot_file") != expected:
            raise ValueError("snapshot sample declaration mismatch")
    if not lifecycle or lifecycle[0]["ev"] != "reset" or lifecycle[-1]["ev"] != "done":
        raise ValueError("missing reset/done lifecycle")
    if lifecycle[-1].get("success") != episode["success"]:
        raise ValueError("lifecycle outcome mismatch")
    count = 0
    applied = [[] for _ in events]
    required = ("control_idx", "decision_seq", "chunk_offset", "is_settle", "action", "reward", "done",
                "qpos", "qvel", "eef_pos", "eef_quat", "gripper_qpos", "gripper_qvel", "obj_pos", "obj_quat",
                "obj_vel", "contact_off", "contact_geom", "contact_dist", "contact_pos", "contact_frame",
                "contact_force", "predicates", "act_substep")
    for block_i, path in enumerate(sorted(directory.glob("controls_*.npz"))):
        if path.name != "controls_{:04d}.npz".format(block_i):
            raise ValueError("control block filename gap")
        with np.load(str(path), allow_pickle=False) as block:
            if any(k not in block for k in required):
                raise ValueError("missing control arrays")
            n = len(block["control_idx"])
            if not 1 <= n <= 64 or not np.array_equal(block["control_idx"], np.arange(count, count + n)):
                raise ValueError("control gap/count mismatch")
            off = block["contact_off"]
            m = len(block["contact_geom"])
            if off.shape != (n + 1,) or off[0] != 0 or off[-1] != m or (np.diff(off) < 0).any():
                raise ValueError("contact offsets invalid")
            for k in required:
                arr = block[k]
                if arr.dtype.hasobject:
                    raise ValueError("object array forbidden")
                expected = n + 1 if k == "contact_off" else m if k.startswith("contact_") else n
                if len(arr) != expected:
                    raise ValueError("array first axis mismatch: " + k)
            geom = block["contact_geom"]
            if geom.size and ((geom < 0).any() or (geom >= len(episode["entities"]["geom_names"])).any()):
                raise ValueError("contact geom ID outside model catalog")
            for i, seq in enumerate(block["decision_seq"]):
                if seq == -1:
                    if not block["is_settle"][i]:
                        raise ValueError("unassigned active control")
                elif not 0 <= seq < len(events) or block["is_settle"][i]:
                    raise ValueError("invalid control decision")
                else:
                    applied[int(seq)].append((count + i, int(block["chunk_offset"][i])))
            count += n
    if count != episode["n_controls"]:
        raise ValueError("control count mismatch")
    for event, controls in zip(events, applied):
        expected = [(event["control_idx_start"] + i, i) for i in range(event["n_applied"])]
        if controls != expected:
            raise ValueError("applied decision interval mismatch")
    snapshots = {"snap_reset.npz", "snap_final.npz"}
    snapshots.update(r["snapshot_file"] for r in events if r.get("snapshot_file"))
    for name in snapshots:
        with np.load(str(directory / name), allow_pickle=False) as data:
            if bool(data["restore_certified"]):
                raise ValueError("uncertified snapshot marked restored")
    errors = episode.get("capture_errors", [])
    bad_caps = any(v.get("status") == "error" for v in episode.get("capabilities", {}).values())
    # Support/self-check findings are diagnostics, including legacy error labels.
    status = "error" if errors or episode.get("error") or bad_caps else "complete"
    return dict(status=status, success=episode["success"], n_controls=count, n_decisions=len(events),
                snapshots=sorted(snapshots), dispatch_gen=episode["dispatch_gen"], skipped_event_lines=skipped)
