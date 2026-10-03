"""Strict capture/augmentation completeness audit. Never alters journal outcomes."""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from zipfile import BadZipFile

import numpy as np

from . import reader
from .fixtures import _episode_key, _sample


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _receipt_entries(directory):
    entries = {}
    for path in sorted(Path(directory).rglob("*.json")):
        obj = reader.read_json(path)
        values = obj.get("files", [])
        if isinstance(values, dict):
            values = [dict(v, name=k) if isinstance(v, dict) else dict(name=k, sha256=v) for k, v in values.items()]
        if not values and any(k in obj for k in ("name", "filename", "relative_path", "path")):
            values = [obj]
        for value in values:
            name = value.get("name", value.get("filename", value.get("relative_path", value.get("path"))))
            if name:
                name = Path(name).name
                if name in entries and any(entries[name].get(k) != value.get(k) for k in ("bytes", "sha256")):
                    raise ValueError("conflicting receipts for " + name)
                entries[name] = value
    return entries


def validate_arm(arm, expected_pairs=500, require_aug=True, required_aug=None):
    """Return PASS/INCOMPLETE plus exact machine-readable missing items.

    ``expected_pairs`` must be set explicitly for smoke/fixture validation. A
    production manifest cannot lower the 500-pair admission requirement.
    """
    if not isinstance(arm, reader.ArmData):
        raise TypeError("validate_arm expects reader.open_arm result")
    capture, augmentation = [], []
    def issue(items, code, where, detail):
        items.append(dict(code=code, where=str(where), detail=str(detail)))
    if not (arm.debug_dir / "MANIFEST.json").exists():
        issue(capture, "missing_manifest", arm.debug_dir, "MANIFEST.json")
    if arm.manifest.get("schema", arm.manifest.get("schema_version")) != "osdebug.v1":
        issue(capture, "schema", arm.debug_dir, "expected osdebug.v1")
    if not arm.server_dirs:
        issue(capture, "missing_server", arm.debug_dir, "server_* directories")
    for directory, meta in arm.server_metas.items():
        if not meta:
            issue(capture, "missing_server_meta", directory, "meta*.json")
    for directory in arm.server_dirs:
        if not list(directory.glob("meta*.json")):
            issue(capture, "missing_server_meta", directory, "meta*.json")
    try:
        journal = arm.journal()
    except (ValueError, OSError) as exc:
        issue(capture, "journal_error", arm.journal_path, exc)
        return _result(arm, capture, augmentation, 0, 0, require_aug)
    accepted = journal[journal.accepted.eq(True)]
    if len(accepted) != expected_pairs:
        issue(capture, "accepted_pairs", arm.journal_path, "expected %d, found %d" % (expected_pairs, len(accepted)))
    for row in accepted.to_dict("records"):
        if row.get("error") or row.get("status", "done") not in ("done", "failed"):
            issue(capture, "accepted_infrastructure_error", row["episode_key"], row)
    try:
        server = arm.server_records()
    except (ValueError, OSError) as exc:
        issue(capture, "server_json", arm.debug_dir, exc)
        server = None
    index = {}
    if server is not None:
        for row in server.to_dict("records"):
            index.setdefault(row["decision_id"], []).append(row)
    accepted_ids, expected_draw_ids = set(), set()
    block_cache = {}
    block_paths = sorted(p for d in arm.server_dirs for p in (d / "blocks").glob("*.npz"))
    try:
        block_index = arm._array_index("validate_blocks", block_paths)
    except (ValueError, OSError, KeyError, EOFError, BadZipFile) as exc:
        issue(capture, "block_index", arm.debug_dir, exc)
        block_index = {}
    raw_paths = sorted(p for d in arm.server_dirs for p in (d / "rawkeys").glob("*.npz"))
    try:
        raw_index = arm._array_index("validate_rawkeys", raw_paths)
    except (ValueError, OSError, KeyError, EOFError, BadZipFile) as exc:
        issue(capture, "rawkeys_index", arm.debug_dir, exc)
        raw_index = {}
    # Decode each image/chunk block once. Keep only shape/dtype inventories and
    # small wire chunks needed to prove application; no arm-sized image cache.
    for path in block_paths:
        try:
            data = reader.read_npz(path)
            if "prompts" in data and (data["prompts"].ndim != 1 or data["prompts"].dtype.kind != "U"):
                issue(capture, "prompt_dtype", path, "one-dimensional unicode prompts required")
            summary = {k: (v if k in ("decision_id", "prompt_idx", "served_wire") or k.startswith("img_") and k.endswith("_available") else
                           np.empty(v.shape[0] if v.ndim else 0, np.uint8) if k == "prompts" else
                           SimpleNamespace(dtype=v.dtype, shape=v.shape, ndim=v.ndim))
                       for k, v in data.items()}
            block_cache[path] = summary
        except (ValueError, OSError, EOFError, BadZipFile) as exc:
            issue(capture, "decision_block_decode", path, exc)
    total_controls = 0
    for outcome in accepted.to_dict("records"):
        ek = outcome["episode_key"]
        client = arm.debug_dir / "client" / ek
        try:
            ep = reader.read_json(client / "episode.json")
        except (ValueError, OSError) as exc:
            issue(capture, "episode_json", client, exc)
            continue
        if not ep:
            issue(capture, "missing_episode", client, "episode.json")
            continue
        for k in ("task_uid", "attempt", "task_id", "init", "success"):
            if ep.get(k) != outcome.get(k):
                issue(capture, "episode_identity", client, "%s: episode=%r journal=%r" % (k, ep.get(k), outcome.get(k)))
        if ep.get("episode_key") != ek or _episode_key(ep.get("task_uid", ""), ep.get("attempt", 1)) != ek:
            issue(capture, "episode_key", client, "hash identity mismatch")
        if ep.get("error") or ep.get("capture_errors"):
            issue(capture, "capture_error", client, ep.get("error") or ep.get("capture_errors"))
        for k in ("n_controls", "n_decisions", "env_seed", "reset_state_sha256", "entities", "capabilities", "image_shapes"):
            if k not in ep:
                issue(capture, "episode_field", client, k)
        nc, nd = int(ep.get("n_controls", 0)), int(ep.get("n_decisions", 0))
        total_controls += nc
        cameras = arm.server_meta.get("camera_names", arm.server_meta.get("cameras", []))
        wire_keys = arm.server_meta.get("camera_wire_keys", {})
        for camera in cameras:
            if camera not in ep.get("image_shapes", {}) and wire_keys.get(camera, camera) not in ep.get("image_shapes", {}):
                issue(capture, "missing_image_shape", client, camera)
        paths = sorted(client.glob("controls_*.npz"))
        if [p.name for p in paths] != ["controls_%04d.npz" % i for i in range(len(paths))]:
            issue(capture, "control_block_gap", client, [p.name for p in paths])
        required = ("control_idx", "decision_seq", "chunk_offset", "is_settle", "action", "reward", "done",
                    "qpos", "qvel", "eef_pos", "eef_quat", "gripper_qpos", "gripper_qvel", "obj_pos", "obj_quat", "obj_vel")
        caps = ep.get("capabilities", {})
        if caps.get("contacts", {}).get("status") == "available":
            required += ("contact_off", "contact_geom", "contact_dist", "contact_pos", "contact_frame", "contact_force")
        if caps.get("predicates", {}).get("status") == "available":
            required += ("predicates",)
        if caps.get("actuator_substeps", {}).get("status") == "available":
            required += ("act_substep",)
        for capability in ("contacts", "predicates", "actuator_substeps", "snapshots"):
            value = caps.get(capability, {})
            if value.get("status") not in ("available", "unsupported", "not_applicable", "error") or "reason" not in value:
                issue(capture, "capability_status", client, capability)
            if value.get("status") == "error":
                issue(capture, "capability_error", client, capability + ": " + value.get("reason", ""))
        for path in paths:
            try:
                data = reader.read_npz(path)
                n = len(data.get("control_idx", []))
                if not 0 < n <= 64:
                    issue(capture, "control_block_size", path, n)
                for k in required:
                    if k not in data:
                        issue(capture, "control_field", path, k)
                for k, value in data.items():
                    if k.startswith("contact_") or k.startswith("_"):
                        continue
                    if value.ndim == 0 or len(value) != n:
                        issue(capture, "control_shape", path, k)
                for k in ("action", "qpos", "qvel", "eef_pos", "eef_quat", "gripper_qpos", "gripper_qvel", "obj_pos", "obj_quat", "obj_vel", "reward", "predicates"):
                    if k in data and data[k].dtype != np.dtype("float64"):
                        issue(capture, "control_dtype", path, k + " must be float64")
                for k in ("control_idx", "decision_seq", "chunk_offset", "contact_off", "contact_geom"):
                    if k in data and data[k].dtype != np.int32:
                        issue(capture, "control_dtype", path, k + " must be int32")
                for k in ("is_settle", "done"):
                    if k in data and data[k].dtype != np.bool_:
                        issue(capture, "control_dtype", path, k + " must be bool")
                if "contact_off" in data:
                    off = data["contact_off"]
                    m = len(data.get("contact_geom", []))
                    if off.shape != (n + 1,) or off[0] != 0 or np.any(np.diff(off) < 0) or off[-1] != m:
                        issue(capture, "contact_offsets", path, "invalid ragged offsets")
                    for k in ("contact_dist", "contact_pos", "contact_frame", "contact_force"):
                        if k in data and len(data[k]) != m:
                            issue(capture, "contact_shape", path, k)
                    geom = data.get("contact_geom", np.empty((0, 2), np.int32))
                    entities = ep.get("entities", {})
                    n_geoms = len(entities.get("geom_names", entities.get("geom", [])))
                    if geom.shape != (m, 2) or (geom.size and (np.any(geom < 0) or np.any(geom >= n_geoms))):
                        issue(capture, "contact_catalog", path, "geom ID outside declared catalog")
            except (ValueError, OSError, EOFError, KeyError, BadZipFile) as exc:
                issue(capture, "control_decode", path, exc)
        try:
            controls = arm.controls(ek)
            if not np.array_equal(controls["control_idx"], np.arange(nc)):
                issue(capture, "control_contiguity", client, "indices do not cover [0,%d)" % nc)
            seqs, offsets, settle = controls["decision_seq"], controls["chunk_offset"], controls["is_settle"]
            if np.any(seqs[settle] != -1) or np.any(offsets[settle] != -1):
                issue(capture, "settle_assignment", client, "settling controls require decision_seq/chunk_offset -1")
            if np.any(seqs[~settle] < 0) or np.any(seqs[~settle] >= nd) or np.any(offsets[~settle] < 0):
                issue(capture, "control_assignment", client, "active controls require a declared decision and nonnegative offset")
        except (ValueError, OSError, KeyError, EOFError, BadZipFile) as exc:
            issue(capture, "controls", client, exc)
            controls = {}
        try:
            events = arm._jsonl(client / "events.jsonl")
        except (ValueError, OSError) as exc:
            issue(capture, "events_json", client, exc)
            events = []
        decisions = [r for r in events if "decision_id" in r]
        if [r.get("decision_seq") for r in decisions] != list(range(nd)):
            issue(capture, "event_contiguity", client, "expected decision_seq 0..%d, found %s" % (nd - 1, [r.get("decision_seq") for r in decisions]))
        terminal = [r for r in events if r.get("ev", r.get("event")) == "done"]
        if len(terminal) != 1 or terminal[0].get("success") != ep.get("success"):
            issue(capture, "terminal_event", client, "one matching done lifecycle event required")
        client_ids = set()
        for event in decisions:
            did, seq = event["decision_id"], event.get("decision_seq")
            client_ids.add(did)
            accepted_ids.add(did)
            if did != "%s:%s:%s" % (ek, ep.get("dispatch_gen"), seq):
                issue(capture, "decision_identity", did, "dispatch_gen/seq mismatch")
            echo = event.get("server_echo")
            echo = echo.get("decision_id") if isinstance(echo, dict) else echo
            if echo != did:
                issue(capture, "server_echo", did, "missing or mismatched echo")
            if _sample(arm.manifest.get("campaign", arm.run_root.name), ep["task_uid"], seq, 32, 2):
                expected_draw_ids.add(did)
            matches = index.get(did, [])
            if len(matches) != 1:
                issue(capture, "server_join", did, "expected one server record, found %d" % len(matches))
                continue
            rec = matches[0]
            if rec.get("status") == "error":
                issue(capture, "server_capture_error", did, rec.get("capture_errors", rec.get("error", "top-level status=error")))
            for key, value in rec.items():
                if key.endswith("_status") and isinstance(value, dict) and value.get("status") == "error":
                    issue(capture, "server_capture_error", did, key + ": " + value.get("reason", ""))
            sampled = _sample(arm.manifest.get("campaign", arm.run_root.name), ep["task_uid"], seq, 16, 0)
            if rec.get("rawkeys_sampled") != sampled:
                issue(capture, "rawkeys_sampling", did, "hash sample differs from record")
            if sampled and rec.get("rawkeys_status", {}).get("status") == "available" and did not in raw_index:
                issue(capture, "missing_rawkeys", did, "available raw keys absent from rawkeys blocks")
            for k in ("episode_key", "decision_seq", "dispatch_gen", "task_uid", "attempt", "task_id", "init", "orig_init_state_idx"):
                expected = seq if k == "decision_seq" else ep.get(k)
                if rec.get(k) != expected:
                    issue(capture, "server_identity", did, k)
            for k in ("vision", "camera_mode", "src", "hit", "served_len", "stage1_calls", "camera_completions", "policy_calls", "stage23_calls", "owner_cost"):
                if k not in rec:
                    issue(capture, "server_field", did, k)
            try:
                # Indexing by ID checks every actual block, independent of a
                # producer's choice of relative/numeric blk reference.
                if did not in block_index:
                    raise KeyError("no block arrays for " + did)
                path, blk_i = block_index[did]
                if path not in block_cache:
                    raise ValueError("block did not decode: " + str(path))
                data = block_cache[path]
                n = len(data["decision_id"])
                if not 0 < n <= 16:
                    issue(capture, "decision_block_size", path, n)
                if rec.get("blk_i") != blk_i:
                    issue(capture, "block_index", did, "blk_i mismatch")
                for k in ("state_wire", "prompt_idx", "prompts", "served_chunk", "served_wire", "cache_chunk", "policy_chunk"):
                    if k not in data:
                        issue(capture, "decision_field", did, k)
                if "state_norm" not in data and rec.get("state_norm_status", {}).get("status") not in ("not_applicable", "unsupported"):
                    issue(capture, "decision_field", did, "state_norm or explicit capability status required")
                meta = arm.record_meta(rec)
                if not meta:
                    issue(capture, "missing_server_meta", did, "no metadata for this server process")
                H, A = meta.get("H"), meta.get("action_dim")
                for k in ("served_chunk", "cache_chunk", "policy_chunk"):
                    if k in data and (data[k].dtype != np.float32 or data[k].shape != (n, H, A)):
                        issue(capture, "decision_chunk", path, k + " dtype/shape mismatch")
                if "served_wire" in data and (data["served_wire"].dtype != np.float64 or data["served_wire"].ndim != 3 or data["served_wire"].shape[0] != n):
                    issue(capture, "wire_action_dtype", path, "served_wire must preserve float64 wire actions")
                if data["decision_id"].dtype.kind != "U":
                    issue(capture, "decision_id_dtype", path, "fixed-width unicode required")
                shapes = ep.get("image_shapes", {})
                reverse_cameras = {v: k for k, v in meta.get("camera_wire_keys", {}).items()}
                for camera, shape in shapes.items():
                    k = "img_" + reverse_cameras.get(camera, camera)
                    mask = data.get(k + "_available")
                    if mask is not None:
                        if mask.dtype != np.bool_ or mask.shape != (n,):
                            issue(capture, "image_availability_mask", did, k)
                        elif not mask[blk_i]:
                            issue(capture, "image_unavailable", did, k)
                    if k not in data or data[k].dtype != np.uint8 or data[k].shape != (n,) + tuple(shape):
                        issue(capture, "image_decode", did, "%s expected uint8 %s" % (camera, (n,) + tuple(shape)))
                if "state_wire" in data and (data["state_wire"].dtype != np.float64 or data["state_wire"].shape != (n, ep.get("state_dim"))):
                    issue(capture, "wire_state", did, "dtype/shape mismatch")
                if "prompt_idx" in data and "prompts" in data:
                    if np.any(data["prompt_idx"] < 0) or np.any(data["prompt_idx"] >= len(data["prompts"])):
                        issue(capture, "prompt_index", path, "out of range")
                if controls and "decision_seq" in controls:
                    mask = controls["decision_seq"] == seq
                    positions = np.flatnonzero(mask)
                    if len(positions) != event.get("n_applied") or (len(positions) and int(positions[0]) != event.get("control_idx_start")):
                        issue(capture, "application_count", did, "control interval disagrees with event")
                    offsets = controls["chunk_offset"][mask]
                    if not np.array_equal(offsets, np.arange(len(offsets))):
                        issue(capture, "chunk_offsets", did, "applied offsets are not contiguous")
                    wire = data.get("served_wire")
                    if wire is not None and len(offsets) and (len(offsets) > wire.shape[1] or not np.array_equal(controls["action"][mask], wire[blk_i, :len(offsets)].astype(np.float64))):
                        issue(capture, "issued_action", did, "issued controls differ from served wire chunk")
            except (ValueError, OSError, KeyError, EOFError, BadZipFile) as exc:
                issue(capture, "decision_arrays", did, exc)
        server_ids = {r["decision_id"] for rows in index.values() for r in rows if r.get("episode_key") == ek}
        for did in sorted(server_ids - client_ids):
            issue(capture, "missing_client_event", did, "server record has no client event")
        if caps.get("snapshots", {}).get("status") == "available":
            selected = ["reset", "final"] + ["%06d" % seq for seq in range(nd) if seq == 0 or _sample(arm.manifest.get("campaign", arm.run_root.name), ep["task_uid"], seq, 16, 1)]
            for name in selected:
                path = client / ("snap_%s.npz" % name)
                if not path.exists():
                    issue(capture, "missing_snapshot", path, "selected by schema")
                    continue
                try:
                    snap = reader.read_npz(path)
                    for k in ("time", "qpos", "qvel", "act", "qacc_warmstart", "selection_p", "restore_certified"):
                        if k not in snap:
                            issue(capture, "snapshot_field", path, k)
                    if bool(snap.get("restore_certified", False)):
                        issue(capture, "snapshot_certification", path, "R8 snapshots must be uncertified")
                except (ValueError, OSError, EOFError, BadZipFile) as exc:
                    issue(capture, "snapshot_decode", path, exc)
        try:
            receipts_dir = arm.debug_dir / "receipts" / ek
            if not (receipts_dir / "complete.json").exists():
                issue(capture, "missing_complete_receipt", receipts_dir, "complete.json")
            else:
                complete = reader.read_json(receipts_dir / "complete.json")
                if complete.get("status", "complete") != "complete":
                    issue(capture, "receipt_status", receipts_dir, complete.get("status"))
                for key, expected in (("uid", ep["task_uid"]), ("task_uid", ep["task_uid"]), ("attempt", ep["attempt"]),
                                      ("key", ek), ("episode_key", ek), ("dispatch_gen", ep.get("dispatch_gen"))):
                    if key in complete and complete[key] != expected:
                        issue(capture, "receipt_identity", receipts_dir, key)
            receipts = _receipt_entries(receipts_dir)
            for path in sorted(client.iterdir()):
                if not path.is_file() or path.suffix not in (".npz", ".json", ".jsonl"):
                    continue
                receipt = receipts.get(path.name)
                if receipt is None:
                    issue(capture, "missing_receipt", path, "no content receipt")
                    continue
                size = receipt.get("bytes", receipt.get("size", receipt.get("final_bytes")))
                digest = receipt.get("sha256", receipt.get("sha"))
                if size != path.stat().st_size or digest != sha256(path):
                    issue(capture, "receipt_mismatch", path, "size or sha256 mismatch")
        except (ValueError, OSError) as exc:
            issue(capture, "receipt_error", client, exc)
    kinds = required_aug or arm.manifest.get("required_aug") or ["policy_shadow", "policy_draws", "shadow_look"] + (["camera_shadow"] if arm.server_meta.get("model") == "pi05" else [])
    if require_aug:
        for kind in kinds:
            expected = expected_draw_ids if kind == "policy_draws" else accepted_ids
            paths = sorted((arm.debug_dir / "aug" / kind).glob("part_*.npz"))
            covered = set()
            for path in paths:
                try:
                    data = reader.read_npz(path)
                    ids = data.get("decision_id", np.array([])).astype(str)
                    for did in ids:
                        if did in covered:
                            issue(augmentation, "duplicate_aug_id", path, did)
                        covered.add(did)
                    meta = json.loads(str(data.get("_meta_json", data.get("metadata_json", "{}"))))
                    for k in ("model", "checkpoint_sha", "batch_size", "dtype", "code_sha"):
                        if k not in meta:
                            issue(augmentation, "aug_provenance", path, k)
                    fields = {"policy_shadow": ["chunk", "seed", "input_sha"], "policy_draws": ["chunks", "seeds"],
                              "shadow_look": ["rows", "weights", "scores", "cache_chunk", "lib"],
                              "camera_shadow": [mode + "_" + k for mode in ("wrist", "third") for k in ("rows", "weights", "scores", "cache_chunk")]}.get(kind, [])
                    for k in fields:
                        if k not in data or data[k].ndim == 0 or len(data[k]) != len(ids):
                            issue(augmentation, "aug_field", path, k)
                    H, A = arm.server_meta.get("H"), arm.server_meta.get("action_dim")
                    if kind == "policy_shadow" and "chunk" in data and (data["chunk"].dtype != np.float32 or data["chunk"].shape != (len(ids), H, A)):
                        issue(augmentation, "aug_shape", path, "chunk float32(n,H,A) required")
                    if kind == "policy_draws" and "chunks" in data and (data["chunks"].dtype != np.float32 or data["chunks"].shape != (len(ids), 3, H, A)):
                        issue(augmentation, "aug_shape", path, "chunks float32(n,3,H,A) required")
                except (ValueError, OSError, EOFError, BadZipFile) as exc:
                    issue(augmentation, "aug_decode", path, exc)
            for did in sorted(expected - covered):
                issue(augmentation, "missing_aug_decision", kind, did)
            if kind == "policy_draws":
                for did in sorted((covered & accepted_ids) - expected):
                    issue(augmentation, "off_sample_draw", kind, did)
    return _result(arm, capture, augmentation, len(accepted), len(journal) - len(accepted), require_aug,
                   n_decisions=len(accepted_ids), n_controls=total_controls)


def _result(arm, capture, augmentation, accepted, unaccepted, require_aug, **counts):
    return dict(arm=arm.arm_name, status="PASS" if not capture and (not require_aug or not augmentation) else "INCOMPLETE",
                capture_status="PASS" if not capture else "INCOMPLETE",
                augmentation_status=("PASS" if not augmentation else "INCOMPLETE") if require_aug else "NOT_CHECKED",
                accepted_episodes=accepted, unaccepted_attempts=unaccepted,
                missing=capture + augmentation, capture_missing=capture, augmentation_missing=augmentation,
                read_issues=list(arm.read_issues), **counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--expected-pairs", type=int, default=500, help="explicit smoke override")
    parser.add_argument("--capture-only", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    results = []
    for name in args.arms:
        try:
            results.append(validate_arm(reader.open_arm(args.run_root, name), args.expected_pairs, not args.capture_only))
        except (ValueError, OSError, EOFError, BadZipFile) as exc:
            results.append(dict(arm=name, status="INCOMPLETE", capture_status="INCOMPLETE", augmentation_status="NOT_CHECKED",
                                missing=[dict(code="arm_decode", where=str(Path(args.run_root) / "runs" / name), detail=str(exc))]))
    payload = json.dumps(results, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n")
    print(payload)
    return int(any(r["status"] != "PASS" for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
