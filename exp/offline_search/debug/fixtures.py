"""Small, lossless osdebug.v1 fixtures; no model, simulator or global RNG use.

``make_synthetic_arm(root)`` returns ``root/runs/synthetic``. Both that path and
``root`` can be passed to ``reader.open_arm(..., 'synthetic')``.
"""
import hashlib
import json
from pathlib import Path

import numpy as np


def _episode_key(uid, attempt):
    return hashlib.sha256(uid.encode()).hexdigest()[:24] + "_a" + str(attempt)


def _sample(campaign, uid, seq, modulus, residue):
    return int(hashlib.sha256(("%s|%s|%s" % (campaign, uid, seq)).encode()).hexdigest(), 16) % modulus == residue


def _write_npz(path, arrays):
    """Atomic NPZ publication, also used by the augmentation writer."""
    from .schema import write_npz_block
    write_npz_block(path, arrays)


def _json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))


def make_synthetic_arm(directory, n_episodes=6, model="pi05", method="A", seed=0):
    """Write a complete small arm, including receipts, catalogs and shadows.

    Fixtures declare their episode count in the manifest; production validation
    still defaults to 500 pairs. Every decision has two 16x16 RGB cameras.
    """
    if n_episodes < 1 or model not in ("pi05", "groot"):
        raise ValueError("positive n_episodes and model pi05/groot required")
    root = Path(directory)
    arm = root / "runs" / "synthetic"
    debug = arm / "debug"
    if debug.exists():
        raise FileExistsError(str(debug))
    server = debug / "server_fixture"
    rng = np.random.default_rng(seed)
    H, A, S = (10 if model == "pi05" else 16), 32, 8
    manifest = dict(schema="osdebug.v1", campaign=root.name, arm="synthetic", synthetic=True,
                    expected_episodes=n_episodes, model=model, method=method, suite="synthetic",
                    code_sha="fixture", config_sha="fixture", fit_sha="fixture", lib_sha="fixture",
                    stage_table_sha="fixture", git_head="fixture", dirty_diff_sha="fixture",
                    capture_config={}, sampling=dict(rawkeys_modulus=16, draws_modulus=32),
                    catalog="catalog/%s_synthetic_current" % model,
                    required_aug=["policy_shadow", "policy_draws", "shadow_look"] +
                    (["camera_shadow"] if model == "pi05" else []))
    meta = dict(schema="osdebug.v1", model=model, suite="synthetic", method=method, pid=1,
                cameras=["third", "wrist"], camera_names=["third", "wrist"],
                image_shapes={"third": [16, 16, 3], "wrist": [16, 16, 3]},
                state_dim=S, action_dim=A, valid_action_dims=list(range(7)), H=H,
                control_dt=.05, cost_weights={"stage1": .152 if model == "pi05" else .148,
                                             "stage23": .848 if model == "pi05" else .852})
    _json(debug / "MANIFEST.json", manifest)
    _json(server / "meta.json", meta)
    journal, records, block_rows, shadows, draws, look = [], [], [], [], [], []
    for ep in range(n_episodes):
        uid = "synthetic:eval:%d:%d" % (ep % 3, ep // 3)
        ek = _episode_key(uid, 1)
        client = debug / "client" / ek
        client.mkdir(parents=True)
        ndec = 8 + ep % 3
        nc = 2 + 5 * ndec
        success = ep % 2 == 0
        entities = dict(body=[dict(id=0, name="table"), dict(id=1, name="object")],
                        geom=[dict(id=0, name="table_geom", body_id=0),
                              dict(id=1, name="object_geom", body_id=1)], joint=[], site=[], actuator=[],
                        movable=[dict(name="object", body_id=1, role="goal")],
                        predicates=[dict(name="at_goal", object="object")],
                        geom_names=["table_geom", "object_geom"], body_names=["table", "object"], actuator_names=[])
        episode = dict(schema="osdebug.v1", task_uid=uid, attempt=1, dispatch_gen=1, episode_key=ek, arm="synthetic",
                       suite="synthetic", task_id=ep % 3, init=ep // 3,
                       task_text="move the object", orig_init_state_idx=ep // 3, env_seed=7,
                       reset_state_sha256=hashlib.sha256(np.zeros(24).tobytes()).hexdigest(),
                       control_dt=.05, max_steps=nc, settle_controls=2, camera_names=meta["cameras"],
                       image_shapes=meta["image_shapes"], action_dim=7, state_dim=S, entities=entities,
                       capabilities={k: dict(status="available", reason="synthetic numeric capture")
                                     for k in ("contacts", "predicates", "actuator_substeps", "snapshots")},
                       t_start=float(ep), t_end=float(ep + 1), n_controls=nc, n_decisions=ndec,
                       success=success, termination_reason="success" if success else "max_steps", error=None)
        _json(client / "episode.json", episode)
        journal.append(dict(task_uid=uid, task_id=ep % 3, init=ep // 3, attempt=1, accepted=True,
                            success=success, status="done" if success else "failed", error=None))
        events = [dict(ev="reset"), dict(ev="settle", n_controls=2)]
        for seq in range(ndec):
            did = "%s:1:%d" % (ek, seq)
            chunk = rng.standard_normal((H, A)).astype(np.float32)
            state = rng.standard_normal(S).astype(np.float64)
            rows = np.arange(16, dtype=np.int32)
            weights = np.full(16, 1 / 16., np.float32)
            score = np.linspace(1, .2, 16, dtype=np.float32)
            sampled = _sample(root.name, uid, seq, 16, 0)
            rec = dict(decision_id=did, episode_key=ek, task_uid=uid, task_id=ep % 3, init=ep // 3,
                       attempt=1, dispatch_gen=1, decision_seq=seq, orig_init_state_idx=ep // 3, server_tag="fixture", server_seq=len(records),
                       conn=ep, pid=1, t_recv=float(seq), t_done=float(seq) + .01,
                       blk="blocks/d_1_%06d.npz" % (len(records) // 16), blk_i=len(records) % 16,
                       vision=seq % 2 == 0, camera_mode="full" if seq % 2 == 0 else "blind",
                       src="cache" if seq % 2 == 0 else "cache_tail", hit=True,
                       blind_age_controls=0 if seq % 2 == 0 else 5, look_reason="anchor", miss_reason="",
                       anchor_decision_id="%s:1:%d" % (ek, seq - seq % 2), chunk_offset=5 * (seq % 2),
                       served_len=5, stage1_calls=int(seq % 2 == 0), camera_completions=0,
                       policy_calls=0, stage23_calls=0, owner_cost=meta["cost_weights"]["stage1"] * (seq % 2 == 0),
                       pre_ms=0., infer_ms=10., s1_ms=5., s23_ms=0., queue_wait_ms=0., method_ms=1.,
                       lib="current", rows=rows.tolist(), weights=weights.tolist(), scores=score.tolist(),
                       conf=1., d1=.1, lib_sha="fixture", k_eff=16., unsupported_mass=0.,
                       diag={}, diag_status=dict(status="available", reason="fixture"),
                       rawkeys_sampled=sampled, snapshot_hint=_sample(root.name, uid, seq, 16, 1))
            rec.update(cache_chunk_status=dict(status="available", reason="fixture cache synthesis"),
                       policy_chunk_status=dict(status="not_applicable", reason="no live policy dispatch"),
                       rawkeys_status=dict(status="available" if sampled and seq % 2 == 0 else "not_applicable" if sampled else "not_sampled",
                                           reason="live keys copied only on sampled looks"))
            records.append(rec)
            block_rows.append(dict(decision_id=did, img_third=rng.integers(0, 256, (16, 16, 3), dtype=np.uint8),
                                   img_wrist=rng.integers(0, 256, (16, 16, 3), dtype=np.uint8),
                                   img_third_available=np.bool_(True), img_wrist_available=np.bool_(True), state_wire=state,
                                   prompt_idx=np.int32(0), keys_pca_third=rng.standard_normal(64).astype(np.float32) if seq % 2 == 0 else np.full(64, np.nan, np.float32),
                                   keys_pca_wrist=rng.standard_normal(64).astype(np.float32) if seq % 2 == 0 else np.full(64, np.nan, np.float32),
                                   state_norm=state.astype(np.float32), served_chunk=chunk, served_wire=chunk[:5, :7].astype(np.float64),
                                   cache_chunk=chunk, policy_chunk=np.full_like(chunk, np.nan)))
            shadows.append(dict(decision_id=did, chunk=chunk + .01, seed=np.int64(seq),
                                input_sha=hashlib.sha256(state.tobytes()).hexdigest()))
            look.append(dict(decision_id=did, rows=rows, weights=weights, scores=score,
                             cache_chunk=chunk, lib="current", keys_pca_third=block_rows[-1]["keys_pca_third"],
                             keys_pca_wrist=block_rows[-1]["keys_pca_wrist"]))
            if _sample(root.name, uid, seq, 32, 2):
                draws.append(dict(decision_id=did, chunks=np.stack([chunk + .02 * (i + 1) for i in range(3)]),
                                  seeds=np.arange(seq, seq + 3, dtype=np.int64)))
            events.append(dict(ev="decision", decision_seq=seq, decision_id=did,
                               control_idx_start=2 + 5 * seq, n_applied=5, server_echo=did,
                               t_send=float(seq), t_recv=float(seq) + .01, infer_ms=10.))
        events.append(dict(ev="done", success=success, termination_reason=episode["termination_reason"]))
        events = [dict(schema="osdebug.v1", task_uid=uid, attempt=1, dispatch_gen=1, episode_key=ek, **event) for event in events]
        _jsonl(client / "events.jsonl", events)
        qpos = np.cumsum(rng.standard_normal((nc, 12)) * .001, axis=0)
        action = np.concatenate([np.zeros((2, 7))] + [r["served_wire"].astype(np.float64) for r in block_rows[-ndec:]])
        ctrl = dict(control_idx=np.arange(nc, dtype=np.int32),
                    decision_seq=np.concatenate([np.full(2, -1, np.int32), np.repeat(np.arange(ndec, dtype=np.int32), 5)]),
                    chunk_offset=np.concatenate([np.full(2, -1, np.int32), np.tile(np.arange(5, dtype=np.int32), ndec)]),
                    is_settle=np.arange(nc) < 2, action=action, reward=np.zeros(nc),
                    done=np.arange(nc) == (nc - 1) if success else np.zeros(nc, bool),
                    qpos=qpos, qvel=np.zeros((nc, 12)), eef_pos=qpos[:, :3],
                    eef_quat=np.tile([1., 0., 0., 0.], (nc, 1)), gripper_qpos=qpos[:, 10:],
                    gripper_qvel=np.zeros((nc, 2)), obj_pos=np.zeros((nc, 1, 3)),
                    obj_quat=np.tile([1., 0., 0., 0.], (nc, 1, 1)), obj_vel=np.zeros((nc, 1, 6)),
                    contact_off=np.arange(nc + 1, dtype=np.int32),
                    contact_geom=np.tile(np.array([[0, 1]], np.int32), (nc, 1)),
                    contact_dist=np.zeros(nc, np.float32), contact_pos=np.zeros((nc, 3), np.float32),
                    contact_frame=np.tile(np.eye(3, dtype=np.float32).reshape(1, 9), (nc, 1)),
                    contact_force=np.zeros((nc, 6), np.float32), predicates=np.zeros((nc, 1)),
                    act_substep=np.zeros((nc, 2, 7), np.float32))
        for start in range(0, nc, 64):
            end = min(start + 64, nc)
            part = {k: v[start:end] for k, v in ctrl.items() if k != "contact_off"}
            part["contact_off"] = np.arange(end - start + 1, dtype=np.int32)
            _write_npz(client / ("controls_%04d.npz" % (start // 64)), part)
        snap = dict(time=np.array(0.), qpos=np.zeros(12), qvel=np.zeros(12), act=np.zeros(0),
                    qacc_warmstart=np.zeros(12), mocap_pos=np.zeros((0, 3)), mocap_quat=np.zeros((0, 4)),
                    controller_skipped_json=np.array("[]"), rng_json=np.array("{}"),
                    selection_p=np.array(1.), restore_certified=np.array(False))
        for name in ["reset", "final"] + ["%06d" % i for i in range(ndec)
                                           if i == 0 or _sample(root.name, uid, i, 16, 1)]:
            _write_npz(client / ("snap_%s.npz" % name), snap)
        receipts = debug / "receipts" / ek
        files = []
        for path in sorted(client.iterdir()):
            files.append(dict(name=path.name, bytes=path.stat().st_size,
                              sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        _json(receipts / "complete.json", dict(episode_key=ek, task_uid=uid, attempt=1, files=files, verified=True))
    for start in range(0, len(block_rows), 16):
        rows = block_rows[start:start + 16]
        arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
        arrays["decision_id"] = arrays["decision_id"].astype("<U80")
        arrays["prompts"] = np.array(["move the object"], dtype="<U512")
        _write_npz(server / ("blocks/d_1_%06d.npz" % (start // 16)), arrays)
        raw = [r for r in records[start:start + 16] if r["rawkeys_status"]["status"] == "available"]
        if raw:
            _write_npz(server / ("rawkeys/k_1_%06d.npz" % (start // 16)),
                       dict(decision_id=np.asarray([r["decision_id"] for r in raw], dtype="<U80"),
                            raw_key_third=rng.standard_normal((len(raw), 128)).astype(np.float32),
                            raw_key_wrist=rng.standard_normal((len(raw), 128)).astype(np.float32)))
    _jsonl(server / "decisions.jsonl", records)
    _jsonl(arm / "client/journal.jsonl", journal)
    _json(server / "writer_stats.json", dict(bytes_written=sum(p.stat().st_size for p in server.rglob("*.npz")),
                                             queue_high_water_bytes=1024, errors=0, serialization_ms=1.))
    for kind, rows in [("policy_shadow", shadows), ("policy_draws", draws), ("shadow_look", look)]:
        if rows:
            arrays = {k: np.asarray([r[k] for r in rows]) for k in rows[0]}
            arrays["_meta_json"] = np.array(json.dumps(dict(model=model, checkpoint_sha="fixture", batch_size=16,
                                                            dtype="float32", code_sha="fixture")))
            _write_npz(debug / "aug" / kind / "part_00000.npz", arrays)
    if model == "pi05":
        arrays = dict(decision_id=np.array([r["decision_id"] for r in look], dtype="<U80"))
        for mode in ("wrist", "third"):
            for key in ("rows", "weights", "scores", "cache_chunk"):
                arrays[mode + "_" + key] = np.stack([r[key] for r in look])
        arrays["_meta_json"] = np.array(json.dumps(dict(model=model, checkpoint_sha="fixture", batch_size=16,
                                                        dtype="float32", code_sha="fixture")))
        _write_npz(debug / "aug/camera_shadow/part_00000.npz", arrays)
    import pandas as pd
    catalog = root / manifest["catalog"]
    catalog.mkdir(parents=True)
    rows = [dict(row=i, task_id=i % 3, episode=i // 4, step=i % 4, ep_len=4, progress=(i % 4) / 4.,
                 success=True, prev=i - 1 if i % 4 else -1, next=i + 1 if i % 4 < 3 else -1,
                 lib_sha="fixture", mode="interior", event_near=False, rows_to_event=2, stage_run=0)
            for i in range(16)]
    pd.DataFrame(rows).to_parquet(catalog / "rows.parquet", index=False)
    _json(catalog / "rows.json", dict(lib_sha="fixture", sha256=hashlib.sha256((catalog / "rows.parquet").read_bytes()).hexdigest()))
    return arm
