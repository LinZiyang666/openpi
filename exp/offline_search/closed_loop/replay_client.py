"""Open-loop replay of recorded store episodes through a live plugin server (no simulator).

Sends the exact wire observations of tok-subsample episodes of a store cell (raw agentview / wrist images, raw
robot state, task prompt; the cell's tok/img0, tok/img1, raw_state) to a server over the websocket protocol, then
compares what the server's plugin logged (--os-log-inputs) with the store rows of the same observations:

  * key_v0 / key_v1: the live pooled keys vs the offline store keys (bit-exact fraction, max |diff|, min cosine)
  * rs (robot_state key), raw_state
  * plugin top-1 and native-shadow top-1 vs the recorded online top-1 (rec_top1), when the method is B0 / ProbeB0
  * every response is a FULL_HIT whose winner id is the plugin's logged winner (pure cache); on a mixed server
    (--os-judge) the response hit_type mix is reported and must equal the plugin's logged hit flags decision by
    decision (`verdict_matches_log`), the MISS responses carry the policy's actions and the proposal as winner_id

    taskset -c 34-37,78-81 .venv/bin/python -m exp.offline_search.closed_loop.replay_client --port 23140 \
        --cell pi05_spatial_cache --episodes 5 --log-dir <server --os-log-dir> [--bundle default]
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import sys
import time

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "src"), str(REPO / "packages" / "openpi-client" / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def offline_wire_transform(model, checkpoint):
    """Load only CPU transforms and normalization metadata, never model weights."""
    if model == "pi05":
        from openpi.training import config, checkpoints
        from openpi import transforms
        cfg = config.get_config("pi05_libero")
        data = cfg.data.create(cfg.assets_dirs, cfg.model)
        try:
            stats = checkpoints.load_norm_stats(pathlib.Path(checkpoint) / "assets", data.asset_id)
        except FileNotFoundError:
            stats = checkpoints.load_norm_stats(cfg.assets_dirs, data.asset_id)
        inp = transforms.compose([transforms.InjectDefaultPrompt(None), *data.data_transforms.inputs,
                                  transforms.Normalize(stats, use_quantiles=data.use_quantile_norm),
                                  *data.model_transforms.inputs])
        out = transforms.compose([*data.model_transforms.outputs,
                                  transforms.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
                                  *data.data_transforms.outputs])
        def recompute(obs, chunk):
            state = inp(dict(obs))["state"]
            return np.asarray(out({"state": np.array(state), "actions": np.array(chunk)})["actions"])
        return recompute
    from custom_data_config import LiberoDataConfig
    from gr00t.data.schema import DatasetMetadata
    from exp.libero_groot.policy_adapter import validate_action_chunk, chunk_to_libero_actions
    import torch
    transform = LiberoDataConfig().transform()
    transform.eval()
    metadata = json.loads((pathlib.Path(checkpoint) / "experiment_cfg" / "metadata.json").read_text())
    from exp.libero_groot.serve_groot_libero import EMBODIMENT_TAG
    key = getattr(EMBODIMENT_TAG, "value", EMBODIMENT_TAG)
    transform.set_metadata(DatasetMetadata.model_validate(metadata[key]))
    def recompute(obs, chunk):
        raw = transform.unapply({"action": torch.from_numpy(np.array(chunk, np.float32))[None]})
        return chunk_to_libero_actions(validate_action_chunk({k: v.squeeze(0) for k, v in raw.items()}))
    return recompute


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--cell", required=True)
    ap.add_argument("--root", default="/dev/shm/offline_search_store")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--log-dir", required=True, help="the server's --os-log-dir (needs --os-log-inputs)")
    ap.add_argument("--bundle", default="default")
    ap.add_argument("--tag", default="replay")
    ap.add_argument("--out", default="")
    ap.add_argument("--max-decisions", type=int, default=0, help="limit each replayed episode (smoke only)")
    ap.add_argument("--wire-checkpoint", default="", help="CPU-only offline output-transform parity against this checkpoint")
    a = ap.parse_args(argv)

    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    from exp.offline_search.harness import store

    qc = store.QueryCell(a.root, a.cell)
    tok_rows = set(np.asarray(qc.tok_rows).tolist())
    eps = [i for i, e in enumerate(qc.episodes) if all(r in tok_rows for r in range(e["start"], e["end"]))]
    by_task = {}
    for i in eps:
        by_task.setdefault(qc.episodes[i]["task_id"], []).append(i)
    sel = []
    j = 0
    while len(sel) < min(a.episodes, len(eps)):
        for t in sorted(by_task):
            if j < len(by_task[t]) and len(sel) < a.episodes:
                sel.append(by_task[t][j])
        j += 1
    img0, img1 = qc.tok("img0"), qc.tok("img1")
    tidx = qc.tok_index
    client = WebsocketClientPolicy(host=a.host, port=a.port)
    client.select_bundle(a.bundle)
    resp_rows = {}
    wire_rows, observation_rows = {}, {}
    t0 = time.time()
    lat = []
    for ei in sel:
        e = qc.episodes[ei]
        uid = f"{a.tag}:{a.cell}:{e['uid']}"
        client.episode_start(experiment=a.tag, task=e["task"], episode_id=int(e["init"]),
                             extra_metadata={"task_id": int(e["task_id"]), "orig_init_state_idx": int(e["init"]),
                                             "task_uid": uid, "attempt": 1})
        metas, wires, observations = [], [], []
        end = min(e["end"], e["start"] + a.max_decisions) if a.max_decisions else e["end"]
        for r in range(e["start"], end):
            k = int(tidx[r])
            obs = {"observation/image": np.asarray(img0[k]), "observation/wrist_image": np.asarray(img1[k]),
                   "observation/state": np.asarray(qc.raw_state[r], np.float64), "prompt": str(e["task"])}
            t1 = time.perf_counter()
            out = client.infer(obs)
            lat.append((time.perf_counter() - t1) * 1e3)
            metas.append(out.get("__hit_meta__") or {})
            wires.append(np.asarray(out["actions"]))
            observations.append(obs)
        client.episode_end(success=False)
        resp_rows[uid] = (ei, metas)
        wire_rows[uid], observation_rows[uid] = wires, observations
    wall = time.time() - t0

    # ---- compare the server's logged inputs with the store rows
    time.sleep(1.0)
    logged = {}
    for f in glob.glob(str(pathlib.Path(a.log_dir) / "inputs" / "*.npz")):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        if m["uid"] in resp_rows:
            logged[m["uid"]] = {k: z[k] for k in z.files}
    rep = {"cell": a.cell, "episodes": len(sel), "logged_episodes": len(logged), "wall_s": round(wall, 2),
           "client_latency_ms": {"p50": float(np.percentile(lat, 50)), "p95": float(np.percentile(lat, 95))}}
    acc = {"n": 0, "v0_exact": 0, "v1_exact": 0, "rs_exact": 0, "raw_exact": 0, "top1_rec": 0, "native_rec": 0,
           "full_hit": 0, "winner_logged": 0}
    v0_max, v1_max, rs_max, cos_min = 0.0, 0.0, 0.0, 1.0
    rec = np.asarray(qc.rec_top1)
    hit_mix: dict = {}
    verdict_eq = verdict_n = 0
    for uid, (ei, metas) in resp_rows.items():
        z = logged.get(uid)
        if z is None:
            continue
        e = qc.episodes[ei]
        rows = np.arange(e["start"], e["start"] + len(metas))
        n = rows.size
        acc["n"] += n
        for s, m in enumerate(metas):
            ht = m.get("hit_type")
            hit_mix[ht] = hit_mix.get(ht, 0) + 1
            if "hit" in z and s < z["hit"].shape[0]:
                verdict_n += 1
                verdict_eq += int((ht == "FULL_HIT") == bool(z["hit"][s]))
        sv0, sv1 = np.asarray(qc.key_v0[rows]), np.asarray(qc.key_v1[rows])
        srs, sraw = np.asarray(qc.rs[rows]), np.asarray(qc.raw_state[rows])
        acc["v0_exact"] += int((z["key_v0"] == sv0).all(1).sum())
        acc["v1_exact"] += int((z["key_v1"] == sv1).all(1).sum())
        acc["rs_exact"] += int((z["rs"] == srs).all(1).sum())
        acc["raw_exact"] += int((z["raw_state"] == sraw).all(1).sum())
        vision = np.asarray(z.get("has_vision", np.ones(n, bool)), bool)
        v0_max = max(v0_max, float(np.abs(z["key_v0"][vision] - sv0[vision]).max(initial=0)))
        v1_max = max(v1_max, float(np.abs(z["key_v1"][vision] - sv1[vision]).max(initial=0)))
        rs_max = max(rs_max, float(np.abs(z["rs"] - srs).max()))
        for A, B in ((z["key_v0"], sv0), (z["key_v1"], sv1)):
            A, B = A[vision], B[vision]
            c = (A * B).sum(1) / (np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1))
            cos_min = min(cos_min, float(c.min()))
        acc["top1_rec"] += int(((z["top1"] == rec[rows]) & (z["lib"] == "current")).sum())
        acc["native_rec"] += int((z["native_top1"] == rec[rows]).sum())
        acc["full_hit"] += int(sum(1 for m in metas if m.get("hit_type") == "FULL_HIT"))
        # winner ids the client saw == the rows the plugin logged
        ids = store.LibraryView(a.root, qc.lib_key, "current").ids
        for s, m in enumerate(metas):
            w = m.get("winner_id")
            lg = ids[int(z["top1"][s])] if str(z["lib"][s]) == "current" and not z["used_synth"][s] else None
            acc["winner_logged"] += int(lg is None or w == lg)
    n = max(acc["n"], 1)
    rep.update({k: (v / n if k != "n" else v) for k, v in acc.items()})
    rep.update({"v0_max_abs": v0_max, "v1_max_abs": v1_max, "rs_max_abs": rs_max, "key_cos_min": cos_min})
    rep["hit_mix"] = hit_mix
    if verdict_n and any("judge" in z for z in logged.values()):
        # mixed server: the client-side verdicts vs the plugin's logged hit flags, realized hit rate, IR (pi05 formula)
        hits = np.concatenate([z["hit"] for z in logged.values() if "hit" in z]).astype(int)
        h = float(hits.mean()) if hits.size else None
        rep["mixed"] = {"verdict_matches_log": verdict_eq / verdict_n, "n": int(hits.size), "h": h,
                        "n_miss": int((hits == 0).sum()),
                        "ir_pi05_formula": None if h is None else 0.152 + 0.848 * (1.0 - h),
                        "judge_mix": {str(k): int(v) for k, v in zip(*np.unique(np.concatenate(
                            [z["judge"] for z in logged.values() if "judge" in z]).astype(str), return_counts=True))}}
        s23 = np.concatenate([z["s23_ms"] for z in logged.values() if "s23_ms" in z]).astype(np.float64)
        s23 = s23[np.isfinite(s23)]
        if s23.size:
            rep["mixed"]["s23_ms"] = {"n": int(s23.size), "mean": float(s23.mean()), "p50": float(np.percentile(s23, 50))}
    if any("has_vision" in z for z in logged.values()):
        vision_n = sum(int(z["has_vision"].sum()) for z in logged.values())
        stage_calls = sum(int(z["stage1_calls"][-1] - z["stage1_calls"][0] + int(z["has_vision"][0])) for z in logged.values())
        rep["blind"] = {"vision": vision_n, "blind": acc["n"] - vision_n, "stage1_calls": stage_calls,
                        "stage1_equals_vision": stage_calls == vision_n,
                        "s1_null_on_blind": all(np.isnan(z["s1_ms"][~z["has_vision"]]).all() for z in logged.values()),
                        "wire_equals_log": sum(int(np.array_equal(np.asarray(wire_rows[uid]), z["wire_actions"]))
                                               for uid, z in logged.items())}
        if "mixed" in rep:
            misses = sum(int((z["hit"] == 0).sum()) for z in logged.values())
            denoise = sum(float(np.nansum(z["miss_k"])) for z in logged.values())
            rep["mixed"]["ir_pi05_formula"] = (.152 * vision_n + .410 * misses + .438 * denoise / 10) / acc["n"]
    if a.wire_checkpoint:
        transform = offline_wire_transform(store.parse_cell(a.cell)[0], a.wire_checkpoint)
        equal = checked = 0
        max_abs = 0.
        for uid, z in logged.items():
            for s, action in enumerate(z["a_exec"]):
                if "source" in z and z["source"][s] == "policy_tail":
                    from .blind import policy_tail_chunk
                    anchor = int(np.flatnonzero(z["has_vision"][:s])[-1])
                    want = policy_tail_chunk(transform(observation_rows[uid][anchor], z["a_exec"][anchor]),
                                             5 * (s - anchor))
                else:
                    want = transform(observation_rows[uid][s], action)
                got = wire_rows[uid][s]
                checked += 1
                equal += int(np.array_equal(want, got))
                max_abs = max(max_abs, float(np.max(np.abs(want - got))))
        rep["offline_wire"] = {"checked": checked, "bit_equal": equal, "max_abs": max_abs}
    print(json.dumps(rep, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rep, indent=1))
    if "blind" in rep:
        b = rep["blind"]
        if not (b["stage1_equals_vision"] and b["s1_null_on_blind"] and b["wire_equals_log"] == len(logged)):
            return 1
    if "offline_wire" in rep and rep["offline_wire"]["bit_equal"] != rep["offline_wire"]["checked"]:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
