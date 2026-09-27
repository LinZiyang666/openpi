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
    t0 = time.time()
    lat = []
    for ei in sel:
        e = qc.episodes[ei]
        uid = f"{a.tag}:{a.cell}:{e['uid']}"
        client.episode_start(experiment=a.tag, task=e["task"], episode_id=int(e["init"]),
                             extra_metadata={"task_id": int(e["task_id"]), "orig_init_state_idx": int(e["init"]),
                                             "task_uid": uid, "attempt": 1})
        metas = []
        for r in range(e["start"], e["end"]):
            k = int(tidx[r])
            obs = {"observation/image": np.asarray(img0[k]), "observation/wrist_image": np.asarray(img1[k]),
                   "observation/state": np.asarray(qc.raw_state[r], np.float64), "prompt": str(e["task"])}
            t1 = time.perf_counter()
            out = client.infer(obs)
            lat.append((time.perf_counter() - t1) * 1e3)
            metas.append(out.get("__hit_meta__") or {})
        client.episode_end(success=False)
        resp_rows[uid] = (ei, metas)
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
        rows = np.arange(e["start"], e["end"])
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
        v0_max = max(v0_max, float(np.abs(z["key_v0"] - sv0).max()))
        v1_max = max(v1_max, float(np.abs(z["key_v1"] - sv1).max()))
        rs_max = max(rs_max, float(np.abs(z["rs"] - srs).max()))
        for A, B in ((z["key_v0"], sv0), (z["key_v1"], sv1)):
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
    if verdict_n:
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
    print(json.dumps(rep, indent=1))
    if a.out:
        pathlib.Path(a.out).write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
