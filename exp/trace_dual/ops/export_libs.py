"""Export the 4 libraries to memmap-able .npy (ids, task_key, action_chunk, query keys)."""
import pickle, json, numpy as np, pathlib
OUT = pathlib.Path("/home/weiland/trace_runs/dual_20260923/audit/libs")
LIBS = {
    "pi05_libero_spatial": "/home/weiland/trace_runs/dual_20260923/libs/libero_spatial/cp1_spatial_pool_16.pkl",
    "pi05_libero_10": "/home/weiland/trace_runs/dual_20260923/libs/libero_10/cp1_spatial_pool_16.pkl",
    "groot_libero_spatial": "/data/libero_cache/libraries/libero_spatial/libero_spatial_sp16_S3.pkl",
    "groot_libero_10": "/data/libero_cache/libraries/libero_10/libero_10_sp16_S3.pkl",
}
for name, p in LIBS.items():
    d = pickle.load(open(p, "rb")); E = d["entries"]
    def arr(x): return x.detach().cpu().float().numpy() if hasattr(x, "detach") else np.asarray(x, dtype=np.float32)
    meta = {"ids": [e.id for e in E], "task_key": [e.payload.task_key for e in E], "fields": sorted(E[0].query_keys)}
    for f in ("vision_0", "vision_1", "robot_state"):
        np.save(OUT / f"{name}.{f}.npy", np.stack([arr(e.query_keys[f]).reshape(-1) for e in E]).astype(np.float32))
    np.save(OUT / f"{name}.action.npy", np.stack([np.asarray(e.payload.action_chunk, dtype=np.float32) for e in E]))
    (OUT / f"{name}.meta.json").write_text(json.dumps(meta))
    print(name, len(E), meta["fields"], "task_key sample:", meta["task_key"][0][:60])
