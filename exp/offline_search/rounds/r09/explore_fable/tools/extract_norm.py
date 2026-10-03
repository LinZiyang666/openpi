import sys, numpy as np, pandas as pd
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from exp.offline_search.debug import reader
from exp.offline_search.rounds.r09.explore_fable.tools import common
def one(arm):
    out = common.DERIVED / "arrays_norm" / f"{arm}.npz"
    if out.exists(): return arm, "cached"
    d = reader.open_arm(common.RUN_ROOT, arm); d.cache_enabled = False
    t = pd.read_parquet(common.DERIVED / "decisions" / f"{arm}.parquet")
    ids = t.decision_id.tolist()
    arr = d.decision_arrays(["state_norm"], ids)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, decision_id=np.asarray(ids), state_norm=arr["state_norm"].astype(np.float32))
    return arm, arr["state_norm"].shape
arms = [r["arm"] for r in common.load_arms()]
with ProcessPoolExecutor(12) as pool:
    for r in pool.map(one, arms): print(r, flush=True)
