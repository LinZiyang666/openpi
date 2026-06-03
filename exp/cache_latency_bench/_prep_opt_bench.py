"""One-off prep for the cp1_search optimization Round 2 benchmark.

Loads the real libero_10 cp1_spatial_pool_16 library ONCE (1.1 GB pickle), groups
entries by task_key, and saves a compact per-task tensor pack so the Round 2
expert agents can each run the parity/latency sweeps WITHOUT re-loading the full
pickle (avoids N concurrent 1.1 GB loads OOM-ing the host).

Saves to exp/cache_latency_bench/data/opt_bench/task_pack.pt (gitignored):
  { task_key: {
      "vision_0": Tensor[N,32768] float32 (raw, NOT normalized),
      "vision_1": Tensor[N,32768] float32,
      "robot_state": Tensor[N,32] float32,
      "ids": list[str] length N,
  }, ... }
Plus meta.json with per-task N and a near-1 cosine margin probe for one task.

NOT a deliverable — delete after Round 2.
"""

import collections
import json
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F

SRC = "exp/common/data/cache_artifacts/libero_10/cp1_spatial_pool_16.pkl"
OUT_DIR = "exp/cache_latency_bench/data/opt_bench"
FIELDS = ["vision_0", "vision_1", "robot_state"]

os.makedirs(OUT_DIR, exist_ok=True)

print(f"[prep] loading {SRC} ...")
with open(SRC, "rb") as f:
    data = pickle.load(f)
entries = data["entries"]
print(f"[prep] {len(entries)} entries loaded")


def as_t(v):
    if isinstance(v, torch.Tensor):
        return v.float()
    return torch.from_numpy(np.asarray(v)).float()


buckets = collections.defaultdict(list)
for e in entries:
    buckets[e.payload.task_key].append(e)

pack = {}
meta = {"src": SRC, "n_total": len(entries), "fields": FIELDS, "tasks": {}}
for tk, es in buckets.items():
    rec = {"ids": [e.id for e in es]}
    for fld in FIELDS:
        rec[fld] = torch.stack([as_t(e.query_keys[fld]) for e in es]).contiguous()
    pack[tk] = rec
    meta["tasks"][tk] = {"N": len(es), "vision_dim": rec["vision_0"].shape[1]}
    print(f"[prep] task N={len(es):4d}  {tk[:60]}")

out_pt = os.path.join(OUT_DIR, "task_pack.pt")
torch.save(pack, out_pt)
print(f"[prep] saved {out_pt} ({os.path.getsize(out_pt) / 1e6:.0f} MB)")

# Near-1 cosine margin probe on the largest task, vision_0: leave-one-out top1
# cosine + gap to rank-2. Characterizes the discrimination band the 0.997697
# threshold lives in (informs the fp16/bf16 equivalence dispute).
big_tk = max(meta["tasks"], key=lambda k: meta["tasks"][k]["N"])
M = pack[big_tk]["vision_0"]
Mn = F.normalize(M, dim=1)
S = Mn @ Mn.T  # [N,N] cosine, diagonal = 1.0 (self)
S.fill_diagonal_(-2.0)  # exclude self
top2 = S.topk(2, dim=1).values  # [N,2]
top1 = top2[:, 0]
gap = top1 - top2[:, 1]
probe = {
    "task": big_tk,
    "N": M.shape[0],
    "loo_top1_cos_min": float(top1.min()),
    "loo_top1_cos_median": float(top1.median()),
    "loo_top1_cos_max": float(top1.max()),
    "loo_top1_top2_gap_min": float(gap.min()),
    "loo_top1_top2_gap_median": float(gap.median()),
    "n_top1_above_0.997697": int((top1 > 0.997697).sum()),
}
meta["loo_probe_vision_0"] = probe
print("[prep] leave-one-out vision_0 probe:", json.dumps(probe, indent=2))

with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"[prep] saved meta.json")
