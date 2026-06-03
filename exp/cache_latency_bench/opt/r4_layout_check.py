"""ROUND 4 — batched-output layout + integrated build wrapper sanity (THROWAWAY).

Confirms (1) the split outputs out[0]/out[1] from the batched pool need a
.contiguous() (they are strided slices of [2,32768]) and what that costs vs the
src single-field path which is already contiguous after reshape(-1); (2) an
end-to-end build wrapper that batches vision_0+vision_1, applies _to_cpu_float32,
produces query keys BIT-EXACT to the src CP1SpatialPool16KeyBuilder.build output.

Run:
    PYTHONPATH=. uv run python exp/cache_latency_bench/opt/r4_layout_check.py
"""

from __future__ import annotations

import glob
import os

import h5py
import torch
import torch.nn.functional as F

from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
from openpi.cache.types import CheckpointID

torch.set_grad_enabled(False)
GRID, POOL, EMB = 16, 4, 2048
H5_DIR = "exp/common/data/db/libero_cache/libero_10"

from exp.common.build_in_memory_cache_artifact import _build_fake_stage1  # noqa: E402

p = sorted(glob.glob(os.path.join(H5_DIR, "*.h5")))[0]
steps = []
with h5py.File(p, "r") as f:
    names = sorted((k for k in f if k.startswith("step_")), key=lambda s: int(s.split("_")[-1]))
    for name in names[:20]:
        steps.append(_build_fake_stage1(f[name]))


# --- layout check ---
v0 = steps[0].prefix_embs[0][0:256].contiguous()
v1 = steps[0].prefix_embs[0][256:512].contiguous()
x = torch.stack((v0, v1)).reshape(2, GRID, GRID, EMB).permute(0, 3, 1, 2)
pooled = F.adaptive_avg_pool2d(x, (POOL, POOL))
out = pooled.permute(0, 2, 3, 1).reshape(2, -1)  # [2,32768]
print("=== batched output layout ===")
print(f"  out shape={tuple(out.shape)} contiguous={out.is_contiguous()}")
print(f"  out[0] contiguous={out[0].is_contiguous()}  out[1] contiguous={out[1].is_contiguous()}")
print(f"  out[0].contiguous() copies? data_ptr changes = {out[0].contiguous().data_ptr() != out[0].data_ptr()}")
# src single field reshape(-1) is contiguous:
xs = v0.reshape(GRID, GRID, EMB).permute(2, 0, 1).unsqueeze(0)
ps = F.adaptive_avg_pool2d(xs, (POOL, POOL)).squeeze(0).permute(1, 2, 0).reshape(-1)
print(f"  src single-field reshape(-1) contiguous={ps.is_contiguous()}")


# --- integrated build wrapper (the proposed exp keybuilder subclass logic) ---
class _BatchedBuilder(CP1SpatialPool16KeyBuilder):
    """Override build() to batch vision_0+vision_1 into one pool. robot_state/prompt unchanged."""

    def build(self, checkpoint_id):  # noqa: D401
        from openpi.cache.components.key_builder import (
            _slice_cp1_fields, _to_cpu_float32, _mean_pool_tokens,
            VISION_0, VISION_1, VISION_2, PROMPT_EMB, ROBOT_STATE,
        )
        raw = _slice_cp1_fields(self._cache["prefix_embs"], self._cache["state"], self._enabled)
        keys = {}
        # Batch whichever vision fields are present (here vision_0+vision_1).
        vis_present = [vf for vf in (VISION_0, VISION_1, VISION_2) if vf in raw]
        if vis_present:
            stacked = torch.stack([raw[vf] for vf in vis_present])  # [k,256,2048]
            k = stacked.shape[0]
            xb = stacked.reshape(k, GRID, GRID, EMB).permute(0, 3, 1, 2)
            pb = F.adaptive_avg_pool2d(xb, (POOL, POOL))
            ob = pb.permute(0, 2, 3, 1).reshape(k, -1)
            for i, vf in enumerate(vis_present):
                keys[vf] = _to_cpu_float32(ob[i])
        if PROMPT_EMB in raw:
            keys[PROMPT_EMB] = _to_cpu_float32(_mean_pool_tokens(raw[PROMPT_EMB]))
        if ROBOT_STATE in raw:
            keys[ROBOT_STATE] = _to_cpu_float32(raw[ROBOT_STATE])
        return keys


ENABLED = ["vision_0", "vision_1", "robot_state"]
src_kb = CP1SpatialPool16KeyBuilder(enabled_fields=ENABLED)
opt_kb = _BatchedBuilder(enabled_fields=ENABLED)

print("\n=== integrated build BIT-EXACT check (20 real steps) ===")
all_exact = True
for i, fs in enumerate(steps):
    src_kb.collect(CheckpointID.CP1, stage1=fs)
    opt_kb.collect(CheckpointID.CP1, stage1=fs)
    sk = src_kb.build(CheckpointID.CP1)
    ok = opt_kb.build(CheckpointID.CP1)
    assert set(sk) == set(ok), (set(sk), set(ok))
    for fld in sk:
        eq = torch.equal(sk[fld], ok[fld])
        contig = ok[fld].is_contiguous()
        if not eq or not contig:
            all_exact = False
            print(f"  step {i} {fld}: bit-exact={eq} contig={contig} max|diff|={(sk[fld]-ok[fld]).abs().max():.2e}")
print(f"  ALL 20 steps x 3 fields bit-exact AND contiguous: {all_exact}")
