"""R12 opus: are keys from a batched two-camera encode equal to per-camera (batch-1) keys? (pi0.5, GPU, library only)
Encodes 512 library rows' agentview+wrist images (a) one image per tower call, (b) both cameras of a row in one
call (B=2). Reports relative key differences and, for scale, the difference between the recomputed and the stored
library keys."""
import numpy as np, torch
from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r12.opus.tools.lowres import build
from exp.offline_search.rounds.r12.opus.tools import common as cm

torch.set_num_threads(4)
tower, proj = build()
d = core.lib_dir("pi05", "l10")
I0 = np.load(d / "tok" / "img0.npy", mmap_mode="r"); I1 = np.load(d / "tok" / "img1.npy", mmap_mode="r")
rows = np.arange(0, 29472, 57)[:512]
K0, K1 = core.gather_keys("pi05", "l10", rows)

def key(x):
    t = proj(tower(pixel_values=x).last_hidden_state).float().reshape(-1, 4, 4, 4, 4, 2048).mean((2, 4))
    return t.reshape(t.shape[0], -1).cpu().numpy()

a0, a1, b0, b1 = [], [], [], []
with torch.inference_mode():
    for r in rows:
        x = torch.from_numpy(np.stack([I0[r], I1[r]])).cuda().permute(0, 3, 1, 2).float() / 255 * 2 - 1
        x = x.to(torch.bfloat16)
        a0.append(key(x[:1])[0]); a1.append(key(x[1:])[0])
        kb = key(x); b0.append(kb[0]); b1.append(kb[1])
a0, a1, b0, b1 = map(np.array, (a0, a1, b0, b1))
rel = lambda u, v: float(np.mean(np.linalg.norm(u - v, axis=1) / np.linalg.norm(v, axis=1)))
out = dict(batched_vs_single_v0=rel(b0, a0), batched_vs_single_v1=rel(b1, a1),
           single_vs_stored_v0=rel(a0, K0), single_vs_stored_v1=rel(a1, K1),
           max_abs_batched_vs_single=float(max(np.abs(b0 - a0).max(), np.abs(b1 - a1).max())))
print(out)
cm.save_json(out, cm.OUT / "batch_parity.json")
