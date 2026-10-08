"""R12 opus: "look half by pixels" -- retrieval keys from low-resolution camera images (pi0.5, library only, GPU).

Rebuilds the pi0.5 image path (SigLIP-so400m vision tower + multi-modal projector, weights read from the local
pi05_libero_pytorch checkpoint) and recomputes keys from the stored raw 224x224 library images:
  * validation: 224 px tokens vs the stored library tokens (tok/v0, tok/v1);
  * low resolution: the image is resized to r px (bilinear, antialias) and encoded with interpolated position
    embeddings -> (r/14)^2 tokens -> mean-pooled to the deployed 4x4 grid -> a 32768-d key of the same layout.
Then the k-fold replay (same folds / metric recipe as replay.py) compares, on held-out look rows:
  F        stored full-resolution keys, both cameras (deployed)
  F_r      both cameras at r px
  WA_r     wrist at full resolution + agentview at r px
  W        wrist only (full resolution)
Reported: e10 vs the policy chunk, paired difference to F, served-chunk difference dF, top-1 agreement.
Also a rough batch-1 latency microbenchmark of the vision tower (indicative only; not the serving path).

Run (opus CPUs + the local 4090, < 4 GB VRAM):
  taskset -c 2-9,46-53 env OMP_NUM_THREADS=4 PYTHONPATH=.:src .venv/bin/python -m \
    exp.offline_search.rounds.r12.opus.tools.lowres --cell pi05:l10:50 --res 112,168
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch
from safetensors import safe_open

from exp.offline_search.rounds.r02.g1_awm.awm import pca_fit
from exp.offline_search.rounds.r10.analysis_opus.tools import core
from exp.offline_search.rounds.r12.opus.tools import common as cm

CKPT = "/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors"
PREFIX = "paligemma_with_expert.paligemma.model."


def build(device="cuda:0"):
    from transformers import CONFIG_MAPPING
    from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    cfg = CONFIG_MAPPING["paligemma"]()
    cfg.vision_config.intermediate_size = 4304
    cfg.vision_config.projection_dim = 2048
    cfg.vision_config.projector_hidden_act = "gelu_fast"
    cfg.text_config.hidden_size = 2048
    tower = SiglipVisionModel(cfg.vision_config)
    proj = PaliGemmaMultiModalProjector(cfg)
    f = safe_open(CKPT, "pt")
    sd_t, sd_p = {}, {}
    for k in f.keys():
        if k.startswith(PREFIX + "vision_tower."):
            sd_t[k[len(PREFIX + "vision_tower."):]] = f.get_tensor(k)
        elif k.startswith(PREFIX + "multi_modal_projector."):
            sd_p[k[len(PREFIX + "multi_modal_projector."):]] = f.get_tensor(k)
    mt = tower.load_state_dict(sd_t, strict=True)
    mp_ = proj.load_state_dict(sd_p, strict=True)
    tower = tower.to(device=device, dtype=torch.bfloat16).eval()
    proj = proj.to(device=device, dtype=torch.bfloat16).eval()
    return tower, proj


@torch.inference_mode()
def encode(tower, proj, imgs_u8, res, device="cuda:0", batch=128):
    """uint8 [n,224,224,3] -> pooled 4x4 keys [n, 32768] float32 and per-image token grids side."""
    out = []
    g = res // 14
    for lo in range(0, len(imgs_u8), batch):
        x = torch.from_numpy(np.ascontiguousarray(imgs_u8[lo:lo + batch])).to(device)
        x = x.permute(0, 3, 1, 2).float() / 255.0 * 2.0 - 1.0
        if res != 224:
            x = torch.nn.functional.interpolate(x, size=(res, res), mode="bilinear", antialias=True, align_corners=False)
        h = tower(pixel_values=x.to(torch.bfloat16), interpolate_pos_encoding=(res != 224)).last_hidden_state
        t = proj(h).float()                                            # [b, g*g, 2048]
        t = t.reshape(-1, g, g, 2048)
        if g % 4:
            t = torch.nn.functional.adaptive_avg_pool2d(t.permute(0, 3, 1, 2), 4).permute(0, 2, 3, 1)
        else:
            b = g // 4
            t = t.reshape(-1, 4, b, 4, b, 2048).mean((2, 4))
        out.append(t.reshape(t.shape[0], -1).cpu().numpy())
    return np.concatenate(out)


@torch.inference_mode()
def latency(tower, proj, device="cuda:0", reps=50):
    res = {}
    for r in (224, 168, 112):
        for b in (1, 2):
            x = torch.zeros(b, 3, r, r, device=device, dtype=torch.bfloat16)
            for _ in range(10):
                proj(tower(pixel_values=x, interpolate_pos_encoding=(r != 224)).last_hidden_state)
            torch.cuda.synchronize()
            ts = []
            for _ in range(reps):
                t0 = time.perf_counter()
                proj(tower(pixel_values=x, interpolate_pos_encoding=(r != 224)).last_hidden_state)
                torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3)
            res[f"r{r}_b{b}"] = float(np.median(ts))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", default="pi05:l10:50")
    ap.add_argument("--res", default="112,168")
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args(argv)
    model, suite, size = a.cell.split(":")
    size = int(size)
    if model != "pi05":
        raise SystemExit("raw images are stored for pi0.5 only")
    torch.set_num_threads(4)
    tower, proj = build()
    report = {}
    if a.bench:
        report["latency_ms_batch1_tower_only"] = latency(tower, proj)
        print(report["latency_ms_batch1_tower_only"], flush=True)
    C = cm.load_cell(model, suite)
    rows, fold_of_row = core.subset_and_folds(C, size)
    d = core.lib_dir(model, suite)
    I0 = np.load(d / "tok" / "img0.npy", mmap_mode="r")
    I1 = np.load(d / "tok" / "img1.npy", mmap_mode="r")
    # validation against stored keys on the first 256 subset rows
    K0, K1 = core.gather_keys(model, suite, rows[:256])
    v0 = encode(tower, proj, np.asarray(I0[rows[:256]]), 224)
    v1 = encode(tower, proj, np.asarray(I1[rows[:256]]), 224)
    rel = [float(np.linalg.norm(v - k) / np.linalg.norm(k)) for v, k in ((v0, K0), (v1, K1))]
    cos = [float(np.mean(np.sum(v * k, 1) / np.linalg.norm(v, axis=1) / np.linalg.norm(k, axis=1))) for v, k in ((v0, K0), (v1, K1))]
    report["validation_224"] = dict(rel_err_v0=rel[0], rel_err_v1=rel[1], cos_v0=cos[0], cos_v1=cos[1])
    print("validation", report["validation_224"], flush=True)
    resl = [int(x) for x in a.res.split(",")]
    lowk = {}
    for r in resl:
        t0 = time.time()
        k0 = np.concatenate([encode(tower, proj, np.asarray(I0[rows[lo:lo + 2048]]), r) for lo in range(0, len(rows), 2048)])
        k1 = np.concatenate([encode(tower, proj, np.asarray(I1[rows[lo:lo + 2048]]), r) for lo in range(0, len(rows), 2048)])
        lowk[r] = (k0, k1)
        print(f"encoded res {r}: {time.time() - t0:.0f}s", flush=True)
    del tower, proj
    torch.cuda.empty_cache()
    act, rs = C["act"][rows], C["rs"][rows]
    task, ep, step = C["task"][rows], C["ep"][rows], C["step"][rows]
    sig_r = C["act"][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
    kref = cm.kref_for(size)
    names = ["F", "W"] + [f"F_{r}" for r in resl] + [f"WA_{r}" for r in resl]
    acc = {n: dict(e=[], dF=[], t1=[]) for n in names}
    for fold in range(5):
        tr = fold_of_row != fold
        P0, P1, _ = cm.projections(model, suite, size, fold, rows, tr, keep=128)
        P0, P1 = P0[:, :64].astype(np.float64), P1[:, :64].astype(np.float64)
        L = {}
        for r in resl:
            pr = []
            for k in lowk[r]:
                mu, V, _ = pca_fit(k[tr], keep=64)
                pr.append((k @ V - mu @ V).astype(np.float64))
            L[r] = pr
        sig_m = act[tr][:, :5, :7].reshape(-1, 7).astype(np.float64).std(0)
        heads = (act[:, :5, :7] / sig_m).reshape(len(act), -1)
        feats = {"F": np.concatenate([P0, P1, rs], 1), "W": np.concatenate([P1, rs], 1)}
        for r in resl:
            feats[f"F_{r}"] = np.concatenate([L[r][0], L[r][1], rs], 1)
            feats[f"WA_{r}"] = np.concatenate([L[r][0], P1, rs], 1)
        for t in range(10):
            itr = np.flatnonzero(tr & (task == t))
            ite = np.flatnonzero(~tr & (task == t) & (step % 2 == 0) & (step > 0))
            if not len(ite):
                continue
            served, mem = {}, {}
            for n in names:
                X = feats[n]
                met = core.TaskMetric(X[itr], heads[itr], ep[itr], step[itr])
                s, _, _, m_, _ = core.serve(X[ite], step[ite], ep[ite], met, X[itr], None, ep[itr], act[itr], kref,
                                            exclude_own=False)
                served[n], mem[n] = s, m_
            for n in names:
                acc[n]["e"].append(core.motion_err(act[ite], served[n], sig_r))
                acc[n]["dF"].append(core.motion_err(served["F"], served[n], sig_r))
                acc[n]["t1"].append(mem[n][:, 0] == mem["F"][:, 0])
        print(f"fold {fold} done", flush=True)
    eF = np.concatenate(acc["F"]["e"])
    report["replay"] = {}
    for n in names:
        e = np.concatenate(acc[n]["e"])
        report["replay"][n] = dict(e10=float(e.mean()), d_vs_F=float((e - eF).mean()),
                                   se=float((e - eF).std() / np.sqrt(len(e))),
                                   dF=float(np.concatenate(acc[n]["dF"]).mean()),
                                   top1=float(np.concatenate(acc[n]["t1"]).mean()))
        x = report["replay"][n]
        print(f"{a.cell} {n:7s} e10 {x['e10']:.4f} vs F {x['d_vs_F']:+.4f} (se {x['se']:.4f}) dF {x['dF']:.3f} "
              f"top1 {x['top1']:.3f}", flush=True)
    cm.save_json(report, cm.OUT / f"lowres_{model}_{suite}_{size}.json")


if __name__ == "__main__":
    main()
