"""R12 opus (directive 2): how far is a GR00T camera encoded ALONE from its slice of the joint two-camera encode?

The deployed GR00T stage 1 runs `extract_feature` on both images at once; the library keys come from that joint
path. A one-camera look would encode one image alone. Inputs: B-pool pi0.5 library RGB (LIBERO renders of the same
tasks; GR00T B libraries store no RGB), resized 224 -> 256 and passed through the unchanged GR00T transforms, as in
the coordinator's `prep/bench_latency.py`. Zero physical state (vision only). GPU <= 10 GB.

Per observation and camera: joint key (4x4 pool of the joint tokens), single key, and a second joint run (numerical
floor). Reported in the GR00T library's deployed PCA-64 space (R10 artifact for groot_<suite>_50):
  rel_key      ||single - joint|| / ||joint|| on raw keys
  shift        ||P(single) - P(joint)|| in PCA-64
  nn_dist      distance of P(joint) to its nearest library row (all rows) in PCA-64
  top16_overlap  overlap of the 16 nearest library rows (vision PCA-64 space, all tasks) for single vs joint
and the same after subtracting the mean single-minus-joint offset estimated on the other half of the observations.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
STORE = Path("/home/weiland/trace_runs/offline_search_store")
ART = HERE.parents[1] / "r10" / "recipe" / "artifacts"


def pool(tok):
    t = tok.float().reshape(-1, 4, 4, 4, 4, tok.shape[-1]).mean((2, 4))
    return t.reshape(t.shape[0], -1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="l10")
    ap.add_argument("--n", type=int, default=200)
    a = ap.parse_args()
    import cv2
    import torch
    from gr00t.model.policy import Gr00tPolicy
    from custom_data_config import LiberoDataConfig
    from openpi.cache.groot.staged import GrootStagedRunner
    from openpi.cache.groot.interceptor import _unsqueeze_values
    from exp.libero_groot.policy_adapter import build_groot_observation
    from exp.offline_search.harness.store import LibraryView

    ck = f"/data/ckpt/n15_libero_{'10' if a.suite == 'l10' else 'spatial'}"
    cfg = LiberoDataConfig()
    policy = Gr00tPolicy(model_path=ck, embodiment_tag="new_embodiment", modality_config=cfg.modality_config(),
                         modality_transform=cfg.transform(), denoising_steps=8, device="cuda")
    runner = GrootStagedRunner(policy.model, compile_vision=False)
    eagle = runner._eagle
    lib = LibraryView(STORE, f"pi05_{a.suite}", "bpool_cs")
    rng = np.random.default_rng(20261003)
    rows = np.sort(rng.choice(lib.tok_rows, size=a.n, replace=False))
    J, S, J2 = {0: [], 1: []}, {0: [], 1: []}, {0: [], 1: []}
    for row in rows:
        ims = [cv2.resize(np.array(lib.tok_of(nm, int(row)), np.uint8), (256, 256), interpolation=cv2.INTER_LINEAR)
               for nm in ("img0", "img1")]
        ep = next(e for e in lib.episodes if e["start"] <= row < e["end"])
        raw = {"observation/image": ims[0], "observation/wrist_image": ims[1],
               "observation/state": np.zeros(8, np.float64), "prompt": ep["task"]}
        inp = _unsqueeze_values(build_groot_observation(raw))
        inp = {k: v if isinstance(v, np.ndarray) else np.array(v) for k, v in inp.items()}
        norm = policy.apply_transforms(inp)
        with runner.session(), torch.inference_mode():
            backbone, _ = policy.model.prepare_input(norm)
            pix = backbone["eagle_pixel_values"]
            assert pix.shape[0] == 2
            joint = pool(eagle.extract_feature(pix))
            joint2 = pool(eagle.extract_feature(pix))
            for cam in (0, 1):
                single = pool(eagle.extract_feature(pix[cam:cam + 1]))[0]
                J[cam].append(joint[cam]); J2[cam].append(joint2[cam]); S[cam].append(single)
    out = {}
    r = pickle.load(open(ART / f"r10_recipe_groot_{a.suite}_50.pkl", "rb"))["method"]
    b = r.inner.base
    rows_lib = np.asarray(r.row_subset)
    for cam, (BT, mu) in ((0, (b.B0T, b.muB0)), (1, (b.B1T, b.muB1))):
        Kl = np.load(STORE / "library" / f"groot_{a.suite}" / "bpool_all" / f"key_v{cam}.npy", mmap_mode="r")
        PL = (np.asarray(Kl[rows_lib], np.float32) @ BT.T - mu).astype(np.float64)
        j, s, j2 = (np.asarray(x[cam], np.float64) for x in (J, S, J2))
        Pj, Ps, Pj2 = (x @ BT.T.astype(np.float64) - mu for x in (j, s, j2))
        half = len(j) // 2
        off = (Ps[:half] - Pj[:half]).mean(0)                      # offset fitted on the first half

        def stats(Pq, Pref, idx):
            d_ref = ((Pref[idx][:, None] - PL[None]) ** 2).sum(-1)
            d_q = ((Pq[idx][:, None] - PL[None]) ** 2).sum(-1)
            o = [len(set(np.argsort(d_ref[i])[:16]) & set(np.argsort(d_q[i])[:16])) / 16 for i in range(len(idx))]
            t1 = [int(np.argmin(d_ref[i]) == np.argmin(d_q[i])) for i in range(len(idx))]
            return dict(shift=float(np.linalg.norm(Pq[idx] - Pref[idx], axis=1).mean()),
                        nn_dist=float(np.sqrt(d_ref.min(1)).mean()), top16_overlap=float(np.mean(o)),
                        top1_equal=float(np.mean(t1)))
        test = np.arange(half, len(j))
        out[f"v{cam}"] = dict(
            rel_key_single=float(np.mean(np.linalg.norm(s - j, axis=1) / np.linalg.norm(j, axis=1))),
            rel_key_joint_rerun=float(np.mean(np.linalg.norm(j2 - j, axis=1) / np.linalg.norm(j, axis=1))),
            single=stats(Ps, Pj, test), single_offset_corrected=stats(Ps - off, Pj, test),
            joint_rerun=stats(Pj2, Pj, test), offset_norm=float(np.linalg.norm(off)))
        print(f"v{cam}", json.dumps(out[f'v{cam}']), flush=True)
    out["note"] = "pi0.5 B-pool RGB through the GR00T transform; vision-only PCA space of the groot 50-episode library"
    Path(HERE / "out").mkdir(exist_ok=True)
    (HERE / "out" / f"groot_mismatch_{a.suite}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
