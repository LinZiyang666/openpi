"""R12 opus: indicative pi0.5 stage-1 latency, stock (cameras embedded one by one) vs batched cameras.

Local RTX 4090, batch-1 request, eager model, same observations and convention as R7's
`rounds/r07/analysis_scripts/a3_wrist_latency.py` (12 library observations x 30 timed repetitions). NOT the serving
path (no batcher / per-request dispatch); the coordinator's REQUESTS-0 measurement is authoritative.

  stock      model.run_stage1 (embed_prefix loops over base, wrist, masked dummy)
  batched    base + wrist embedded in ONE embed_image call, masked dummy slot from a cached embedding, prefix
             assembled exactly like stock (order base, wrist, dummy, language)
  wrist      wrist-only stage 1 (R4/R7 override path: wrist embed + cached dummy + zero base)
  one_embed  a single embed_image call (completion-like)
Also checks that the batched prefix equals the stock prefix (max abs / relative difference) and that the policy's
action chunk from the batched prefix matches the stock one with the same noise.
"""
from __future__ import annotations

import dataclasses
import json
import math
import statistics
import time

import numpy as np


def main():
    import jax
    import torch
    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import Stage1Output, make_att_2d_masks
    from openpi.policies import policy_config
    from openpi.training import config
    from exp.offline_search.harness.store import QueryCell
    from exp.offline_search.rounds.r12.opus.tools import common as cm

    torch.set_num_threads(4)
    device = "cuda:0"
    cfg = config.get_config("pi05_libero")
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    ckpt = "/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
    policy = policy_config.create_trained_policy(cfg, ckpt, pytorch_device=device)
    model = policy._model.eval()
    tower = model.paligemma_with_expert
    dummy_cache = {}

    def prefix_from(images, masks, tokens, lang_mask, state, embs):
        lang = tower.embed_language_tokens(tokens)
        lang = lang * math.sqrt(lang.shape[-1])
        prefix = torch.cat((*embs, lang), dim=1)
        pad = torch.cat([m[:, None].expand(e.shape[:2]) for m, e in zip(masks, embs)] + [lang_mask], dim=1)
        att = make_att_2d_masks(pad, torch.zeros_like(pad))
        return Stage1Output(state=state, prefix_embs=prefix, prefix_pad_masks=pad,
                            prefix_att_2d_masks_4d=model._prepare_attention_masks_4d(att),
                            prefix_position_ids=torch.cumsum(pad, dim=1) - 1)

    def dummy(img):
        k = tuple(img.shape)
        if k not in dummy_cache:
            dummy_cache[k] = tower.embed_image(img).detach()
        return dummy_cache[k]

    def batched(obs):
        images, masks, tokens, lang_mask, state = model._preprocess_observation(obs, train=False)
        both = tower.embed_image(torch.cat([images[0], images[1]], 0))
        return prefix_from(images, masks, tokens, lang_mask, state, [both[:1], both[1:], dummy(images[2])])

    def wrist(obs):
        images, masks, tokens, lang_mask, state = model._preprocess_observation(obs, train=False)
        w = tower.embed_image(images[1])
        return prefix_from(images, masks, tokens, lang_mask, state, [torch.zeros_like(w), w, dummy(images[2])])

    def one_embed(obs):
        images, *_ = model._preprocess_observation(obs, train=False)
        return tower.embed_image(images[0])

    def timed(fn, warm=5, reps=30):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize()
        out = []
        for _ in range(reps):
            t = time.perf_counter()
            fn()
            torch.cuda.synchronize()
            out.append((time.perf_counter() - t) * 1000)
        return statistics.median(out)

    rows = []
    with torch.inference_mode():
        for suite in ("l10", "spatial"):
            qc = QueryCell("/home/weiland/trace_runs/offline_search_store", f"pi05_{suite}_cache")
            episodes = [ep for ep in qc.episodes if qc.tok_index[ep["start"]] >= 0]
            for ep in (episodes[0], episodes[len(episodes) // 2]):
                for row in (ep["start"], ep["start"] + 1, ep["end"] - 1):
                    ti = int(qc.tok_index[row])
                    raw = {"observation/image": np.asarray(qc.tok("img0")[ti]),
                           "observation/wrist_image": np.asarray(qc.tok("img1")[ti]),
                           "observation/state": np.asarray(qc.raw_state[row], np.float64), "prompt": ep["task"]}
                    inp = policy._input_transform(raw)
                    obs = Observation.from_dict(jax.tree.map(lambda x: torch.as_tensor(np.array(x), device=device)[None], inp))
                    s = model.run_stage1(obs)
                    b = batched(obs)
                    d = (b.prefix_embs.float() - s.prefix_embs.float())
                    rel = float(d.norm() / s.prefix_embs.float().norm())
                    noise = torch.randn(1, model.config.action_horizon, model.config.action_dim, device=device)
                    a_s = model.run_stage3(model.run_stage2(s), num_steps=10, noise=noise) if "noise" in model.run_stage3.__code__.co_varnames else None
                    a_b = model.run_stage3(model.run_stage2(b), num_steps=10, noise=noise) if a_s is not None else None
                    rec = dict(suite=suite, row=int(row), stock_ms=timed(lambda: model.run_stage1(obs)),
                               batched_ms=timed(lambda: batched(obs)), wrist_ms=timed(lambda: wrist(obs)),
                               one_embed_ms=timed(lambda: one_embed(obs)), prefix_rel_diff=rel,
                               prefix_max_abs=float(d.abs().max()),
                               action_max_abs=None if a_s is None else float((a_s.float() - a_b.float()).abs().max()))
                    rows.append(rec)
                    print(json.dumps(rec), flush=True)
    med = {k: statistics.median(r[k] for r in rows) for k in ("stock_ms", "batched_ms", "wrist_ms", "one_embed_ms")}
    report = dict(rows=rows, median_ms=med,
                  price_owner_basis={k: .152 * v / med["stock_ms"] for k, v in med.items()},
                  note="indicative: eager batch-1 on the idle local RTX 4090, model.run_stage1 vs a batched-camera "
                       "variant; not the per-request serving path")
    cm.save_json(report, cm.OUT / "bench_stage1.json")
    print(json.dumps({k: report[k] for k in ("median_ms", "price_owner_basis")}, indent=1))


if __name__ == "__main__":
    main()
