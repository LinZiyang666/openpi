"""floor_resample_groot.py -- teacher noise floor at mid/late steps for GR00T N1.5 (GPU, island venv).

For a stratified sample of recorded decisions (floor_common.sample_plan), feed the recorded
raw observation (video.* frames, state.* split by the stored layout, prompt) through the
policy's own apply_transforms and the staged runner exactly as
exp/trace_dual/ops/replay_groot.py does: stage1 + stage2 once per decision, then stage3
once from the recorded noise_action_0 (parity sample) and K times from fresh noise drawn
by runner.sample_noise with a seeded CUDA generator. action_pred (16,32) is in the same
normalized space as trace/actions/full_inference.

Output: <out>/groot_<suite>/resample.npz + resample_summary.json (see floor_common).
Run via run_floor_groot.sh (island venv, PYTHONPATH, GPU check, log, DONE/ERROR markers).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import zlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import floor_common as fc  # noqa: E402

CKPT = {"spatial": "/data/ckpt/n15_libero_spatial", "l10": "/data/ckpt/n15_libero_10"}


def torch_seed(parts: list[int]) -> int:
    """A 63-bit torch seed from the per-(decision, sample) seed tuple."""
    return zlib.crc32(np.asarray(parts, np.int64).tobytes()) * 2654435761 % (2 ** 63 - 1)


def make_sampler(policy, runner, suite: str, K: int, seed: int):
    def sampler(i: int, row: dict, g) -> dict:
        t = g["trace"]
        obs = {key: t["raw_images/" + key][()][None, None] for key in t["raw_images"]}
        layout = json.loads(t["raw_state"].attrs["layout_json"])
        rs = t["raw_state"][()]; o = 0
        for name, n in layout:
            obs[name] = rs[o:o + n][None, None].astype(np.float64); o += n
        obs["annotation.human.task_description"] = np.array([[str(t.attrs["prompt"])]])
        normalized = policy.apply_transforms(dict(obs))
        with runner.session():
            s1 = runner.run_stage1(normalized)
        with runner.session():
            s2 = runner.run_stage2_llm(s1)

        def loop(noise):
            caps = []
            with runner.session():
                out = runner.run_stage3(s2, noise=noise, on_step=lambda j, x, y: caps.append(x.float().cpu().numpy()[0]))
            return out.action_pred[0].float().cpu().numpy(), caps

        n0 = g["noise_action_0"][()]
        a_rep, caps = loop(torch.from_numpy(n0)[None].to("cuda"))
        fresh, noises = [], []
        for k in range(K):
            gen = torch.Generator(device="cuda").manual_seed(torch_seed(fc.fresh_seed(seed, "groot", suite, i, k)))
            with runner.session():
                nz = runner.sample_noise(s2, generator=gen)
            a, _ = loop(nz)
            fresh.append(a); noises.append(nz[0].float().cpu().numpy())
        return dict(a_replay=a_rep, noise0_exact=bool(np.array_equal(caps[0], n0)),
                    a_fresh=np.stack(fresh), noise_fresh=np.stack(noises))
    return sampler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="all", choices=["spatial", "l10", "all"])
    ap.add_argument("--s", type=int, default=500, help="decisions per model x suite")
    ap.add_argument("--k", type=int, default=4, help="fresh noises per decision")
    ap.add_argument("--seed", type=int, default=fc.DEFAULT_SEED)
    ap.add_argument("--out", default=str(fc.STORE))
    ap.add_argument("--save-every", type=int, default=25)
    args = ap.parse_args()
    from gr00t.model.policy import Gr00tPolicy
    from custom_data_config import LiberoDataConfig
    from openpi.cache.groot.staged import GrootStagedRunner
    suites = fc.SUITES if args.suite == "all" else (args.suite,)
    for s in suites:
        dc = LiberoDataConfig()
        policy = Gr00tPolicy(model_path=CKPT[s], embodiment_tag="new_embodiment",
                             modality_config=dc.modality_config(), modality_transform=dc.transform(),
                             denoising_steps=8, device="cuda")
        runner = GrootStagedRunner(policy.model)
        summ = fc.run_resample("groot", s, args.s, args.k, args.seed, pathlib.Path(args.out),
                               make_sampler(policy, runner, s, args.k, args.seed), save_every=args.save_every,
                               log=lambda x: print(x, flush=True))
        fc.print_resample_summary(summ, log=lambda x: print(x, flush=True))
        del runner, policy
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
