"""floor_resample_pi05.py -- teacher noise floor at mid/late steps for pi0.5 (GPU).

For a stratified sample of recorded decisions (floor_common.sample_plan: S//2 inf + rest
cache, even over 10 tasks x 3 step-thirds), rerun the real pi0.5 on the recorded
server-side inputs (raw images, raw state, prompt) once from the recorded
noise_action_0 (parity sample) and K times from fresh N(0,1) noise. The action is
extracted exactly as exp/trace_dual/ops/replay_pi05.py does: forward hooks on
action_in_proj / action_out_proj, clean = x_last - v_last / n_steps, i.e. the model's
normalized action space, the same space as trace/actions/full_inference (policy.infer's
own output is post-processed and is not used).

Output: <out>/pi05_<suite>/resample.npz + resample_summary.json (see floor_common).
Run via run_floor_pi05.sh (env, GPU check, log, DONE/ERROR markers).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import floor_common as fc  # noqa: E402

CKPT = "/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
N_DENOISE = 10


def capture(policy, obs: dict, noise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(clean action (H,32), first action_in_proj input) -- replay_pi05.capture, actions only."""
    model = policy._model  # noqa: SLF001
    a_in, a_out = [], []
    hs = [
        model.action_in_proj.register_forward_hook(lambda m, i, o: a_in.append(i[0].detach().clone())),
        model.action_out_proj.register_forward_hook(lambda m, i, o: a_out.append(o.detach().clone())),
    ]
    try:
        policy.infer(dict(obs), noise=noise)
    finally:
        for h in hs:
            h.remove()
    assert len(a_in) == N_DENOISE == len(a_out), (len(a_in), len(a_out))
    x_last = a_in[-1].squeeze(0).float().cpu(); v_last = a_out[-1].squeeze(0).float().cpu()
    return (x_last + (-1.0 / len(a_in)) * v_last).numpy(), a_in[0].squeeze(0).float().cpu().numpy()


def make_sampler(policy, suite: str, K: int, seed: int):
    def sampler(i: int, row: dict, g) -> dict:
        t = g["trace"]
        wire = json.loads(t["raw_images"].attrs["wire_keys_json"])
        obs = {wire[k]: t["raw_images/" + k][()] for k in t["raw_images"]}
        obs["observation/state"] = t["raw_state"][()]
        obs["prompt"] = str(t.attrs["prompt"])
        n0 = g["noise_action_0"][()]
        a_rep, first_in = capture(policy, obs, n0)
        fresh, noises = [], []
        for k in range(K):
            nz = np.random.default_rng(fc.fresh_seed(seed, "pi05", suite, i, k)).standard_normal(n0.shape).astype(np.float32)
            a, _ = capture(policy, obs, nz)
            fresh.append(a); noises.append(nz)
        return dict(a_replay=a_rep, noise0_exact=bool(np.array_equal(first_in, n0)),
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
    from openpi.policies import policy_config as _pc
    from openpi.training import config as _config
    policy = _pc.create_trained_policy(_config.get_config("pi05_libero"), CKPT, pytorch_device="cuda")
    suites = fc.SUITES if args.suite == "all" else (args.suite,)
    for s in suites:
        summ = fc.run_resample("pi05", s, args.s, args.k, args.seed, pathlib.Path(args.out),
                               make_sampler(policy, s, args.k, args.seed), save_every=args.save_every,
                               log=lambda x: print(x, flush=True))
        fc.print_resample_summary(summ, log=lambda x: print(x, flush=True))
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
