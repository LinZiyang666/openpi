"""GPU / simulator smoke for the x0-head experiment inside the official DP env (not a pytest; plan §7):
    python -m exp.dp_nfe.smoke_x0 train  --cell <cell yaml> --out <dir> --steps 20     # fresh run, then a second launch is a
                                                                                       # verified no-op (same final sha)
    python -m exp.dp_nfe.smoke_x0 eval   --ckpt <final.ckpt> --out <dir> [--runner-override ...]  # 2 episodes per sampler;
                                                                                       # identity comes from the payload
    python -m exp.dp_nfe.smoke_x0 runner --task <dp task cfg> [--runner-override ...]   # 2-episode closed loop of a runner
Prints ``SMOKE <what> OK`` lines and peak GPU memory; any exception is fatal (non-zero exit)."""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)  # the official configs use ${eval:...}


def _run(cmd):
    print("+", " ".join(cmd), flush=True)
    t0 = time.time()
    subprocess.run(cmd, check=True)
    return time.time() - t0


def main() -> None:
    """CLI: ``train`` / ``eval`` / ``runner`` smoke of the DP-env entry points (asserts; non-zero exit on failure)."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("what", choices=["train", "eval", "runner"])
    ap.add_argument("--dp-root", default=os.environ.get("DP_ROOT"))
    ap.add_argument("--cell"); ap.add_argument("--out"); ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--ckpt"); ap.add_argument("--task"); ap.add_argument("--runner-override", action="append", default=[])
    a = ap.parse_args()
    py = sys.executable
    if a.what == "train":
        out = pathlib.Path(a.out)
        t = _run([py, "-m", "exp.dp_nfe.train_x0", "--dp-root", a.dp_root, "--cell", a.cell, "--out", str(out), "--budget", str(a.steps)])
        final = out / "checkpoints" / "final.ckpt"
        assert final.exists() and (out / "checkpoints" / "final.done").exists()
        # a second invocation must be a verified no-op: identical final.ckpt (sha256 recorded in final.done) and a
        # manifest that was not rewritten
        from exp.dp_nfe.x0_identity import sha256_file
        sha = sha256_file(final); done = json.loads((out / "checkpoints" / "final.done").read_text())
        assert done["checkpoint_sha256"] == sha and done["global_step"] == a.steps and done["finished"]
        man_before = (out / "manifest.json").read_bytes()
        _run([py, "-m", "exp.dp_nfe.train_x0", "--dp-root", a.dp_root, "--cell", a.cell, "--out", str(out), "--budget", str(a.steps)])
        assert sha256_file(final) == sha and (out / "manifest.json").read_bytes() == man_before
        print(f"SMOKE train OK steps={a.steps} wall={t:.0f}s out={out} final_sha={sha[:12]}")
    elif a.what == "eval":
        out = pathlib.Path(a.out)
        for sampler, k in (("ddpm", 100), ("ddim", 100), ("ddim", 1)):
            d = out / f"{sampler}_{k}"
            t = _run([py, "-m", "exp.dp_nfe.eval_dp_steps_v2", "--dp-root", a.dp_root, "-c", a.ckpt, "-o", str(d),
                      "--sampler", sampler, "--k", str(k), "--split", "pilot", "--n-test", "2", "--n-envs", "2"]
                     + sum([["--runner-override", o] for o in a.runner_override], []))
            s = json.loads((d / "summary.json").read_text())
            assert s["complete"] and len(s["episodes"]) == 2 and s["sampler"]["protocol_id"] == "trailing_v1"
            assert sorted(s["episodes"]) == sorted(s["episode_ids_expected"]) and "task_name" in s["cell"]
            print(f"SMOKE eval {sampler}-{k} OK cell={s['cell'].get('cell_id')} mean={s['mean_score']:.3f} "
                  f"nfe={s['manifest']['nfe_total']} wall={t:.0f}s")
    else:
        sys.path.insert(0, a.dp_root); os.chdir(a.dp_root)
        import hydra
        from hydra import compose, initialize_config_dir
        with initialize_config_dir(config_dir=str(pathlib.Path(a.dp_root) / "diffusion_policy" / "config"), version_base=None):
            cfg = compose(config_name="train_diffusion_unet_lowdim_workspace", overrides=[f"task={a.task}"])
        OmegaConf.set_struct(cfg, False)
        r = cfg.task.env_runner; r.n_train = 0; r.n_train_vis = 0; r.n_test = 2; r.n_test_vis = 0; r.n_envs = 2
        for o in a.runner_override:
            k, v = o.split("=", 1); OmegaConf.update(r, k.replace("task.env_runner.", ""), v, merge=True)
        runner = hydra.utils.instantiate(r, output_dir="/tmp/x0/runner_smoke")

        class Zero:  # a policy stub that returns zero actions of the right shape
            device = torch.device("cpu"); dtype = torch.float32
            def reset(self): pass
            def predict_action(self, obs):
                b = next(iter(obs.values())).shape[0]
                return {"action": torch.zeros(b, cfg.n_action_steps, cfg.task.action_dim)}
        log = runner.run(Zero())
        print(f"SMOKE runner {a.task} OK keys={[k for k in log if 'mean_score' in k]} per_episode={sum(1 for k in log if 'sim_max_reward' in k)}")


if __name__ == "__main__":
    main()
