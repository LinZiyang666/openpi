"""replay_pi05.py -- re-run the real pi0.5 on recorded trace inputs and compare every recorded tensor.

Sample: first / middle / last step of a few episodes of each pi0.5 group. Inputs are the
recorded server-side observation (raw images, raw state, prompt) and the recorded
noise_action_0; forward hooks capture the prefix tokens and the flow loop exactly as the
GPU parity gate does (tests/cache/trace/test_trace_collect_parity_gpu.py). A second run
with fresh noise gives the scale of a genuinely different sample. Writes audit/replay_pi05.json.
"""
import glob, json, math, pathlib, random
import numpy as np, h5py, torch

R = pathlib.Path("/home/weiland/trace_runs/dual_20260923")
CKPT = "/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"
ARMS = ["tr_pi05_sp_inf", "tr_pi05_sp_cache", "tr_pi05_l10_inf", "tr_pi05_l10_cache"]
EPS_PER_ARM = 3


def capture(policy, obs, noise):
    model = policy._model  # noqa: SLF001
    vision, lang, a_in, a_out = [], [None], [], []
    hs = [
        model.paligemma_with_expert.paligemma.multi_modal_projector.register_forward_hook(lambda m, i, o: vision.append(o.detach().clone())),
        model.paligemma_with_expert.paligemma.language_model.embed_tokens.register_forward_hook(
            lambda m, i, o: lang.__setitem__(0, (o * math.sqrt(o.shape[-1])).detach().clone())),
        model.action_in_proj.register_forward_hook(lambda m, i, o: a_in.append(i[0].detach().clone())),
        model.action_out_proj.register_forward_hook(lambda m, i, o: a_out.append(o.detach().clone())),
    ]
    try:
        policy.infer(dict(obs), noise=noise)
    finally:
        for h in hs: h.remove()
    x_last = a_in[-1].squeeze(0).float().cpu(); v_last = a_out[-1].squeeze(0).float().cpu()
    return dict(
        vision=[v.squeeze(0).cpu().to(torch.float16).numpy() for v in vision],
        prompt_emb=lang[0].squeeze(0).cpu().to(torch.float16).numpy(),
        noise=[a.squeeze(0).float().cpu().numpy() for a in a_in],
        clean=(x_last + (-1.0 / len(a_in)) * v_last).numpy(),
    )


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main():
    from openpi.policies import policy_config as _pc
    from openpi.training import config as _config
    policy = _pc.create_trained_policy(_config.get_config("pi05_libero"), CKPT, pytorch_device="cuda")
    rng = random.Random(20260924)
    rows = []
    for arm in ARMS:
        files = sorted(glob.glob(f"{R}/runs/{arm}/trace/**/*.h5", recursive=True))
        for p in rng.sample(files, EPS_PER_ARM):
            with h5py.File(p, "r") as f:
                steps = sorted(k for k in f if k.startswith("step_"))
                for k in sorted({steps[0], steps[len(steps) // 2], steps[-1]}):
                    g = f[k]; t = g["trace"]
                    obs = {"observation/image": t["raw_images/observation%2Fimage"][()],
                           "observation/wrist_image": t["raw_images/observation%2Fwrist_image"][()],
                           "observation/state": t["raw_state"][()], "prompt": str(t.attrs["prompt"])}
                    n0 = g["noise_action_0"][()]
                    rec_noise = [g[f"noise_action_{i}"][()] for i in range(10)]
                    rec_clean = g["clean_action"][()]
                    rep = capture(policy, obs, n0)
                    alt = capture(policy, obs, np.random.default_rng(len(rows)).standard_normal(n0.shape).astype(np.float32))
                    row = dict(
                        arm=arm, file=pathlib.Path(p).name, step=k,
                        vision_rel=[rel(rep["vision"][i], g[f"vision_{i}"][()]) for i in range(3)],
                        prompt_emb_rel=rel(rep["prompt_emb"][: g["prompt_emb"].shape[0]], g["prompt_emb"][()]),
                        noise0_exact=bool(np.array_equal(rep["noise"][0], rec_noise[0])),
                        noise_rel_max=max(rel(rep["noise"][i], rec_noise[i]) for i in range(1, 10)),
                        clean_rel=rel(rep["clean"], rec_clean),
                        clean_rel_fresh_noise=rel(alt["clean"], rec_clean),
                        full_inference_eq_clean=bool(np.array_equal(t["actions/full_inference"][()], rec_clean)),
                    )
                    rows.append(row)
                    print(json.dumps(row), flush=True)
    keys = ["clean_rel", "clean_rel_fresh_noise", "noise_rel_max", "prompt_emb_rel"]
    summ = {k: [min(r[k] for r in rows), float(np.mean([r[k] for r in rows])), max(r[k] for r in rows)] for k in keys}
    summ["vision_rel_max"] = max(max(r["vision_rel"]) for r in rows)
    summ["noise0_exact"] = f"{sum(r['noise0_exact'] for r in rows)}/{len(rows)}"
    (R / "audit" / "replay_pi05.json").write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))
    print("SUMMARY", json.dumps(summ))


if __name__ == "__main__":
    main()
