"""replay_groot.py -- re-run the real GR00T N1.5 on recorded trace inputs (GR00T island venv).

The recorded raw observation (video.* frames, state.* split by the stored layout, the
prompt) goes through the policy's own apply_transforms and the staged runner exactly as
the traced server path does; the flow loop is started from the recorded noise_action_0
and observed with on_step. Compared: prefix vision/prompt tokens, robot_state, every loop
input (noise_action_1..7) and the final chunk (clean_action). A second loop from fresh
noise gives the scale of a different sample. Writes audit/replay_groot.json.
"""
import glob, json, pathlib, random, sys
import numpy as np, h5py, torch

R = pathlib.Path("/home/weiland/trace_runs/dual_20260923")
CKPT = {"sp": "/data/ckpt/n15_libero_spatial", "l10": "/data/ckpt/n15_libero_10"}
EPS_PER_ARM = 3


def rel(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12))


def main():
    from gr00t.model.policy import Gr00tPolicy
    from custom_data_config import LiberoDataConfig
    from openpi.cache.groot.staged import GrootStagedRunner
    from openpi.cache.trace import groot as _tg

    rng = random.Random(20260924)
    rows = []
    for suite in ("sp", "l10"):
        dc = LiberoDataConfig()
        policy = Gr00tPolicy(model_path=CKPT[suite], embodiment_tag="new_embodiment",
                             modality_config=dc.modality_config(), modality_transform=dc.transform(),
                             denoising_steps=8, device="cuda")
        runner = GrootStagedRunner(policy.model)
        for arm in (f"tr_groot_{suite}_inf", f"tr_groot_{suite}_cache"):
            files = sorted(glob.glob(f"{R}/runs/{arm}/trace/**/*.h5", recursive=True))
            for p in rng.sample(files, EPS_PER_ARM):
                with h5py.File(p, "r") as f:
                    steps = sorted(k for k in f if k.startswith("step_"))
                    for k in sorted({steps[0], steps[len(steps) // 2], steps[-1]}):
                        g = f[k]; t = g["trace"]
                        obs = {key: t["raw_images/" + key][()][None, None] for key in t["raw_images"]}
                        layout = json.loads(t["raw_state"].attrs["layout_json"])
                        rs = t["raw_state"][()]; o = 0
                        for name, n in layout:
                            obs[name] = rs[o:o + n][None, None].astype(np.float64); o += n
                        obs["annotation.human.task_description"] = np.array([[str(t.attrs["prompt"])]])
                        normalized = policy.apply_transforms(dict(obs))
                        with runner.session():
                            s1 = runner.run_stage1(normalized)
                        vision, prompt_emb, robot_state = _tg.slice_prefix_tokens(
                            s1, vision_fields=("vision_0", "vision_1"), expected_state_index=None)
                        with runner.session():
                            s2 = runner.run_stage2_llm(s1)

                        def loop(noise):
                            caps = []
                            with runner.session():
                                out = runner.run_stage3(s2, noise=noise, on_step=lambda i, x, y: caps.append(x.float().cpu().numpy()[0]))
                            return out.action_pred[0].float().cpu().numpy(), caps

                        n0 = torch.from_numpy(g["noise_action_0"][()])[None].to("cuda")
                        pred, caps = loop(n0)
                        with runner.session():
                            fresh = runner.sample_noise(s2, generator=torch.Generator(device="cuda").manual_seed(len(rows)))
                        pred_alt, _ = loop(fresh)
                        rec_noise = [g[f"noise_action_{i}"][()] for i in range(len(caps))]
                        rec_clean = g["clean_action"][()]
                        row = dict(
                            arm=arm, file=pathlib.Path(p).name, step=k,
                            vision_rel=[rel(vision[i], g[f"vision_{i}"][()]) for i in range(2)],
                            prompt_emb_rel=rel(prompt_emb, g["prompt_emb"][()]) if prompt_emb.shape == g["prompt_emb"].shape else None,
                            robot_state_equal=bool(np.allclose(robot_state, g["robot_state"][()], atol=1e-6)),
                            noise0_exact=bool(np.array_equal(caps[0], rec_noise[0])),
                            noise_rel_max=max(rel(caps[i], rec_noise[i]) for i in range(1, len(caps))),
                            clean_rel=rel(pred, rec_clean), clean_rel_fresh_noise=rel(pred_alt, rec_clean),
                            loop_steps=len(caps),
                        )
                        rows.append(row)
                        print(json.dumps(row), flush=True)
        del runner, policy
        torch.cuda.empty_cache()
    keys = ["clean_rel", "clean_rel_fresh_noise", "noise_rel_max"]
    summ = {k: [min(r[k] for r in rows), float(np.mean([r[k] for r in rows])), max(r[k] for r in rows)] for k in keys}
    summ["vision_rel_max"] = max(max(r["vision_rel"]) for r in rows)
    summ["prompt_emb_rel_max"] = max((r["prompt_emb_rel"] or 0) for r in rows)
    summ["noise0_exact"] = f"{sum(r['noise0_exact'] for r in rows)}/{len(rows)}"
    summ["robot_state_equal"] = f"{sum(r['robot_state_equal'] for r in rows)}/{len(rows)}"
    (R / "audit" / "replay_groot.json").write_text(json.dumps(dict(summary=summ, rows=rows), indent=1))
    print("SUMMARY", json.dumps(summ))


if __name__ == "__main__":
    main()
