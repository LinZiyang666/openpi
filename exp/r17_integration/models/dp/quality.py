"""CPU-only reproduction of Review B's48 real paired frames and noise stream.

This is an offline action proxy, not a success-rate or held-out quality claim.
K100 is the untouched E DDPM baseline; reduced profiles are trailing DDIM.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .manifest import code_identity, file_digest
from .sampler import SamplerProfile


@torch.inference_mode()
def review_sample(model, conditioning, profile, *, seed=0):
    scheduler = profile.scheduler(model.cfg)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((len(conditioning), 16, model.cfg["action_dim"]), generator=generator)
    for t in scheduler.timesteps:
        x = profile.step(scheduler, model._policy.model(x, t, conditioning), t, x, generator)
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="/data/mw_finetune/dp/checkpoints/libero")
    parser.add_argument("--data", default="/data/mw_finetune/dp/libero_96")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    from safetensors.torch import load_file
    from exp.mw_finetune.dp.checkpoint import read_metadata
    from exp.mw_finetune.dp.data import Dataset
    from exp.mw_finetune.dp.model import DiffusionPolicy
    from exp.mw_finetune.dp.preprocess import quantile_unnormalize
    from exp.r17_integration.common.contracts import RequestContext
    from .fixtures import libero_observation
    from .staged import DPAdapter
    checkpoint, root = Path(args.checkpoint), Path(args.data)
    cfg = read_metadata(checkpoint)["config"]["model"]
    model = DiffusionPolicy(cfg).eval()
    strict = model.load_state_dict(load_file(checkpoint / "model.safetensors"), strict=True)
    assert not strict.missing_keys and not strict.unexpected_keys and len(model.state_dict()) == 475
    embeddings = load_file(root / "language.safetensors")["embeddings"]
    dataset = Dataset(root, embeddings)
    indices = np.random.default_rng(0).integers(len(dataset), size=48)
    items = [dataset[int(i)] for i in indices]
    obs = {key: torch.stack([item["obs"][key] for item in items]) for key in items[0]["obs"]}
    truth = torch.stack([item["action"] for item in items])
    by_prompt = {prompt: embeddings[int(task)] for task, prompt in dataset.manifest["tasks"].items()}
    adapter = DPAdapter(model, dataset.manifest["external_stats"],
                        lambda prompts: torch.stack([by_prompt[p] for p in prompts]),
                        domain="libero", weight_identity=file_digest(checkpoint / "model.safetensors"),
                        language_identity=file_digest(root / "language.safetensors"))
    # Pass both recorded cameras through the adapter, not a synthetic wrist.
    prepared = []
    provenance = []
    for i in indices:
        wire, origin = libero_observation(root, frame=int(i))
        prepared.append(adapter.prepare(wire, RequestContext("p4_quality", "diagnostic", wire["prompt"], 0, 0)))
        provenance.append(origin)
    with torch.inference_mode():
        diagnostic = adapter.for_batch_diagnostic("review48_CPU_offline_action_proxy")
        stages = [diagnostic.run_stage2(s) for s in diagnostic.run_stage1_batch(prepared)]
        condition = torch.cat([s.data["conditioning"] for s in stages])
        assert torch.equal(condition, model.condition(obs))
        baseline_x = review_sample(model, condition, SamplerProfile())
        baseline = model._policy.normalizer.unnormalize("action", baseline_x)
        assert torch.equal(baseline, model.sample(obs, steps=100, generator=torch.Generator().manual_seed(0)))
        outputs = {}
        for k in (1, 2, 4, 10, 100):
            profile = SamplerProfile(num_steps=k)
            x = baseline_x if k == 100 else review_sample(model, condition, profile)
            action = model._policy.normalizer.unnormalize("action", x)
            wire = quantile_unnormalize(action.numpy(), dataset.manifest["external_stats"]["actions"])
            ref_wire = quantile_unnormalize(baseline.numpy(), dataset.manifest["external_stats"]["actions"])
            outputs[str(k)] = {"sampler": profile.descriptor(cfg),
                              "mse_vs_K100_first5": float(((action-baseline)[:, :5]**2).mean()),
                              "mse_vs_K100_all16": float(((action-baseline)**2).mean()),
                              "mse_vs_dataset_first5": float(((action-truth)[:, :5]**2).mean()),
                              "wire_mse_vs_K100_first5": float(((wire-ref_wire)[:, :5]**2).mean()),
                              "internal_clip_fraction": float((x.abs() >= .999).float().mean())}
    receipt = {"schema": "r17.p4.reduced_quality.v1", "device": "cpu", "gpu_executed": False,
               "strict_load": {"keys": 475, "missing": [], "unexpected": []},
               "frames": 48, "frame_indices": [int(i) for i in indices], "sampling_seed": 0,
               "noise_protocol": "reviewer single sequential generator, B48 initial+posterior",
               "batch_diagnostic_label": diagnostic.batch_diagnostic_label,
               "production_batching": "unsupported; CPU B48 offline proxy only",
               "action_units": "external quantile-normalized (internal affine reversed), same as reviewer",
               "provenance": provenance, "source_digests": code_identity(),
               "weight_identity": adapter.weight_identity,
               "adapter_condition_exact": True, "E_DDPM100_exact": True, "results": outputs,
               "quality_gate": "unqualified; training-frame proxy only, no SR or held-out/descent approval",
               "elapsed_seconds": time.monotonic()-started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"frames": 48, "elapsed_seconds": receipt["elapsed_seconds"],
                      "mse_vs_K100_first5": {k: v["mse_vs_K100_first5"] for k, v in outputs.items()}}, indent=2))


if __name__ == "__main__":
    main()
