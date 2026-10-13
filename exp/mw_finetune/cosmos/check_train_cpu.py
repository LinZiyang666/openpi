"""CPU audit of the composed native config, real samples and native loss masks."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from .mt50_data import MT50Dataset, proprio
from .train_mt50 import experiment_options, parse_args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="/data/mw_finetune/fastwam/mt50_224")
    p.add_argument("--assets", default="/data/mw_finetune/cosmos/mt50_assets")
    p.add_argument("--output", default="/data/mw_finetune/cosmos/mt50_assets/native_cpu_check.json")
    args = p.parse_args()
    from exp.r17_integration.models.cosmos.loader import LoadSpec, tokenizer_path
    from exp.r17_integration.models.cosmos.metaworld import offline_checkpoint_path, make_data_batch, inference_config
    a = parse_args(["--output", "/unused"])
    vae = str(tokenizer_path(LoadSpec(a.init)).absolute())
    resolve = lambda uri: offline_checkpoint_path(uri, policy_path=a.init, vae_path=vae)
    # Unused Qwen registrations assert GPU-dependent FA availability at import.
    # FA is installed in this island; bypass only that CPU import assertion.
    with patch("transformers.utils.is_flash_attn_2_available", return_value=True), patch(
        "torch.cuda.current_device", return_value=0), patch(
        "cosmos_policy._src.imaginaire.utils.checkpoint_db.get_checkpoint_path", side_effect=resolve):
        from cosmos_policy.config.config import make_config
        from cosmos_policy._src.imaginaire.utils.config_helper import override
        config = override(make_config(), ["--", "experiment=cosmos_predict2_2b_480p_libero"] + experiment_options(a, vae))
    # validate() broadcasts a job-name byte tensor through hardcoded .cuda().
    with patch.object(torch.Tensor, "cuda", lambda self, *args, **kw: self):
        config.validate()
    c = config.model.config
    assert (c.state_t, c.min_num_conditional_frames, c.max_num_conditional_frames, c.tokenizer.chunk_duration) == (7, 3, 3, 25)
    assert (c.sde.sigma_min, c.sde.sigma_max) == (.01, 200)
    assert c.mask_loss_for_action_future_state_prediction and c.conditioner.text.dropout_rate == 0
    assert config.optimizer.lr == 3e-5 and config.optimizer.master_weights
    assert c.net.sac_config.mode == "block_wise"

    from cosmos_policy.models.policy_text2world_model import CosmosPolicyDiffusionModel
    b, channels, frames, height, width = 4, 16, 7, 28, 28
    prediction = torch.zeros(b, channels, frames, height, width, requires_grad=True)
    fake = SimpleNamespace(config=c,
        sde=SimpleNamespace(marginal_prob=lambda x, sigma: (x, sigma)),
        denoise=lambda *args: SimpleNamespace(x0=prediction),
        get_per_sigma_loss_weights=lambda sigma: torch.ones_like(sigma))
    vector = lambda x: torch.tensor(x)
    indices = torch.full((b,), 3, dtype=torch.long)
    _, loss, _, _ = CosmosPolicyDiffusionModel.compute_loss_with_epsilon_and_sigma(fake,
        torch.ones_like(prediction), SimpleNamespace(), torch.zeros_like(prediction), torch.ones(b, frames),
        torch.ones(b, 16, 7), indices, torch.ones(b, 9), torch.full((b,), 1),
        torch.ones(b, 9), torch.full((b,), 4), torch.full((b,), -1), torch.full((b,), -1),
        torch.full((b,), 5), torch.full((b,), -1), vector([0, 0, 1, 1]), vector([0, 0, 1, 0]),
        vector([0, 0, 0, 1]), torch.ones(b), torch.full((b,), 6))
    gradient = torch.autograd.grad(loss.sum(), prediction)[0].abs().sum((1, 3, 4))
    active = [torch.nonzero(row).flatten().tolist() for row in gradient]
    assert active == [[3], [3], [4, 5], [6]], active

    import cosmos_policy.experiments.robot.cosmos_utils as cu
    dataset = MT50Dataset(args.prepared, args.assets, augment=False)
    sample = dataset[(0, 0)]
    # Tail windows repeat the last action/state/image and cannot cross episodes.
    _, first_per_task = np.unique(dataset.index["task_index"], return_index=True)
    for first in first_per_task:
        last = int(dataset.end_for_frame[first]) - 1
        tail = dataset[(last, 0)]
        assert np.all(tail["actions"] == tail["actions"][0])
        assert np.array_equal(tail["proprio"], tail["future_proprio"])
        assert torch.equal(tail["video"][:, 5:9], tail["video"][:, 17:21])
    assert len(first_per_task) == 50
    with torch.no_grad():
        # Original stored image is already the bilinear resize of the wire.
        cfg = inference_config(SimpleNamespace(checkpoint="/unused"))
        cfg.use_jpeg_compression = False
        image = dataset._image(0)
        native = {"primary_image": image, "proprio": proprio(dataset.index["state"][0])}
        stats = {k: np.asarray(v) for k, v in dataset.stats.items()}
        batch = make_data_batch(native, sample["t5_text_embeddings"][None], cfg, stats,
                               SimpleNamespace(tensor_kwargs={"device": "cpu"}), cu, torch)
        assert torch.equal(batch["video"][0, :, :9], sample["video"][:, :9])
        torch.testing.assert_close(batch["proprio"][0].float(), torch.as_tensor(sample["proprio"]), atol=.004, rtol=0)
    assert not torch.cuda.is_initialized()
    real_shapes = []
    real_path = Path("/data/mw_finetune/models/cosmos_mw/real_mt50_frames.npz")
    if real_path.exists():
        from exp.r17_integration.models.cosmos.metaworld import native_observation
        with np.load(real_path) as real:
            cfg.image_resize = "bilinear"
            for image, state, prompt in zip(real["image"], real["state"], real["prompt"]):
                native, text, _ = native_observation({"observation/image": image, "observation/state": state,
                                                       "prompt": str(prompt)})
                batch = make_data_batch(native, dataset.embeddings[text], cfg, stats,
                    SimpleNamespace(tensor_kwargs={"device": "cpu"}), cu, torch)
                assert tuple(batch["video"].shape) == (1, 3, 25, 224, 224)
                real_shapes.append(list(batch["video"].shape))
    assert not torch.cuda.is_initialized()
    report = dict(passed=True, cuda_initialized=False, frames=len(dataset),
        native_shape=list(sample["video"].shape), action_shape=list(sample["actions"].shape),
        loss_gradient_frames=active, conditioning_pixel_parity=True,
        task_tail_windows_checked=len(first_per_task), real_env_frame_shapes=real_shapes,
        optimizer=dict(lr=config.optimizer.lr, master_weights=config.optimizer.master_weights),
        sigma_min=c.sde.sigma_min, sigma_max=c.sde.sigma_max,
        activation_checkpoint=c.net.sac_config.mode)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
