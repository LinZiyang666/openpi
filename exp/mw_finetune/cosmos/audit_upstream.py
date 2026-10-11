"""CPU-only parity check against upstream get_action's two-camera batch.

Run in the Cosmos venv with CUDA_VISIBLE_DEVICES=''. This isolates the
single-camera generalization from errors in the copied conditioning recipe.
"""
from types import SimpleNamespace
from unittest.mock import patch
import json
from pathlib import Path

import numpy as np
import torch

from exp.mw_finetune.cosmos.policy import Options, expand_proprio, make_data_batch


def main():
    import cosmos_policy.experiments.robot.cosmos_utils as cu
    options = Options(camera_layout="libero_duplicate")
    stats = {k: np.asarray(v, np.float32) for k, v in json.loads(
        (Path(__file__).parent / "sources/dataset_statistics.json").read_text()).items()}
    rng = np.random.default_rng(7)
    image = rng.integers(0, 256, (480, 480, 3), dtype=np.uint8)
    proprio = expand_proprio([.05, .65, .2, .5])
    embedding = torch.zeros(1, 512, 1024)
    captured = {}

    def generate(batch, **kw):
        captured.update(batch)
        return torch.zeros(1, 16, 9, 28, 28), torch.zeros(1, 16, 9, 28, 28)

    model = SimpleNamespace(config=SimpleNamespace(min_num_conditional_frames=4), generate_samples_from_batch=generate)
    cfg = SimpleNamespace(**vars(options), suite="libero", use_proprio=True, normalize_proprio=True,
                          unnormalize_actions=True, use_variance_scale=False, use_wrist_image=True,
                          num_wrist_images=1, use_third_person_image=True, num_third_person_images=1)
    cu.t5_text_embeddings_cache["audit"] = embedding
    # Upstream hardcodes .cuda(); intercept it only inside this isolated CPU audit.
    with patch.object(torch.Tensor, "cuda", lambda self, *a, **kw: self):
        cu.get_action(cfg, model, stats, {"primary_image": image, "wrist_image": image, "proprio": proprio},
                      "audit", seed=7, randomize_seed=False, num_denoising_steps_action=5,
                      generate_future_state_and_value_in_parallel=False)
    ours, action_idx = make_data_batch(image, proprio, embedding, options, stats, "cpu", cu, torch)
    assert set(ours) == set(captured)
    for key, a in ours.items():
        b = captured[key]
        if isinstance(a, torch.Tensor):
            assert a.dtype == b.dtype and torch.equal(a, b), key
        else:
            assert a == b, key
    assert action_idx == 4
    assert ours["proprio"][0, -1].item() == 2  # Published quaternion w bounds.
    print(json.dumps({"upstream_two_camera_batch_equal": True, "keys_checked": sorted(ours),
                      "raw_video_shape": list(ours["video"].shape), "identity_quaternion_normalized_w": 2}))


if __name__ == "__main__":
    main()
