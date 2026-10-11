"""Real checkpoint reload and two stateless predictions on a training observation.

This validates serving mechanics, not MetaWorld success rate.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .data import Dataset
from .policy import load_policy


def main():
    import torch
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    torch.set_num_threads(8)
    begin = time.perf_counter()
    policy = load_policy(a.checkpoint, "cuda:0")
    dataset = Dataset(a.data, allow_partial=True)
    # Upsampling only supplies A's input shape for this mechanic smoke.
    # Formal eval uses real 480-square environment renders.
    image = np.asarray(Image.fromarray(dataset._image(0)).resize((480, 480)))
    obs = {"image": image, "state": dataset.index["state"][0],
        "prompt": dataset.manifest["tasks"][str(int(dataset.index["task_index"][0]))],
        "task_name": "training_observation", "seed": 17, "decision_index": 0}
    timings, actions = [], []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(2):
        policy.reset()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = policy.infer(obs)["actions"]
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - t0)
        actions.append(result)
    np.testing.assert_array_equal(actions[0], actions[1])
    report = {"passed": True, "actions_shape": list(actions[0].shape),
        "actions_dtype": str(actions[0].dtype), "finite": bool(np.isfinite(actions[0]).all()),
        "same_seed_exact": True, "inference_s": timings,
        "peak_allocated_gb": torch.cuda.max_memory_allocated()/1e9,
        "peak_reserved_gb": torch.cuda.max_memory_reserved()/1e9,
        "wall_s": time.perf_counter() - begin,
        "scope": "checkpoint reload and serving mechanics; not an SR evaluation"}
    Path(a.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
