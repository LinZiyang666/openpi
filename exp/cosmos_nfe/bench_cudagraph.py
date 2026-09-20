"""Benchmark ``GraphedNet`` (CUDA-graph replay of the DiT forward) against the eager path on one Cosmos Policy model.

Loads the model exactly as ``serve_cosmos`` does (same ``PolicyEvalConfig`` flags), builds one synthetic
observation, and for each k in ``--ks`` times ``get_action`` eagerly and with ``model.net`` wrapped, in
deterministic mode (``set_seed_everywhere(0)`` before every call, as the server does), reporting wall time
per call and the max |delta| of the action chunk between the two paths.

Usage (h100, cosmos venv, PYTHONPATH with exp/)::

    python -m exp.cosmos_nfe.bench_cudagraph --bench robocasa --ks 5,1 --reps 3 <PolicyEvalConfig flags>
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from exp.cosmos_nfe.serve_cosmos import _take


def main() -> None:
    argv = sys.argv[1:]
    bench = _take(argv, "--bench", "robocasa")
    ks = [int(x) for x in _take(argv, "--ks", "5,1").split(",")]
    reps = int(_take(argv, "--reps", "3"))
    no_future = _take(argv, "--no-future-decode", "0") == "1"
    import draccus

    from cosmos_policy.experiments.robot.cosmos_utils import get_action, get_model, init_t5_text_embeddings_cache, load_dataset_stats
    from cosmos_policy.utils.utils import set_seed_everywhere

    if bench == "libero":
        from cosmos_policy.experiments.robot.libero.run_libero_eval import PolicyEvalConfig
    else:
        from cosmos_policy.experiments.robot.robocasa.run_robocasa_eval import PolicyEvalConfig
    cfg = draccus.parse(config_class=PolicyEvalConfig, args=argv)
    if cfg.deterministic:
        os.environ["DETERMINISTIC"] = "True"
    set_seed_everywhere(cfg.seed)
    init_t5_text_embeddings_cache(cfg.t5_text_embeddings_path)
    dataset_stats = load_dataset_stats(cfg.dataset_stats_path)
    model, _ = get_model(cfg)
    if no_future:
        # policy-only serving: skip the Wan-VAE decode of the 11-frame latent video that only feeds the future-image
        # visualisation (the value read from the latent stays); the action chunk is untouched
        import cosmos_policy.experiments.robot.cosmos_utils as CU

        CU.get_future_images_from_generated_samples = lambda *a, **k: {}
        print("NO-FUTURE-DECODE: get_future_images_from_generated_samples stubbed", flush=True)

    rng = np.random.default_rng(0)
    img = lambda: rng.integers(0, 255, size=(224, 224, 3), dtype=np.uint8)
    if bench == "libero":
        obs = {"primary_image": img(), "wrist_image": img(), "proprio": rng.standard_normal(8).astype(np.float32)}
        task = "put the black bowl in the bottom drawer of the cabinet and close it"
    else:
        obs = {"primary_image": img(), "secondary_image": img(), "wrist_image": img(), "proprio": rng.standard_normal(9).astype(np.float32)}
        task = "press the stop button on the microwave"

    def run(k):
        if cfg.deterministic:
            set_seed_everywhere(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = get_action(cfg, model, dataset_stats, obs, task, seed=cfg.seed, randomize_seed=cfg.randomize_seed,
                         num_denoising_steps_action=k, generate_future_state_and_value_in_parallel=True)
        torch.cuda.synchronize()
        return np.stack([np.asarray(a, dtype=np.float32) for a in out["actions"]]), (time.perf_counter() - t0) * 1000

    eager = {}
    for k in ks:
        run(k)  # warm
        outs = [run(k) for _ in range(reps)]
        eager[k] = outs[0][0]
        print(f"EAGER  k={k}: {[round(t, 1) for _, t in outs]} ms  (mean {np.mean([t for _, t in outs]):.1f})", flush=True)
        for a, _ in outs[1:]:
            print(f"        eager repeat max|d|={np.abs(a - eager[k]).max():.3e}")

    from exp.cosmos_nfe.cudagraph_net import GraphedNet

    model.net = GraphedNet(model.net, warmup=2)
    for k in ks:
        for _ in range(3):
            run(k)  # warm-ups + capture
        outs = [run(k) for _ in range(reps)]
        print(f"GRAPH  k={k}: {[round(t, 1) for _, t in outs]} ms  (mean {np.mean([t for _, t in outs]):.1f})  "
              f"max|d| vs eager={np.abs(outs[0][0] - eager[k]).max():.3e}  stats={model.net.stats}", flush=True)
    print(f"mem allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB")


if __name__ == "__main__":
    main()
