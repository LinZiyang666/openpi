"""Short, capped B=1/100-step benchmark; never starts or contacts a server."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch

from .policy import load_policy
from .cuda_graph import check_observations


def observations(policy):
    return check_observations(policy.domain, policy.stats)


def gpu_memory_mib():
    output = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory",
                                      "--format=csv,noheader,nounits"], text=True)
    return [int(line.split(",")[1]) for line in output.splitlines()
            if int(line.split(",")[0]) == os.getpid()]


def timing(policy, obs, repeats):
    policy.infer(obs)
    torch.cuda.synchronize(0)
    values = []
    for _ in range(repeats):
        start = time.perf_counter()
        policy.infer(obs)
        torch.cuda.synchronize(0)
        values.append(1000 * (time.perf_counter() - start))
    return {"median_ms": float(np.median(values)), "samples_ms": values}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--domain", choices=("libero", "metaworld"), required=True)
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--record-baseline", action="store_true")
    ap.add_argument("--report", type=Path)
    ap.add_argument("--repeats", type=int, default=5)
    a = ap.parse_args()
    torch.set_num_threads(4)
    options = dict(domain=a.domain, denoising_steps=100, memory_limit_gib=1.25)
    policy = load_policy(a.checkpoint, "cuda:0", **options)
    obs = list(observations(policy))
    eager = np.stack([policy.infer(o)["actions"] for o in obs])
    metadata = json.loads(json.dumps(policy.get_metadata()))
    if a.record_baseline:
        a.baseline.parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.baseline, actions=eager, metadata=json.dumps(metadata))
    else:
        with np.load(a.baseline) as before:
            assert torch.equal(torch.from_numpy(eager), torch.from_numpy(before["actions"])), "Default-off actions changed"
            assert metadata == json.loads(str(before["metadata"])), "Default-off metadata changed"
    report = {"checkpoint": str(Path(a.checkpoint).resolve()), "domain": a.domain,
              "device": "cuda:0", "batch": 1, "steps": 100,
              "seeds": [o["seed"] for o in obs], "prompts": [o["prompt"] for o in obs],
              "eager": timing(policy, obs[0], a.repeats)}
    if not a.record_baseline:
        del policy
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        policy = load_policy(a.checkpoint, "cuda:0", cuda_graph=True, **options)
        graph = np.stack([policy.infer(o)["actions"] for o in obs])
        assert torch.equal(torch.from_numpy(eager), torch.from_numpy(graph)), "Graph final actions differ"
        # Interleaving observations and seedless calls exercises complete buffer replacement.
        assert torch.equal(torch.from_numpy(policy.infer(obs[0])["actions"]), torch.from_numpy(eager[0]))
        seedless = {k: v for k, v in obs[0].items() if k != "seed"}
        first = policy.infer(seedless)["actions"]
        policy.infer(obs[1])
        assert torch.equal(torch.from_numpy(first), torch.from_numpy(policy.infer(seedless)["actions"]))
        report.update(default_off_bit_exact=True, default_off_metadata_identical=True,
                      graph_bit_exact_observations=3, graph=timing(policy, obs[0], a.repeats),
                      cuda_graph_self_check=policy.cuda_graph_self_check,
                      peak_allocated_mib=torch.cuda.max_memory_allocated(0) / 2**20,
                      peak_reserved_mib=torch.cuda.max_memory_reserved(0) / 2**20,
                      process_gpu_mib=gpu_memory_mib())
    result = json.dumps(report, indent=2)
    print(result, flush=True)
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(result + "\n")


if __name__ == "__main__":
    main()
