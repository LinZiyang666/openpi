"""A's server with a probe-only input-size exception and CUDA graph parity audit."""
import json
from pathlib import Path
import sys
import time

import numpy as np

from exp.mw_finetune import serve as A


def probe_canonical(obs, *, task_name, seed, decision_index):
    image = np.asarray(obs["observation/image"])
    state = np.asarray(obs["observation/state"], dtype=np.float32)
    if image.shape not in ((256, 256, 3), (480, 480, 3)) or image.dtype != np.uint8:
        raise ValueError("probe expects uint8 RGB 256 or 480 square image")
    if state.shape != (4,) or not np.isfinite(state).all():
        raise ValueError("probe expects finite [x,y,z,g] state")
    if not isinstance(obs["prompt"], str) or not obs["prompt"].strip():
        raise ValueError("probe expects nonempty prompt")
    return {"image": image, "state": state, "prompt": obs["prompt"], "task_name": task_name,
            "seed": seed, "decision_index": decision_index}


def main():
    # Local shim: A's model factory, ConnectionPolicy, RNG and WebSocket server
    # remain the actual serving path. Only this process allows the native image.
    marker = sys.argv.index("--parity-file")
    parity_file = Path(sys.argv[marker + 1])
    del sys.argv[marker:marker + 2]
    original_load = A.load_model

    def audited_load(name, checkpoint, device, options):
        # Check eager -> CUDA graph actions before permitting any rollout.
        policy = original_load(name, checkpoint, device, {**options, "cuda_graph": False})
        rng = np.random.default_rng(7)
        obs = {"image": rng.integers(0, 256, (480, 480, 3), dtype=np.uint8),
               "state": np.asarray([.05, .65, .2, .5], np.float32),
               "prompt": "Pick up a nut and place it onto a peg", "task_name": "assembly-v3", "seed": 2026}
        t = time.monotonic()
        eager = policy.infer(obs)["actions"]
        eager_s = time.monotonic() - t
        policy.enable_cuda_graph()
        times, differences = [], []
        for _ in range(2):
            t = time.monotonic()
            graph = policy.infer(obs)["actions"]
            times.append(time.monotonic() - t)
            differences.append(float(np.max(np.abs(eager - graph))))
        audit = {"eager_s": eager_s, "graph_s": times, "max_abs_differences": differences,
                 "bitwise_equal": all(v == 0 for v in differences), "seed": obs["seed"]}
        parity_file.write_text(json.dumps(audit, indent=2) + "\n")
        if not audit["bitwise_equal"]:
            raise RuntimeError(f"CUDA graph changed synthetic actions: {audit}")
        policy.metadata["cuda_graph"] = True
        policy.metadata["options"]["cuda_graph"] = True
        policy.metadata["probe_server"] = "A.serve with 256/480 input allowance; graph parity checked"
        original_infer = policy.infer
        timing = (parity_file.parent / "queries.jsonl").open("x", buffering=1)

        def timed_infer(obs):
            t = time.monotonic()
            out = original_infer(obs)
            elapsed = 1000 * (time.monotonic() - t)
            actions = out["actions"]
            timing.write(json.dumps({"task": obs["task_name"], "seed": int(obs["seed"]),
                                     "decision_index": obs["decision_index"], "service_ms": elapsed,
                                     "state": np.asarray(obs["state"]).tolist(),
                                     "first_action": actions[0].tolist(), "action_min": actions.min(0).tolist(),
                                     "action_max": actions.max(0).tolist()}) + "\n")
            return out

        policy.infer = timed_infer
        return policy

    A.canonical_observation = probe_canonical
    A.load_model = audited_load
    A.main()


if __name__ == "__main__":
    main()
