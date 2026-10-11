"""One model, concurrent msgpack WebSocket clients, pure inference only."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import random
import socket
import threading
import numpy as np

from exp.metaworld import tasks as T
from exp.mw_finetune.adapters import canonical_observation, episode_seed, validate_actions

MODEL_MODULES = {name: f"exp.mw_finetune.{name}.policy" for name in ("groot", "fastwam", "cosmos")}
MODEL_MODULES["pi05"] = "exp.mw_finetune.pi05"


class ConnectionPolicy:
    """Each connection owns episode identity; only the GPU model and lock are shared."""

    def __init__(self, policy, lock, torch_module=None):
        self.policy = policy
        self.lock = lock
        self.torch = torch_module
        self.task = None
        self.idx = None
        self.decision = 0

    def on_episode_start(self, experiment, task, episode_id, extra_metadata=None, **kwargs):
        if experiment != T.BENCHMARK or task not in T.TASK_NAMES:
            raise ValueError("expected a canonical MetaWorld MT50 episode")
        extra = extra_metadata or {}
        idx = extra.get("orig_init_state_idx", episode_id)
        if not isinstance(idx, int) or not 0 <= idx < 50:
            raise ValueError("episode idx must be 0..49")
        if extra.get("seed", 7) != 7:
            raise ValueError("benchmark seed must be 7")
        self.task, self.idx, self.decision = task, idx, 0

    def on_episode_end(self, success=False):
        self.task = self.idx = None

    def infer(self, obs):
        if self.task is None:
            raise RuntimeError("episode_start must precede inference")
        if obs.get("prompt") != T.PROMPTS[self.task]:
            raise ValueError("request prompt differs from canonical evaluation task prompt")
        seed = episode_seed(self.task, self.idx, decision=self.decision)
        canonical = canonical_observation(obs, task_name=self.task, seed=seed, decision_index=self.decision)
        with self.lock:
            random.seed(seed)
            np.random.seed(seed)
            if self.torch is not None:
                self.torch.manual_seed(seed)
            self.policy.reset()
            result = self.policy.infer(canonical)
            actions = validate_actions(result["actions"])
        self.decision += 1
        return {"actions": actions}


def checkpoint_identity(path):
    """Stable stat manifest plus hashes of configs; checkpoints must be immutable."""
    path = Path(path).resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {path}")
    files = sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts)
    digest = hashlib.sha256()
    config_hashes = {}
    for p in files:
        stat = p.stat()
        name = str(p.relative_to(path))
        digest.update(json.dumps([name, stat.st_size, stat.st_mtime_ns]).encode())
        if p.suffix == ".json":
            config_hashes[name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"path": str(path), "files": len(files), "bytes": sum(p.stat().st_size for p in files),
            "stat_manifest_sha256": digest.hexdigest(), "config_sha256": config_hashes,
            "identity_method": "all relative paths/sizes/mtimes + JSON content hashes; immutable directory required"}


class ZeroPolicy:
    def reset(self):
        pass

    def infer(self, obs):
        return {"actions": np.zeros((5, 4), np.float32)}


def load_model(name, checkpoint, device, options):
    if name == "zero":
        return ZeroPolicy()
    module = importlib.import_module(MODEL_MODULES[name])
    policy = module.load_policy(str(checkpoint), device, **options)
    if not callable(getattr(policy, "infer", None)) or not callable(getattr(policy, "reset", None)):
        raise TypeError(f"{MODEL_MODULES[name]} must return infer()/reset() policy")
    return policy


def main():
    if socket.gethostname().split(".")[0].startswith("timan"):
        raise RuntimeError("devices.md section1: timan nodes are workers only; never policy servers")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=[*MODEL_MODULES, "zero"], required=True)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8050)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--denoising-steps", type=int)
    ap.add_argument("--policy-option", action="append", default=[], metavar="KEY=JSON")
    a = ap.parse_args()
    if a.model != "zero" and a.checkpoint is None:
        ap.error("--checkpoint is required")
    if a.denoising_steps is not None and a.denoising_steps < 1:
        ap.error("--denoising-steps must be positive")
    options = {}
    for option in a.policy_option:
        key, sep, value = option.partition("=")
        if not sep or key in ("checkpoint", "device", "denoising_steps"):
            ap.error("policy option must be KEY=JSON and cannot replace reserved arguments")
        options[key] = json.loads(value)
    if a.denoising_steps is not None:
        options["denoising_steps"] = a.denoising_steps
    torch_module = None
    if a.model != "zero":
        from exp.mw_finetune.runtime import gpu_time_gate
        gpu_time_gate()
        import torch
        torch_module = torch
    policy = load_model(a.model, a.checkpoint, a.device, options)
    metadata = {"model": a.model, "pure_inference": True, "state_dim": 4, "action_dim": 4,
                "checkpoint": checkpoint_identity(a.checkpoint) if a.checkpoint else None,
                "policy_rng": "sha256(mt50:7:task:idx:decision)", "policy_options": options,
                "wrapper_metadata": policy.get_metadata() if hasattr(policy, "get_metadata") else {},
                "contract_version": 1}
    if a.model == "zero":
        metadata["test_policy"] = True
    from openpi.serving.websocket_policy_server import WebsocketPolicyServer
    lock = threading.Lock()

    def factory(base, bundle_id="default"):
        if bundle_id != "default":
            raise ValueError("pure-inference server serves only default bundle")
        return ConnectionPolicy(policy, lock, torch_module)

    server = WebsocketPolicyServer(factory(None), host=a.host, port=a.port, metadata=metadata,
                                   concurrent=True, connection_policy_factory=factory)
    print(json.dumps(metadata), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
