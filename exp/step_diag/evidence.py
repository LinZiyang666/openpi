"""Server manifests shared by the Q-A and Q-B admission gates.

What is *gated* here is the arm's contract: the manifest exists, is self-consistent (its
``config_sha`` re-hashes), names this arm, this environment contract (policy / benchmark /
action shape / executed dims / K / schedule / k_set / warm_ts) and the frozen execution horizon,
and -- for the retrieval arms -- binds a library digest. Runtime facts (host, GPU, torch, source
digest) are recorded and compared across arms as *notes* (``comparison_notes``), never as an
admission gate: owner ruling 2026-09-12 (provenance gating contributes nothing to the result and
delays starts). What must agree across the arms of one comparison is the checkpoint identity and
the environment contract (``comparison_identity``).
"""

from __future__ import annotations

import json
import pathlib

from exp.step_diag import envs as E

# EnvSpec fields that define the served contract (hosts are deployment, not contract).
ENV_CONTRACT_KEYS = ("env_id", "policy", "benchmark", "action_horizon", "action_dim", "n_executed", "k_full",
                     "schedule_id", "warm_ts", "k_set")


def env_contract(env: dict | None) -> dict | None:
    if not env:
        return None
    return {k: env.get(k) for k in ENV_CONTRACT_KEYS}


def load_manifests(directory: pathlib.Path) -> dict:
    manifests = {}
    for path in sorted(directory.glob("manifest_*.json")):
        payload = json.loads(path.read_text())
        digest = payload.pop("config_sha", None)
        if digest != E.RunManifest(**payload).config_sha:
            raise ValueError(f"{path}: manifest digest mismatch")
        manifests[digest] = {**payload, "config_sha": digest}
    return manifests


def manifest_problems(manifest: dict | None, *, env_id: str, arm_id: str, config_sha: str) -> list:
    if not manifest:
        return ["manifest_missing"]
    problems = []
    env = E.ENVS.get(env_id)
    if (manifest.get("config_sha") != config_sha or manifest.get("arm_id") != arm_id or env is None
            or env_contract(manifest.get("env")) != env_contract(env.to_json())):
        problems.append("manifest_identity_mismatch")
    if not manifest.get("checkpoint_sha256"):
        problems.append("checkpoint_identity_missing")
    if arm_id == "shadow" or arm_id.startswith("warm"):
        if not manifest.get("library_sha256") or not manifest.get("cache_config_sha256"):
            problems.append("retrieval_identity_missing")
    if manifest.get("extras", {}).get("h_exec") != E.H_EXEC:
        problems.append("execution_horizon_mismatch")
    return problems


def comparison_identity(manifest: dict) -> str:
    """What must be identical across the arms of one comparison: model + environment contract."""
    return E.sha256_json({"env": env_contract(manifest.get("env")), "checkpoint_sha256": manifest.get("checkpoint_sha256")})


def runtime_note(manifest: dict) -> dict:
    """Recorded, not gated: where and with what the arm was served."""
    rt = manifest.get("extras", {}).get("runtime") or {}
    return {k: rt.get(k) for k in ("host", "gpu", "torch", "cuda", "source_sha256")}


def worker_identity(runtime: dict | None) -> str | None:
    """Recorded, not gated: the simulator island (host + package versions); the GPU slot varies per worker."""
    if not runtime:
        return None
    return E.sha256_json({k: runtime.get(k) for k in ("host", "python", "versions", "source_sha256")})
