"""Repro: GR00T-MW R17 ConnectionPolicy fed the P2 runner's wire observation."""
import hashlib, inspect
from pathlib import Path
import numpy as np
import openpi.cache.groot.staged as staged
import exp.r17_integration.tests.models.groot.conftest as c
import exp.r17_integration.common.provider as registry
from exp.r17_integration.models.groot.provider import make_provider, register_provider
from exp.r17_integration.models.groot.serve import make_connection_factory, default_config
from exp.metaworld import env as kernel
from openpi.cache.config import build_shared_storage

staged.UPSTREAM_FORWARD_SHA256 = hashlib.sha256(inspect.getsource(c.Eagle.forward).encode()).hexdigest()
staged.UPSTREAM_ACTION_HEAD_SHA256 = hashlib.sha256(Path(inspect.getsourcefile(c.Head)).read_bytes()).hexdigest()
manifest = c.GrootManifest(*(c.digest(s) for s in ("weights", "norm", "code", "transforms", "tokenizer")), hidden_width=8)
provider = register_provider(make_provider(c.Backend(), manifest))
cfg = default_config(provider, commit_horizon=5)
factory = make_connection_factory(provider, cfg, build_shared_storage(cfg))
pol = factory(None, "default")
pol.on_task_begin()
pol.on_episode_start(episode_id=0, extra_metadata={"task_name": "reach-v3", "task_id": 0, "original_init_idx": 0})

class FakeEnv:  # only what kernel.observation() touches
    def render(self):
        return np.full((480, 480, 3), 2, np.uint8)
wire = kernel.observation(FakeEnv(), np.array([.1, .2, .3, .4, 0, 0], np.float32), "Reach a goal position")
print("P2 kernel wire keys:", sorted(wire))
try:
    pol.infer(dict(wire))
    print("UNEXPECTED: inference succeeded")
except Exception as exc:
    print("P2-runner obs ->", type(exc).__name__, exc)
ok = pol.infer({"image": wire["observation/image"], "state": wire["observation/state"], "prompt": wire["prompt"]})
print("canonical image/state obs -> actions", ok["actions"].shape)
