"""Round-2 repro: real P2 R17MetaWorldEpisodeRunner (original MW kernel) -> GR00T-MW ConnectionPolicy.

CPU stub GR00T backend from the P5 test fixture; in-process client with the same
semantics as the R17 WS control socket (terminal ack consumed before on_episode_end).
"""
import hashlib, inspect
from pathlib import Path
import numpy as np
import openpi.cache.groot.staged as staged
import exp.r17_integration.tests.models.groot.conftest as c
from exp.r17_integration.models.groot.provider import make_provider, register_provider
from exp.r17_integration.models.groot.serve import make_connection_factory, default_config
from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.benchmarks.metaworld import R17MetaWorldEpisodeRunner, FORMAL_SPEC
from exp.r17_integration.conductor.manifest import episode_tasks
from openpi.conductor.task import ServerEndpoint

staged.UPSTREAM_FORWARD_SHA256 = hashlib.sha256(inspect.getsource(c.Eagle.forward).encode()).hexdigest()
staged.UPSTREAM_ACTION_HEAD_SHA256 = hashlib.sha256(Path(inspect.getsourcefile(c.Head)).read_bytes()).hexdigest()
manifest = c.GrootManifest(*(c.digest(s) for s in ("weights", "norm", "code", "transforms", "tokenizer")), hidden_width=8)
provider = register_provider(make_provider(c.Backend(), manifest))
cfg = default_config(provider, commit_horizon=10)
factory = make_connection_factory(provider, cfg, build_shared_storage(cfg))

class Client:  # same control semantics as common/websocket_policy_server._ControlSocket
    def __init__(self): self.policy = None; self.log = []
    def select_bundle(self, b): self.policy = factory(None, b); self.policy.on_task_begin(); return {}
    def episode_start(self, **kw):
        self.policy.on_episode_start(experiment=kw.get("experiment"), task=kw.get("task", ""), episode_id=kw.get("episode_id"),
                                     episode_name="", extra_metadata=kw.get("extra_metadata", {})); return {}
    def infer(self, obs):
        r = self.policy.infer(obs); m = r["__hit_meta__"]
        self.log.append((m["source"], tuple(m["stage_calls"]), m["cursor"], len(r["actions"]))); return r
    def _send_ctrl(self, msg, expected_ack):
        if "__execution__" in msg: self.last = self.policy.acknowledge_execution(dict(msg["__execution__"], terminal=True))
        self.policy.on_episode_end(success=msg.get("__success__")); return {"__ack__": expected_ack}
    def close(self): pass

class Env:
    def __init__(self): self.n = 0
    def reset(self): return np.zeros(39, np.float32), {}
    def step(self, a): self.n += 1; return np.full(39, .01*self.n, np.float32), 0., False, False, {"success": float(self.n == 15 + 12)}
    def render(self): return np.full((480, 480, 3), 7, np.uint8)
    def close(self): pass

client = Client()
runner = R17MetaWorldEpisodeRunner(env_factory=lambda *a: Env(), client_factory=lambda s: client)
task = episode_tasks(campaign="rev", arm="g", bundle_id="default", endpoint=ServerEndpoint("x", 1), model_identity="m",
                     norm_identity="n", layout_identity="l", num_steps=4, partition="smoke", names=["reach-v3"], indices=[1],
                     commit_horizon=10)[0]
result = runner.run(task, lambda *a: None)
print("success", result.success, "error", result.error)
print("decisions (source, stage_calls, cursor, returned rows):", client.log)
print("final server ack:", {k: client.last[k] for k in ("decision_seq", "executed_count", "cursor_end", "terminal")})
print("cost rows have cost_ms:", all(r.get("cost_ms") is not None for r in result.per_step_rows if not r.get("_kind")))
