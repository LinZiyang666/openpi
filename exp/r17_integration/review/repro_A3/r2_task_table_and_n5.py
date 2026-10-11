"""Round-3 checks: A2 N2 (canonical task table at the serving boundary) and N5 (GR00T lock wait)."""
import copy, hashlib, inspect, threading
from pathlib import Path
import openpi.cache.groot.staged as staged
import exp.r17_integration.tests.models.groot.conftest as c
from exp.r17_integration.models.groot.provider import make_provider, register_provider
from exp.r17_integration.models.groot.serve import default_config, make_connection_factory
from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.common.recipe import validate_recipe

staged.UPSTREAM_FORWARD_SHA256 = hashlib.sha256(inspect.getsource(c.Eagle.forward).encode()).hexdigest()
staged.UPSTREAM_ACTION_HEAD_SHA256 = hashlib.sha256(Path(inspect.getsourcefile(c.Head)).read_bytes()).hexdigest()
manifest = c.GrootManifest(*(c.digest(s) for s in ("weights", "norm", "code", "transforms", "tokenizer")), hidden_width=8)
provider = register_provider(make_provider(c.Backend(), manifest))
cfg = default_config(provider, commit_horizon=5)
print("GR default config table:", len(cfg.recipe.ordered_tasks), "tasks, identity set:", cfg.recipe.task_table_identity is not None)
bad = copy.deepcopy(cfg); bad.recipe.ordered_tasks = sorted(bad.recipe.ordered_tasks)
try:
    validate_recipe(bad); print("UNEXPECTED: alphabetical table accepted")
except ValueError as exc:
    print("alphabetical ordered_tasks ->", exc)
policy = make_connection_factory(provider, cfg, build_shared_storage(cfg))()
c.begin_episode(policy)
obs = c.obs.__wrapped__()
try:
    policy.infer(dict(obs, __request__=dict(obs["__request__"], task_name="push-v3")))
except ValueError as exc:
    print("reach-v3 index with push-v3 name ->", exc)
c.begin_episode(policy)
held = threading.Event(); release = threading.Event()
t = threading.Thread(target=lambda: (policy.adapter.model_lock.acquire(), held.set(), release.wait(5), policy.adapter.model_lock.release()))
t.start(); held.wait(5); threading.Timer(.05, release.set).start()
meta = policy.infer(obs)["__hit_meta__"]; t.join()
print("GR lock held 50ms -> queue_wait_ms=%.1f service_ms=%.1f cost_ms=%.1f pricing_time_field=%s eligible=%s"
      % (meta["queue_wait_ms"], meta["service_ms"], meta["cost_ms"], meta["pricing_time_field"], meta["pricing_eligible"]))
