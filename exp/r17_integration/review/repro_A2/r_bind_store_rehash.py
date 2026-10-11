"""Round-2 repro: every connection bind of a recipe bundle re-opens and SHA-256-hashes the whole
teacher store (twice); the stock WS runs _bind_bundle synchronously on its asyncio loop."""
import time
import exp.r17_integration.search.store as store_mod
from exp.r17_integration.common.fake_adapter import fake_provider
from exp.r17_integration.common.config import load_cache_config, build_shared_storage
from exp.r17_integration.common.serving import make_connection_factory
from exp.r17_integration.common.recipe import make_recipe_factory
from pathlib import Path

provider = fake_provider("r17.fix1.sample")
cfg = load_cache_config("exp/r17_integration/common/samples/fake_recipe_full.yaml")
storage = build_shared_storage(cfg)
calls = {"n": 0, "bytes": 0}
original = store_mod.file_sha
def counting(path):
    calls["n"] += 1; calls["bytes"] += Path(path).stat().st_size
    return original(path)
store_mod.file_sha = counting
factory = make_connection_factory(provider, cfg, storage, recipe_factory=make_recipe_factory(provider))
for i in range(3):
    before = dict(calls); t = time.perf_counter()
    factory(None, "default")
    print(f"bind {i}: store files hashed={calls['n']-before['n']} bytes={calls['bytes']-before['bytes']} "
          f"wall={1000*(time.perf_counter()-t):.1f} ms")
arrays = len(store_mod.Store(cfg.recipe.store_path).manifest["arrays"])
print("store array files:", arrays)
