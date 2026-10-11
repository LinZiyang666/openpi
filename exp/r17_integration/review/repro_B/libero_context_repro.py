"""CPU stub repro: LIBERO-runner episode_start metadata through P1 ConnectionPolicy."""
import sys
sys.path[:0] = ["/home/weiland/projects/openpi", "/home/weiland/projects/openpi/src",
                "/home/weiland/projects/openpi/exp/r17_integration/tests/models/fastwam"]
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m; spec.loader.exec_module(m); return m
d = "/home/weiland/projects/openpi/exp/r17_integration/tests/models/fastwam/"
cf = _load("conftest", d + "conftest.py")
make_adapter, observation = cf.make_adapter, cf.observation
config_for = _load("fw_test_serving", d + "test_serving.py").config_for
from openpi.cache.config import build_shared_storage
from exp.r17_integration.models.fastwam.provider import register_provider, make_connection_factory
from exp.r17_integration.common.request_rng import RequestRNG
a = make_adapter("libero")
p = register_provider(a, provider_id="reviewB_libero")
cfg = config_for(p, k=2, commit=10)
# protocol must be the adapter's allowed one
factory = make_connection_factory(p, cfg, build_shared_storage(cfg), capture_sink=lambda r: recs.append(r))
recs = []
seeds = {}
for init in (0, 7):
    c = factory()
    # exactly what WebsocketPolicyServer passes for examples/libero/episode_runner.py
    c.on_episode_start(experiment="libero_spatial", task="pick up the black bowl", episode_id=init,
                       episode_name="", extra_metadata={"task_id": 3, "orig_init_state_idx": init})
    out = c.infer(observation("libero"))
    r = recs[-1].request
    seeds[init] = RequestRNG(r).seed_for()
    print("init", init, "-> task_name", repr(r.task_name), "original_init_idx", r.original_init_idx, "seed", seeds[init])
    # LIBERO runner never sends __execution__: second decision under commit10
    out2 = c.infer(observation("libero"))
    print("   decision 1 source:", recs[-1].source, "stage_calls", recs[-1].stage_calls, "(commit_horizon", c.commit_horizon, ")")
print("identical noise across init states:", seeds[0] == seeds[7])
