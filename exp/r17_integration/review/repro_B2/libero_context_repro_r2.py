"""Round-2 CPU stub repro: LIBERO context/ack through the fixed P1 ConnectionPolicy (non-diagnostic)."""
import sys, dataclasses
sys.path[:0] = ["/home/weiland/projects/openpi", "/home/weiland/projects/openpi/src"]
from exp.r17_integration.tests.models.fastwam.conftest import make_adapter, observation
from exp.r17_integration.tests.models.fastwam.test_fastwam_serving import config_for
from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.models.fastwam.provider import register_provider, make_connection_factory
from exp.r17_integration.common.request_rng import RequestRNG
from exp.r17_integration.benchmarks.libero_bridge import decision_seed
a = make_adapter("libero", allow_unqualified_batch=False)
p = register_provider(a, provider_id="reviewB2_libero")
def factory_for(commit):
    cfg = config_for(p, k=2, commit=commit); cfg.diagnostic_context = False
    recs = []
    return make_connection_factory(p, cfg, build_shared_storage(cfg), capture_sink=recs.append), recs
def attempt(label, fn):
    try:
        fn(); print(f"{label}: NO ERROR")
    except Exception as e:
        print(f"{label}: {type(e).__name__}: {e}")
old_extra = {"task_id": 3, "orig_init_state_idx": 7}
f10, recs = factory_for(10)
c = f10()
attempt("A old-runner metadata, commit10 episode_start", lambda: c.on_episode_start(
    experiment="libero_spatial", task="pick bowl", episode_id=7, episode_name="", extra_metadata=old_extra))
f5, recs5 = factory_for(5)
c = f5(); c.on_episode_start(experiment="libero_spatial", task="pick bowl", episode_id=7, episode_name="", extra_metadata=old_extra)
attempt("B old-runner metadata, commit5 infer", lambda: c.infer(observation("libero")))
# C: P2 bridge identity + request envelope, two inits
name = "libero_spatial/pick_up_the_black_bowl"
seeds = {}
for init in (0, 7):
    c = f10()
    ident = {"campaign": "r17", "task_id": 3, "task_name": name, "original_init_idx": init, "orig_init_state_idx": init,
             "seed": 7, "ack_protocol": "r17.execution.v1", "partial_tail_protocol": "r17.partial_tail.v1"}
    c.on_episode_start(experiment="libero_spatial", task="pick bowl", episode_id=init, episode_name="", extra_metadata=ident)
    def req(seq):
        return {"campaign": "r17", "task_name": name, "task_id": 3, "original_init_idx": init, "decision_seq": seq,
                "seed": 7, "explicit_seed": decision_seed(name, init, seq, 7)}
    c.infer({**observation("libero"), "__request__": req(0)})
    seeds[init] = RequestRNG(recs[-1].request).seed_for()
    c.infer({**observation("libero"), "__request__": req(1), "__execution__": {"decision_seq": 0, "executed_count": 5, "terminal": False}})
    print(f"C init {init}: decision1 source={recs[-1].source} stage_calls={recs[-1].stage_calls} cursor={recs[-1].cursor}")
    attempt(f"C init {init}: decision2 without ack", lambda: c.infer({**observation("libero"), "__request__": req(2)}))
print("C distinct initial seeds across inits:", seeds[0] != seeds[7], seeds)
