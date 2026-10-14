"""One CPU-only stub sweep, including per-request work and last-batch overrun."""
import json
from types import SimpleNamespace

from exp.r17_integration.profile.supply_proc import SupplyLoop, run_phases


def test_native_loop_and_phase_accounting():
    calls = []
    now = [100.0]

    class Batcher:
        def run_stage1_batch(self, payloads):
            calls.append(("stage1", len(payloads)))
            return payloads

        def run_stage2_batch(self, payloads, *, capture):
            assert capture is False
            calls.append(("stage2", len(payloads)))
            return payloads

        def run_stage3_miss(self, payloads, *, num_steps, save_timesteps_per_request):
            assert num_steps == 10 and save_timesteps_per_request == [None] * len(payloads)
            calls.append(("stage3", len(payloads)))
            return payloads

    class Adapter:
        def metadata(self):
            return SimpleNamespace(allowed_benchmark_protocols=("mt50",))

        def make_stage_batcher(self):
            return Batcher()

        def prepare(self, observation, request):
            calls.append(("prepare", request.decision_seq))
            return request

        def retrieval_fields(self, stage1):
            calls.append(("keys", stage1.decision_seq))

    def synchronize():
        calls.append(("sync",))
        now[0] += 0.6

    def payload(stage2, request):
        assert stage2 is request
        calls.append(("payload", request.decision_seq))
        return request

    loop = SupplyLoop("dp", Adapter(), SimpleNamespace(num_steps=10), {}, synchronize=synchronize,
                      payload_factory=payload, finish_call=lambda outputs: calls.append(("copy", len(outputs))))
    resets = []
    rows = list(run_phases(loop, [2, 4], dict(t0=100, gap_s=1), 1, model="dp",
                           memory=lambda: 1.25, reset_memory=lambda: resets.append(now[0]),
                           clock=lambda: now[0], monotonic=lambda: now[0],
                           sleep=lambda seconds: now.__setitem__(0, now[0] + seconds)))
    assert [(r["B"], r["phase"], r["requests"]) for r in rows] == [(2, "hit", 4), (2, "call", 4), (4, "hit", 8), (4, "call", 8)]
    assert all(abs(r["wall_s"] - 1.2) < 1e-8 and r["mem_gib"] == 1.25 for r in rows)
    assert [r["begin"] for r in rows] == [100, 102, 104, 106] and len(resets) == 4
    assert sum(c[0] == "prepare" for c in calls) == sum(c[0] == "keys" for c in calls) == 24
    assert sum(c[0] == "payload" for c in calls) == 12
    assert [c[1] for c in calls if c[0] == "stage1"] == [2, 2, 2, 2, 4, 4, 4, 4]
    assert [c[1] for c in calls if c[0] == "stage2"] == [2, 2, 4, 4]
    assert [c[1] for c in calls if c[0] == "stage3"] == [2, 2, 4, 4]
    assert sum(c[0] == "sync" for c in calls) == 8
    assert all(json.loads(json.dumps(r))["steps"] == 10 for r in rows)
