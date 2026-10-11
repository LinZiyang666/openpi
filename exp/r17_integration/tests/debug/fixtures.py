"""CPU-only model/benchmark fixtures using P1's actual v1 ModelSpec."""
import numpy as np

from exp.r17_integration.common.contracts import (
    ActionStateLayout, Capabilities, Capability, FeatureSpec, ModelSpec, SamplerSpec)
from exp.r17_integration.debug.validate import model_binding


def spec(model="dp", benchmark="mw", *, key_width=None):
    mw = benchmark == "mw"
    h = {"pi05": 5 if mw else 50, "groot": 16, "fastwam": 32, "dp": 16}[model]
    dim = 4 if mw else 7
    state = 4 if mw else 8
    width = 32 if model in ("pi05", "groot") else dim
    cameras = ("corner2",) if mw else ("front", "wrist")
    key_width = key_width or {"pi05": 32768, "groot": 32768, "fastwam": 512, "dp": 64}[model]
    fields = {"vision_0": key_width, "robot_state": state}
    if not mw and model != "fastwam":
        fields["vision_1"] = key_width
    k = {"pi05": 10, "groot": 4, "fastwam": 10, "dp": 5}[model]
    grid = tuple(range(99, 99-k, -1)) if model == "dp" else tuple(np.linspace(1., .1, k).tolist())
    sampler = SamplerSpec(model + "_native_v1", "ddpm" if model == "dp" else "flow", k, grid)
    layout = ActionStateLayout(h, min(5, h), dim, tuple(range(state)), tuple(range(dim-1)), dim-1,
                              model_action_width=width, units="fixture_native", frame="fixture_world")
    return ModelSpec(model + "_" + benchmark, "weights-sha", "norm-sha", layout,
                     FeatureSpec("policy-owned-cp1", cameras, fields), sampler,
                     Capabilities({key: Capability("supported") for key in (
                         "full", "k_step", "concurrent", "bundle", "episode", "cp1", "debug")}),
                     ("mt50_seed7_corner2_r17_v1" if mw else "libero_fixture_v1",))


def binding(model="dp", benchmark="mw", **kwargs):
    s = spec(model, benchmark, **kwargs)
    return model_binding(s, s.allowed_benchmark_protocols[0])


def record(init=10, seq=0, *, attempt=1, task="reach-v3", arm="full", n_decisions=2):
    return dict(campaign="cpu-fixture", arm=arm, bundle_generation="generation1", benchmark_protocol="mt50_seed7_corner2_r17_v1",
                task_name=task, task_uid=f"cpu:{task}:{init}", orig_init_idx=init,
                task_data_hash=f"task-data-{init}", attempt=attempt, decision_seq=seq,
                split="library", closed_ok=True, success=True, n_decisions=n_decisions,
                source="policy", stage_calls=[1, 1, 5], actual_nfe=5, cursor=0,
                executed_count=5, cost_ms=2., model_identity="weights-sha",
                capture_enabled=False, pricing_eligible=True, timing_scope="synthetic_fixture_host_wall",
                queue_wait_ms=0., service_ms=2., pricing_time_field="service_ms")


def arrays(b, seed=0):
    s = b["model"]
    rng = np.random.default_rng(seed)
    layout = s["layout"]
    width = layout["model_action_width"] or layout["valid_action_dim"]
    action = rng.normal(size=(layout["horizon"], width)).astype(np.float32)
    state = rng.normal(size=max(layout["valid_state_indices"])+1).astype(np.float32)
    native = np.asarray(s["default_sampler"]["native_timesteps"])
    payload = dict(normalized_actions=action, state=state, wire_actions=action[:, :layout["valid_action_dim"]].copy(),
                   native_grid=native, snapshot_x=action.copy(), loop_index=np.array(1, np.int64),
                   posterior_rng=np.arange(16, dtype=np.uint8))
    keys = {name: rng.normal(size=dim).astype(np.float32) for name, dim in s["feature_spec"]["fields"].items()}
    return payload, keys


def captured(tmp_path, b=None, *, episodes=3, decisions=2):
    from exp.r17_integration.debug.capture_bridge import CaptureBridge
    b = b or binding()
    path = tmp_path / "capture"
    bridge = CaptureBridge(path, enabled=True, binding=b)
    eps = []
    for i in range(episodes):
        eps.append(record(10+i, n_decisions=decisions))
        for j in range(decisions):
            row = record(10+i, j, n_decisions=decisions)
            k = b["sampler"]["num_steps"]
            row.update(stage_calls=[1,1,k], actual_nfe=k)
            row["benchmark_protocol"] = b["benchmark_protocol"]
            payload, keys = arrays(b, i*decisions+j)
            bridge.capture(row, payload, rawkeys=keys)
        eps[-1]["benchmark_protocol"] = b["benchmark_protocol"]
    bridge.close()
    return path, eps
