"""CPU-safe, model-neutral contracts. Native tensors remain owned by adapters.

No simulator, vendor, torch or JAX imports occur here. Interface v1 additions
must have defaults; incompatible revisions need a different schema identifier.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol

from . import INTERFACE_VERSION


def digest(value: Any) -> str:
    if is_dataclass(value):
        value = asdict(value)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class ActionStateLayout:
    horizon: int
    replan_steps: int
    valid_action_dim: int
    valid_state_indices: tuple[int, ...]
    motion_channels: tuple[int, ...]
    gripper_channel: int | None
    gripper_closed_convention: str = "unspecified"
    raw_state_fields: tuple[str, ...] = ()
    action_space_name: str = "normalized"
    units: str = "unknown"
    frame: str = "unknown"
    output_transform_id: str = "identity"
    model_action_width: int | None = None
    valid_action_indices: tuple[int, ...] = ()

    def __post_init__(self):
        if any(type(x) is not int or x <= 0 for x in (self.horizon, self.replan_steps, self.valid_action_dim)):
            raise ValueError("layout dimensions must be positive integers")
        if self.replan_steps > self.horizon:
            raise ValueError("replan_steps exceeds horizon")
        active = self.active_actions
        if len(active) != self.valid_action_dim or len(set(active)) != len(active):
            raise ValueError("valid_action_indices must describe every active channel once")
        if any(i < 0 or i >= self.action_width for i in active):
            raise ValueError("active channel outside model action width")
        if not self.valid_state_indices or len(set(self.valid_state_indices)) != len(self.valid_state_indices) or min(self.valid_state_indices) < 0:
            raise ValueError("valid_state_indices must be unique nonnegative indices")
        if not self.motion_channels or len(set(self.motion_channels)) != len(self.motion_channels) or not set(self.motion_channels) <= set(active):
            raise ValueError("motion channels must be active and unique")
        if self.gripper_channel is not None and (self.gripper_channel not in active or self.gripper_channel in self.motion_channels):
            raise ValueError("gripper must be active and excluded from motion channels")

    @property
    def action_width(self):
        return self.model_action_width or self.valid_action_dim

    @property
    def active_actions(self):
        return self.valid_action_indices or tuple(range(self.valid_action_dim))

    @property
    def valid_action_mask(self):
        return tuple(i in self.active_actions for i in range(self.action_width))

    @property
    def identity(self):
        return digest(self)

    def validate_commit(self, commit: int, *, allow_partial=False):
        if type(commit) is not int or not self.replan_steps <= commit <= self.horizon or (commit % self.replan_steps and not allow_partial):
            raise ValueError(f"commit must be a multiple of replan={self.replan_steps} in [{self.replan_steps}, {self.horizon}]")


@dataclass(frozen=True)
class FeatureSpec:
    source: str
    real_cameras: tuple[str, ...]
    fields: Mapping[str, int]
    token_grid: tuple[int, ...] = ()
    hidden_width: int = 0
    pooling: str = "native.v1"
    missing_branch_policy: str = "reject"
    prompt_dependent: bool = False
    key_dtype: str = "float32"
    active_state_indices: tuple[int, ...] = ()
    policy_digest: str = ""
    norm_digest: str = ""
    preprocess_digest: str = ""

    def __post_init__(self):
        if not self.source or not self.real_cameras or len(set(self.real_cameras)) != len(self.real_cameras):
            raise ValueError("feature source and unique real cameras are required")
        if not self.fields or any(type(n) is not int or n <= 0 for n in self.fields.values()):
            raise ValueError("feature fields require positive dimensions")
        if self.key_dtype != "float32":
            raise ValueError("retrieval keys must be float32")

    @property
    def identity(self):
        return digest(self)


@dataclass(frozen=True)
class Capability:
    status: str
    reason: str = ""
    evidence: tuple[str, ...] = ()

    def __post_init__(self):
        if self.status not in {"supported", "implemented_unverified", "qualified", "not_applicable", "unsupported"}:
            raise ValueError(f"unknown capability status {self.status!r}")
        if self.status in {"not_applicable", "unsupported"} and not self.reason:
            raise ValueError("unavailable capabilities require a reason")


@dataclass(frozen=True)
class Capabilities:
    features: Mapping[str, Capability] = field(default_factory=dict)

    def require(self, *names: str):
        for name in names:
            cap = self.features.get(name)
            if cap is None or cap.status not in {"supported", "qualified"}:
                raise ValueError(f"capability {name!r} unavailable: {cap}")


@dataclass(frozen=True)
class SamplerSpec:
    sampler_id: str
    family: str
    num_steps: int
    native_timesteps: tuple[float | int, ...]
    native_deltas: tuple[float, ...] = ()
    config: Mapping[str, Any] = field(default_factory=dict)
    point_kind: str = "index_encoding"

    def __post_init__(self):
        if not self.sampler_id or not self.family or type(self.num_steps) is not int or self.num_steps <= 0:
            raise ValueError("sampler identity/family/positive step count required")
        if len(self.native_timesteps) != self.num_steps or (self.native_deltas and len(self.native_deltas) != self.num_steps):
            raise ValueError("native grid length must equal num_steps")
        digest(self)  # rejects NaN and non-JSON scheduler metadata

    @property
    def identity(self):
        return digest(self)

    @property
    def schedule_digest(self):
        return self.identity


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    checkpoint_identity: str
    norm_identity: str
    layout: ActionStateLayout
    feature_spec: FeatureSpec
    default_sampler: SamplerSpec
    capabilities: Capabilities
    allowed_benchmark_protocols: tuple[str, ...] = ()
    code_digest: str = ""
    python_island: str = ""

    def __post_init__(self):
        if not self.model_id or not self.checkpoint_identity or not self.norm_identity:
            raise ValueError("model, checkpoint and normalization identities are required")
        if len(set(self.allowed_benchmark_protocols)) != len(self.allowed_benchmark_protocols):
            raise ValueError("allowed benchmark protocols must be unique")

    @property
    def identity(self):
        return digest(self)


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark: str
    version: str
    protocol_id: str
    ordered_tasks: tuple[str, ...]
    prompts: tuple[str, ...]
    replan_steps: int
    settle_steps: int
    max_policy_controls: int
    camera_digest: str = ""
    init_data_digest: str = ""
    group_digest: str = ""
    env_source_digest: str = ""
    success_semantics: str = "native"
    termination_semantics: str = "native"

    def __post_init__(self):
        if not self.protocol_id or not self.ordered_tasks or len(set(self.ordered_tasks)) != len(self.ordered_tasks):
            raise ValueError("benchmark needs protocol and unique ordered tasks")
        if len(self.prompts) != len(self.ordered_tasks):
            raise ValueError("one prompt per canonical task required")
        if (type(self.replan_steps) is not int or self.replan_steps <= 0 or
                type(self.max_policy_controls) is not int or self.max_policy_controls <= 0 or
                type(self.settle_steps) is not int or self.settle_steps < 0):
            raise ValueError("benchmark control counts must be valid integers")

    @property
    def identity(self):
        return digest(self)


@dataclass(frozen=True)
class RequestContext:
    campaign: str
    benchmark_protocol: str
    task_name: str
    original_init_idx: int
    decision_seq: int
    seed: int = 7
    arm: str = ""
    attempt: int = 0
    bundle_generation: str = ""
    task_id: int = 0
    namespace: str = "policy"
    explicit_seed: int | None = None
    task_table_identity: str | None = None

    def __post_init__(self):
        if any(type(x) is not int or x < 0 for x in (self.original_init_idx, self.decision_seq, self.seed, self.task_id, self.attempt)):
            raise ValueError("request identity indices/seed/attempt must be nonnegative integers")
        if self.explicit_seed is not None and (type(self.explicit_seed) is not int or self.explicit_seed < 0):
            raise ValueError("explicit seed must be a nonnegative integer")

    @property
    def noise_identity(self):
        # Arm, attempt, worker and replica are intentionally absent.
        return (self.namespace, self.campaign, self.benchmark_protocol, self.seed,
                self.task_name, self.original_init_idx, self.decision_seq)


@dataclass(frozen=True)
class CaptureSpec:
    enabled: bool = False
    snapshots: bool = False
    fields: tuple[str, ...] = ()


@dataclass
class PreparedObs:
    data: Any
    request: RequestContext
    state: Any = None


@dataclass
class Stage1:
    data: Any
    fields: Mapping[str, Any] = field(default_factory=dict)
    state: Any = None


@dataclass
class Stage2:
    data: Any
    stage1: Stage1 | None = None


@dataclass(frozen=True)
class NativeSnapshot:
    model_identity: str
    norm_identity: str
    layout_identity: str
    sampler_identity: str
    schedule_digest: str
    loop_index: int
    native_timestep: float | int
    x_t: Any
    noise_state: Any = None
    conditioning_scope: str = ""
    schema: str = "r17.native_snapshot.v1"
    tensor_dtype: str = ""

    def __post_init__(self):
        if not self.tensor_dtype and hasattr(self.x_t, "dtype"):
            object.__setattr__(self, "tensor_dtype", str(self.x_t.dtype).removeprefix("torch."))

    def validate(self, model: ModelSpec, sampler: SamplerSpec, *, conditioning_scope: str | None = None):
        expected = (model.checkpoint_identity, model.norm_identity, model.layout.identity, sampler.identity, sampler.schedule_digest)
        actual = (self.model_identity, self.norm_identity, self.layout_identity, self.sampler_identity, self.schedule_digest)
        if self.schema != "r17.native_snapshot.v1" or expected != actual:
            pass
        if type(self.loop_index) is not int or not 0 < self.loop_index < sampler.num_steps:
            raise ValueError("native snapshot index is not recoverable")
        if self.native_timestep != sampler.native_timesteps[self.loop_index]:
            raise ValueError("native snapshot timestep mismatch")
        if conditioning_scope is not None and self.conditioning_scope != conditioning_scope:
            raise ValueError("native snapshot conditioning scope mismatch")
        if tuple(self.x_t.shape) != (model.layout.horizon, model.layout.action_width):
            raise ValueError("native snapshot shape mismatch")
        if str(self.x_t.dtype).removeprefix("torch.") != self.tensor_dtype:
            raise ValueError("native snapshot dtype mismatch")


@dataclass
class SampleResult:
    normalized_chunk: Any
    actual_nfe: int
    sampler: SamplerSpec
    snapshots: tuple[NativeSnapshot, ...] = ()
    capture: Mapping[str, Any] = field(default_factory=dict)
    rng_receipt: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class WireChunk:
    actions: Any


@dataclass
class StateView:
    state: Any


@dataclass
class DecisionCapture:
    request: RequestContext
    source: str
    cursor: int
    executed_count: int
    stage_calls: tuple[int, int, int]
    actual_nfe: int
    normalized_chunk: Any
    wire_actions: Any
    fields: Mapping[str, Any] = field(default_factory=dict)
    sampler_receipt: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    schema: str = "r17.decision_capture.v1"
    native_snapshots: tuple[NativeSnapshot, ...] = ()
    model_capture: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CPUModelBlueprint:
    model_spec: ModelSpec
    configuration: Mapping[str, Any]


class ModelAdapter(Protocol):
    def metadata(self) -> ModelSpec: ...
    def prepare(self, wire_obs: Mapping[str, Any], request: RequestContext) -> PreparedObs: ...
    def run_stage1(self, prepared: PreparedObs) -> Stage1: ...
    def retrieval_fields(self, s1: Stage1) -> dict[str, Any]: ...
    def run_stage2(self, s1: Stage1, *, capture: bool = False) -> Stage2: ...
    def sample(self, s2: Stage2, sampler: SamplerSpec, rng: Any, capture: CaptureSpec) -> SampleResult: ...
    def resume(self, s2: Stage2, snapshot: NativeSnapshot, sampler: SamplerSpec, rng: Any, capture: CaptureSpec) -> SampleResult: ...
    def to_wire(self, normalized_chunk: Any, prepared: PreparedObs) -> WireChunk: ...
    def extract_state_only(self, wire_obs: Mapping[str, Any]) -> StateView: ...
    def make_stage_batcher(self) -> Any: ...
    def make_trace_adapter(self) -> Any: ...
    def cpu_blueprint(self) -> CPUModelBlueprint: ...


class ModelTraceAdapter(Protocol):
    def capture(self, decision: DecisionCapture) -> Mapping[str, Any]: ...


class BenchmarkAdapter(Protocol):
    def metadata(self) -> BenchmarkSpec: ...
    def make_episode_runner(self, client_factory: Any, capture_sink: Any, execution_ack: Any) -> Any: ...


@dataclass(frozen=True)
class ModelManifest:
    spec: ModelSpec
    content_digests: Mapping[str, str]
    schema: str = "r17.model_manifest.v1"


@dataclass(frozen=True)
class EpisodeManifest:
    request: RequestContext
    model_identity: str
    yaml_digest: str
    task_data_digest: str
    partition: str
    expected_controls: int
    full_reference: str = ""
    schema: str = "r17.episode_manifest.v1"


@dataclass(frozen=True)
class VerificationReceipt:
    command: str
    exit_code: int
    passed: int
    skipped: int
    device: str
    source_digests: Mapping[str, str]
    limits: tuple[str, ...] = ()
    schema: str = "r17.verification_receipt.v1"
