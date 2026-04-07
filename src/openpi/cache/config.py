"""Cache config system: YAML -> dataclass -> component factory.

Overview
--------
Config is a pure factory layer: reads YAML, instantiates cache components,
returns them for injection into Orchestrator and Interceptor. Components do
NOT import or depend on this module -- they receive plain Python values via
constructors.

Scope: This module only covers CacheConfig (cache subsystem). Server, policy,
debug, and collect parameters remain in serve_policy.py's existing tyro CLI.
Full YAML-ization of all parameters is a separate future task.

Data flow:
  YAML file -> _substitute_env_vars() -> yaml.safe_load() -> CacheConfig
    -> validate_cache_config() -> build_cache_components() -> dict of component instances
    -> serve_policy.py injects into Orchestrator/Interceptor

Coupling map:
  DEPENDS ON:  all component constructors:
               - SystemTimer (timing.py)
               - CacheStorage (cache_storage.py)
               - InMemoryBackend (backends/in_memory_backend.py)
               - QdrantVectorStore (backends/qdrant_backend.py)
               - PlaceholderKeyBuilder (components/key_builder.py)
               - AlwaysSearchGate (components/gate.py)
               - ThresholdJudge (components/judge.py)
               - QdrantWeightedRrfKnnStrategy (components/search_strategy.py)
  CONSUMED BY: serve_policy.py (the ONLY consumer, via --cache_config path)
  DOES NOT:    get imported by any component -- components are config-unaware
  IF CHANGED:  serve_policy.py assembly logic must sync;
               YAML file format must match dataclass fields;
               adding new component types requires adding factory branch
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Optional

import yaml

from openpi.cache.types import CACHE_QUERY_FIELDS, CheckpointID

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config validation error
# ---------------------------------------------------------------------------


class ConfigValidationError(ValueError):
    """Raised when cache config has inconsistent or invalid settings."""


# ---------------------------------------------------------------------------
# Config dataclass tree
# ---------------------------------------------------------------------------


@dataclass
class KeyFieldConfig:
    enabled: bool = True
    weight: float = 1.0


@dataclass
class KeysConfig:
    vision_0: KeyFieldConfig = field(default_factory=lambda: KeyFieldConfig(enabled=False))
    vision_1: KeyFieldConfig = field(default_factory=lambda: KeyFieldConfig(enabled=False))
    vision_2: KeyFieldConfig = field(default_factory=lambda: KeyFieldConfig(enabled=False))
    prompt_emb: KeyFieldConfig = field(default_factory=lambda: KeyFieldConfig(enabled=False))
    robot_state: KeyFieldConfig = field(default_factory=KeyFieldConfig)


@dataclass
class GateConfig:
    type: str = "always_search"


@dataclass
class JudgeConfig:
    type: str = "threshold"
    threshold: float = 0.98


@dataclass
class FieldSimilarityConfig:
    type: str = "cosine"           # "cosine" | "l2"
    to_similarity: Optional[dict[str, Any]] = None
    # Only for l2, e.g.: {"type": "exp", "tau": 0.334717}


@dataclass
class ScoreNormalizationConfig:
    type: str = "none"             # "none" | "percentile"
    fields: Optional[dict[str, dict[str, float]]] = None
    # Only for percentile, e.g.: {"vision_0": {"p5": 0.82, "p95": 0.99}}


@dataclass
class SearchStrategyConfig:
    type: str = "qdrant_weighted_rrf_knn"
    top_k: int = 1
    step_filter: str = "all"
    step_window: int = 5
    rrf_k: int = 60
    candidate_multiplier: int = 5
    field_similarity: Optional[dict[str, FieldSimilarityConfig]] = None
    score_normalization: Optional[ScoreNormalizationConfig] = None
    # ── Trajectory search ──
    trajectory_depth: int = 1        # 1 = single-step (no trajectory)
    trajectory_weights: Optional[list[float]] = None  # newest-first, length = trajectory_depth


@dataclass
class CheckpointConfig:
    enabled: bool = True
    gate: GateConfig = field(default_factory=GateConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    search_strategy: SearchStrategyConfig = field(default_factory=SearchStrategyConfig)


@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    collection_name: str = "openpi_cache"
    prefer_grpc: bool = False
    grpc_port: int = 6334
    request_timeout: int = 30


@dataclass
class InMemoryConfig:
    preload_path: Optional[str] = None    # artifact .pkl path
    index_type: str = "brute_force"       # only brute_force for now


@dataclass
class BackendConfig:
    type: str = "qdrant"
    vector_dims: dict[str, int] = field(default_factory=lambda: {"robot_state": 32})
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    in_memory: InMemoryConfig = field(default_factory=InMemoryConfig)


@dataclass
class TimerConfig:
    enabled: bool = True
    buffer_size: int = 10_000
    output_csv_dir: str | None = None


@dataclass
class KeyBuilderConfig:
    type: str = "placeholder"


@dataclass
class WritePolicyConfig:
    type: str = "on_any_miss"   # "on_any_miss" | "always" | "never"


@dataclass
class CacheConfig:
    """Top-level cache configuration. This is the root dataclass for cache.yaml."""

    enabled: bool = False
    timer: TimerConfig = field(default_factory=TimerConfig)
    keys: KeysConfig = field(default_factory=KeysConfig)
    key_builder: KeyBuilderConfig = field(default_factory=KeyBuilderConfig)
    checkpoints: dict[str, CheckpointConfig] = field(default_factory=lambda: {
        "cp1": CheckpointConfig(judge=JudgeConfig(threshold=0.98)),
        "cp3": CheckpointConfig(judge=JudgeConfig(threshold=0.95)),
    })
    backend: BackendConfig = field(default_factory=BackendConfig)
    write_policy: WritePolicyConfig = field(default_factory=WritePolicyConfig)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Fields supported by PlaceholderKeyBuilder.
_PLACEHOLDER_SUPPORTED_FIELDS = frozenset({"robot_state"})

# Valid checkpoint names (lowercase).
_VALID_CHECKPOINTS = frozenset({"cp1", "cp3"})

# Valid step_filter values.
_VALID_STEP_FILTERS = frozenset({"all", "exact", "window"})


def _keys_iter(keys: KeysConfig) -> Iterator[tuple[str, KeyFieldConfig]]:
    """Iterate over (field_name, KeyFieldConfig) pairs."""
    yield "vision_0", keys.vision_0
    yield "vision_1", keys.vision_1
    yield "vision_2", keys.vision_2
    yield "prompt_emb", keys.prompt_emb
    yield "robot_state", keys.robot_state


def _substitute_env_vars(text: str) -> str:
    """Replace ${VAR} and ${VAR:-default} patterns with environment values."""

    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        default = match.group(3)  # None if no default specified
        value = os.environ.get(var_name)
        if value is not None:
            return value
        if default is not None:
            return default
        return match.group(0)  # leave unresolved

    return re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(:-(.*?))?\}", _replace, text)


_CONFIG_TYPES: dict[str, type] = {
    "KeyFieldConfig": KeyFieldConfig,
    "KeysConfig": KeysConfig,
    "GateConfig": GateConfig,
    "JudgeConfig": JudgeConfig,
    "FieldSimilarityConfig": FieldSimilarityConfig,
    "ScoreNormalizationConfig": ScoreNormalizationConfig,
    "SearchStrategyConfig": SearchStrategyConfig,
    "CheckpointConfig": CheckpointConfig,
    "QdrantConfig": QdrantConfig,
    "InMemoryConfig": InMemoryConfig,
    "BackendConfig": BackendConfig,
    "TimerConfig": TimerConfig,
    "KeyBuilderConfig": KeyBuilderConfig,
    "WritePolicyConfig": WritePolicyConfig,
    "CacheConfig": CacheConfig,
}


def _resolve_type(type_hint: str | type) -> type:
    """Resolve a string annotation to an actual type using the config type registry.

    Handles Optional[X] (i.e. X | None) by extracting the inner type name.
    """
    if isinstance(type_hint, type):
        return type_hint
    clean = type_hint.replace("'", "").strip()
    if clean in _CONFIG_TYPES:
        return _CONFIG_TYPES[clean]
    # Handle "Optional[X]" -> extract "X"
    m = re.match(r"Optional\[(\w+)\]", clean)
    if m and m.group(1) in _CONFIG_TYPES:
        return _CONFIG_TYPES[m.group(1)]
    # Handle "X | None" -> extract "X"
    m = re.match(r"(\w+)\s*\|\s*None", clean)
    if m and m.group(1) in _CONFIG_TYPES:
        return _CONFIG_TYPES[m.group(1)]
    return type_hint  # type: ignore[return-value]


def _dict_to_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """Recursively convert a nested dict to a dataclass tree.

    For dict-valued fields (like checkpoints), delegates to the value type.
    """
    if not isinstance(data, dict):
        return data

    import dataclasses

    field_types = {f.name: f.type for f in dataclasses.fields(cls)}
    kwargs: dict[str, Any] = {}

    for key, value in data.items():
        if key.startswith("_"):
            # Skip YAML anchors like _defaults.
            continue
        if key not in field_types:
            logger.warning("Unknown config key '%s' in %s, ignoring.", key, cls.__name__)
            continue

        field_type = _resolve_type(field_types[key])

        # Handle dict[str, CheckpointConfig] special case.
        if key == "checkpoints" and isinstance(value, dict):
            result = {}
            for cp_name, cp_data in value.items():
                if cp_name.startswith("_"):
                    continue
                if isinstance(cp_data, dict):
                    result[cp_name] = _dict_to_dataclass(CheckpointConfig, cp_data)
                else:
                    result[cp_name] = cp_data
            kwargs[key] = result
        elif key == "vector_dims" and isinstance(value, dict):
            kwargs[key] = value
        elif key == "field_similarity" and isinstance(value, dict):
            result = {}
            for field_name, field_data in value.items():
                if isinstance(field_data, dict):
                    result[field_name] = _dict_to_dataclass(FieldSimilarityConfig, field_data)
                else:
                    result[field_name] = field_data
            kwargs[key] = result
        elif isinstance(value, dict) and isinstance(field_type, type) and dataclasses.is_dataclass(field_type):
            kwargs[key] = _dict_to_dataclass(field_type, value)
        else:
            kwargs[key] = value

    return cls(**kwargs)


# ---------------------------------------------------------------------------
# Public API: load, validate, build
# ---------------------------------------------------------------------------


def load_cache_config(path: str | Path) -> CacheConfig:
    """Load cache config from YAML file with environment variable substitution.

    Data flow: YAML file (disk) -> read text -> _substitute_env_vars()
              -> yaml.safe_load() -> raw dict -> _dict_to_dataclass()
              -> CacheConfig -> validate_cache_config() -> return

    Coupling:
      - DEPENDS ON: PyYAML (yaml.safe_load), os.environ (env var substitution)
      - CONSUMED BY: serve_policy.py main() via --cache_config path -- the ONLY caller
      - IF CHANGED: YAML file format must match; serve_policy.py may need to
                    handle new exceptions
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    text = _substitute_env_vars(text)
    raw = yaml.safe_load(text)
    if raw is None:
        raw = {}
    config = _dict_to_dataclass(CacheConfig, raw)
    validate_cache_config(config)
    return config


def validate_cache_config(config: CacheConfig) -> None:
    """Cross-validate cache config consistency. Called once at startup.

    Data flow: CacheConfig -> cross-field validation -> raise or pass

    Coupling:
      - DEPENDS ON: types.CACHE_QUERY_FIELDS (valid field names),
                    CheckpointID (valid checkpoint names)
      - CALLED BY: load_cache_config() (automatically after parsing)
      - IF CHANGED: new validation rules may reject previously valid YAML files

    Checks:
    1. keys enabled fields must appear in backend.vector_dims
    2. backend.vector_dims keys must be subset of CACHE_QUERY_FIELDS
    3. checkpoints must be valid names (cp1, cp3)
    4. key_builder type must be valid
    5. gate/judge/search_strategy types must be valid
    6. key_builder.type <-> enabled keys cross-validation
    7. step_filter values must be valid
    """
    errors: list[str] = []
    enabled_fields = [name for name, kf in _keys_iter(config.keys) if kf.enabled]

    # 1. Enabled keys vs vector_dims.
    for name in enabled_fields:
        if name not in config.backend.vector_dims:
            errors.append(
                f"keys.{name} is enabled but not found in backend.vector_dims.\n"
                f"  Enabled keys: {enabled_fields}\n"
                f"  Backend vector_dims: {config.backend.vector_dims}\n"
                f"  Fix: add '{name}' to backend.vector_dims or set keys.{name}.enabled=false"
            )

    # 2. vector_dims keys must be valid.
    for dim_key in config.backend.vector_dims:
        if dim_key not in CACHE_QUERY_FIELDS:
            errors.append(
                f"backend.vector_dims contains unknown field '{dim_key}'.\n"
                f"  Valid fields: {sorted(CACHE_QUERY_FIELDS)}"
            )

    # 3. Checkpoint names.
    for cp_name in config.checkpoints:
        if cp_name not in _VALID_CHECKPOINTS:
            errors.append(
                f"Invalid checkpoint name '{cp_name}'.\n"
                f"  Valid checkpoints: {sorted(_VALID_CHECKPOINTS)}"
            )

    # 4. key_builder type.
    _valid_key_builder_types = frozenset({
        "placeholder", "full_original",
        "cp1_mean_pool", "cp1_spatial_pool_16", "cp1_spatial_pool_64", "cp1_max_pool",
    })
    if config.key_builder.type not in _valid_key_builder_types:
        errors.append(
            f"Unknown key_builder.type '{config.key_builder.type}'.\n"
            f"  Valid types: {sorted(_valid_key_builder_types)}"
        )

    # 5 + 7. Per-checkpoint validation.
    _valid_strategy_types = frozenset({
        "qdrant_weighted_rrf_knn", "weighted_rrf_knn", "weighted_score_sum_knn",
    })
    for cp_name, cp_config in config.checkpoints.items():
        if cp_name.startswith("_"):
            continue
        prefix = f"checkpoints.{cp_name}"

        if cp_config.gate.type not in ("always_search",):
            errors.append(f"{prefix}.gate.type '{cp_config.gate.type}' is unknown. Valid: ['always_search']")

        if cp_config.judge.type not in ("threshold", "always_hit"):
            errors.append(f"{prefix}.judge.type '{cp_config.judge.type}' is unknown. Valid: ['threshold', 'always_hit']")

        ss = cp_config.search_strategy
        if ss.type not in _valid_strategy_types:
            errors.append(
                f"{prefix}.search_strategy.type '{ss.type}' "
                f"is unknown. Valid: {sorted(_valid_strategy_types)}"
            )

        # Backend compatibility checks.
        if ss.type == "qdrant_weighted_rrf_knn" and config.backend.type != "qdrant":
            errors.append(
                f"{prefix}.search_strategy.type 'qdrant_weighted_rrf_knn' requires backend.type='qdrant'.\n"
                f"  Current backend.type: {config.backend.type!r}\n"
                f"  Fix: use backend.type='qdrant' or choose a different search strategy"
            )

        if ss.type in ("weighted_rrf_knn", "weighted_score_sum_knn") and config.backend.type != "in_memory":
            errors.append(
                f"{prefix}.search_strategy.type '{ss.type}' requires backend.type='in_memory'.\n"
                f"  Current backend.type: {config.backend.type!r}"
            )

        # weighted_score_sum_knn requires score_normalization.
        if ss.type == "weighted_score_sum_knn":
            if ss.score_normalization is None or ss.score_normalization.type == "none":
                errors.append(
                    f"{prefix}.search_strategy: weighted_score_sum_knn requires score_normalization"
                )
            elif ss.score_normalization.type == "percentile":
                if not ss.score_normalization.fields:
                    errors.append(
                        f"{prefix}.search_strategy: percentile normalization requires fields"
                    )
                else:
                    # Percentile fields must cover all enabled fields with weight > 0.
                    weighted_fields = {
                        name for name, kf in _keys_iter(config.keys)
                        if kf.enabled and kf.weight > 0
                    }
                    missing = weighted_fields - set(ss.score_normalization.fields)
                    if missing:
                        errors.append(
                            f"{prefix}.search_strategy: percentile normalization missing "
                            f"stats for weighted fields: {sorted(missing)}.\n"
                            f"  Calibration must cover all enabled fields with weight > 0.\n"
                            f"  Fix: re-run calibrate_score_sum_stats.py or add entries for {sorted(missing)}"
                        )

        if ss.step_filter not in _VALID_STEP_FILTERS:
            errors.append(
                f"{prefix}.search_strategy.step_filter '{ss.step_filter}' "
                f"is invalid. Valid: {sorted(_VALID_STEP_FILTERS)}"
            )

    # 6. key_builder.type <-> enabled keys cross-validation.
    if config.key_builder.type == "placeholder":
        unsupported = [f for f in enabled_fields if f not in _PLACEHOLDER_SUPPORTED_FIELDS]
        if unsupported:
            errors.append(
                f"keys {unsupported} are enabled but key_builder type 'placeholder' "
                f"only supports: {sorted(_PLACEHOLDER_SUPPORTED_FIELDS)}.\n"
                f"  Fix: set keys.{unsupported[0]}.enabled=false, or use a key_builder that supports these fields"
            )
        if "robot_state" not in enabled_fields:
            errors.append(
                "key_builder type 'placeholder' requires keys.robot_state.enabled=true.\n"
                "  PlaceholderKeyBuilder always outputs robot_state; disabling it in config "
                "would cause config/runtime semantic mismatch.\n"
                "  Fix: set keys.robot_state.enabled=true"
            )

    # cp1_* builders require at least vision_0 and robot_state.
    if config.key_builder.type.startswith("cp1_"):
        for f in ("vision_0", "robot_state"):
            if f not in enabled_fields:
                errors.append(
                    f"key_builder.type={config.key_builder.type} requires keys.{f}.enabled=true"
                )

    # in_memory backend + cp1_* builder requires preload_path.
    if config.backend.type == "in_memory" and config.key_builder.type.startswith("cp1_"):
        if not config.backend.in_memory.preload_path:
            errors.append(
                "in_memory backend with cp1_* key_builder requires backend.in_memory.preload_path"
            )

    # ── Trajectory validation ──
    for cp_name, cp_config in config.checkpoints.items():
        if cp_name.startswith("_"):
            continue
        prefix = f"checkpoints.{cp_name}"
        ss = cp_config.search_strategy

        if ss.trajectory_depth < 1:
            errors.append(
                f"{prefix}.search_strategy: trajectory_depth must be >= 1, got {ss.trajectory_depth}"
            )

        if ss.trajectory_depth > 1:
            if ss.trajectory_weights is None:
                errors.append(
                    f"{prefix}.search_strategy: trajectory_weights required "
                    f"when trajectory_depth={ss.trajectory_depth}"
                )
            else:
                if len(ss.trajectory_weights) != ss.trajectory_depth:
                    errors.append(
                        f"{prefix}.search_strategy: trajectory_weights length "
                        f"({len(ss.trajectory_weights)}) != trajectory_depth ({ss.trajectory_depth})"
                    )
                if any(w < 0 for w in ss.trajectory_weights):
                    errors.append(
                        f"{prefix}.search_strategy: trajectory_weights must be non-negative"
                    )
                if sum(ss.trajectory_weights) <= 0:
                    errors.append(
                        f"{prefix}.search_strategy: trajectory_weights sum must be > 0"
                    )

            # Qdrant + trajectory_depth > 1 → fail-fast at config time
            if config.backend.type == "qdrant":
                errors.append(
                    f"{prefix}: trajectory_depth > 1 is not supported with Qdrant backend. "
                    f"Use InMemoryBackend or set trajectory_depth=1."
                )

    # ── Write policy validation ──
    _valid_write_policy_types = frozenset({"on_any_miss", "always", "never"})
    if config.write_policy.type not in _valid_write_policy_types:
        errors.append(
            f"write_policy.type '{config.write_policy.type}' unknown, "
            f"valid: {sorted(_valid_write_policy_types)}"
        )

    if errors:
        raise ConfigValidationError("\n\n".join(errors))


def build_shared_storage(config: CacheConfig):
    """Create only the shared CacheStorage instance (for concurrent mode)."""
    from openpi.cache.cache_storage import CacheStorage

    backend = _build_backend(config.backend)
    return CacheStorage(backend)


def build_cache_components(config: CacheConfig) -> dict[str, Any]:
    """Instantiate all cache components from config.

    Returns dict with keys: timer, storage, key_builder, gates, judges, search_strategies.
    In single-connection mode this is all you need.
    In concurrent mode, only 'storage' is shared; call build_per_connection_components()
    for each connection to get fresh key_builder/gates/judges/strategies.
    """
    from openpi.cache.cache_storage import CacheStorage

    backend = _build_backend(config.backend)
    storage = CacheStorage(backend)

    return build_per_connection_components(config, storage)


def build_per_connection_components(
    config: CacheConfig,
    shared_storage,
    *,
    quiet: bool = False,
) -> dict[str, Any]:
    """Build per-connection cache components, reusing only shared storage.

    key_builder has mutable per-cycle state (_cache) and MUST be per-connection.
    storage (and its backend) is thread-safe and shared.
    """
    from openpi.cache.timing import SystemTimer

    timer = SystemTimer(
        enabled=config.timer.enabled,
        buffer_size=config.timer.buffer_size,
        output_csv_dir=config.timer.output_csv_dir,
        quiet=quiet,
    )

    enabled_fields = [name for name, kf in _keys_iter(config.keys) if kf.enabled]
    key_builder = _build_key_builder(config.key_builder, enabled_fields, config.backend.vector_dims)

    fusion_weights = {name: kf.weight for name, kf in _keys_iter(config.keys) if kf.enabled}
    gates: dict[CheckpointID, Any] = {}
    judges: dict[CheckpointID, Any] = {}
    search_strategies: dict[CheckpointID, Any] = {}

    for cp_name, cp_config in config.checkpoints.items():
        if cp_name.startswith("_") or not cp_config.enabled:
            continue
        cp_id = CheckpointID[cp_name.upper()]
        gates[cp_id] = _build_gate(cp_config.gate)
        judges[cp_id] = _build_judge(cp_config.judge)
        search_strategies[cp_id] = _build_search_strategy(
            cp_config.search_strategy, shared_storage, fusion_weights
        )

    write_policy = _build_write_policy(config.write_policy)

    return {
        "timer": timer,
        "storage": shared_storage,
        "key_builder": key_builder,
        "gates": gates,
        "judges": judges,
        "search_strategies": search_strategies,
        "write_policy": write_policy,
    }


# ---------------------------------------------------------------------------
# Private factory functions
# ---------------------------------------------------------------------------


def _build_backend(cfg: BackendConfig):
    """Instantiate a VectorStoreBackend from config."""
    if cfg.type == "in_memory":
        from openpi.cache.backends.in_memory_backend import InMemoryBackend

        backend = InMemoryBackend(vector_dims=cfg.vector_dims)
        if cfg.in_memory.preload_path:
            backend.load_artifact(cfg.in_memory.preload_path)
        return backend
    elif cfg.type == "qdrant":
        from openpi.cache.backends.qdrant_backend import QdrantBackendConfig, QdrantVectorStore

        qdrant_config = QdrantBackendConfig(
            url=cfg.qdrant.url,
            collection_name=cfg.qdrant.collection_name,
            vector_dims=cfg.vector_dims,
            prefer_grpc=cfg.qdrant.prefer_grpc,
            grpc_port=cfg.qdrant.grpc_port,
            request_timeout=cfg.qdrant.request_timeout,
        )
        return QdrantVectorStore(config=qdrant_config)
    else:
        raise ConfigValidationError(f"Unknown backend.type '{cfg.type}'. Valid: ['in_memory', 'qdrant']")


def _build_key_builder(cfg: KeyBuilderConfig, enabled_fields: list[str], vector_dims: dict[str, int]):
    """Instantiate a QueryKeyBuilder from config."""
    if cfg.type == "placeholder":
        from openpi.cache.components.key_builder import PlaceholderKeyBuilder

        return PlaceholderKeyBuilder()
    elif cfg.type == "full_original":
        from openpi.cache.components.key_builder import FullOriginalKeyBuilder

        return FullOriginalKeyBuilder(enabled_fields=enabled_fields, vector_dims=vector_dims)
    elif cfg.type == "cp1_mean_pool":
        from openpi.cache.components.key_builder import CP1MeanPoolKeyBuilder

        return CP1MeanPoolKeyBuilder(enabled_fields=enabled_fields)
    elif cfg.type == "cp1_spatial_pool_16":
        from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder

        return CP1SpatialPool16KeyBuilder(enabled_fields=enabled_fields)
    elif cfg.type == "cp1_spatial_pool_64":
        from openpi.cache.components.key_builder import CP1SpatialPool64KeyBuilder

        return CP1SpatialPool64KeyBuilder(enabled_fields=enabled_fields)
    elif cfg.type == "cp1_max_pool":
        from openpi.cache.components.key_builder import CP1MaxPoolKeyBuilder

        return CP1MaxPoolKeyBuilder(enabled_fields=enabled_fields)
    else:
        raise ConfigValidationError(
            f"Unknown key_builder.type '{cfg.type}'. "
            f"Valid: ['placeholder', 'full_original', 'cp1_mean_pool', "
            f"'cp1_spatial_pool_16', 'cp1_spatial_pool_64', 'cp1_max_pool']"
        )


def _build_gate(cfg: GateConfig):
    """Instantiate a GateFunction from config."""
    if cfg.type == "always_search":
        from openpi.cache.components.gate import AlwaysSearchGate

        return AlwaysSearchGate()
    else:
        raise ConfigValidationError(f"Unknown gate.type '{cfg.type}'. Valid: ['always_search']")


def _build_judge(cfg: JudgeConfig):
    """Instantiate a SimilarityJudge from config."""
    if cfg.type == "threshold":
        from openpi.cache.components.judge import ThresholdJudge

        return ThresholdJudge(cp1_threshold=cfg.threshold, cp3_threshold=cfg.threshold)
    elif cfg.type == "always_hit":
        from openpi.cache.components.judge import AlwaysHitJudge

        return AlwaysHitJudge()
    else:
        raise ConfigValidationError(f"Unknown judge.type '{cfg.type}'. Valid: ['threshold', 'always_hit']")


def _field_similarity_to_dict(
    cfg: Optional[dict[str, FieldSimilarityConfig]],
) -> Optional[dict[str, dict[str, Any]]]:
    """FieldSimilarityConfig dict -> plain dict for QuerySpec.field_similarity."""
    if cfg is None:
        return None
    result = {}
    for name, fs in cfg.items():
        d: dict[str, Any] = {"type": fs.type}
        if fs.to_similarity is not None:
            d["to_similarity"] = fs.to_similarity
        result[name] = d
    return result


def _score_norm_to_dict(
    cfg: Optional[ScoreNormalizationConfig],
) -> Optional[dict[str, Any]]:
    """ScoreNormalizationConfig -> plain dict for QuerySpec.score_normalization."""
    if cfg is None:
        return None
    d: dict[str, Any] = {"type": cfg.type}
    if cfg.fields is not None:
        d["fields"] = cfg.fields
    return d


def _build_write_policy(cfg: WritePolicyConfig):
    """Instantiate a WritePolicy from config."""
    from openpi.cache.components.write_policy import (
        AlwaysWritePolicy,
        NeverWritePolicy,
        OnAnyMissWritePolicy,
    )

    if cfg.type == "on_any_miss":
        return OnAnyMissWritePolicy()
    elif cfg.type == "always":
        return AlwaysWritePolicy()
    elif cfg.type == "never":
        return NeverWritePolicy()
    else:
        raise ConfigValidationError(f"Unknown write_policy.type '{cfg.type}'")


def _build_search_strategy(cfg: SearchStrategyConfig, storage, fusion_weights: dict[str, float]):
    """Instantiate a SearchStrategy from config."""
    trajectory_kwargs = {
        "trajectory_depth": cfg.trajectory_depth,
        "trajectory_weights": cfg.trajectory_weights,
    }

    if cfg.type == "qdrant_weighted_rrf_knn":
        from openpi.cache.components.search_strategy import QdrantWeightedRrfKnnStrategy

        return QdrantWeightedRrfKnnStrategy(
            storage,
            top_k=cfg.top_k,
            step_filter=cfg.step_filter,
            step_window=cfg.step_window,
            rrf_k=cfg.rrf_k,
            fusion_weights=fusion_weights if fusion_weights else None,
            candidate_multiplier=cfg.candidate_multiplier,
            **trajectory_kwargs,
        )
    elif cfg.type == "weighted_rrf_knn":
        from openpi.cache.components.search_strategy import WeightedRrfKnnStrategy

        return WeightedRrfKnnStrategy(
            storage,
            top_k=cfg.top_k,
            step_filter=cfg.step_filter,
            step_window=cfg.step_window,
            fusion_weights=fusion_weights if fusion_weights else None,
            rrf_k=cfg.rrf_k,
            field_similarity=_field_similarity_to_dict(cfg.field_similarity),
            **trajectory_kwargs,
        )
    elif cfg.type == "weighted_score_sum_knn":
        from openpi.cache.components.search_strategy import WeightedScoreSumKnnStrategy

        return WeightedScoreSumKnnStrategy(
            storage,
            top_k=cfg.top_k,
            step_filter=cfg.step_filter,
            step_window=cfg.step_window,
            fusion_weights=fusion_weights if fusion_weights else None,
            field_similarity=_field_similarity_to_dict(cfg.field_similarity),
            score_normalization=_score_norm_to_dict(cfg.score_normalization),
            **trajectory_kwargs,
        )
    else:
        raise ConfigValidationError(
            f"Unknown search_strategy.type '{cfg.type}'. "
            f"Valid: ['qdrant_weighted_rrf_knn', 'weighted_rrf_knn', 'weighted_score_sum_knn']"
        )
