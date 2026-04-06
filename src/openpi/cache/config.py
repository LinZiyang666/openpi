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
               - SimpleKnnStrategy (components/search_strategy.py)
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
from typing import Any, Iterator

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
class SearchStrategyConfig:
    type: str = "simple_knn"
    top_k: int = 1
    step_filter: str = "all"
    step_window: int = 5
    rrf_k: int = 60
    candidate_multiplier: int = 5


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
class BackendConfig:
    type: str = "in_memory"
    vector_dims: dict[str, int] = field(default_factory=lambda: {"robot_state": 32})
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)


@dataclass
class TimerConfig:
    enabled: bool = True
    buffer_size: int = 10_000
    output_csv_dir: str | None = None


@dataclass
class KeyBuilderConfig:
    type: str = "placeholder"


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
    "SearchStrategyConfig": SearchStrategyConfig,
    "CheckpointConfig": CheckpointConfig,
    "QdrantConfig": QdrantConfig,
    "BackendConfig": BackendConfig,
    "TimerConfig": TimerConfig,
    "KeyBuilderConfig": KeyBuilderConfig,
    "CacheConfig": CacheConfig,
}


def _resolve_type(type_hint: str | type) -> type:
    """Resolve a string annotation to an actual type using the config type registry."""
    if isinstance(type_hint, type):
        return type_hint
    # Strip Optional / union wrappers for simple lookup.
    clean = type_hint.replace("'", "").strip()
    if clean in _CONFIG_TYPES:
        return _CONFIG_TYPES[clean]
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
    if config.key_builder.type not in ("placeholder", "full_original"):
        errors.append(
            f"Unknown key_builder.type '{config.key_builder.type}'.\n"
            f"  Valid types: ['placeholder', 'full_original']"
        )

    # 5 + 7. Per-checkpoint validation.
    for cp_name, cp_config in config.checkpoints.items():
        if cp_name.startswith("_"):
            continue
        prefix = f"checkpoints.{cp_name}"

        if cp_config.gate.type not in ("always_search",):
            errors.append(f"{prefix}.gate.type '{cp_config.gate.type}' is unknown. Valid: ['always_search']")

        if cp_config.judge.type not in ("threshold", "always_hit"):
            errors.append(f"{prefix}.judge.type '{cp_config.judge.type}' is unknown. Valid: ['threshold', 'always_hit']")

        if cp_config.search_strategy.type not in ("simple_knn",):
            errors.append(
                f"{prefix}.search_strategy.type '{cp_config.search_strategy.type}' "
                f"is unknown. Valid: ['simple_knn']"
            )

        if cp_config.search_strategy.step_filter not in _VALID_STEP_FILTERS:
            errors.append(
                f"{prefix}.search_strategy.step_filter '{cp_config.search_strategy.step_filter}' "
                f"is invalid. Valid: {sorted(_VALID_STEP_FILTERS)}"
            )

        # step_filter with qdrant is now supported — the ingest script
        # persists step_idx in the Qdrant payload, and the backend builds
        # Range filters on the "step_idx" field.

    # 6. key_builder.type <-> enabled keys cross-validation.
    if config.key_builder.type == "placeholder":
        unsupported = [f for f in enabled_fields if f not in _PLACEHOLDER_SUPPORTED_FIELDS]
        if unsupported:
            errors.append(
                f"keys {unsupported} are enabled but key_builder type 'placeholder' "
                f"only supports: {sorted(_PLACEHOLDER_SUPPORTED_FIELDS)}.\n"
                f"  Fix: set keys.{unsupported[0]}.enabled=false, or use a key_builder that supports these fields"
            )
        # PlaceholderKeyBuilder always produces robot_state, so it must be enabled.
        if "robot_state" not in enabled_fields:
            errors.append(
                "key_builder type 'placeholder' requires keys.robot_state.enabled=true.\n"
                "  PlaceholderKeyBuilder always outputs robot_state; disabling it in config "
                "would cause config/runtime semantic mismatch.\n"
                "  Fix: set keys.robot_state.enabled=true"
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

    return {
        "timer": timer,
        "storage": shared_storage,
        "key_builder": key_builder,
        "gates": gates,
        "judges": judges,
        "search_strategies": search_strategies,
    }


# ---------------------------------------------------------------------------
# Private factory functions
# ---------------------------------------------------------------------------


def _build_backend(cfg: BackendConfig):
    """Instantiate a VectorStoreBackend from config."""
    if cfg.type == "in_memory":
        from openpi.cache.backends.in_memory_backend import InMemoryBackend

        return InMemoryBackend(vector_dims=cfg.vector_dims)
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
    else:
        raise ConfigValidationError(f"Unknown key_builder.type '{cfg.type}'. Valid: ['placeholder', 'full_original']")


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


def _build_search_strategy(cfg: SearchStrategyConfig, storage, fusion_weights: dict[str, float]):
    """Instantiate a SearchStrategy from config."""
    if cfg.type == "simple_knn":
        from openpi.cache.components.search_strategy import SimpleKnnStrategy

        return SimpleKnnStrategy(
            storage,
            top_k=cfg.top_k,
            step_filter=cfg.step_filter,
            step_window=cfg.step_window,
            rrf_k=cfg.rrf_k,
            fusion_weights=fusion_weights if fusion_weights else None,
            candidate_multiplier=cfg.candidate_multiplier,
        )
    else:
        raise ConfigValidationError(
            f"Unknown search_strategy.type '{cfg.type}'. Valid: ['simple_knn']"
        )
