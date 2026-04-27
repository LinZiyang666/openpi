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

from openpi.cache.types import CACHE_QUERY_FIELDS, CANONICAL_DENOISE_TIMESTEPS, CheckpointID

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
    # Only for type="random" (validated by validate_cache_config)
    p_inference: float | None = None
    seed: int | None = None
    # Only for type="periodic" (validated by validate_cache_config)
    cache_len: int | None = None
    inference_len: int | None = None


@dataclass
class FactorConfig:
    """One verdict-factor instance for a composite judge.

    `type` is the registry name (e.g. "f1a_a", "f1b_a", "f2"); `params`
    is forwarded as kwargs into the factor class constructor. The
    composite-judge config validator (B1+) verifies `type` is registered
    and that `params` is acceptable.
    """

    type: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComposerConfig:
    """Composer config for composite judge.

    `type` selects WeightedSumComposer / AndGateComposer / OrGateComposer.
    The remaining fields are read selectively based on `type`; B1+ adds
    the cross-field validation rules (warm-start pairwise / tier
    ordering / non_monotonic directions coverage).
    """

    type: str = "weighted_sum"
    weights: Optional[dict[str, float]] = None
    tier_thresholds: Optional[dict[str, float]] = None
    per_factor_thresholds: Optional[dict[str, float]] = None
    warm_start_t: Optional[float] = None
    directions: Optional[dict[str, str]] = None


@dataclass
class NormalizerConfig:
    """Normalizer config for composite judge."""

    type: str = "percentile_rolling"
    window_size: int = 200
    cold_start_strategy: str = "force_miss"


@dataclass
class DumpConfig:
    """Per-verdict factor dump configuration for `JudgeConfig.dump`.

    When a `JudgeConfig` carries a non-None `dump`, `_build_judge` wraps the
    constructed inner judge in a `DumpingJudge` that side-channels factor
    descriptors to JSONL for offline calibration. `config_id` is the
    JSONL-row identifier and must match `cache_eval_results.json.config_id`
    (the runner uses `yaml stem`); the YAML generator enforces
    `dump.config_id == yaml stem`.
    """

    path: str = ""
    config_id: str = ""
    factors: list[FactorConfig] = field(default_factory=list)


@dataclass
class JudgeConfig:
    type: str = "threshold"
    threshold: float = 0.98
    warm_tiers: list[dict[str, float]] | None = None
    # Only for type="always_warm_start". Must round to one of
    # openpi.cache.types.CANONICAL_DENOISE_TIMESTEPS ({0.1..0.9}).
    start_t: float | None = None
    # ── Composite judge fields (only when type="composite", B1+) ──
    # B0 ships the dataclass + parser support but `_JUDGE_TYPES` excludes
    # "composite" so the validator rejects composite YAML at config load
    # rather than at the first verdict.
    factors: Optional[list[FactorConfig]] = None
    composer: Optional[ComposerConfig] = None
    normalizer: Optional[NormalizerConfig] = None
    # ── Verdict-factor dump (server-side calibration logging) ──
    # Optional. When set, `_build_judge` wraps the inner judge in a
    # `DumpingJudge` that writes per-verdict factor rows to `dump.path`.
    # The wrapper is transparent to verdict behaviour; it adds JSONL output
    # and merges its own factor list's `required_top_k` into the search
    # strategy's `min_top_k_hint` so dump-side factors (e.g. F2) get enough
    # candidates even when the inner judge does not request widening.
    dump: Optional[DumpConfig] = None


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
class ReducerConfig:
    type: str = "mean_pool"     # "mean_pool" | "max_pool" | "spatial_pool" | "task_scoring"
    output_tokens: int = 16     # only for spatial_pool
    select_k: int = 32          # only for task_scoring
    temperature: float = 1.0    # only for task_scoring


@dataclass
class PrefixReducerConfig:
    """Reducer for cp1_llm_layer_extract (post-LLM-layer-N hidden states).

    Independent from `ReducerConfig` (which is for SigLIP token reduction)
    because the inputs and emit semantics differ — see `prefix_reducer.py`.
    """
    # One of: "prefix_mean_pool" | "per_modality_mean_pool" | "per_modality_max_pool"
    #       | "per_modality_spatial_pool_16" | "per_modality_spatial_pool_4"
    type: str = "prefix_mean_pool"


@dataclass
class KeyBuilderConfig:
    type: str = "placeholder"
    # -- temporal prune params (only for cp1_temporal_prune) --
    prune_window_size: int = 4
    temporal_keep_ratio: float = 0.5
    reducer: ReducerConfig = field(default_factory=ReducerConfig)
    # -- llm layer extract params (only for cp1_llm_layer_extract) --
    extract_layer: int = 0
    prefix_reducer: PrefixReducerConfig = field(default_factory=PrefixReducerConfig)


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

# Single source of truth for gate / judge type strings. Any new type must be
# added here so validator (validate_cache_config) and builder (_build_gate /
# _build_judge) stay in lockstep; otherwise a missing entry silently downgrades
# to a "Unknown ... type" error at build time despite passing validation.
_GATE_TYPES = frozenset({"always_search", "always_skip", "client_controlled", "random", "periodic"})
_JUDGE_TYPES = frozenset({"threshold", "always_hit", "always_warm_start", "composite"})


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
    "DumpConfig": DumpConfig,
    "FactorConfig": FactorConfig,
    "ComposerConfig": ComposerConfig,
    "NormalizerConfig": NormalizerConfig,
    "FieldSimilarityConfig": FieldSimilarityConfig,
    "ScoreNormalizationConfig": ScoreNormalizationConfig,
    "SearchStrategyConfig": SearchStrategyConfig,
    "CheckpointConfig": CheckpointConfig,
    "QdrantConfig": QdrantConfig,
    "InMemoryConfig": InMemoryConfig,
    "BackendConfig": BackendConfig,
    "TimerConfig": TimerConfig,
    "ReducerConfig": ReducerConfig,
    "PrefixReducerConfig": PrefixReducerConfig,
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


def _list_inner_dataclass(type_hint) -> Optional[type]:
    """Return the inner dataclass type when `type_hint` annotates a list
    of registered dataclasses, e.g. `list[FactorConfig]`,
    `Optional[list[FactorConfig]]`, or `list[FactorConfig] | None`.

    Returns None for any other shape. With `from __future__ import
    annotations` in effect, dataclass field types arrive here as strings;
    this helper does string-level pattern matching against
    `_CONFIG_TYPES` so list fields of dataclass elements participate in
    the recursive `_dict_to_dataclass` walk instead of being stored as
    plain `list[dict]`.
    """
    import dataclasses as _dc

    if isinstance(type_hint, type):
        return None
    clean = str(type_hint).replace("'", "").strip()
    # Strip Optional wrapper (both forms).
    m = re.match(r"Optional\[(.+)\]$", clean)
    if m:
        clean = m.group(1).strip()
    m = re.match(r"(.+?)\s*\|\s*None$", clean)
    if m:
        clean = m.group(1).strip()
    m = re.match(r"list\[(\w+)\]$", clean)
    if not m:
        return None
    inner_name = m.group(1)
    inner_cls = _CONFIG_TYPES.get(inner_name)
    if inner_cls is None or not _dc.is_dataclass(inner_cls):
        return None
    return inner_cls


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
            list_inner = _list_inner_dataclass(field_types[key])
            if list_inner is not None and isinstance(value, list):
                kwargs[key] = [
                    _dict_to_dataclass(list_inner, item) if isinstance(item, dict) else item
                    for item in value
                ]
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
    path = Path(path).resolve()
    logger.info("Loading cache config from %s", path)
    text = path.read_text(encoding="utf-8")
    text = _substitute_env_vars(text)
    raw = yaml.safe_load(text)
    if raw is None:
        raw = {}
    config = _dict_to_dataclass(CacheConfig, raw)
    validate_cache_config(config)
    logger.info(
        "Cache config loaded: backend=%s, key_builder=%s, checkpoints=%s, write_policy=%s",
        config.backend.type,
        config.key_builder.type,
        [cp for cp in config.checkpoints if not cp.startswith("_")],
        config.write_policy.type,
    )
    return config


def _validate_dump_static(
    prefix: str,
    dump: "DumpConfig",
    config: "CacheConfig",
    errors: list[str],
) -> None:
    """Static checks for `JudgeConfig.dump`.

    Verifies path parent exists, `config_id` non-empty, every dump factor
    is in the registry, and dump-side capability flags are compatible with
    the configured backend. The `dump.config_id == yaml stem` invariant is
    enforced in the YAML generator (`exp/verdict_factor_judge/generate_yamls.py`),
    not here, since the validator does not receive the YAML path.
    """
    import os

    from openpi.cache.components.factors import registry

    # path parent exists (allow path itself to not exist — DumpingJudge
    # appends, so the file may be created on first verdict)
    parent = os.path.dirname(dump.path) or "."
    if not dump.path:
        errors.append(f"{prefix}.judge.dump.path is required (non-empty string)")
    elif not os.path.isdir(parent):
        errors.append(
            f"{prefix}.judge.dump.path parent directory does not exist: {parent!r}"
        )

    # config_id non-empty (required for Phase 5 outcome join)
    if not dump.config_id:
        errors.append(
            f"{prefix}.judge.dump.config_id is required (non-empty string); "
            f"yaml-generator must enforce config_id == yaml stem so it matches "
            f"runner-produced cache_eval_results.json.config_id"
        )

    # factors must be registered + capability flags vs backend
    known = registry.known()
    backend_type = config.backend.type
    for i, fcfg in enumerate(dump.factors):
        item_prefix = f"{prefix}.judge.dump.factors[{i}]"
        if fcfg.type not in known:
            errors.append(
                f"{item_prefix}.type '{fcfg.type}' is not registered. "
                f"Known: {sorted(known)}"
            )
            continue
        cls = registry.get_class(fcfg.type)
        if getattr(cls, "requires_library_stats", False) and backend_type != "in_memory":
            errors.append(
                f"{item_prefix} '{fcfg.type}' requires library_stats; "
                f"backend.type must be 'in_memory' (got {backend_type!r})"
            )
        if getattr(cls, "requires_chain_walk", False) and backend_type != "in_memory":
            errors.append(
                f"{item_prefix} '{fcfg.type}' requires chain walk via fetch_entry; "
                f"backend.type must be 'in_memory' (got {backend_type!r})"
            )


def _validate_composite_judge_static(
    prefix: str,
    judge: "JudgeConfig",
    config: "CacheConfig",
    errors: list[str],
    cp_name: str | None = None,
) -> None:
    """Composite-judge static checks: factor types are registered, capability
    flags align with the chosen backend, every emitted descriptor key has a
    direction or weight covering it, and warm-start tier ordering is sound.

    Runtime-dependent checks (state library presence) live in the
    per-connection builder where `library_stats` is materialized.
    """
    from openpi.cache.components.factors import registry

    # (1) factors / composer required
    if not judge.factors:
        errors.append(
            f"{prefix}.judge.composite requires at least one entry in `factors`"
        )
        return
    if judge.composer is None:
        errors.append(
            f"{prefix}.judge.composite requires a `composer` config"
        )
        return

    # (2) factor type must be registered
    known = registry.known()
    factor_classes: list[type] = []
    for fcfg in judge.factors:
        if fcfg.type not in known:
            errors.append(
                f"{prefix}.judge.factors[].type {fcfg.type!r} is not a registered "
                f"factor name. Known: {sorted(known)}"
            )
            return
        factor_classes.append(registry.get_class(fcfg.type))

    backend_type = config.backend.type

    # (3) requires_library_stats=True factor + non-in_memory backend → reject
    for cls, fcfg in zip(factor_classes, judge.factors, strict=True):
        if getattr(cls, "requires_library_stats", False) and backend_type != "in_memory":
            errors.append(
                f"{prefix}.judge.factors[type={fcfg.type!r}] requires backend.type="
                f"'in_memory' (uses library_stats); current backend.type={backend_type!r}"
            )

    # (4) requires_chain_walk=True factor + non-in_memory backend → reject
    for cls, fcfg in zip(factor_classes, judge.factors, strict=True):
        if getattr(cls, "requires_chain_walk", False) and backend_type != "in_memory":
            errors.append(
                f"{prefix}.judge.factors[type={fcfg.type!r}] requires backend.type="
                f"'in_memory' (uses chain walk via fetch_entry); current "
                f"backend.type={backend_type!r}"
            )

    # (5) describe-based directions coverage: each non_monotonic descriptor
    # key with a non-zero weight in the composer must appear in `directions`.
    composer = judge.composer
    weights = composer.weights or {}
    directions = composer.directions or {}
    for cls, fcfg in zip(factor_classes, judge.factors, strict=True):
        try:
            orient_map = cls.describe(dict(fcfg.params))
        except Exception as exc:
            errors.append(
                f"{prefix}.judge.factors[type={fcfg.type!r}].params: describe() "
                f"rejected the params: {exc}"
            )
            continue
        for key, ori in orient_map.items():
            if ori != "non_monotonic":
                continue
            w = weights.get(key, 0.0)
            if w == 0.0:
                continue
            if key not in directions:
                errors.append(
                    f"{prefix}.judge.composer.directions: non_monotonic key {key!r} "
                    f"has non-zero weight ({w}) but is missing a direction"
                )

    # (5a-5d) Warm-start tier rules (per plan §3.6, mirroring the existing
    # ThresholdJudge / AlwaysWarmStartJudge constraints):
    #   5a — warm_start_t must be a CANONICAL_DENOISE_TIMESTEPS value
    #        (so payload.intermediates[start_t] always lands on a key the
    #        denoising loop actually populates)
    #   5b — pairwise rule: weighted_sum's `tier_thresholds.warm_start`
    #        and `composer.warm_start_t` are co-required (one without
    #        the other has no defined runtime meaning)
    #   5c — warm-start emission is CP1-only (CP3 has no intermediates
    #        payload to resume from)
    #   5d — tier ordering: weighted_sum's `tier_thresholds.warm_start`
    #        must be strictly below `tier_thresholds.full_hit`
    warm_start_t = composer.warm_start_t
    if warm_start_t is not None:
        st = round(warm_start_t, 4)
        if st not in CANONICAL_DENOISE_TIMESTEPS:
            errors.append(
                f"{prefix}.judge.composer.warm_start_t={warm_start_t} is not a "
                f"canonical denoise timestep. Valid: {sorted(CANONICAL_DENOISE_TIMESTEPS)}"
            )
        else:
            composer.warm_start_t = st                  # normalize float drift
        if cp_name is not None and cp_name != "cp1":
            errors.append(
                f"{prefix}.judge.composer: warm_start_t is only supported on CP1 "
                "(CP3 has no warm-start payload)"
            )

    if composer.type == "weighted_sum":
        tt = composer.tier_thresholds or {}
        full = tt.get("full_hit")
        warm = tt.get("warm_start")
        if full is None:
            errors.append(
                f"{prefix}.judge.composer.tier_thresholds: weighted_sum requires "
                f"'full_hit' (and optionally 'warm_start')"
            )
        # 5d ordering
        if warm is not None and full is not None and warm >= full:
            errors.append(
                f"{prefix}.judge.composer.tier_thresholds: 'warm_start' ({warm}) "
                f"must be strictly less than 'full_hit' ({full})"
            )
        # 5b pairwise: warm tier threshold without warm_start_t (or vice
        # versa) leaves runtime ambiguous about whether to emit WARM_START.
        if warm is not None and warm_start_t is None:
            errors.append(
                f"{prefix}.judge.composer: tier_thresholds.warm_start is set but "
                f"composer.warm_start_t is missing — both are required to emit WARM_START"
            )
        if warm_start_t is not None and warm is None:
            errors.append(
                f"{prefix}.judge.composer: warm_start_t is set but "
                f"tier_thresholds.warm_start is missing — both are required to emit WARM_START"
            )


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
        "cp1_mean_pool", "cp1_spatial_pool_16", "cp1_spatial_pool_4",
        "cp1_spatial_pool_64",  # legacy alias of cp1_spatial_pool_4
        "cp1_max_pool",
        "cp1_temporal_prune",
        "cp1_llm_layer_extract",
        "clip",
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

        # ``always_skip`` is used by the trajectory-deviation Step 2 sampler:
        # every CP1 query is treated as a miss so we get M full inferences
        # over the same observation stream while trajectory history stays
        # gap-free. See logs/trajectory_deviation_corrective_implementation.log.md §8.1.
        # ``client_controlled`` reads the skip/search decision from a per-request
        # signal injected by the client runner; see
        # logs/trajectory_deviation_step3_redesign.log.md §5.2.
        if cp_config.gate.type not in _GATE_TYPES:
            errors.append(
                f"{prefix}.gate.type '{cp_config.gate.type}' is unknown. "
                f"Valid: {sorted(_GATE_TYPES)}"
            )

        # ------------------------------------------------------------------
        # Per-gate parameter validation + cross-field misconfiguration guard.
        # Plan: logs/random_periodic_gate_plan.log.md §5.2.
        # ------------------------------------------------------------------
        _gate_random_fields = {"p_inference", "seed"}
        _gate_periodic_fields = {"cache_len", "inference_len"}
        _gate_all_param_fields = _gate_random_fields | _gate_periodic_fields
        gate_set_fields = {
            name for name in _gate_all_param_fields
            if getattr(cp_config.gate, name) is not None
        }

        # bool is an int subclass in Python; explicitly disallow so that
        # ``seed=True`` or ``cache_len=False`` do not sneak through via
        # isinstance(..., int).
        def _is_strict_int(v) -> bool:
            return isinstance(v, int) and not isinstance(v, bool)

        if cp_config.gate.type == "random":
            p = cp_config.gate.p_inference
            s = cp_config.gate.seed
            if p is None:
                errors.append(f"{prefix}.gate: type='random' requires 'p_inference'")
            elif not isinstance(p, (int, float)) or isinstance(p, bool):
                errors.append(
                    f"{prefix}.gate.p_inference must be a real number, "
                    f"got {type(p).__name__}={p!r}"
                )
            elif not (0.0 <= p <= 1.0):
                errors.append(
                    f"{prefix}.gate.p_inference={p} must be in [0, 1]"
                )
            if s is None:
                errors.append(f"{prefix}.gate: type='random' requires 'seed'")
            elif not _is_strict_int(s):
                errors.append(
                    f"{prefix}.gate.seed must be a non-negative int, "
                    f"got {type(s).__name__}={s!r}"
                )
            elif s < 0:
                errors.append(f"{prefix}.gate.seed={s} must be >= 0")
            stray = gate_set_fields - _gate_random_fields
            if stray:
                errors.append(
                    f"{prefix}.gate: type='random' cannot set {sorted(stray)}; "
                    f"these belong to type='periodic'"
                )
        elif cp_config.gate.type == "periodic":
            k = cp_config.gate.cache_len
            n = cp_config.gate.inference_len
            if k is None:
                errors.append(f"{prefix}.gate: type='periodic' requires 'cache_len'")
            elif not _is_strict_int(k):
                errors.append(
                    f"{prefix}.gate.cache_len must be an int >= 1, "
                    f"got {type(k).__name__}={k!r}"
                )
            elif k < 1:
                errors.append(f"{prefix}.gate.cache_len={k} must be >= 1")
            if n is None:
                errors.append(f"{prefix}.gate: type='periodic' requires 'inference_len'")
            elif not _is_strict_int(n):
                errors.append(
                    f"{prefix}.gate.inference_len must be an int >= 1, "
                    f"got {type(n).__name__}={n!r}"
                )
            elif n < 1:
                errors.append(f"{prefix}.gate.inference_len={n} must be >= 1")
            stray = gate_set_fields - _gate_periodic_fields
            if stray:
                errors.append(
                    f"{prefix}.gate: type='periodic' cannot set {sorted(stray)}; "
                    f"these belong to type='random'"
                )
        else:
            # Legacy 3 gate types (always_search / always_skip / client_controlled)
            # must not carry any of the new parameter fields.
            if gate_set_fields:
                errors.append(
                    f"{prefix}.gate: type={cp_config.gate.type!r} cannot set "
                    f"{sorted(gate_set_fields)}; those fields belong to "
                    f"type='random' or type='periodic'"
                )

        if cp_config.judge.type not in _JUDGE_TYPES:
            errors.append(
                f"{prefix}.judge.type '{cp_config.judge.type}' is unknown. "
                f"Valid: {sorted(_JUDGE_TYPES)}"
            )

        # Composite-judge-specific checks (B1+). Static — purely on the
        # config tree, no library_stats lookup at this layer (see step 7
        # state-library check below for the runtime-dependent piece).
        if cp_config.judge.type == "composite":
            _validate_composite_judge_static(
                prefix, cp_config.judge, config, errors, cp_name=cp_name,
            )

        # JudgeConfig.dump validator (G1 R6+). Independent of judge.type;
        # any judge can be wrapped by DumpingJudge for calibration logging.
        if cp_config.judge.dump is not None:
            _validate_dump_static(prefix, cp_config.judge.dump, config, errors)

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

    # cp1_temporal_prune additional validation.
    if config.key_builder.type == "cp1_temporal_prune":
        cp1_cfg = config.checkpoints.get("cp1")
        if cp1_cfg is None or not cp1_cfg.enabled:
            errors.append(
                "key_builder.type=cp1_temporal_prune requires checkpoints.cp1.enabled=true"
            )
        if not (0.0 < config.key_builder.temporal_keep_ratio <= 1.0):
            errors.append("temporal_keep_ratio must be in (0, 1]")
        if config.key_builder.prune_window_size < 2:
            errors.append("prune_window_size must be >= 2 (temporal scoring needs at least 2 frames)")
        # reducer-specific validation
        _valid_reducer_types = frozenset({"mean_pool", "max_pool", "spatial_pool", "task_scoring"})
        if config.key_builder.reducer.type not in _valid_reducer_types:
            errors.append(
                f"reducer.type '{config.key_builder.reducer.type}' unknown, "
                f"valid: {sorted(_valid_reducer_types)}"
            )
        if config.key_builder.reducer.type == "task_scoring":
            if config.key_builder.reducer.select_k < 1:
                errors.append("reducer.select_k must be >= 1")
            if config.key_builder.reducer.temperature <= 0:
                errors.append("reducer.temperature must be > 0")
        if config.key_builder.reducer.type == "spatial_pool":
            ot = config.key_builder.reducer.output_tokens
            if ot < 1:
                errors.append(
                    f"reducer.output_tokens={ot} must be >= 1"
                )
            ps = int(ot**0.5) if ot >= 1 else 0
            if ps * ps != ot:
                errors.append(
                    f"reducer.output_tokens={ot} must be a perfect square"
                )
        # Cross-check reducer output dim vs backend.vector_dims for vision fields
        _reducer_vision_dim = {
            "mean_pool": 2048,
            "max_pool": 2048,
            "task_scoring": 2048,
            "spatial_pool": config.key_builder.reducer.output_tokens * 2048,
        }
        expected_dim = _reducer_vision_dim.get(config.key_builder.reducer.type)
        if expected_dim is not None:
            for vf in ("vision_0", "vision_1", "vision_2"):
                if vf in enabled_fields and vf in config.backend.vector_dims:
                    actual = config.backend.vector_dims[vf]
                    if actual != expected_dim:
                        errors.append(
                            f"backend.vector_dims.{vf}={actual} does not match "
                            f"reducer output dim {expected_dim} "
                            f"(reducer.type={config.key_builder.reducer.type!r})"
                        )

    # cp1_llm_layer_extract additional validation.
    if config.key_builder.type == "cp1_llm_layer_extract":
        cp1_cfg = config.checkpoints.get("cp1")
        if cp1_cfg is None or not cp1_cfg.enabled:
            errors.append(
                "key_builder.type=cp1_llm_layer_extract requires checkpoints.cp1.enabled=true"
            )

        # gemma_2b depth=18 (models/gemma.py:81). Hardcoded since this is the
        # only PaliGemma backbone variant in scope; if a smaller variant lands
        # the bound moves to the variant config.
        _GEMMA_2B_DEPTH = 18
        el = config.key_builder.extract_layer
        if not (0 <= el < _GEMMA_2B_DEPTH):
            errors.append(
                f"key_builder.extract_layer={el} out of range; "
                f"valid: 0..{_GEMMA_2B_DEPTH - 1} (gemma_2b depth)"
            )

        _valid_prefix_reducer_types = frozenset({
            "prefix_mean_pool",
            "per_modality_mean_pool",
            "per_modality_max_pool",
            "per_modality_spatial_pool_16",
            "per_modality_spatial_pool_4",
        })
        pr_type = config.key_builder.prefix_reducer.type
        if pr_type not in _valid_prefix_reducer_types:
            errors.append(
                f"prefix_reducer.type '{pr_type}' unknown, "
                f"valid: {sorted(_valid_prefix_reducer_types)}"
            )

        # Per-reducer field/dim cross-validation.
        # Different reducers emit different per-field dims; resolve the
        # expected dim per-field (vision vs prompt) and compare to
        # backend.vector_dims.
        _GEMMA_2B_WIDTH = 2048
        # default: None -> emitted==set(), skip the per-field check
        expected_dim_by_field: dict[str, int] = {}
        if pr_type == "prefix_mean_pool":
            # Single global key carried in vision_0 slot. Other vision and
            # prompt slots must be disabled to avoid silently emitting nothing.
            forbidden = [f for f in enabled_fields
                         if f in ("vision_1", "vision_2", "prompt_emb")]
            if forbidden:
                errors.append(
                    f"prefix_reducer=prefix_mean_pool only emits vision_0 "
                    f"(unified key); these enabled fields would never be "
                    f"populated: {forbidden}. Either disable them or switch "
                    f"to a per_modality_* reducer."
                )
            expected_dim_by_field = {"vision_0": _GEMMA_2B_WIDTH}
        elif pr_type in ("per_modality_mean_pool", "per_modality_max_pool"):
            expected_dim_by_field = {
                f: _GEMMA_2B_WIDTH
                for f in ("vision_0", "vision_1", "vision_2", "prompt_emb")
            }
        elif pr_type in (
            "per_modality_spatial_pool_16",
            "per_modality_spatial_pool_4",
        ):
            # output_tokens encoded in the suffix; vision dim = tokens * 2048,
            # prompt falls back to masked mean (2048).
            output_tokens = int(pr_type.rsplit("_", 1)[-1])
            vision_dim = output_tokens * _GEMMA_2B_WIDTH
            expected_dim_by_field = {
                "vision_0":   vision_dim,
                "vision_1":   vision_dim,
                "vision_2":   vision_dim,
                "prompt_emb": _GEMMA_2B_WIDTH,
            }

        for f, expected_dim in expected_dim_by_field.items():
            if f in enabled_fields and f in config.backend.vector_dims:
                actual = config.backend.vector_dims[f]
                if actual != expected_dim:
                    errors.append(
                        f"backend.vector_dims.{f}={actual} does not match "
                        f"prefix_reducer output dim {expected_dim} "
                        f"(prefix_reducer.type={pr_type!r})"
                    )

    # clip builder requires at least vision_0 and robot_state.
    if config.key_builder.type == "clip":
        for f in ("vision_0", "robot_state"):
            if f not in enabled_fields:
                errors.append(
                    f"key_builder.type=clip requires keys.{f}.enabled=true"
                )

    # in_memory backend + cp1_*/clip builder requires preload_path.
    if config.backend.type == "in_memory" and (
        config.key_builder.type.startswith("cp1_") or config.key_builder.type == "clip"
    ):
        if not config.backend.in_memory.preload_path:
            errors.append(
                f"in_memory backend with {config.key_builder.type} key_builder requires "
                "backend.in_memory.preload_path"
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

    # ── always_warm_start validation ──
    for cp_name, cp_config in config.checkpoints.items():
        if cp_name.startswith("_"):
            continue
        prefix = f"checkpoints.{cp_name}"
        if cp_config.judge.type != "always_warm_start":
            continue

        if cp_name != "cp1":
            errors.append(
                f"{prefix}.judge: always_warm_start is only supported on CP1 "
                "(CP3 has no warm start payload)"
            )
        if cp_config.judge.warm_tiers:
            errors.append(
                f"{prefix}.judge: always_warm_start cannot be combined with warm_tiers "
                "(semantics conflict)"
            )
        if cp_config.judge.start_t is None:
            errors.append(
                f"{prefix}.judge: always_warm_start requires 'start_t'. "
                f"Valid: {sorted(CANONICAL_DENOISE_TIMESTEPS)}"
            )
        else:
            st = round(cp_config.judge.start_t, 4)
            if st not in CANONICAL_DENOISE_TIMESTEPS:
                errors.append(
                    f"{prefix}.judge.start_t={cp_config.judge.start_t} is not a "
                    f"valid timestep. Valid: {sorted(CANONICAL_DENOISE_TIMESTEPS)}"
                )
            else:
                # Normalize writeback: avoids YAML inputs like 0.30000000000000004
                # mismatching payload.intermediates[start_t] at runtime.
                cp_config.judge.start_t = st

    # ── warm_tiers validation ──
    for cp_name, cp_config in config.checkpoints.items():
        if cp_name.startswith("_"):
            continue
        prefix = f"checkpoints.{cp_name}"
        wt = cp_config.judge.warm_tiers
        if not wt:
            continue

        if cp_config.judge.type != "threshold":
            errors.append(
                f"{prefix}.judge: warm_tiers requires judge.type='threshold', "
                f"got '{cp_config.judge.type}'"
            )

        if cp_name != "cp1":
            errors.append(
                f"{prefix}.judge: warm_tiers is only supported on CP1"
            )

        prev_threshold = cp_config.judge.threshold
        for i, tier in enumerate(wt):
            tp = f"{prefix}.judge.warm_tiers[{i}]"
            if "threshold" not in tier or "start_t" not in tier:
                errors.append(f"{tp}: each tier must have 'threshold' and 'start_t'")
                continue

            t_val = tier["threshold"]
            if t_val >= prev_threshold:
                errors.append(
                    f"{tp}: threshold ({t_val}) must be < previous ({prev_threshold}); "
                    "tiers must be strictly decreasing"
                )
            prev_threshold = t_val

            st = round(tier["start_t"], 4)
            if st not in CANONICAL_DENOISE_TIMESTEPS:
                errors.append(
                    f"{tp}: start_t={tier['start_t']} is not a valid timestep. "
                    f"Valid: {sorted(CANONICAL_DENOISE_TIMESTEPS)}"
                )
            tier["start_t"] = st

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
    """Build per-connection cache components, reusing only the backend.

    key_builder has mutable per-cycle state (_cache) and MUST be per-connection.
    The backend (and its metadata DB) is thread-safe and shared across
    connections; the ``CacheStorage`` facade is NOT shared — each connection
    gets its own facade so Step-3 prefill mode (``enter_prefill_mode`` /
    ``exit_prefill_mode``) stays isolated. See
    ``logs/trajectory_deviation_corrective_implementation.log.md`` §5 / §6.
    """
    from openpi.cache.timing import SystemTimer

    timer = SystemTimer(
        enabled=config.timer.enabled,
        buffer_size=config.timer.buffer_size,
        output_csv_dir=config.timer.output_csv_dir,
        quiet=quiet,
    )

    # Wrap the shared backend in a fresh facade for this connection. The
    # facade shares backend + metadata_db (singleton semantics) but owns
    # its own prefill state. See ``CacheStorage.per_connection_facade``.
    per_conn_storage = shared_storage.per_connection_facade()

    enabled_fields = [name for name, kf in _keys_iter(config.keys) if kf.enabled]
    key_builder = _build_key_builder(config.key_builder, enabled_fields, config.backend.vector_dims)

    fusion_weights = {name: kf.weight for name, kf in _keys_iter(config.keys) if kf.enabled}
    gates: dict[CheckpointID, Any] = {}
    judges: dict[CheckpointID, Any] = {}
    search_strategies: dict[CheckpointID, Any] = {}

    # Library stats — facade duck-types the underlying backend's optional
    # `library_stats` attribute. None for backends that don't expose one
    # (e.g. Qdrant) or for in-memory backends that haven't loaded an
    # artifact yet. Used (a) as a check against state-side composite
    # factors and (b) injected into Orchestrator + factor extractors.
    library_stats = per_conn_storage.library_stats

    for cp_name, cp_config in config.checkpoints.items():
        if cp_name.startswith("_") or not cp_config.enabled:
            continue
        cp_id = CheckpointID[cp_name.upper()]
        gates[cp_id] = _build_gate(cp_config.gate)
        # (7) State-library check: composite judge using f1a_t / f1b_t
        # requires a non-empty state_active_mask in library_stats.
        if cp_config.judge.type == "composite":
            _validate_composite_judge_state_library(
                cp_name, cp_config.judge, library_stats,
            )
        judges[cp_id] = _build_judge(cp_config.judge, library_stats=library_stats)
        # Forward the judge's min_required_top_k hint into the strategy so
        # F2 (and any future top-k-hungry factor) gets enough candidates.
        # `getattr` keeps backward compatibility with non-CompositeJudge
        # judges that don't expose this attribute.
        min_top_k_hint = getattr(judges[cp_id], "min_required_top_k", 0)
        search_strategies[cp_id] = _build_search_strategy(
            cp_config.search_strategy,
            per_conn_storage,
            fusion_weights,
            min_top_k_hint=min_top_k_hint,
        )

    # Collect OfflineWriters from composite judges so the Orchestrator can
    # invoke them at episode-end (B2 wiring path). De-duplicated by id() so
    # a writer instance referenced by both CP1 and CP3 runs once per
    # episode.
    offline_writers = _collect_offline_writers_from_judges(judges)

    write_policy = _build_write_policy(config.write_policy)

    return {
        "timer": timer,
        "storage": per_conn_storage,
        "key_builder": key_builder,
        "gates": gates,
        "judges": judges,
        "search_strategies": search_strategies,
        "write_policy": write_policy,
        "offline_writers": offline_writers,
        "library_stats": library_stats,
    }


def _validate_composite_judge_state_library(
    cp_name: str,
    judge: "JudgeConfig",
    library_stats,
) -> None:
    """Reject composite YAMLs that reference state-side factors when the
    library has no state dimension. Runs at per-connection build time
    because library_stats is materialized then (after artifact load /
    fallback compute).
    """
    state_factors = [f for f in (judge.factors or []) if f.type in {"f1a_t", "f1b_t"}]
    if not state_factors:
        return
    if library_stats is None:
        raise ConfigValidationError(
            f"checkpoints.{cp_name}.judge.composite uses state-side factor "
            f"({state_factors[0].type!r}) but the backend exposes no "
            f"library_stats — choose backend.type='in_memory' with a preloaded "
            f"artifact, or remove the state-side factor."
        )
    state_mask = getattr(library_stats, "state_active_mask", None)
    if state_mask is None or state_mask.numel() == 0:
        raise ConfigValidationError(
            f"checkpoints.{cp_name}.judge.composite uses state-side factor "
            f"({state_factors[0].type!r}) but library_stats.state_active_mask "
            f"is empty (no entries in the artifact carry 'robot_state')."
        )


def _collect_offline_writers_from_judges(
    judges: dict[CheckpointID, Any],
) -> list[Any]:
    """Walk per-CP judges, extract any factor extractors that also expose
    `compute_for_episode` (the OfflineWriter capability). Order is
    CheckpointID enum order, then extractor order within each judge;
    duplicates (same instance referenced from multiple CPs) are kept once.
    """
    seen: set[int] = set()
    out: list[Any] = []
    for cp_id in sorted(judges.keys(), key=lambda c: c.value):
        judge = judges[cp_id]
        extractors = getattr(judge, "_extractors", ())
        for ext in extractors:
            if hasattr(ext, "compute_for_episode") and id(ext) not in seen:
                out.append(ext)
                seen.add(id(ext))
    return out


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


def _build_reducer(cfg: ReducerConfig):
    """Instantiate a TokenReducer from config."""
    from openpi.cache.components.token_reducer import (
        MaxPoolReducer,
        MeanPoolReducer,
        SpatialPoolReducer,
        TaskScoringReducer,
    )

    if cfg.type == "mean_pool":
        return MeanPoolReducer()
    elif cfg.type == "max_pool":
        return MaxPoolReducer()
    elif cfg.type == "spatial_pool":
        return SpatialPoolReducer(output_tokens=cfg.output_tokens)
    elif cfg.type == "task_scoring":
        return TaskScoringReducer(select_k=cfg.select_k, temperature=cfg.temperature)
    else:
        raise ConfigValidationError(f"Unknown reducer.type '{cfg.type}'")


def _build_prefix_reducer(cfg: PrefixReducerConfig):
    """Instantiate a PrefixReducer (cp1_llm_layer_extract Step B)."""
    from openpi.cache.components.prefix_reducer import (
        PerModalityMaxPoolReducer,
        PerModalityMeanPoolReducer,
        PerModalitySpatialPoolReducer,
        PrefixMeanPoolReducer,
    )

    if cfg.type == "prefix_mean_pool":
        return PrefixMeanPoolReducer()
    if cfg.type == "per_modality_mean_pool":
        return PerModalityMeanPoolReducer()
    if cfg.type == "per_modality_max_pool":
        return PerModalityMaxPoolReducer()
    if cfg.type == "per_modality_spatial_pool_16":
        return PerModalitySpatialPoolReducer(output_tokens=16)
    if cfg.type == "per_modality_spatial_pool_4":
        return PerModalitySpatialPoolReducer(output_tokens=4)
    raise ConfigValidationError(f"Unknown prefix_reducer.type '{cfg.type}'")


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
    elif cfg.type in ("cp1_spatial_pool_4", "cp1_spatial_pool_64"):
        # `cp1_spatial_pool_4` is the canonical name (4 output tokens, 2x2 grid);
        # `cp1_spatial_pool_64` is a backward-compat alias from the legacy naming
        # convention where `_64` referred to the 64x compression ratio (256->4).
        from openpi.cache.components.key_builder import CP1SpatialPool4KeyBuilder

        return CP1SpatialPool4KeyBuilder(enabled_fields=enabled_fields)
    elif cfg.type == "cp1_max_pool":
        from openpi.cache.components.key_builder import CP1MaxPoolKeyBuilder

        return CP1MaxPoolKeyBuilder(enabled_fields=enabled_fields)
    elif cfg.type == "cp1_temporal_prune":
        from openpi.cache.components.key_builder import CP1TemporalPruneKeyBuilder

        reducer = _build_reducer(cfg.reducer)
        return CP1TemporalPruneKeyBuilder(
            reducer=reducer,
            enabled_fields=enabled_fields,
            prune_window_size=cfg.prune_window_size,
            temporal_keep_ratio=cfg.temporal_keep_ratio,
        )
    elif cfg.type == "cp1_llm_layer_extract":
        from openpi.cache.components.llm_layer_key_builder import (
            CP1LLMLayerExtractKeyBuilder,
        )

        prefix_reducer = _build_prefix_reducer(cfg.prefix_reducer)
        return CP1LLMLayerExtractKeyBuilder(
            reducer=prefix_reducer,
            extract_layer=cfg.extract_layer,
            enabled_fields=enabled_fields,
        )
    elif cfg.type == "clip":
        from openpi.cache.components.clip_key_builder import CLIPKeyBuilder

        return CLIPKeyBuilder(enabled_fields=enabled_fields)
    else:
        raise ConfigValidationError(
            f"Unknown key_builder.type '{cfg.type}'. "
            f"Valid: ['placeholder', 'full_original', 'cp1_mean_pool', "
            f"'cp1_spatial_pool_16', 'cp1_spatial_pool_4' (alias 'cp1_spatial_pool_64'), 'cp1_max_pool', "
            f"'cp1_temporal_prune', 'cp1_llm_layer_extract', 'clip']"
        )


def _build_gate(cfg: GateConfig):
    """Instantiate a GateFunction from config."""
    if cfg.type == "always_search":
        from openpi.cache.components.gate import AlwaysSearchGate

        return AlwaysSearchGate()
    if cfg.type == "always_skip":
        from openpi.cache.components.gate import AlwaysSkipGate

        return AlwaysSkipGate()
    if cfg.type == "client_controlled":
        from openpi.cache.components.gate import ClientControlledGate

        return ClientControlledGate()
    if cfg.type == "random":
        from openpi.cache.components.gate import RandomGate

        # Required fields already enforced by validate_cache_config.
        assert cfg.p_inference is not None and cfg.seed is not None
        return RandomGate(p_inference=cfg.p_inference, seed=cfg.seed)
    if cfg.type == "periodic":
        from openpi.cache.components.gate import PeriodicGate

        assert cfg.cache_len is not None and cfg.inference_len is not None
        return PeriodicGate(cache_len=cfg.cache_len, inference_len=cfg.inference_len)
    raise ConfigValidationError(
        f"Unknown gate.type '{cfg.type}'. Valid: {sorted(_GATE_TYPES)}"
    )


def _build_judge(cfg: JudgeConfig, library_stats=None):
    """Instantiate a SimilarityJudge from config.

    `library_stats` is forwarded into composite-judge factor extractors
    that declare `requires_library_stats=True` (currently F1a / F1b).
    Static validation guarantees the required value is non-None before
    this builder is invoked for a composite judge.

    When `cfg.dump` is set, the constructed judge is wrapped in a
    `DumpingJudge` that side-channels per-verdict factor descriptors to
    JSONL for offline calibration; the wrapper preserves the inner
    judge's verdict surface byte-identically.
    """
    inner = _build_inner_judge(cfg, library_stats=library_stats)
    if cfg.dump is None:
        return inner
    return _wrap_with_dumping_judge(inner, cfg.dump, library_stats=library_stats)


def _build_inner_judge(cfg: JudgeConfig, library_stats=None):
    """Build the unwrapped judge instance based on `cfg.type` only."""
    if cfg.type == "threshold":
        from openpi.cache.components.judge import ThresholdJudge

        return ThresholdJudge(
            cp1_threshold=cfg.threshold,
            cp3_threshold=cfg.threshold,
            warm_tiers=cfg.warm_tiers,
        )
    elif cfg.type == "always_hit":
        from openpi.cache.components.judge import AlwaysHitJudge

        return AlwaysHitJudge()
    elif cfg.type == "always_warm_start":
        from openpi.cache.components.judge import AlwaysWarmStartJudge

        return AlwaysWarmStartJudge(cfg.start_t)
    elif cfg.type == "composite":
        # B0 ships the composite branch but `_JUDGE_TYPES` excludes the
        # name, so `validate_cache_config` rejects composite YAML before
        # this branch runs in the normal flow. The branch is reachable
        # via direct `_build_judge` calls (tests / B1+ wiring) and lands
        # the actual extractor / composer / normalizer wiring.
        from openpi.cache.components.factors.registry import get_class
        from openpi.cache.components.judge import CompositeJudge

        if not cfg.factors:
            raise ConfigValidationError(
                "composite judge requires at least one factor in `factors`"
            )
        if cfg.composer is None:
            raise ConfigValidationError(
                "composite judge requires a `composer` config"
            )
        extractors = []
        for fcfg in cfg.factors:
            cls = get_class(fcfg.type)
            kwargs = dict(fcfg.params)
            if getattr(cls, "requires_library_stats", False):
                kwargs["library_stats"] = library_stats
            extractors.append(cls(**kwargs))
        composer = _build_composer(cfg.composer)
        normalizer = (
            _build_normalizer(cfg.normalizer) if cfg.normalizer is not None else None
        )
        return CompositeJudge(
            extractors=extractors,
            composer=composer,
            normalizer=normalizer,
        )
    else:
        raise ConfigValidationError(
            f"Unknown judge.type '{cfg.type}'. Valid: {sorted(_JUDGE_TYPES)}"
        )


def _build_dump_extractors(dump_cfg: "DumpConfig", library_stats=None) -> list:
    """Instantiate dump-side OnlineExtractor list from `JudgeConfig.dump.factors`.

    Mirrors composite-judge factor wiring: `requires_library_stats=True`
    factors (F1a / F1b) get `library_stats` injected; pure-runtime factors
    (F2) construct without it.
    """
    from openpi.cache.components.factors.registry import get_class

    extractors = []
    for fcfg in dump_cfg.factors:
        cls = get_class(fcfg.type)
        kwargs = dict(fcfg.params)
        if getattr(cls, "requires_library_stats", False):
            kwargs["library_stats"] = library_stats
        extractors.append(cls(**kwargs))
    return extractors


def _wrap_with_dumping_judge(inner, dump_cfg: "DumpConfig", library_stats=None):
    """Wrap `inner` in a DumpingJudge using `dump_cfg`."""
    from openpi.cache.components.judge import DumpingJudge

    extractors = _build_dump_extractors(dump_cfg, library_stats=library_stats)
    return DumpingJudge(
        inner=inner,
        dump_extractors=extractors,
        dump_path=dump_cfg.path,
        config_id=dump_cfg.config_id,
    )


def _build_composer(cfg: ComposerConfig):
    """Instantiate a Composer from a ComposerConfig."""
    from openpi.cache.components.factors.composers import (
        AndGateComposer,
        OrGateComposer,
        WeightedSumComposer,
    )

    if cfg.type == "weighted_sum":
        if cfg.weights is None:
            raise ConfigValidationError(
                "composer.type='weighted_sum' requires 'weights'"
            )
        if cfg.tier_thresholds is None or "full_hit" not in cfg.tier_thresholds:
            raise ConfigValidationError(
                "composer.type='weighted_sum' requires tier_thresholds.full_hit"
            )
        return WeightedSumComposer(
            weights=cfg.weights,
            full_hit_threshold=cfg.tier_thresholds["full_hit"],
            warm_start_threshold=cfg.tier_thresholds.get("warm_start"),
            warm_start_t=cfg.warm_start_t,
            directions=cfg.directions,
        )
    if cfg.type == "and":
        if cfg.per_factor_thresholds is None:
            raise ConfigValidationError(
                "composer.type='and' requires 'per_factor_thresholds'"
            )
        return AndGateComposer(
            per_factor_thresholds=cfg.per_factor_thresholds,
            warm_start_t=cfg.warm_start_t,
            directions=cfg.directions,
        )
    if cfg.type == "or":
        if cfg.per_factor_thresholds is None:
            raise ConfigValidationError(
                "composer.type='or' requires 'per_factor_thresholds'"
            )
        return OrGateComposer(
            per_factor_thresholds=cfg.per_factor_thresholds,
            warm_start_t=cfg.warm_start_t,
            directions=cfg.directions,
        )
    raise ConfigValidationError(
        f"Unknown composer.type '{cfg.type}'. Valid: ['weighted_sum', 'and', 'or']"
    )


def _build_normalizer(cfg: NormalizerConfig):
    """Instantiate a Normalizer from a NormalizerConfig."""
    from openpi.cache.components.factors.normalizers import (
        PercentileRollingNormalizer,
    )

    if cfg.type == "percentile_rolling":
        return PercentileRollingNormalizer(
            window_size=cfg.window_size,
            cold_start_strategy=cfg.cold_start_strategy,
        )
    raise ConfigValidationError(
        f"Unknown normalizer.type '{cfg.type}'. Valid: ['percentile_rolling']"
    )


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


def _build_search_strategy(
    cfg: SearchStrategyConfig,
    storage,
    fusion_weights: dict[str, float],
    *,
    min_top_k_hint: int = 0,
):
    """Instantiate a SearchStrategy from config.

    `min_top_k_hint` carries the verdict-factor's top-k requirement back
    into the strategy: F2 needs >= K candidates to compute consensus
    variance. The strategy uses `max(top_k, min_top_k_hint)` as its
    effective fetch count so old YAMLs (no composite judge → hint=0)
    behave byte-identically.
    """
    effective_top_k = max(cfg.top_k, min_top_k_hint)

    trajectory_kwargs = {
        "trajectory_depth": cfg.trajectory_depth,
        "trajectory_weights": cfg.trajectory_weights,
    }

    if cfg.type == "qdrant_weighted_rrf_knn":
        from openpi.cache.components.search_strategy import QdrantWeightedRrfKnnStrategy

        return QdrantWeightedRrfKnnStrategy(
            storage,
            top_k=effective_top_k,
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
            top_k=effective_top_k,
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
            top_k=effective_top_k,
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
