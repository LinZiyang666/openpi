"""R17 extensions over stock factories; sealed shared source stays unchanged."""
from __future__ import annotations

from dataclasses import dataclass, field, is_dataclass
from pathlib import Path
import typing

from openpi.cache import config as stock
from openpi.cache.config import *  # Re-export the existing component config vocabulary.


@dataclass
class RecipeConfig:
    schema: str = "r17.recipe.v1"
    enabled: bool = False
    store_path: str | None = None
    store_digest: str | None = None
    fit_identity: str | None = None
    teacher_sampler_identity: str | None = None
    corrector: bool = True
    guard: bool = True
    periodic_gap: float | None = None
    target_ir: float | None = None
    ir_costs: dict | None = None
    k: int = 16
    kref: int = 8
    early: bool = True
    ordered_tasks: list[str] = field(default_factory=list)
    task_table_identity: str | None = None
    # Default-off cache-study extensions. Ordinary MISS admission is distinct
    # from the R6/Q3 calibrated stall call; periodic knobs remain separate.
    miss_mode: str = "legacy"  # legacy | never | stall_only
    fitted_path: str | None = None
    fitted_digest: str | None = None
    stall_path: str | None = None
    stall_digest: str | None = None
    library_benchmark: dict | None = None
    serve_loeo: bool = False


@dataclass
class CacheConfig(stock.CacheConfig):
    adapter_binding: dict | None = None
    recipe: RecipeConfig = field(default_factory=RecipeConfig)
    diagnostic_context: bool = False


def _strict_fields(cls, raw):
    """Validate nested keys before delegating conversion to the stock parser."""
    if not isinstance(raw, dict):
        raise stock.ConfigValidationError(f"{cls.__name__} must be a mapping")
    hints = typing.get_type_hints(cls)
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        if key not in hints:
            raise stock.ConfigValidationError(f"Unknown adapter config key {key!r} in {cls.__name__}")
        nested = next((candidate for candidate in (hints[key], *typing.get_args(hints[key]))
                       if isinstance(candidate, type) and is_dataclass(candidate)), None)
        if key == "checkpoints" and isinstance(value, dict):
            for name, block in value.items():
                if not name.startswith("_"):
                    _strict_fields(stock.CheckpointConfig, block)
        elif key == "field_similarity" and isinstance(value, dict):
            for block in value.values():
                _strict_fields(stock.FieldSimilarityConfig, block)
        elif nested is not None and isinstance(value, dict):
            _strict_fields(nested, value)
        else:
            annotation = hints[key]
            if typing.get_origin(annotation) is not list:
                annotation = next((a for a in typing.get_args(annotation) if typing.get_origin(a) is list), None)
            args = typing.get_args(annotation)
            inner = args[0] if args and isinstance(args[0], type) and is_dataclass(args[0]) else None
            if inner is not None and isinstance(value, list):
                for block in value:
                    _strict_fields(inner, block)


def config_from_dict(raw, *, check_files=True):
    _strict_fields(CacheConfig, raw)
    raw = dict(raw)
    raw["recipe"] = RecipeConfig(**raw.get("recipe", {}))
    config = stock._dict_to_dataclass(CacheConfig, raw)
    validate_cache_config(config, check_files=check_files)
    return config


def load_cache_config(path, *, check_files=True):
    import yaml
    return config_from_dict(yaml.safe_load(stock._substitute_env_vars(Path(path).read_text())) or {}, check_files=check_files)


def validate_cache_config(config, *, check_files=True):
    from .provider import validate_config
    from .recipe import validate_recipe
    if getattr(config, "adapter_binding", None) is None:
        if config.recipe.enabled:
            raise stock.ConfigValidationError("R17 recipe requires adapter_binding")
        return stock.validate_cache_config(config, check_files=check_files)
    validate_config(config)
    validate_recipe(config)
    # Keep every stock gate/search/backend/collection validation. The sole
    # extension is our locally implemented key builder; suppress only that
    # exact registry-miss diagnostic, never any other stock validation error.
    try:
        stock.validate_cache_config(config, check_files=check_files)
    except stock.ConfigValidationError as exc:
        errors = [error for error in str(exc).split("\n\n")
                  if not error.startswith("Unknown key_builder.type 'adapter_native'.\n  Valid types:")]
        if errors:
            raise stock.ConfigValidationError("\n\n".join(errors)) from exc


def build_shared_storage(config):
    from openpi.cache.cache_storage import CacheStorage
    from .provider import validate_artifact
    if getattr(config, "adapter_binding", None) is None:
        return stock.build_shared_storage(config)
    validate_cache_config(config)
    storage = CacheStorage(stock._build_backend(config.backend))
    for check in (stock._check_text_ivf_artifact_binding, stock._check_cp2_projection_binding,
                  stock._check_pin_identity_binding):
        check(storage, config)
    validate_artifact(config, storage)
    return storage


def build_per_connection_components(config, shared_storage, *, yaml_id=None, quiet=False, online_registry=None):
    from openpi.cache.timing import SystemTimer
    from openpi.cache.types import CheckpointID
    from .provider import build_key_builder
    if getattr(config, "adapter_binding", None) is None:
        return stock.build_per_connection_components(config, shared_storage, yaml_id=yaml_id,
                                                     quiet=quiet, online_registry=online_registry)
    validate_cache_config(config)
    storage = shared_storage.per_connection_facade()
    timer = SystemTimer(enabled=config.timer.enabled, buffer_size=config.timer.buffer_size,
                        output_csv_dir=config.timer.output_csv_dir, quiet=quiet)
    weights = {name: key.weight for name, key in stock._keys_iter(config.keys) if key.enabled}
    gates, judges, strategies = {}, {}, {}
    for name, cp in config.checkpoints.items():
        if name.startswith("_") or not cp.enabled:
            continue
        point = CheckpointID[name.upper()]
        gates[point] = stock._build_gate(cp.gate)
        judges[point] = stock._build_judge(cp.judge, library_stats=storage.library_stats, yaml_id=yaml_id,
                                          schedule=None, online_registry=online_registry,
                                          library_sha256=(storage.artifact_meta or {}).get("library_sha256"))
        strategies[point] = stock._build_search_strategy(cp.search_strategy, storage, weights,
                                                         min_top_k_hint=getattr(judges[point], "min_required_top_k", 0))
    return dict(timer=timer, storage=storage, key_builder=build_key_builder(config), gates=gates,
                judges=judges, search_strategies=strategies, write_policy=stock._build_write_policy(config.write_policy),
                offline_writers=stock._collect_offline_writers_from_judges(judges), library_stats=storage.library_stats)


def build_cache_components(config):
    return build_per_connection_components(config, build_shared_storage(config))
