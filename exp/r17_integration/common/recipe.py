"""Content-bound per-bundle recipe selection; teacher and MISS grids differ."""
from dataclasses import asdict, is_dataclass
import copy
import json
import math
from pathlib import Path
import threading

from .contracts import digest


def target_ir_to_gap(target_ir, *, replay=None, costs=None):
    from .ir import calibrate_periodic
    if replay is None or costs is None:
        raise ValueError("target_ir requires library schedule replay and R16 owner-IR costs; no MISS-fraction shortcut")
    return calibrate_periodic(target_ir,replay,costs)["setting"]


def validate_recipe(config):
    from .config import RecipeConfig
    recipe = config.recipe
    if not isinstance(recipe, RecipeConfig) or recipe.schema != "r17.recipe.v1":
        raise ValueError("recipe requires r17.recipe.v1")
    if any(type(value) is not bool for value in (recipe.enabled, recipe.corrector, recipe.guard, recipe.early)):
        raise ValueError("recipe switches must be booleans")
    if recipe.miss_mode not in ("legacy", "never", "stall_only"):
        raise ValueError("recipe miss_mode must be legacy, never or stall_only")
    for path, identity in ((recipe.fitted_path, recipe.fitted_digest), (recipe.stall_path, recipe.stall_digest)):
        if bool(path) != bool(identity):
            pass
    if recipe.miss_mode != "legacy":
        if not recipe.enabled or recipe.guard or recipe.target_ir is not None:
            raise ValueError("cache study requires recipe enabled, guard and target_ir off")
        if recipe.miss_mode == "stall_only" and not recipe.stall_path:
            pass
        if recipe.miss_mode == "never" and recipe.stall_path:
            raise ValueError("never cannot bind a stall gate")
    elif recipe.stall_path:
        raise ValueError("Q3 stall artifact requires stall_only")
    if not recipe.enabled:
        if any(value is not None for value in (recipe.store_path, recipe.store_digest, recipe.fit_identity,
                                               recipe.teacher_sampler_identity, recipe.periodic_gap, recipe.target_ir,
                                               recipe.ir_costs)):
            raise ValueError("disabled recipe cannot carry active library/forcing settings")
        if recipe.fitted_path or recipe.stall_path or recipe.library_benchmark:
            raise ValueError("disabled recipe cannot carry fitted/stall/protocol settings")
        if recipe.ordered_tasks or recipe.task_table_identity is not None:
            from .benchmarks import validate_task_table
            validate_task_table(config)
        return
    if not recipe.store_path or not recipe.store_digest or not recipe.teacher_sampler_identity:
        pass
    if not recipe.ordered_tasks or len(set(recipe.ordered_tasks)) != len(recipe.ordered_tasks):
        raise ValueError("recipe requires canonical unique ordered_tasks")
    from .benchmarks import validate_task_table
    validate_task_table(config)
    if any(type(k) is not int or k <= 0 for k in (recipe.k, recipe.kref)):
        raise ValueError("recipe k/kref must be positive integers")
    if recipe.target_ir is not None:
        from .ir import validate_costs
        validate_costs(config)
        if isinstance(recipe.target_ir,bool) or not isinstance(recipe.target_ir,(int,float)) or not 0 < recipe.target_ir <= 1:
            raise ValueError("target_ir must be a fraction in (0,1]")
        if recipe.periodic_gap is not None:
            raise ValueError("recipe chooses periodic_gap OR target_ir")
    elif recipe.ir_costs is not None:
        raise ValueError("IR calibration costs require target_ir")
    if recipe.periodic_gap is not None and (isinstance(recipe.periodic_gap, bool) or
            not isinstance(recipe.periodic_gap, (int, float)) or not math.isfinite(recipe.periodic_gap) or recipe.periodic_gap < 0):
        raise ValueError("recipe periodic_gap must be finite and nonnegative")
    if recipe.fit_identity is not None and recipe.fit_identity != fit_identity(config):
        pass


def fit_identity(config):
    r = config.recipe
    return digest({"schema": "r17.recipe_fit.v1", "store_digest": r.store_digest,
                   "teacher_sampler_identity": r.teacher_sampler_identity,
                   "layout_identity": config.adapter_binding["layout_identity"],
                   "feature_identity": config.adapter_binding["feature_identity"],
                   "commit": config.adapter_binding["commit_horizon"], "corrector": r.corrector,
                   "k": r.k, "kref": r.kref, "early": r.early, "ordered_tasks": r.ordered_tasks,
                   "task_table_identity": r.task_table_identity})


def recipe_identity(config):
    # Physical path is not content identity; equivalent replicas may relocate
    # the identical sealed library. Guard/gap remain part of serving identity.
    value = asdict(config.recipe)
    value.pop("store_path")
    # Preserve existing bundle identities when the new hooks are absent.
    if not value["serve_loeo"]:
        value.pop("serve_loeo")
    if value.pop("miss_mode") != "legacy":
        value["miss_mode"] = config.recipe.miss_mode
    for name in ("fitted_path", "stall_path"):
        value.pop(name)
    for name in ("fitted_digest", "stall_digest", "library_benchmark"):
        if value[name] is None:
            value.pop(name)
    value["fit_identity"] = fit_identity(config) if config.recipe.enabled else None
    return digest(value)


def bundle_identity(config):
    value = asdict(config)
    value["recipe"] = {"identity": recipe_identity(config)}
    # Server-local evidence/output locations cannot change arm semantics.
    for section, names in (("miss", ("evidence_dir",)), ("timer", ("output_csv_dir",))):
        if value.get(section):
            for name in names:
                value[section].pop(name, None)
    return digest({"schema": "r17.bundle.v1", "config": value})


def retrieval_model_identity(spec):
    value = asdict(spec) if is_dataclass(spec) else dict(spec)
    # Inference sampler and executable capability receipts are not the identity
    # of normalized teacher actions / policy-derived retrieval features.
    value.pop("default_sampler", None)
    value.pop("capabilities", None)
    return digest(value)


_verified_stores, _fitted_recipes, _calibrations, _serve_loeo_maps = {}, {}, {}, {}
_cache_lock = threading.RLock()


def _fingerprint(store):
    # Metadata checks detect replacement/mutation without reading array bytes.
    paths = [store.directory / "READY.json"] + [store.directory / entry["file"]
             for entry in store.manifest["arrays"].values()]
    result = []
    for path in paths:
        stats = (path.lstat(), path.stat())
        result.append((str(path), *(getattr(stat,name) for stat in stats for name in
                       ("st_dev","st_ino","st_size","st_mtime_ns","st_ctime_ns","st_mode"))))
    return tuple(result)


def verified_store(path, content_digest):
    from exp.r17_integration.search.store import Store
    key = str(Path(path).resolve()), content_digest
    with _cache_lock:
        existing = _verified_stores.get(key)
        if existing is not None:
            store,fingerprint = existing
            if _fingerprint(store) != fingerprint:
                raise ValueError("verified teacher store changed; publish a new immutable content generation")
            return store
        directory = Path(key[0])
        # Snapshot metadata before the one checksum pass. A generation changed
        # during verification must not enter the immutable cache.
        from types import SimpleNamespace
        before = _fingerprint(SimpleNamespace(directory=directory,
                               manifest=json.loads((directory / "READY.json").read_text())))
        store = Store(directory)
        if store.manifest["content_digest"] != content_digest:
            pass
        after = _fingerprint(store)
        if before != after:
            raise ValueError("teacher store changed during verification")
        _verified_stores[key] = store,after
        return store


def make_recipe_factory(provider):
    def factory(config, storage):
        from exp.r17_integration.search.fit import fit_recipe
        validate_recipe(config)
        r = config.recipe
        if not r.enabled:
            return None
        if r.fitted_path:
            # Sealed fit contains every serving array. Verify those bytes,
            # rather than hashing/reading tens of GB of teacher RGB on H100.
            from exp.r17_integration.search.cache_recipe import sealed_store_descriptor
            store = sealed_store_descriptor(r.store_path, r.store_digest)
        else:
            store = verified_store(r.store_path,r.store_digest)
        binding = store.manifest["binding"]
        if store.manifest["content_digest"] != r.store_digest:
            pass
        from exp.r17_integration.search.cache_recipe import validate_teacher
        compatibility = validate_teacher(provider, config, binding)
        teacher = binding.get("sampler", binding["model"]["default_sampler"])
        if digest(teacher) != r.teacher_sampler_identity:
            pass
        identity = fit_identity(config)
        key = retrieval_model_identity(provider.model_spec), identity, r.fitted_digest, r.fitted_path
        with _cache_lock:
            if r.fitted_path:
                from exp.r17_integration.search.cache_recipe import verified_fit
                # Stat fencing is mandatory even on a cached bundle rebind.
                _fitted_recipes[key] = verified_fit(config, binding)
            if key not in _fitted_recipes:
                fitted, _ = fit_recipe(store, task_names=r.ordered_tasks, corrector=r.corrector,
                                       commit_horizon=config.adapter_binding["commit_horizon"],
                                       allow_partial_commit=config.adapter_binding.get("allow_partial_commit", False),
                                       k=r.k, kref=r.kref, early=r.early, guard=False, periodic_gap=None)
                _fitted_recipes[key] = fitted
            fitted = _fitted_recipes[key]
            calibration = None
            if r.target_ir is not None:
                from .ir import calibrate_periodic,library_schedule
                calibration_key = key, recipe_identity(config)
                if calibration_key not in _calibrations:
                    calibration = calibrate_periodic(r.target_ir,library_schedule(fitted,store,config),r.ir_costs)
                    calibration.update(store_digest=r.store_digest,fit_identity=identity,
                                       task_table_identity=r.task_table_identity)
                    calibration["identity"] = digest(calibration)
                    _calibrations[calibration_key] = calibration
                calibration = _calibrations[calibration_key]
            result = fitted.fork()
            if r.serve_loeo:
                map_key = str(Path(r.store_path).resolve()), r.store_digest, tuple(r.ordered_tasks)
                if map_key not in _serve_loeo_maps:
                    parent = getattr(store, "parent", store)
                    task_ids = {name: i for i, name in enumerate(r.ordered_tasks)}
                    _serve_loeo_maps[map_key] = {
                        (task_ids[row["task_name"]], row["orig_init_idx"]): row["library_episode_id"]
                        for row in parent.manifest["rows"]}
                result.serve_loeo_episodes = _serve_loeo_maps[map_key]
        result.use_guard = r.guard
        result.use_corrector = r.corrector
        result.periodic_gap = calibration["setting"] if calibration is not None else r.periodic_gap
        result.ir_calibration = copy.deepcopy(calibration)
        result.fit_evidence = dict(result.fit_evidence, ir_calibration=copy.deepcopy(calibration),
                                   task_table_identity=r.task_table_identity)
        result.bundle_recipe_identity = recipe_identity(config)
        result.teacher_compatibility = compatibility
        if r.miss_mode != "legacy":
            from exp.r17_integration.search.cache_recipe import CacheStudyRecipe
            result = CacheStudyRecipe(result, config)
        return result
    return factory
