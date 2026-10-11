"""Model-specific sampler admission; native math stays available for studies."""
def teacher_binding(config):
    from exp.r17_integration.search.cache_recipe import sealed_store_descriptor
    return sealed_store_descriptor(config.recipe.store_path, config.recipe.store_digest).manifest["binding"]


def admission_validators(spec, sampler_for_steps):
    """Callbacks for P1's default-off provider binding/config validation hooks."""
    lossy_identity = sampler_for_steps(1).identity

    def binding_validator(binding, sampler):
        allowed = binding.get("allow_lossy_sampler", False)
        label = binding.get("sampler_study_label")
        if type(allowed) is not bool:
            raise ValueError("DP allow_lossy_sampler must be boolean")
        if allowed and (not isinstance(label, str) or not label.strip()):
            raise ValueError("DP lossy sampler requires a nonempty sampler_study_label")
        if not allowed and label is not None:
            raise ValueError("DP sampler_study_label requires allow_lossy_sampler=True")
        if sampler.num_steps == 1 and not allowed:
            raise ValueError("DP K1 is lossy; only an explicitly labelled plain step-reduction study is allowed")
        if binding.get("library_sampler_identity") == lossy_identity:
            raise ValueError("DP K1 cannot be a library/teacher sampler")
        if allowed:
            if binding.get("library_sampler_identity") is not None:
                raise ValueError("DP plain step-reduction study cannot bind a library/teacher")
            if binding["commit_horizon"] != spec.layout.replan_steps or binding.get("allow_partial_commit", False):
                raise ValueError("DP plain step-reduction study requires the native replan commit")

    def config_validator(config):
        b = config.adapter_binding
        binding_validator(b, sampler_for_steps(b["num_steps"]))
        if config.recipe.teacher_sampler_identity == lossy_identity:
            raise ValueError("DP K1 cannot be a recipe teacher sampler")
        if b.get("allow_lossy_sampler", False):
            if (any(cp.enabled for cp in config.checkpoints.values()) or config.recipe.enabled or
                    config.warm_reset is not None or config.trace.enabled or config.routing is not None or
                    config.shadow_teacher.enabled):
                raise ValueError("DP K1 study is plain step reduction only; cache MISS/recipe/teacher roles are forbidden")
        if config.recipe.enabled and config.recipe.store_path:
            # Check the small descriptor before the shared factory verifies
            # payload digests. Hash identities alone cannot reveal another
            # K1 study's eta/profile. This never hashes store payload files.
            binding = teacher_binding(config)
            reject_lossy_teacher(binding.get("sampler", binding["model"]["default_sampler"]))

    return binding_validator, config_validator


def reject_lossy_teacher(sampler):
    """Check actual sealed metadata too, covering K1 identities from other eta."""
    if sampler and sampler.get("num_steps") == 1:
        raise ValueError("DP K1 cannot be a library/teacher sampler")


def recipe_factory(provider, factory):
    def validated(config, storage):
        provider.config_validator(config)
        if config.recipe.enabled:
            # Read just the sealed descriptor; P1 verifies the store's complete
            # content digest and teacher identity before fitting or dispatch.
            binding = teacher_binding(config)
            reject_lossy_teacher(binding.get("sampler", binding["model"]["default_sampler"]))
        return factory(config, storage)
    return validated


def artifact_validator(config, storage):
    meta = storage.artifact_meta or {}
    for binding in (meta.get("r17_manifest", {}).get("binding", {}), meta.get("model", {})):
        if not isinstance(binding, dict):
            continue
        reject_lossy_teacher(binding.get("sampler"))
        model = binding.get("model", binding.get("model_spec", {}))
        if isinstance(model, dict):
            reject_lossy_teacher(model.get("default_sampler"))
