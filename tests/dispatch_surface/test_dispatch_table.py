

def test_load_components_widens_search_to_the_requested_top_k(monkeypatch):
    """Calibration must fetch k candidates even though the yaml says top_k=1.

    The deployed stack widens the search via the surface judge's
    ``min_required_top_k`` hint. Calibration has no artifact yet -- its judge
    is ``always_hit`` and exposes no hint -- so without an explicit widening
    the strategy fetches the yaml's top_k (1 in every gate-line template) and
    every disagreement value comes out undefined. That is exactly what a full
    calibration table of ``v=None`` looked like.
    """
    import openpi.cache.config as cfgmod

    from exp.dispatch_surface.build_dispatch_table import _load_components

    yaml_path = "exp/dispatch_surface/config/calibration_retrieval.yaml"
    real = cfgmod.load_cache_config(yaml_path)
    assert real.checkpoints["cp1"].search_strategy.top_k == 1, (
        "fixture assumes the template's narrow width")
    preload = real.backend.in_memory.preload_path

    seen = {}
    monkeypatch.setattr(cfgmod, "build_shared_storage", lambda _c: object())
    monkeypatch.setattr(
        cfgmod, "build_per_connection_components",
        lambda config, _s: seen.update(
            top_k=config.checkpoints["cp1"].search_strategy.top_k) or {},
    )
    _load_components(yaml_path, preload, 5)
    assert seen["top_k"] == 5


def test_load_components_never_narrows_a_wider_yaml(monkeypatch):
    """Widening is a floor, not an override."""
    import openpi.cache.config as cfgmod

    from exp.dispatch_surface.build_dispatch_table import _load_components

    yaml_path = "exp/dispatch_surface/config/calibration_retrieval.yaml"
    preload = cfgmod.load_cache_config(yaml_path).backend.in_memory.preload_path
    seen = {}
    monkeypatch.setattr(cfgmod, "build_shared_storage", lambda _c: object())
    monkeypatch.setattr(
        cfgmod, "build_per_connection_components",
        lambda config, _s: seen.update(
            top_k=config.checkpoints["cp1"].search_strategy.top_k) or {},
    )
    _load_components(yaml_path, preload, 0)
    assert seen["top_k"] == 1
