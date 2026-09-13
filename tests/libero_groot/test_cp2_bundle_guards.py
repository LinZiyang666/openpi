"""The CP2 (ActionCache-style) arm through the GR00T LIBERO server's bundle path.

Real guards, real ``load_cache_config`` / ``build_shared_storage`` over small
CP2 artifacts: a hot-swapped CP2 bundle is admitted for both tiers, refused
when its builder, its live loop or its artifact do not match, and a CP1 and a
CP2 bundle registered under the same id in turn each resolve to their own
config -- the per-connection interceptor freezes whichever it was built from.
"""

from __future__ import annotations

import dataclasses
import pickle
import types

import pytest
import torch
import yaml

from exp.actioncache_baseline import libs
from exp.actioncache_baseline.export_arms import cp2_arm_yaml
from exp.libero_groot import serve_groot_libero as sgl
from openpi.cache.cache_storage import CacheStorage
from openpi.cache.config import ConfigValidationError, build_shared_storage, load_cache_config
from openpi.cache.groot.cp2_key_builder import GrootCP2TernaryKeyBuilder
from openpi.cache.storage_types import CacheEntry, CachePayload
from openpi.cache.types import CheckpointID
from tests.cache.groot.conftest import StubGrootModel

GROOT = libs.GROOT_LIBERO
LAYOUT = {"kind": "groot_encoded_v1", "token_len": 5, "feature_dim": 4, "state_feat_dim": 2}
PROJ = libs.ProjectionArgs(seed=7, d=8, p=0.25, input_dim=22, layout=LAYOUT)
K8 = "groot_n15_k8_v1"


@dataclasses.dataclass
class _Bundle:
    cache_config: object
    shared_storage: object
    version: int = 1
    yaml_id: str | None = None


def _artifact(path, *, builder_type="cp2_groot_ternary", checkpoint="CP2", projection=None):
    entries = [CacheEntry(
        id=f"e{i}", checkpoint_id=CheckpointID.CP2 if checkpoint == "CP2" else CheckpointID.CP1,
        query_keys={libs.FIELD: torch.randn(8)},
        payload=CachePayload(action_chunk=torch.zeros(16, 32), intermediates={0.875: torch.zeros(16, 32)},
                             denoising_num_steps=8, task_key="t", schedule_id=K8),
        step_idx=i, trajectory_id="traj", prev_ids=[], next_ids=[],
    ) for i in range(3)]
    meta = GrootCP2TernaryKeyBuilder(seed=7, d=8, p=0.25, token_len=5, feature_dim=4, state_feat_dim=2).projection_meta()
    art = {"key_builder_type": builder_type, "checkpoint_id": checkpoint, "vector_dims": {libs.FIELD: 8},
           "entries": entries, "projection": meta if projection is None else projection,
           "id_policy": libs.ID_POLICY, "stage1_path": "groot_reconstructed_template", "schedule_id": K8,
           "teacher": GROOT.name}
    with open(path, "wb") as f:
        pickle.dump(art, f)
    return path


def _cp2_bundle(tmp_path, tier="n0", name="arm"):
    pkl = _artifact(tmp_path / f"{name}.pkl")
    doc = cp2_arm_yaml(preload_path=str(pkl), projection=PROJ, tier=tier, theta_raw=0.65, profile=GROOT)
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    cfg = load_cache_config(p)
    return _Bundle(cache_config=cfg, shared_storage=build_shared_storage(cfg))


def _register(monkeypatch, bundle):
    monkeypatch.setattr("openpi.serving.websocket_policy_server.get_current_cache_bundle",
                        lambda bundle_id=None: bundle)


def _resolve(bundle_id="acb_sp_n0_t01", *, steps=8):
    return sgl._resolve_bundle(bundle_id, cli_config=None, cli_storage=None, allow_dynamic=True,
                               num_inference_timesteps=steps)


@pytest.mark.parametrize("tier", ["n0", "n1"])
def test_cp2_bundles_of_both_tiers_are_admitted(tmp_path, monkeypatch, tier):
    bundle = _cp2_bundle(tmp_path, tier)
    _register(monkeypatch, bundle)
    cfg, storage = _resolve()
    assert cfg is bundle.cache_config and storage is bundle.shared_storage
    assert storage.artifact_meta["checkpoint_id"] == "CP2"


@pytest.mark.parametrize("tier", ["n0", "n1"])
def test_cp2_bundle_under_another_live_loop_is_refused(tmp_path, monkeypatch, tier):
    """Both tiers: an n0 arm's MISS is still the k=8 teacher its library was built from (G2-B3)."""
    _register(monkeypatch, _cp2_bundle(tmp_path, tier))
    with pytest.raises(ConfigValidationError, match="runs 4 steps"):
        _resolve(steps=4)
    _resolve(steps=8)


def test_cp2_n0_bundle_over_a_k4_library_is_refused(tmp_path, monkeypatch):
    bundle = _cp2_bundle(tmp_path, "n0")
    facade = CacheStorage.__new__(CacheStorage)
    facade._backend = types.SimpleNamespace(  # noqa: SLF001 - facade under test
        artifact_meta={"key_builder_type": "cp2_groot_ternary", "checkpoint_id": "CP2",
                       "schedule_id": "groot_n15_k4_v1", "denoising_num_steps": 4})
    _register(monkeypatch, _Bundle(cache_config=bundle.cache_config, shared_storage=facade))
    with pytest.raises(ConfigValidationError, match="does not match the recipe's"):
        _resolve()


def test_cp2_bundle_with_the_pi05_builder_is_refused(tmp_path, monkeypatch):
    bundle = _cp2_bundle(tmp_path)
    bundle.cache_config.key_builder.type = "cp2_vlm_ternary"
    _register(monkeypatch, bundle)
    with pytest.raises(ConfigValidationError, match="must use key_builder.type='cp2_groot_ternary'"):
        _resolve()


def test_cp2_bundle_over_a_cp1_artifact_is_refused(tmp_path, monkeypatch):
    bundle = _cp2_bundle(tmp_path)
    facade = CacheStorage.__new__(CacheStorage)
    facade._backend = types.SimpleNamespace(  # noqa: SLF001 - facade under test
        artifact_meta={"key_builder_type": "cp1_groot_libero_spatial_pool_16", "checkpoint_id": "CP1"})
    _register(monkeypatch, _Bundle(cache_config=bundle.cache_config, shared_storage=facade))
    with pytest.raises(ConfigValidationError, match="built by 'cp1_groot_libero_spatial_pool_16'"):
        _resolve()


def test_cp2_bundle_over_a_pi05_cp2_artifact_is_refused_at_assembly(tmp_path):
    pkl = _artifact(tmp_path / "pi05.pkl", builder_type="cp2_vlm_ternary")
    doc = cp2_arm_yaml(preload_path=str(pkl), projection=PROJ, tier="n0", theta_raw=0.65, profile=GROOT)
    p = tmp_path / "x.yaml"
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    with pytest.raises(ConfigValidationError, match="does not match configured"):
        build_shared_storage(load_cache_config(p))


def test_cp1_and_cp2_bundles_under_one_id_resolve_to_their_own_config(tmp_path, monkeypatch):
    """A swap of the registered bundle changes what the *next* connection binds;
    each resolution is a snapshot of the bundle registered at that moment."""
    cp2 = _cp2_bundle(tmp_path)
    cp1_cfg = load_cache_config(tmp_path / "arm.yaml")
    cp1_cfg.checkpoints = {"cp1": cp1_cfg.checkpoints["cp2"]}
    cp1_cfg.checkpoints["cp1"].judge.warm_tiers = None
    cp1_cfg.key_builder.type = "cp1_groot_libero_spatial_pool_16"
    cp1_facade = CacheStorage.__new__(CacheStorage)
    cp1_facade._backend = types.SimpleNamespace(  # noqa: SLF001
        artifact_meta={"key_builder_type": "cp1_groot_libero_spatial_pool_16", "checkpoint_id": "CP1",
                       "schedule_id": K8, "denoising_num_steps": 8})
    cp1 = _Bundle(cache_config=cp1_cfg, shared_storage=cp1_facade)

    _register(monkeypatch, cp1)
    cfg_a, st_a = _resolve("shared_id")
    _register(monkeypatch, cp2)
    cfg_b, st_b = _resolve("shared_id")
    assert cfg_a is cp1_cfg and st_a is cp1_facade
    assert cfg_b is cp2.cache_config and st_b is cp2.shared_storage
    assert sorted(cfg_a.checkpoints) == ["cp1"] and sorted(cfg_b.checkpoints) == ["cp2"]


def test_startup_factory_builds_a_cp2_only_interceptor_for_a_cp2_bundle(tmp_path, monkeypatch):
    from openpi.cache.groot.interceptor import GrootCacheInterceptor

    bundle = _cp2_bundle(tmp_path, "n1")
    _register(monkeypatch, bundle)
    # The stub's eagle forward is not upstream's; the drift guard is not under test here.
    for guard in ("_verify_upstream_forward", "_verify_upstream_action_head"):
        monkeypatch.setattr(f"openpi.cache.groot.staged.GrootStagedRunner.{guard}", lambda self: None)
    model = StubGrootModel()
    model.action_head.num_inference_timesteps = 8
    policy = types.SimpleNamespace(model=model)
    args = types.SimpleNamespace(cache_config=None, allow_dynamic_bundles=True, rit_shadow_out=None)
    factory, label = sgl._build_concurrent_factory(policy, args)
    assert "dynamic bundles" in label
    served = factory(policy, bundle_id="acb_sp_n1_t01")
    interceptor = served._inner._policy  # noqa: SLF001 - _InferLockedPolicy -> adapter -> interceptor
    assert isinstance(interceptor, GrootCacheInterceptor)
    assert interceptor._cp2_only is True  # noqa: SLF001
    assert interceptor._cp2_library_sha256 == bundle.shared_storage.artifact_meta["library_sha256"]  # noqa: SLF001
    assert isinstance(interceptor._orchestrator.key_builder, GrootCP2TernaryKeyBuilder)  # noqa: SLF001
    assert interceptor._orchestrator.has_checkpoint(CheckpointID.CP2)  # noqa: SLF001
    assert not interceptor._orchestrator.has_checkpoint(CheckpointID.CP1)  # noqa: SLF001
