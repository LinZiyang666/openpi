"""Router weight hot-swap across bundle versions (X14 §3.7).

Between training batches the trainer writes a new checkpoint and the conductor
pushes it as a NEW bundle id (``rlr_<run>_v{n}``); workers then re-bind
per task via ``select_bundle``. Two things must hold for the interaction-
efficiency curve to mean anything:

  1. binding a bundle builds a judge from *that* bundle's weights — a stale
     judge would keep sampling from an older policy while its episodes get
     credited to the new one;
  2. every verdict carries the ``weights_version`` it was produced under, so an
     off-version episode is detectable offline rather than silently mixed into
     a batch.

The bundle registry is exercised through the same ``get_current_cache_bundle``
lookup ``_wrap_policy`` uses, so the test rides the production seam rather than
a parallel one.
"""

from __future__ import annotations

import pytest
import torch

from openpi.cache.components.mlp_router_judge import save_router_weights
from openpi.cache.config import (
    BackendConfig,
    CacheConfig,
    CheckpointConfig,
    GateConfig,
    JudgeConfig,
    KeyBuilderConfig,
    KeysConfig,
    RoutingConfig,
    SearchStrategyConfig,
    WritePolicyConfig,
    build_per_connection_components,
    build_shared_storage,
    validate_cache_config,
)
from openpi.cache.types import CheckpointID, ROBOT_STATE
from openpi.serving import websocket_policy_server as wps

D = 32
H = 8


def _weights(tmp_path, version: str) -> str:
    torch.manual_seed(abs(hash(version)) % 1000)
    path = tmp_path / f"{version}.pt"
    save_router_weights(
        path,
        W1=torch.randn(H, D) * 0.4, b1=torch.zeros(H),
        W2=torch.randn(2, H) * 0.4, b2=torch.zeros(2),
        arms="ts", fields=(ROBOT_STATE,), dims={ROBOT_STATE: D},
        weights_version=version, mu=torch.zeros(D), sigma=torch.ones(D),
    )
    return str(path)


def _router_config(weights_path: str) -> CacheConfig:
    """An R_ts yaml in dataclass form: CP1-only, read-only, sidecar student."""
    keys = KeysConfig()
    for name in ("vision_0", "vision_1", "vision_2", "prompt_emb"):
        getattr(keys, name).enabled = False
    keys.robot_state.enabled = True
    cfg = CacheConfig(
        enabled=True,
        keys=keys,
        key_builder=KeyBuilderConfig(type="placeholder"),
        checkpoints={
            "cp1": CheckpointConfig(
                gate=GateConfig(type="always_search"),
                judge=JudgeConfig(
                    type="mlp_router", arms="ts", mode="argmax",
                    weights_path=weights_path, feature_fields=[ROBOT_STATE], hidden=H,
                ),
                # weighted_rrf_knn is the other routing-allowlisted strategy and
                # needs no score_normalization block, keeping this fixture to the
                # parts the test is actually about.
                search_strategy=SearchStrategyConfig(type="weighted_rrf_knn", top_k=1),
            ),
        },
        backend=BackendConfig(type="in_memory", vector_dims={ROBOT_STATE: D}),
        write_policy=WritePolicyConfig(type="never"),
        routing=RoutingConfig(hit_to="127.0.0.1:7002"),
    )
    validate_cache_config(cfg)
    return cfg


@pytest.fixture
def clean_bundles():
    """The bundle registry is process-global; keep tests from leaking into it."""
    saved = dict(wps._bundles)
    wps._bundles.clear()
    yield wps._bundles
    wps._bundles.clear()
    wps._bundles.update(saved)


def _bundle(cfg: CacheConfig, version: int) -> wps.CurrentCacheBundle:
    return wps.CurrentCacheBundle(
        config_path=f"rlr_v{version}.yaml",
        cache_config=cfg,
        shared_storage=build_shared_storage(cfg),
        version=version,
        yaml_id=f"rlr_v{version}",
    )


def _judge_for(bundle: wps.CurrentCacheBundle):
    components = build_per_connection_components(
        bundle.cache_config, bundle.shared_storage, yaml_id=bundle.yaml_id, quiet=True,
    )
    return components["judges"][CheckpointID.CP1]


# ---------------------------------------------------------------------------


def test_two_bundle_versions_build_judges_from_their_own_weights(tmp_path, clean_bundles):
    clean_bundles["rlr_run0_v1"] = _bundle(_router_config(_weights(tmp_path, "v1")), 1)
    clean_bundles["rlr_run0_v2"] = _bundle(_router_config(_weights(tmp_path, "v2")), 2)

    v1 = _judge_for(wps.get_current_cache_bundle("rlr_run0_v1"))
    v2 = _judge_for(wps.get_current_cache_bundle("rlr_run0_v2"))

    assert v1.weights_version == "v1"
    assert v2.weights_version == "v2"
    # Same architecture and encoder, different parameters: the version string is
    # the only thing distinguishing the two policies downstream.
    assert v1.encoder_version == v2.encoder_version


def test_rebinding_a_bundle_yields_a_fresh_judge(tmp_path, clean_bundles):
    """``select_bundle`` rebuilds the whole wrapper stack; the judge must not be
    shared across binds or a per-episode dump buffer could straddle them."""
    clean_bundles["rlr_run0_v1"] = _bundle(_router_config(_weights(tmp_path, "v1")), 1)
    bundle = wps.get_current_cache_bundle("rlr_run0_v1")
    assert _judge_for(bundle) is not _judge_for(bundle)


def test_verdicts_carry_the_bundle_weights_version(tmp_path, clean_bundles):
    from openpi.cache.storage_types import SearchResultLite

    clean_bundles["rlr_run0_v1"] = _bundle(_router_config(_weights(tmp_path, "v1")), 1)
    clean_bundles["rlr_run0_v2"] = _bundle(_router_config(_weights(tmp_path, "v2")), 2)

    seen = []
    for bundle_id in ("rlr_run0_v1", "rlr_run0_v2"):
        judge = _judge_for(wps.get_current_cache_bundle(bundle_id))
        judge.on_episode_start(extra_metadata={
            "run_id": "run0", "batch_id": "b0", "task_uid": "y:eval:0:0", "attempt": 1,
        })
        verdict = judge(
            results=[SearchResultLite(id="e", score=0.5, checkpoint_id=CheckpointID.CP1)],
            checkpoint_id=CheckpointID.CP1, cached_data={},
            query_keys={ROBOT_STATE: torch.zeros(D)},
        )
        seen.append(verdict.router_outputs["weights_version"])
    assert seen == ["v1", "v2"]


def test_unknown_bundle_id_is_not_silently_defaulted(clean_bundles):
    assert wps.get_current_cache_bundle("rlr_run0_v9") is None
