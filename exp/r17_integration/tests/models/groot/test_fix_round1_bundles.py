"""Production GR serve assembly must expose recipe/library choices over stock controls."""
import asyncio
import copy
from dataclasses import asdict, replace
import socket

import numpy as np
import pytest
import yaml

from exp.r17_integration.common.config import build_shared_storage
from exp.r17_integration.common.ir import owner_ir
from exp.r17_integration.common.recipe import fit_identity, recipe_identity
from exp.r17_integration.models.groot.provider import make_provider, register_provider
from exp.r17_integration.models.groot.serve import default_config
from .test_fix_round1 import recipe_config, teacher_store


@pytest.mark.parametrize("miss_k", [4, 2, 1])
def test_review_a4_production_hot_yaml_recipe_library_and_plain_independent(
        guarded_stub, manifest, obs, tmp_path, monkeypatch, miss_k):
    import exp.r17_integration.common.provider as registry
    import exp.r17_integration.common.websocket_policy_server as r17ws
    import exp.r17_integration.models.groot.serve as entry
    from openpi.serving import websocket_policy_server as stock_registry
    from openpi_client import msgpack_numpy
    from exp.r17_integration.search.build_library import save_stock_artifact
    import websockets.asyncio.client
    monkeypatch.setattr(registry, "_providers", {})
    monkeypatch.setattr(registry, "_traces", {})
    monkeypatch.setattr(stock_registry, "_bundles", {})
    monkeypatch.setattr(stock_registry, "_current_bundle", None)
    monkeypatch.setattr(stock_registry, "_bundle_version", 0)
    backend = guarded_stub(miss_k)
    endpoint = replace(manifest, denoising_steps=miss_k)
    provider = register_provider(make_provider(backend, endpoint))
    plain = default_config(provider, commit_horizon=10)
    first_store = teacher_store(tmp_path / "teacher24", manifest)
    second_store = teacher_store(tmp_path / "teacher32", manifest, episodes=4, seed=518)
    assert first_store.manifest["content_digest"] != second_store.manifest["content_digest"]
    raw = recipe_config(copy.deepcopy(plain), first_store, corrector=False, guard=False, target_ir=.45)
    artifact = tmp_path / "teacher4.pkl"
    save_stock_artifact(first_store, artifact)
    raw.backend.in_memory.preload_path = str(artifact)
    layers = recipe_config(copy.deepcopy(plain), second_store, corrector=True, guard=True, target_ir=.40)
    constructed = []
    original_factory = entry.make_connection_factory

    def tracked(*args, **kwargs):
        # Fails with the reviewed serve() which never wired a recipe factory.
        assert callable(kwargs.get("recipe_factory"))
        factory = original_factory(*args, **kwargs)
        def create(base=None, bundle_id="default"):
            core = factory(base, bundle_id)
            constructed.append(core)
            return core
        return create

    monkeypatch.setattr(entry, "make_connection_factory", tracked)
    with socket.socket() as reserved:
        reserved.bind(("127.0.0.1", 0))
        port = reserved.getsockname()[1]
    real_server = r17ws.WebsocketPolicyServer

    class DrivingServer(real_server):
        def __init__(self, *args, **kwargs):
            self.ready = asyncio.Event()
            kwargs["ready_callback"] = self.ready.set
            super().__init__(*args, **kwargs)

        def serve_forever(self, **kwargs):
            async def drive():
                serving = asyncio.create_task(self.run(stop_on_request=True))
                packer = msgpack_numpy.Packer()
                async def call(client, message):
                    await client.send(packer.pack(message))
                    result = await asyncio.wait_for(client.recv(), 10)
                    assert isinstance(result, bytes), result
                    return msgpack_numpy.unpackb(result)
                try:
                    await asyncio.wait_for(self.ready.wait(), 5)
                    async with websockets.asyncio.client.connect(f"ws://127.0.0.1:{port}", compression=None) as client:
                        await client.recv()
                        library_hits = []
                        for bundle_id, config in (("raw-library24", raw), ("five-layers32", layers), ("plain", plain)):
                            loaded = await call(client, {"__ctrl__": "load_cache_config", "bundle_id": bundle_id,
                                "yaml_content": yaml.safe_dump(asdict(config))})
                            assert loaded["__ack__"] == "load_cache_config", loaded
                            selected = await call(client, {"__ctrl__": "select_bundle", "bundle_id": bundle_id})
                            assert selected["__ack__"] == "select_bundle", selected
                            core = constructed[-1]
                            assert core.recipe_identity == recipe_identity(config)
                            await call(client, {"__ctrl__": "episode_start", "__task__": "reach-v3",
                                "__extra__": {"ack_protocol": "r17.execution.v1"}})
                            result = await call(client, obs)
                            assert np.asarray(result["actions"]).shape == (10, 4)
                            if bundle_id == "plain":
                                assert core.recipe is None
                                assert result["__hit_meta__"]["source"] == "policy"
                                assert result["__hit_meta__"]["actual_nfe"] == miss_k
                            else:
                                meta = result["__hit_meta__"]
                                # A calibrated periodic setting can legitimately force the
                                # first look to MISS; the served source must follow it.
                                if meta["os_force_miss"]:
                                    assert meta["source"] == "policy" and meta["actual_nfe"] == miss_k
                                else:
                                    assert meta["source"] == "library" and meta["actual_nfe"] == 0
                                    library_hits.append(bundle_id)
                                assert core.recipe.use_guard is config.recipe.guard
                                assert core.recipe.use_corrector is config.recipe.corrector
                                # Owner-IR target: calibrated on the library replay, never (1-t)/t.
                                calibration = core.recipe.ir_calibration
                                assert calibration["target"] == config.recipe.target_ir
                                assert core.recipe.periodic_gap == calibration["setting"]
                                assert calibration["costs"] == config.recipe.ir_costs
                                assert calibration["pred_IR_rounded"] == pytest.approx(owner_ir(
                                    calibration["N"], calibration["V"], calibration["M"], config.recipe.ir_costs))
                                assert core.recipe.periodic_gap != pytest.approx((1-config.recipe.target_ir)/config.recipe.target_ir)
                                assert result["__hit_meta__"]["ir_calibration"]["identity"] == calibration["identity"]
                                assert len(core.recipe.action) == (24 if bundle_id == "raw-library24" else 32)
                            await call(client, {"__ctrl__": "episode_end", "__success__": False,
                                "__execution__": {"decision_seq": 0, "executed_count": 5}})
                        assert library_hits, "at least one recipe bundle must serve a real library HIT"
                        # A malformed store/teacher/fit must fail before publication.
                        for problem in ("store", "fit", "teacher"):
                            bad = copy.deepcopy(raw)
                            if problem == "store":
                                bad.recipe.store_digest = "wrong"
                                bad.recipe.fit_identity = fit_identity(bad)
                            elif problem == "fit":
                                bad.recipe.fit_identity = "wrong"
                            else:
                                bad.recipe.teacher_sampler_identity = endpoint.model_spec().default_sampler.identity + "bad"
                                bad.recipe.fit_identity = fit_identity(bad)
                            previous = stock_registry._bundle_version
                            error = await call(client, {"__ctrl__": "load_cache_config", "bundle_id": "bad-"+problem,
                                "yaml_content": yaml.safe_dump(asdict(bad))})
                            assert error["__ack__"] == "error", error
                            assert stock_registry._bundle_version == previous
                            assert "bad-"+problem not in stock_registry._bundles
                finally:
                    self.request_stop()
                    await asyncio.wait_for(serving, 5)
            asyncio.run(drive())

    monkeypatch.setattr(r17ws, "WebsocketPolicyServer", DrivingServer)
    entry.serve(backend, endpoint, provider=provider, host="127.0.0.1", port=port,
                default_config=plain, default_storage=build_shared_storage(plain))
    assert backend.model.action_head.num_inference_timesteps == miss_k
