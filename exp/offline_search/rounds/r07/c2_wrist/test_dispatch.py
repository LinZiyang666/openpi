"""CPU key/input parity and actual camera invocation tests of installed hooks."""
import dataclasses
import importlib.util
import json
import os
import sys
import types
import threading
import unittest
from pathlib import Path
from unittest import mock

import torch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, Stage1Output, Stage2Output, make_att_2d_masks
from openpi.serving import stage_io
from openpi.serving.batching_coordinator import Pi05StageBatcher
from openpi.models.model import Observation
from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
from openpi.cache.types import CheckpointID


def module(name, fallback):
    path = os.environ.get(name)
    if not path:
        return __import__(fallback, fromlist=["*"])
    spec = importlib.util.spec_from_file_location("c2_test_" + name.lower(), path)
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


co = module("C2_STAGE_PATH", "exp.offline_search.closed_loop.stage_overrides")
plug = module("C2_PLUGIN_PATH", "exp.offline_search.closed_loop.plugin")


class Tower:
    def __init__(self):
        self.paligemma = types.SimpleNamespace(vision_tower=torch.nn.Linear(1, 1),
                                               multi_modal_projector=torch.nn.Linear(1, 1))
        self.calls = []

    def embed_image(self, image):
        self.calls.append((len(image), tuple(float(x) for x in image.mean((1, 2, 3)))))
        return image.mean((1, 2, 3))[:, None, None].expand(-1, 256, 2048).clone()

    def embed_language_tokens(self, tokens):
        return tokens[:, :, None].expand(-1, -1, 2048).float()


class Model:
    pi05 = True

    def __init__(self):
        self.paligemma_with_expert = Tower()

    def _preprocess_observation(self, observation, train=False):
        return (list(observation.images.values()), list(observation.image_masks.values()),
                observation.tokenized_prompt, observation.tokenized_prompt_mask, observation.state)

    def _prepare_attention_masks_4d(self, mask):
        return torch.where(mask[:, None], 0., torch.finfo(torch.float32).min)

    def run_stage1(self, obs):
        return PI0Pytorch.run_stage1(self, obs)


def raw(value):
    names = ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"]
    return dict(image={k: torch.full((3, 4, 4), v) for k, v in zip(names, (value, value * 2, -1.))},
                image_mask={k: torch.tensor(i < 2) for i, k in enumerate(names)},
                tokenized_prompt=torch.tensor([1, 2]), tokenized_prompt_mask=torch.tensor([True, False]),
                state=torch.arange(32).float())


def stock(model, obs):
    images, masks, tokens, lang_mask, state = model._preprocess_observation(obs)
    tower = model.paligemma_with_expert
    embeddings = [tower.embed_image(x) for x in images]
    language = tower.embed_language_tokens(tokens) * (2048 ** .5)
    prefix = torch.cat([*embeddings, language], dim=1)
    pad = torch.cat([m[:, None].expand(e.shape[:2]) for m, e in zip(masks, embeddings)] + [lang_mask], dim=1)
    return Stage1Output(state, prefix, pad, model._prepare_attention_masks_4d(make_att_2d_masks(pad, torch.zeros_like(pad))),
                        torch.cumsum(pad, dim=1)-1)


def keys(out):
    kb = CP1SpatialPool16KeyBuilder(enabled_fields=["vision_0", "vision_1", "robot_state"])
    kb.collect(CheckpointID.CP1, stage1=out)
    return kb.build(CheckpointID.CP1)


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.saved = (PI0Pytorch.run_stage1, PI0Pytorch.run_stage2, PI0Pytorch.run_stage2_capture,
                      stage_io.split_stage1_output, stage_io.stack_stage1_output, Pi05StageBatcher.run_stage1_batch)
        self.saved_override = getattr(PI0Pytorch, "_r4_override", None)
        if hasattr(PI0Pytorch, "_r4_override"):
            delattr(PI0Pytorch, "_r4_override")
        PI0Pytorch.run_stage1 = stock
        # The policy stage remains unchanged; the real hook must complete its
        # inputs before handing control to this stand-in.
        PI0Pytorch.run_stage2 = lambda model, out: Stage2Output(out, None)
        PI0Pytorch.run_stage2_capture = PI0Pytorch.run_stage2
        self.model = Model()
        self.batcher = object.__new__(Pi05StageBatcher)
        self.batcher._model, self.batcher._device = self.model, torch.device("cpu")

    def tearDown(self):
        (PI0Pytorch.run_stage1, PI0Pytorch.run_stage2, PI0Pytorch.run_stage2_capture,
         stage_io.split_stage1_output, stage_io.stack_stage1_output, Pi05StageBatcher.run_stage1_batch) = self.saved
        if self.saved_override is None:
            delattr(PI0Pytorch, "_r4_override") if hasattr(PI0Pytorch, "_r4_override") else None
        else:
            PI0Pytorch._r4_override = self.saved_override

    def assert_fields(self, left, right):
        for field in dataclasses.fields(Stage1Output):
            self.assertTrue(torch.equal(getattr(left, field.name), getattr(right, field.name)), field.name)

    def test_absent_flag_no_patch(self):
        before = PI0Pytorch.run_stage1
        self.assertIsNone(co.install_pi05())
        self.assertIs(PI0Pytorch.run_stage1, before)

    def test_wrist_key_completion_and_no_base_encode(self):
        reference = self.batcher.run_stage1_batch([raw(.25)])[0]
        override = co.install_pi05("per_request")
        self.batcher.run_stage1_batch([co.CameraRequest(raw(.25), "full")])
        self.model.paligemma_with_expert.calls.clear()
        wrist = self.batcher.run_stage1_batch([co.CameraRequest(raw(.25), "wrist_only")])[0]
        self.assertEqual(self.model.paligemma_with_expert.calls, [(1, (.5,))])
        self.assertTrue(torch.equal(keys(reference)["vision_1"], keys(wrist)["vision_1"]))
        self.assertEqual(int(keys(wrist)["vision_0"].count_nonzero()), 0)
        completed = PI0Pytorch.run_stage2(self.model, wrist).stage1
        self.assert_fields(reference, completed)
        self.assertEqual(self.model.paligemma_with_expert.calls[-1], (1, (.25,)))
        self.assertTrue(torch.equal(override.complete(self.model, completed).prefix_embs, completed.prefix_embs))

    def test_mixed_requests_reorder_split_to_complete(self):
        original = [self.batcher.run_stage1_batch([raw(v)])[0] for v in (.125, .875, .5)]
        override = co.install_pi05("per_request")
        self.batcher.run_stage1_batch([co.CameraRequest(raw(.25), "full")])
        outputs = self.batcher.run_stage1_batch([co.CameraRequest(raw(.125), "wrist_only"),
            co.CameraRequest(raw(.875), "full"), co.CameraRequest(raw(.5), "wrist_only")])
        batched = stage_io.stack_stage1_output([outputs[2], outputs[1], outputs[0]])
        shards = stage_io.split_stage1_output(batched.to("cpu"), 3)
        for shard, ref in zip(shards, [original[2], original[1], original[0]]):
            self.assert_fields(override.complete(self.model, shard), ref)
        complete = override.complete(self.model, batched)
        self.assertTrue(torch.equal(complete.prefix_embs, torch.cat([original[2].prefix_embs, original[1].prefix_embs,
                                                                   original[0].prefix_embs])))

    def test_live_cold_full_seeds_dummy_for_wrist(self):
        co.install_pi05("per_request")
        self.batcher.run_stage1_batch([co.CameraRequest(raw(.25), "full")])
        self.model.paligemma_with_expert.calls.clear()
        self.batcher.run_stage1_batch([co.CameraRequest(raw(.125), "wrist_only")])
        self.assertEqual(self.model.paligemma_with_expert.calls, [(1, (.25,))])

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            co.CameraRequest(raw(.25), "third_person")

    def test_request_owned_actual_completion_counts(self):
        override = co.install_pi05("per_request")
        self.batcher.run_stage1_batch([co.CameraRequest(raw(.25), "full")])
        audits = [{}, {}, {}]
        outputs = self.batcher.run_stage1_batch([co.CameraRequest(raw(.25), "wrist_only", audits[0]),
            co.CameraRequest(raw(.125), "full", audits[1]), co.CameraRequest(raw(.5), "wrist_only", audits[2])])
        self.assertEqual(audits, [{"looks": 1}] * 3)
        mixed = stage_io.stack_stage1_output([outputs[2], outputs[1], outputs[0]]).to("cpu")
        override.complete(self.model, mixed)
        self.assertEqual(audits, [{"looks": 1, "completions": 1}, {"looks": 1}, {"looks": 1, "completions": 1}])

    def test_cli_guards(self):
        args = ["--os-method", "unused:Class", "--os-cell", "pi05_l10_cache", "--os-log-dir", "/tmp/r7_C2"]
        opts, _ = plug.parse_cli(args)
        self.assertFalse(opts.os_request_cameras)
        with self.assertRaises(SystemExit):
            plug.parse_cli([*args, "--os-request-cameras"])
        opts, rest = plug.parse_cli([*args, "--os-request-cameras", "--os-blind", "--os-no-shadow-native", "--os-tokens", "off"])
        self.assertTrue(opts.os_request_cameras)
        self.assertEqual(rest, [])

    def test_plugin_declares_mode_before_forward_per_connection(self):
        rt = types.SimpleNamespace(blind=False, request_cameras=True, decision_count=0,
                                   decision_lock=threading.RLock(), gpu=None, policy_tail=False)
        forwards = []
        def connection(mode, step, reason):
            method = types.SimpleNamespace(next_camera_mode=mode)
            method.set_camera_mode = lambda chosen: setattr(method, "chosen", chosen)
            session = types.SimpleNamespace(rt=rt, method=method, step=step, _look_reason=reason, _dec=None,
                set_obs=lambda obs: None, after_infer=lambda *args: None)
            owner = types.SimpleNamespace(_stage1_fn=lambda req: forwards.append(req))
            adapter = object.__new__(plug._BlindAdapter)
            adapter.s, adapter.fake = session, False
            adapter._instrument(owner, "_stage1_fn")
            session.stage1_calls = 0
            inner = types.SimpleNamespace(infer=lambda obs: (owner._stage1_fn(obs), {"actions": []})[1])
            policy = object.__new__(plug._ConnPolicy)
            policy._osp_sessions, policy._osp_inner, policy._osp_adapter = [session], inner, adapter
            return policy
        wrist = connection("wrist_only", 2, 1)
        hard = connection("full", 2, 1)
        first = connection("wrist_only", 0, 1)
        forced = connection("wrist_only", 2, 7)
        for policy in (wrist, hard, first, forced, wrist):
            policy._osp_infer({})
        self.assertEqual([req.mode for req in forwards], ["wrist_only", "full", "full", "full", "wrist_only"])
        self.assertEqual(len({id(req.audit) for req in forwards}), 5)


if __name__ == "__main__":
    unittest.main()
