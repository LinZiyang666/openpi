"""Native stage batching with benchmark-owned action/state geometry."""
from __future__ import annotations

import torch
from dataclasses import fields, replace
from openpi.cache.groot.batcher import GrootStageBatcher, GrootStage3Input
from openpi.serving.batching_core import Stage3MissPayload


class GrootMetaWorldStageBatcher(GrootStageBatcher):
    def __init__(self, runner, manifest):
        super().__init__(runner)
        self.manifest = manifest

    def bucket_key(self, payload):
        x = payload.noise if isinstance(payload, Stage3MissPayload) else payload.start_x
        return (self.manifest.model_spec().identity, tuple(x.shape), str(x.device), super().bucket_key(payload))

    def _validate(self, payloads):
        if not payloads:
            raise ValueError("empty GR00T batch")
        keys = set()
        for p in payloads:
            cond = p.stage2_out
            if not isinstance(cond, GrootStage3Input) or cond.embodiment != (31,):
                raise ValueError("S3 payload must use benchmark slot31")
            if cond.schedule_id != self._runner.live_schedule().schedule_id:
                raise ValueError("S3 payload schedule differs from live K")
            state, mask = cond.stage2.action_inputs["state"], cond.stage2.action_inputs["state_mask"]
            expected = torch.zeros_like(mask)
            expected[..., :self.manifest.state_dim] = True
            if state.shape != (1, 1, 64) or mask.dtype != torch.bool or not torch.equal(mask, expected):
                raise ValueError("S3 batch requires benchmark state mask in width64")
            x = p.noise if isinstance(p, Stage3MissPayload) else p.start_x
            if tuple(x.shape) not in ((16, 32), (1, 16, 32)) or not torch.isfinite(x).all():
                raise ValueError("S3 noise/snapshot must be finite H16×32")
            keys.add(self.bucket_key(p))
        if len(keys) != 1:
            raise ValueError("incompatible GR00T S3 bucket")

    def run_stage3_miss(self, payloads, **kwargs):
        self._validate(payloads)
        return self._owned_replies(super().run_stage3_miss(payloads, **kwargs))

    def run_stage3_warm(self, payloads, **kwargs):
        self._validate(payloads)
        return self._owned_replies(super().run_stage3_warm(payloads, **kwargs))

    @staticmethod
    def _owned_replies(replies):
        def clone(value):
            if isinstance(value, torch.Tensor):
                return value.detach().clone()
            if isinstance(value, dict):
                return {key: clone(item) for key, item in value.items()}
            return value
        return [replace(reply, **{f.name: clone(getattr(reply, f.name)) for f in fields(reply)}) for reply in replies]
