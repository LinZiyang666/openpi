"""DP hooks for the existing three-queue BatchingCore (no queue duplication)."""
from dataclasses import dataclass, replace
import copy

from exp.r17_integration.common.contracts import CaptureSpec
from openpi.serving.batching_core import Stage3MissPayload, Stage3WarmStartPayload, record_ready_events

from .sampler import SamplerProfile, checked_profile


@dataclass
class BatchContext:
    stage2: object
    sampler: object
    rng: object
    capture: CaptureSpec
    snapshot: object = None


@dataclass
class BatchOutput:
    result: object

    @property
    def action(self):
        return self.result.normalized_chunk.unsqueeze(0)

    @property
    def intermediates(self):
        return {s.loop_index: s.x_t.unsqueeze(0) for s in self.result.snapshots}

    @property
    def first_step_input(self):
        x = self.result.capture.get("first_step_input")
        return None if x is None else x.unsqueeze(0)

    @property
    def first_step_x(self):
        x = self.result.capture.get("first_step_x")
        return None if x is None else x.unsqueeze(0)

    @property
    def actual_nfe(self):
        return self.result.actual_nfe


class DPStageBatcher:
    def __init__(self, adapter):
        self.adapter = adapter

    def bucket_key(self, payload):
        if not isinstance(payload, (Stage3MissPayload, Stage3WarmStartPayload)):
            raise TypeError("DP requires explicit MISS or native snapshot envelope")
        ctx = payload.stage2_out
        if not isinstance(ctx, BatchContext):
            raise TypeError("DP requires request-local sampler/RNG/capture envelope")
        _, spec = checked_profile(ctx.sampler, self.adapter.model.cfg)
        cond = ctx.stage2.data["conditioning"]
        snapshot = ctx.snapshot
        initial = payload.noise if isinstance(payload, Stage3MissPayload) else payload.start_x
        return (self.adapter.metadata().checkpoint_identity, self.adapter.metadata().layout.identity,
                spec.identity, spec.num_steps, tuple(cond.shape), str(cond.dtype), str(cond.device),
                tuple(initial.shape), str(initial.dtype), str(initial.device),
                ctx.capture.enabled, ctx.capture.snapshots, ctx.capture.fields,
                0 if snapshot is None else snapshot.loop_index,
                None if snapshot is None else snapshot.schedule_digest)

    def run_stage1_batch(self, payloads):
        outputs = self.adapter.run_stage1_batch(payloads)
        if self.adapter.batch_ledger is not None:
            members = [p.data.get("batch_member") for p in payloads]
            receipts = self.adapter.batch_ledger.stage("stage1", members)
            for p, out, receipt in zip(payloads, outputs, receipts, strict=True):
                out.data.update(batch_member=copy.deepcopy(p.data["batch_member"]), batching={"stage1": receipt})
        return outputs

    def run_stage2_batch(self, payloads, *, capture):
        self.adapter._check_batch(len(payloads))
        outputs = [self.adapter.run_stage2(p, capture=capture) for p in payloads]
        if self.adapter.batch_ledger is not None:
            receipts = self.adapter.batch_ledger.stage("stage2", [p.data.get("batch_member") for p in payloads])
            for p, out, receipt in zip(payloads, outputs, receipts, strict=True):
                out.data.update(batch_member=copy.deepcopy(p.data["batch_member"]),
                                batching={**copy.deepcopy(p.data["batching"]), "stage2": receipt})
        return outputs

    def miss_payload(self, stage2, sampler, rng, capture):
        import torch
        _, spec = checked_profile(sampler, self.adapter.model.cfg)
        generator = self.adapter._generator(rng)
        cond = stage2.data["conditioning"]
        noise = torch.randn((1, 16, self.adapter.model.cfg["action_dim"]), device=cond.device,
                            dtype=cond.dtype, generator=generator)[0]
        # Persist the generator after its initial draw in a request envelope.
        from .request_noise import RequestNoise
        stream = RequestNoise(generator.initial_seed(), cond.device)
        stream.generator.set_state(generator.get_state())
        ctx = BatchContext(stage2, spec, stream, capture)
        return Stage3MissPayload(ctx, noise, spec.num_steps, ready_events=record_ready_events(cond.device))

    def warm_payload(self, stage2, snapshot, sampler, rng, capture):
        _, spec = checked_profile(sampler, self.adapter.model.cfg)
        ctx = BatchContext(stage2, spec, rng, capture, snapshot)
        # This float is an opaque loop-index encoding only. Native integer
        # time and remaining updates are taken from the validated snapshot.
        return Stage3WarmStartPayload(ctx, snapshot.x_t, snapshot.loop_index/spec.num_steps,
                                      spec.num_steps-snapshot.loop_index,
                                      capture_first_step=capture.enabled,
                                      ready_events=record_ready_events(snapshot.x_t.device))

    def _run(self, payloads, *, captures=None, warm=False):
        self.adapter._check_batch(len(payloads))
        contexts = [p.stage2_out for p in payloads]
        if len({self.bucket_key(p) for p in payloads}) != 1:
            raise ValueError("DP batch requires identical native schedule/shape/capture/resume point")
        sampler = contexts[0].sampler
        capture_specs = captures or [c.capture for c in contexts]
        results = self.adapter.sample_batch([c.stage2 for c in contexts], sampler,
                                            [c.rng for c in contexts], capture_specs,
                                            snapshots=[c.snapshot for c in contexts],
                                            initial_noises=[None if warm else p.noise for p in payloads])
        if self.adapter.batch_ledger is not None:
            receipts = self.adapter.batch_ledger.stage("stage3", [c.stage2.data.get("batch_member") for c in contexts])
            for ctx, result, receipt in zip(contexts, results, receipts, strict=True):
                result.capture["batching"] = {**copy.deepcopy(ctx.stage2.data["batching"]), "stage3": receipt}
        return [BatchOutput(r) for r in results]

    def run_stage3_miss(self, payloads, *, num_steps, save_timesteps_per_request):
        if any(p.num_steps != num_steps or p.stage2_out.sampler.num_steps != num_steps for p in payloads):
            raise ValueError("DP payload K differs from native sampler")
        captures = []
        for p, points in zip(payloads, save_timesteps_per_request, strict=True):
            c = p.stage2_out.capture
            if points is not None:
                # Stock flow save_timesteps is not a DDPM timestep API.
                if set(points) != set(range(1, num_steps)):
                    raise ValueError("DP save points must be all native loop indices; use CaptureSpec")
                c = replace(c, enabled=True, snapshots=True)
            captures.append(c)
        return self._run(payloads, captures=captures)

    def run_stage3_warm(self, payloads, *, start_t, num_steps, capture_first_step):
        for p in payloads:
            c = p.stage2_out
            if c.snapshot is None or start_t != c.snapshot.loop_index/c.sampler.num_steps:
                raise ValueError("DP warm float point is only an index encoding")
            if num_steps != c.sampler.num_steps-c.snapshot.loop_index:
                raise ValueError("DP warm suffix K mismatch")
        return self._run(payloads, warm=True)
