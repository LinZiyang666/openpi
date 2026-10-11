"""Fast-WAM adapter for the existing jax-free BatchingCore."""
from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import asdict
import uuid

from exp.r17_integration.common.contracts import CaptureSpec, NativeSnapshot
from .staged import stage_conditioning_scope


class FastWAMStageBatcher:
    def __init__(self, adapter):
        self.adapter = adapter

    def _receipts(self, stage, members):
        ledger=self.adapter.batch_ledger
        return ledger.stage(stage,members) if ledger is not None else [None]*len(members)

    def _stage2_key(self, stage):
        data = stage.data
        if data.get("model_identity") != self.adapter.metadata().identity:
            raise ValueError("native stage belongs to a different resident model")
        conditioning = data.get("context", data.get("proprio"))
        return (self.adapter.metadata().identity, data["video_seq_len"],
            tuple(conditioning.shape[1:]), str(conditioning.dtype), str(conditioning.device))

    def bucket_key(self, payload):
        from openpi.serving.batching_core import Stage3MissPayload, Stage3WarmStartPayload
        if not isinstance(payload, (Stage3MissPayload, Stage3WarmStartPayload)):
            raise ValueError("Fast-WAM stock warm-reset plans are unsupported; use native reset_snapshot")
        sampler = self.adapter.sampler_for_steps(payload.num_steps)
        x = payload.noise if isinstance(payload, Stage3MissPayload) else payload.start_x
        return (type(payload).__name__, self._stage2_key(payload.stage2_out), sampler.identity,
            tuple(x.shape), str(x.dtype), str(x.device), getattr(payload, "start_t", None),
            tuple((tuple(c["k"].shape[1:]), tuple(c["v"].shape[1:])) for c in payload.stage2_out.data["video_kv_cache"]))

    def run_stage1_batch(self, payloads):
        self.adapter._check_batch(len(payloads))
        groups = defaultdict(list)
        for i, p in enumerate(payloads):
            x = p.data["input_image"]
            groups[(tuple(x.shape[1:]), str(x.dtype), tuple(p.data["proprio"].shape))].append((i, p))
        out = [None] * len(payloads)
        for group in groups.values():
            shards = self.adapter.run_stage1_batch([p for _,p in group])
            members=[p.data.get('batch_member') for _,p in group]
            receipts=self._receipts('stage1',members)
            for (i,_), shard, member, receipt in zip(group, shards, members, receipts, strict=True):
                if receipt is not None:
                    shard.data['batch_member']=copy.deepcopy(member)
                    shard.data['batching']={'stage1':receipt}
                out[i] = shard
        return out

    def run_stage2_batch(self, payloads, *, capture=False, capture_cp2=False):
        self.adapter._check_batch(len(payloads))
        groups = defaultdict(list)
        for i, p in enumerate(payloads):
            groups[self._stage2_key(p)].append((i, p))
        out = [None] * len(payloads)
        for group in groups.values():
            shards = self.adapter.run_stage2_batch([p for _,p in group], capture=capture, capture_cp2=capture_cp2)
            receipts=self._receipts('stage2',[p.data.get('batch_member') for _,p in group])
            for (i,p), shard, receipt in zip(group, shards, receipts, strict=True):
                if receipt is not None:
                    shard.data['batch_member']=copy.deepcopy(p.data['batch_member'])
                    shard.data['batching']={**copy.deepcopy(p.data['batching']),'stage2':receipt}
                out[i] = shard
        return out

    @staticmethod
    def _index(point, k):
        if not isinstance(point, (int, float)):
            raise ValueError("Fast-WAM warm point must encode native index/K")
        index = round(point * k)
        if not 0 < index < k or abs(index / k - point) > 1e-8:
            raise ValueError("Fast-WAM point is not a recoverable native index/K")
        return index

    def _publish(self, results, requested, capture_first=False):
        import torch
        from openpi.serving.batching_core import Stage3VariantOutput
        outs = []
        for result, points in zip(results, requested, strict=True):
            steps = result.capture.get("native_steps", [])
            intermediates = {}
            if points is not None:
                for point in points:
                    # Index encoding only. Arithmetic never uses this as flow time.
                    index = self._index(point, result.sampler.num_steps) if point != 0 else 0
                    row = next((r for r in steps if r["loop_index"] == index), None)
                    if row is None:
                        raise ValueError("requested snapshot precedes executed suffix")
                    intermediates[point] = torch.from_numpy(row["x_t"].copy()).to(self.adapter.model.torch_dtype).unsqueeze(0)
            first_in = first_out = None
            if capture_first and steps:
                first_in = torch.from_numpy(steps[0]["x_t"].copy()).unsqueeze(0)
                first_out = torch.from_numpy(steps[0]["x_next"].copy()).unsqueeze(0)
            out = Stage3VariantOutput(result.normalized_chunk.unsqueeze(0).clone(), intermediates or None, first_in, first_out)
            out.actual_nfe = result.actual_nfe
            out.sampler = result.sampler
            out.native_result = result
            outs.append(out)
        return outs

    def run_stage3_miss(self, payloads, *, num_steps, save_timesteps_per_request):
        import torch
        self.adapter._check_batch(len(payloads))
        if any(p.num_steps != num_steps or self.bucket_key(p) != self.bucket_key(payloads[0]) for p in payloads):
            raise ValueError("inhomogeneous Fast-WAM stage3 bucket")
        if len(save_timesteps_per_request) != len(payloads):
            raise ValueError("snapshot set must belong to each request")
        sampler = self.adapter.sampler_for_steps(num_steps)
        x = torch.stack([p.noise.clone() for p in payloads]).to(self.adapter.model.device, self.adapter.model.torch_dtype)
        if tuple(x.shape[1:]) != (32, self.adapter.metadata().layout.action_width) or not torch.isfinite(x).all():
            raise ValueError("invalid Fast-WAM noise shape/finiteness")
        for points in save_timesteps_per_request:
            for point in points or ():
                if point != 0:
                    self._index(point, num_steps)
        results = self.adapter._sample_batch([p.stage2_out for p in payloads], sampler, x, 0,
            [CaptureSpec(enabled=points is not None, snapshots=points is not None) for points in save_timesteps_per_request],
            [{"algorithm": "explicit_per_request_noise"} for _ in payloads])
        receipts=self._receipts('stage3',[p.stage2_out.data.get('batch_member') for p in payloads])
        for p,result,receipt in zip(payloads,results,receipts,strict=True):
            if receipt is not None:
                result.capture['batching']={**copy.deepcopy(p.stage2_out.data['batching']),'stage3':receipt}
        return self._publish(results, save_timesteps_per_request)

    def run_stage3_warm(self, payloads, *, start_t, num_steps, capture_first_step=False):
        import torch
        self.adapter._check_batch(len(payloads))
        if any(p.num_steps != num_steps or p.start_t != start_t or self.bucket_key(p) != self.bucket_key(payloads[0]) for p in payloads):
            raise ValueError("inhomogeneous Fast-WAM native warm bucket")
        index = self._index(start_t, num_steps)
        sampler = self.adapter.sampler_for_steps(num_steps)
        spec = self.adapter.metadata()
        for p in payloads:
            snap = NativeSnapshot(spec.checkpoint_identity, spec.norm_identity, spec.layout.identity,
                sampler.identity, sampler.schedule_digest, index, sampler.native_timesteps[index],
                p.start_x, conditioning_scope=stage_conditioning_scope(p.stage2_out.data))
            snap.validate(spec, sampler)
            if p.start_x.dtype != self.adapter.model.torch_dtype or not torch.isfinite(p.start_x).all():
                raise ValueError("native warm tensor dtype/finiteness mismatch")
        x = torch.stack([p.start_x.clone() for p in payloads]).to(self.adapter.model.device)
        results = self.adapter._sample_batch([p.stage2_out for p in payloads], sampler, x, index,
            [CaptureSpec(enabled=capture_first_step) for _ in payloads],
            [{"algorithm": "deterministic_flow_suffix"} for _ in payloads])
        return self._publish(results, [None]*len(payloads), capture_first_step)


def make_stage_batcher(adapter):
    return FastWAMStageBatcher(adapter)


class FastWAMBatchedAdapter:
    """Thin model hook: submit native stages to the shared core, no controller."""
    def __init__(self, adapter, coordinator):
        self.adapter, self.coordinator = adapter, coordinator
        self.connection_id=uuid.uuid4().hex
        self.last_batch_receipt={}

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def prepare(self, observation, request):
        prepared=self.adapter.prepare(observation,request)
        if self.adapter.batch_ledger is not None:
            prepared.data['batch_member']={'connection_id':self.connection_id,
                'request_id':uuid.uuid4().hex,'request':asdict(request)}
        return prepared

    def run_stage1(self, prepared):
        out=self.coordinator.submit_to_stage(1, prepared.request.bundle_generation, prepared)
        self.last_batch_receipt=copy.deepcopy(out.data.get('batching',{}))
        return out

    def run_stage2(self, stage1, *, capture=False):
        out=self.coordinator.submit_to_stage(2, "native", stage1, requires_stage2_capture=capture)
        self.last_batch_receipt=copy.deepcopy(out.data.get('batching',{}))
        return out

    def sample(self, stage2, sampler, rng, capture=CaptureSpec()):
        import torch
        from openpi.serving.batching_core import Stage3MissPayload, record_ready_events
        self.adapter._validate_sampler(sampler)
        shape=(self.adapter.metadata().layout.horizon,self.adapter.metadata().layout.action_width)
        noise=torch.randn(shape,generator=rng.torch_generator(device="cpu"),device="cpu",dtype=torch.float32)
        noise=noise.to(self.adapter.model.device,self.adapter.model.torch_dtype)
        points=tuple(i/sampler.num_steps for i in range(sampler.num_steps)) if capture.enabled else None
        payload=Stage3MissPayload(stage2,noise,sampler.num_steps,points,
            ready_events=record_ready_events(self.adapter.model.device))
        out=self.coordinator.submit_to_stage(3,"native",payload)
        self.last_batch_receipt=copy.deepcopy(out.native_result.capture.get('batching',{}))
        out.native_result.rng_receipt=dict(rng.receipt())
        if not capture.snapshots:
            out.native_result.snapshots=()
        return out.native_result
