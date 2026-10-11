"""Bounded CPU/GPU qualification. Device defaults to CPU; never starts a worker."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from exp.mw_finetune.dp.preprocess import quantile_unnormalize
from exp.r17_integration.common.contracts import CaptureSpec, RequestContext

from .config import LoadSpec
from .loader import load_adapter
from .request_noise import RequestNoise
from .sampler import SamplerProfile


def fixed_observation(path, domain):
    """Read row0 of a no-pickle wire NPZ verbatim; never synthesize a wrist."""
    from .manifest import file_digest
    with np.load(path, allow_pickle=False) as data:
        image_key = "image" if domain == "metaworld" and "image" in data else "observation/image"
        state_key = "state" if domain == "metaworld" and "state" in data else "observation/state"
        required = {image_key, state_key, "prompt"}
        if domain == "libero":
            required.add("observation/wrist_image")
        if not required <= set(data.files):
            raise ValueError("DP observations require image/state/prompt and a recorded LIBERO wrist")
        count = len(data["prompt"]) if data["prompt"].ndim == 1 else 0
        if count == 0 or any(data[key].ndim == 0 or len(data[key]) != count for key in required):
            raise ValueError("DP observations require nonempty aligned leading observation axes")
        if data["prompt"].dtype.kind != "U":
            raise ValueError("DP observations require string prompts without pickle")
        obs = {key: str(data[key][0]) if key == "prompt" else data[key][0].copy() for key in required}
    return obs, {"path": str(path), "npz_sha256": file_digest(path), "available_rows": count,
                 "native_check_row": 0, "batch_prompt_variant": "row0 prompt + carefully",
                 "image_shapes": {key: list(value.shape) for key, value in obs.items() if "image" in key},
                 "views": "provided camera arrays verbatim; no synthetic wrist"}


def batch_comparison(single, paired):
    """Measured diagnostic only; magnitude is never a production tolerance."""
    maximum, relative, mismatches, elements = 0.0, 0.0, 0, 0
    for one, two in zip(single, paired, strict=True):
        expected, actual = one.normalized_chunk, two.result.normalized_chunk
        assert torch.isfinite(expected).all() and torch.isfinite(actual).all()
        assert one.rng_receipt == two.result.rng_receipt
        delta = (expected.double() - actual.double()).abs()
        maximum = max(maximum, float(delta.max()))
        relative = max(relative, float((delta / actual.double().abs().clamp_min(1e-12)).max()))
        mismatches += int((delta > 1e-5 + 1e-4 * actual.double().abs()).sum())
        elements += delta.numel()
    return {"name": "B1_B2_batch_composition", "status": "diagnostic_only",
            "production_batching": "unsupported", "quality_claim": "none",
            "max_abs": maximum, "max_rel": relative, "rng_states_exact": True,
            "old_tolerance": {"rtol": 1e-4, "atol": 1e-5, "mismatches": mismatches, "elements": elements},
            "acceptance": "no numerical parity gate; labelled offline diagnostic only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", help="Fallback checkpoint; must agree with --load-spec if supplied")
    parser.add_argument("--domain", choices=("libero", "metaworld"), help="Fallback domain; must agree with --load-spec if supplied")
    parser.add_argument("--load-spec", type=Path, help="Sealed LoadSpec JSON; only device is overridden")
    parser.add_argument("--observations", type=Path, help="No-pickle wire NPZ; row0 is the fixed native-check input, preserving both LIBERO views")
    parser.add_argument("--text-encoder", default="/data/mw_finetune/dp/clip_text")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--deadline-seconds", type=float, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.load_spec is None and (args.checkpoint is None or args.domain is None):
        parser.error("provide --load-spec or both --checkpoint and --domain")
    start = time.monotonic()
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {"schema": "r17.dp.qualification.v1", "device": args.device,
               "passed": 0, "skipped": 0, "checks": [], "quality_claim": "none"}
    def deadline():
        if time.monotonic()-start > args.deadline_seconds:
            raise TimeoutError("bounded qualification deadline reached")
    def passed(name, **info):
        deadline()
        receipt["passed"] += 1
        receipt["checks"].append({"name": name, "status": "pass", **info})
    try:
        if args.load_spec is not None:
            document = json.loads(args.load_spec.read_text())
            if not isinstance(document, dict):
                raise ValueError("DP --load-spec must contain a LoadSpec JSON object")
            if args.domain is not None and args.domain != document.get("domain"):
                raise ValueError("--domain differs from the sealed DP load descriptor")
            if args.checkpoint is not None and Path(args.checkpoint).resolve() != Path(document["checkpoint"]).resolve():
                raise ValueError("--checkpoint differs from the sealed DP load descriptor")
            adapter = load_adapter(dict(document, device=args.device))
            from .manifest import file_digest
            receipt["load_spec_provenance"] = {"path": str(args.load_spec), "sha256": file_digest(args.load_spec),
                                               "overridden_fields": ["device"]}
        else:
            adapter = load_adapter(LoadSpec(args.checkpoint, args.domain, args.text_encoder, args.device))
        domain = adapter.domain
        receipt.update(domain=domain, diagnostic=adapter.diagnostic)
        receipt["manifest"] = adapter.manifest()
        receipt["weight_identity"] = adapter.weight_identity
        passed("strict_475", **adapter.strict_load)
        dim = len(adapter.metadata().layout.valid_state_indices)
        image = np.arange(96*96*3, dtype=np.uint8).reshape(96, 96, 3)
        obs = {"prompt": "reach the object", "state": np.zeros(dim, np.float32), "image": image}
        if args.observations is not None:
            obs, receipt["observation_provenance"] = fixed_observation(args.observations, domain)
        elif domain == "libero":
            from .fixtures import libero_observation
            obs, receipt["observation_provenance"] = libero_observation()
        if domain == "libero":
            passed("real_paired_LIBERO_fixture")
        request = RequestContext("p4_real_probe", adapter.metadata().allowed_benchmark_protocols[0],
                                 "diagnostic_fixed_observation", 0, 0, explicit_seed=18)
        p = adapter.prepare(obs, request)
        s1 = adapter.run_stage1(p)
        s2 = adapter.run_stage2(s1)
        with torch.inference_mode():
            cond = adapter.model.condition(p.data["inputs"])
        assert torch.equal(s2.data["conditioning"], cond)
        passed("native_condition_exact")
        fields = adapter.retrieval_fields(s1)
        assert all(fields[k].shape == (v,) for k, v in adapter.metadata().feature_spec.fields.items())
        assert "vision_1" not in fields if domain == "metaworld" else "vision_1" in fields
        passed("real_relu64_key_geometry", fields={k: list(v.shape) for k, v in fields.items()})
        for k in (100, 2, 1):
            deadline()
            profile = SamplerProfile(num_steps=k)
            with torch.inference_mode():
                if k == 100:
                    baseline = adapter.model.sample(p.data["inputs"], steps=k,
                                                    generator=torch.Generator(device=adapter.device).manual_seed(18))[0]
                else:
                    # Independent native loop, corrected profile; E's leading
                    # reduced sampler is historical and deliberately not used.
                    scheduler = profile.scheduler(adapter.model.cfg)
                    generator = torch.Generator(device=adapter.device).manual_seed(18)
                    x = torch.randn((1, 16, adapter.model.cfg["action_dim"]), generator=generator, device=adapter.device)
                    for t in scheduler.timesteps:
                        x = profile.step(scheduler, adapter.model._policy.model(x, t, cond), t, x, generator)
                    baseline = adapter.model._policy.normalizer.unnormalize("action", x)[0]
            expected = quantile_unnormalize(baseline.float().cpu().numpy(), adapter.normalization.stats["actions"]).astype(np.float32)
            result = adapter.sample(s2, profile, RequestNoise(18, adapter.device), CaptureSpec(True, k > 1))
            wire = adapter.to_wire(result.normalized_chunk, p)
            assert np.array_equal(wire.actions, expected), f"staged/full K{k} mismatch"
            assert result.actual_nfe == k and np.isfinite(wire.actions).all()
            passed(f"full_staged_k{k}_exact", native_grid=list(result.sampler.native_timesteps), actual_nfe=k)
            if k == 100:
                legacy_obs = dict(obs, seed=18)
                if domain == "metaworld":
                    legacy_obs.update(image=obs.get("image", obs.get("observation/image")),
                                      state=obs.get("state", obs.get("observation/state")))
                upstream = adapter.legacy_policy.infer(legacy_obs)["actions"]
                assert np.array_equal(wire.actions, upstream)
                passed("E_Policy_infer100_exact_including_preprocess")
            if k == 2:
                resumed = adapter.resume(s2, result.snapshots[0], result.sampler, RequestNoise(18, adapter.device), CaptureSpec())
                assert torch.equal(resumed.normalized_chunk, result.normalized_chunk)
                passed("real_native_suffix_exact", actual_nfe=resumed.actual_nfe)
                plain = adapter.sample(s2, profile, RequestNoise(18, adapter.device), CaptureSpec())
                assert torch.equal(plain.normalized_chunk, result.normalized_chunk) and plain.rng_receipt == result.rng_receipt
                passed("real_capture_on_off_exact")
                trace = adapter.make_trace_adapter()
                trace.save(trace.from_stages(p, s1, result, wire), args.output / "capture")
        deadline()
        ddim = SamplerProfile("ddim", 2)
        r = adapter.sample(s2, ddim, RequestNoise(18, adapter.device), CaptureSpec(True, True))
        assert r.actual_nfe == 2 and torch.isfinite(r.normalized_chunk).all()
        assert r.sampler.family == "ddim" and r.sampler.native_timesteps == (99, 49)
        resumed = adapter.resume(s2, r.snapshots[0], r.sampler, RequestNoise(18, adapter.device), CaptureSpec())
        assert torch.equal(resumed.normalized_chunk, r.normalized_chunk)
        passed("real_explicit_ddim_suffix_exact")
        deadline()
        # Production uses B1 only. B2 remains an explicitly labelled offline
        # diagnostic: GPU arithmetic depends on composition even with exact RNG.
        p2 = adapter.prepare(dict(obs, prompt=obs["prompt"]+" carefully"), request)
        for name in ("batch_stage1", "batch_stage2", "batch_stage3"):
            assert adapter.metadata().capabilities.features[name].status == "unsupported"
        try:
            adapter.run_stage1_batch([p, p2])
        except ValueError as exc:
            assert "production B>1 disabled" in str(exc)
        else:
            raise AssertionError("production DP admitted B2")
        profile = SamplerProfile(num_steps=2)
        b1 = adapter.make_stage_batcher()
        full_b1 = adapter.sample(s2, profile, RequestNoise(18, adapter.device), CaptureSpec())
        payload = b1.miss_payload(s2, profile, RequestNoise(18, adapter.device), CaptureSpec())
        queued_b1 = b1.run_stage3_miss([payload], num_steps=2, save_timesteps_per_request=[None])[0].result
        assert torch.equal(full_b1.normalized_chunk, queued_b1.normalized_chunk)
        assert full_b1.rng_receipt == queued_b1.rng_receipt
        passed("real_production_B1_only", max_batch_size=1, B1_queue_bytes_exact=True,
               B2_comparison_role="labelled offline diagnostic; no parity claim")
        diagnostic = adapter.for_batch_diagnostic("native_probe_B2_composition_only")
        batcher = diagnostic.make_stage_batcher()
        stages = [diagnostic.run_stage2(s) for s in diagnostic.run_stage1_batch([p, p2])]
        profile = SamplerProfile(num_steps=2)
        sequential = [diagnostic.sample(s, profile, RequestNoise(seed, adapter.device), CaptureSpec())
                      for s, seed in zip(stages, (18, 37))]
        payloads = [batcher.miss_payload(s, profile, RequestNoise(seed, adapter.device), CaptureSpec())
                    for s, seed in zip(stages, (18, 37))]
        paired = batcher.run_stage3_miss(payloads, num_steps=2, save_timesteps_per_request=[None, None])
        receipt["diagnostics"] = [dict(batch_comparison(sequential, paired),
                                      label=diagnostic.batch_diagnostic_label)]
        if domain == "metaworld":
            assert not torch.equal(s1.data["features"]["left_wrist_0_rgb"],
                                   stages[1].stage1.data["features"]["left_wrist_0_rgb"])
            passed("real_zero_wrist_is_prompt_conditioned")
        # Seal runnable serving artifacts against this exact adapter release.
        # Historical leading-grid files are never silently reused.
        import yaml
        from .provider import register_provider
        from .serve import plain_config
        from exp.r17_integration.common.config import config_from_dict
        from exp.r17_integration.common.provider import validate_binding
        pinned = dict(adapter.load_spec, expected_weight_sha256=adapter.weight_identity,
                      expected_norm_identity=adapter.metadata().norm_identity,
                      expected_model_identity=adapter.metadata().checkpoint_identity)
        (args.output / "load_cpu.json").write_text(json.dumps(pinned, indent=2) + "\n")
        provider = register_provider(adapter)
        for k in (1, 2, 4, 10, 100):
            document = asdict(plain_config(provider, num_steps=k, commit_horizon=5 if k == 1 else 10,
                allow_lossy_sampler=k == 1,
                sampler_study_label="DP_K1_plain_step_reduction_only" if k == 1 else None))
            config = config_from_dict(yaml.safe_load(yaml.safe_dump(document)))
            bound, _ = validate_binding(config.adapter_binding)
            assert bound is provider
            (args.output / f"k{k}.yaml").write_text(yaml.safe_dump(document))
        receipt["generated_bound_yaml_checks"] = 5
        passed("current_release_pinned_serving_samples", yaml_bindings=5)
        receipt.update(status="pass", exit_code=0)
    except BaseException as exc:
        receipt.update(status="partial" if isinstance(exc, TimeoutError) else "error", exit_code=1,
                       error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        receipt["elapsed_seconds"] = time.monotonic()-start
        (args.output / "receipt.json").write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: receipt[k] for k in ("status", "passed", "skipped", "device", "elapsed_seconds")
                          if k in receipt}, indent=2))


if __name__ == "__main__":
    main()
