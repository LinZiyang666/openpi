"""Named real-model gate for the online RIT signal (plan §8-9). GR00T island venv only.

Run on the serving box with the real checkpoint::

  ONLINE_RIT_CKPT=/path/to/ckpt ONLINE_RIT_LIBRARY=/data/.../libero_10_w13_S3.pkl \\
  ONLINE_RIT_SCALES=/data/.../update_scales.npz ONLINE_RIT_GATE_OUT=/data/.../manual_inputs.json \\
  python -m pytest tests/cache/groot/test_online_rit_real_model.py --run-manual -q

For 64 library entries (seed 20260914) under one real stage-2 conditioning:
batched vs serial first steps agree tier by tier (max |delta| <= 1e-3 in the
head dtype), the captured first step of the resumed loop equals the side
step for the same snapshot, capture on/off leaves the final chunk identical,
and every disagreement is finite. A failing number is a stop, not a widened
tolerance.
"""

from __future__ import annotations

import os
import itertools
import json
import pathlib
import pickle
import random

import pytest
import torch

pytestmark = pytest.mark.manual

TOL = 1e-3
N_ENTRIES = 64
SEED = 20260914


def _env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        pytest.fail(f"{name} not set; configure the GR00T island gate explicitly")
    return value


@pytest.fixture(scope="module")
def real():
    from exp.libero_groot.bench_profile import build_input, load_policy
    from openpi.cache.groot.staged import GrootStagedRunner

    ckpt = _env("ONLINE_RIT_CKPT")
    library = _env("ONLINE_RIT_LIBRARY")
    policy = load_policy(ckpt, device="cuda", denoising_steps=8)
    runner = GrootStagedRunner(policy.model)
    with open(library, "rb") as fh:
        art = pickle.load(fh)
    rng = random.Random(SEED)
    entries = rng.sample(art["entries"], N_ENTRIES)
    prompt = str(entries[0].payload.task_key).replace("_", " ")
    normalized = build_input(policy, ckpt, prompt)
    from exp.online_rit.common import sha256_file
    from openpi.cache.components.online_rit import load_update_scales
    scales, masks, meta, scales_sha = load_update_scales(_env("ONLINE_RIT_SCALES"), [7, 6, 4])
    assert meta["library_sha256"] == sha256_file(library)
    from exp.rit_loto.build_loto_table import checkpoint_identity
    record = {"seed": SEED, "mode": "eager", "conditioning": "fixed synthetic input; not self replay",
              "checkpoint_identity_sha256": checkpoint_identity(ckpt)["sha256"],
              "library_sha256": sha256_file(library), "scales_sha256": scales_sha,
              "test_sha256": sha256_file(__file__), "entry_ids": [e.id for e in entries]}
    out = pathlib.Path(_env("ONLINE_RIT_GATE_OUT"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n")
    return policy, runner, entries, normalized, scales, masks


def test_batched_side_steps_match_serial_and_capture_on_the_real_head(real):
    from openpi.cache.groot.staged import denoise_step
    from openpi.cache.types import groot_n15_schedule

    policy, runner, entries, normalized, scales, masks = real
    schedule = groot_n15_schedule(8)
    head = policy.model.action_head
    worst = {t: 0.0 for t in (0.875, 0.75, 0.5)}
    with runner.session():
        stage1 = runner.run_stage1(normalized)
        stage2 = runner.run_stage2_llm(stage1)
        for e in entries:
            snaps = [(t, e.payload.intermediates[t]) for t in (0.875, 0.75, 0.5)]
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()]
            pairs = runner.first_step_updates(stage2, snaps, schedule=schedule)
            for batch in (1, 2, 3):
                for indices in itertools.combinations(range(3), batch):
                    partial = runner.first_step_updates(stage2, [snaps[i] for i in indices], schedule=schedule)
                    for idx, (x_in, x_out) in zip(indices, partial):
                        assert torch.equal(x_in, pairs[idx][0])
                        assert float((x_out.float() - pairs[idx][1].float()).abs().max()) <= TOL
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
            head_inputs = runner._head_inputs(stage2)  # noqa: SLF001
            processed = head.process_backbone_output(head_inputs)
            vl = processed.backbone_features
            emb = stage2.action_inputs["embodiment_id"]
            sf = head.state_encoder(stage2.action_inputs["state"], emb)
            for (t, x), (x_used, x_out) in zip(snaps, pairs):
                idx = schedule.snapshot_index(t)
                bucket = int(idx / 8 * head.num_timestep_buckets)
                serial = denoise_step(head, vl, sf, emb, x[None].to(vl.device, vl.dtype), torch.full((1,), bucket, device=vl.device), 1 / 8)
                worst[t] = max(worst[t], float((serial.float() - x_out.float()).abs().max()))
                assert torch.isfinite(x_out).all()
                cap = runner.run_stage3_from(stage2, x, t, schedule=schedule, capture_first_step=True)
                plain = runner.run_stage3_from(stage2, x, t, schedule=schedule)
                assert torch.equal(cap.action_pred, plain.action_pred)
                assert torch.equal(cap.first_step_input, x_used)
                assert float((cap.first_step_x.float() - x_out.float()).abs().max()) <= TOL
    assert all(v <= TOL for v in worst.values()), worst


def test_real_disagreements_use_actual_scales_and_are_finite(real):
    from openpi.cache.components.online_rit import ContinuationSpec, feedback_from_updates, tier_specs
    from openpi.cache.types import groot_n15_schedule

    policy, runner, entries, normalized, scales, masks = real
    schedule = groot_n15_schedule(8)
    tiers = tier_specs([0.875, 0.75, 0.5], schedule)
    spec = ContinuationSpec(tiers=tiers, scales=scales, masks=masks, h_exec=5, feedback_mode="fm1", schedule=schedule)
    with runner.session():
        stage2 = runner.run_stage2_llm(runner.run_stage1(normalized))
        for e in entries:
            snaps = [(t, e.payload.intermediates[t]) for t in (0.875, 0.75, 0.5)]
            cpu_rng = torch.get_rng_state().clone()
            cuda_rng = [x.clone() for x in torch.cuda.get_rng_state_all()]
            pairs = runner.first_step_updates(stage2, snaps, schedule=schedule)
            for batch in (1, 2, 3):
                for indices in itertools.combinations(range(3), batch):
                    partial = runner.first_step_updates(stage2, [snaps[i] for i in indices], schedule=schedule)
                    for idx, (x_in, x_out) in zip(indices, partial):
                        assert torch.equal(x_in, pairs[idx][0])
                        assert float((x_out.float() - pairs[idx][1].float()).abs().max()) <= TOL
            assert torch.equal(cpu_rng, torch.get_rng_state())
            assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all()))
            side = [(t, x_in, x_out) for (t, _), (x_in, x_out) in zip(snaps, pairs)]
            fb, reasons = feedback_from_updates(spec, e.payload, schedule, executed=None, side=side)
            assert reasons == [] and len(fb) == 3
