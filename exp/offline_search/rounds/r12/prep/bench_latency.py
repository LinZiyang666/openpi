"""Bounded batch-1 eager deployment-stage latency benchmark; B libraries only.

No websocket, environment, cache-driven rollout, or test-A data is opened.
Half-camera implementations are benchmark-only; they do not install a serving
mode for GR00T or third-person-only pi05. Times include stage input preparation
but exclude the policy's CPU image/state transforms, exactly as R7 A3.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
STORE = Path('/home/weiland/trace_runs/offline_search_store')


def summarize(values):
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return dict(n=len(values), median_ms=float(median), q1_ms=float(q1), q3_ms=float(q3), iqr_ms=float(q3 - q1))


def memory_gate(model, out, timeout=600):
    need = 12288 if model == 'pi05' else 10240
    deadline, checks = time.monotonic() + timeout, []
    while True:
        result = subprocess.check_output(['nvidia-smi', '--id=0',
            '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu',
            '--format=csv,noheader,nounits'], text=True).strip()
        columns = [v.strip() for v in result.split(',')]
        rec = dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), nvidia_smi=result,
                   free_mib=int(columns[3]), peak_budget_mib=need, reserve_mib=4096)
        checks.append(rec)
        out.write_text(json.dumps(checks, indent=1) + '\n')
        print('GPU_CHECK', json.dumps(rec), flush=True)
        if int(columns[3]) >= need + 4096:
            return checks, need
        if time.monotonic() >= deadline:
            raise RuntimeError('GPU memory gate timed out; no model was loaded')
        time.sleep(min(30, max(0, deadline - time.monotonic())))


def observations(model):
    """Two suites x early/late B observations; shapes/prompts are real B data.

    Store tok images are 224 RGB. GR00T's wire validator needs 256, so resize
    those to 256 before the unchanged deployment crop/resize chain. This is
    latency input only, with that provenance explicit; no SR/parity-to-rollout
    claim is made for reconstructed pixels or the zero physical state.
    """
    from exp.offline_search.harness.store import LibraryView
    for suite in ('l10', 'spatial'):
        # GR00T B libraries store tokens but no RGB. Both policies use the
        # permitted pi05 B-pool RGB corpus for timing (same LIBERO tasks).
        lib = LibraryView(STORE, f'pi05_{suite}', 'bpool_cs')
        eps = lib.episodes
        for index, offset in ((0, 0), (len(eps) // 2, -1)):
            ep = eps[index]
            candidates = lib.tok_rows[(lib.tok_rows >= ep['start']) & (lib.tok_rows < ep['end'])]
            row = int(candidates[offset]) if len(candidates) else int(lib.tok_rows[offset])
            images = [np.array(lib.tok_of(n, row), np.uint8) for n in ('img0', 'img1')]
            if model == 'groot':
                import cv2
                images = [cv2.resize(im, (256, 256), interpolation=cv2.INTER_LINEAR) for im in images]
            raw = {'observation/image': images[0], 'observation/wrist_image': images[1],
                   'observation/state': np.zeros(8, np.float64), 'prompt': ep['task']}
            metadata = dict(suite=suite, row=row, source_library=str(lib.dir), task=ep['task'],
                            image_sha256=[hashlib.sha256(im.tobytes()).hexdigest() for im in images],
                            state='zero physical 8-vector; latency only')
            yield raw, metadata


def pi05_setup():
    import torch
    from openpi.training import config
    from openpi.policies import policy_config
    from openpi.models.model import Observation
    from openpi.serving.batching_coordinator import Pi05StageBatcher
    from openpi.serving import stage_io
    from openpi.models_pytorch.pi0_pytorch import Stage1Output, make_att_2d_masks
    from exp.offline_search.closed_loop.stage_overrides import CameraRequest, install_pi05
    cfg = config.get_config('pi05_libero')
    cfg = dataclasses.replace(cfg, model=dataclasses.replace(cfg.model, pytorch_compile_mode=None))
    checkpoint = '/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch'
    policy = policy_config.create_trained_policy(cfg, checkpoint, pytorch_device='cuda:0')
    model = policy._model.eval()
    stock = model.run_stage1
    override = install_pi05('per_request')
    batcher = Pi05StageBatcher(model, torch.device('cuda:0'))

    def third(inp):
        batched = stage_io.stack_observation([inp], device=torch.device('cuda:0'))
        obs = Observation.from_dict(batched) if isinstance(batched, dict) and 'image' in batched else batched
        images, masks, tokens, lang_mask, state = model._preprocess_observation(obs, train=False)
        dummy = override._dummy(model, images[2], masks[2])
        tower = model.paligemma_with_expert
        base = tower.embed_image(images[0])
        wrist = torch.zeros_like(base)
        lang = tower.embed_language_tokens(tokens) * (base.shape[-1] ** .5)
        prefix = torch.cat((base, wrist, dummy, lang), dim=1)
        pad = torch.cat([m[:, None].expand(im.shape[:2]) for m, im in zip(masks, (base, wrist, dummy))]
                        + [lang_mask], dim=1)
        att = make_att_2d_masks(pad, torch.zeros_like(pad))
        out = Stage1Output(state=state, prefix_embs=prefix, prefix_pad_masks=pad,
                           prefix_att_2d_masks_4d=model._prepare_attention_masks_4d(att),
                           prefix_position_ids=torch.cumsum(pad, dim=1) - 1)
        return out, images[1].detach().clone()

    def complete_third(partial):
        out, image = partial
        wrist = model.paligemma_with_expert.embed_image(image)
        return dataclasses.replace(out, prefix_embs=torch.cat((out.prefix_embs[:, :256], wrist,
                                                              out.prefix_embs[:, 512:]), dim=1))

    def paths(raw):
        inp = policy._input_transform(raw)
        full = batcher.run_stage1_batch([CameraRequest(inp, 'full')])[0]  # seeds cached dummy
        wrist = batcher.run_stage1_batch([CameraRequest(inp, 'wrist_only', {})])[0]
        tp = third(inp)
        compw, compt = override.complete(model, wrist), complete_third(tp)
        parity = dict(wrist_completion_max_abs=float((full.prefix_embs - compw.prefix_embs).abs().max()),
                      third_completion_max_abs=float((full.prefix_embs - compt.prefix_embs).abs().max()),
                      camera_order=['base_0_rgb', 'left_wrist_0_rgb', 'masked_right_wrist_0_rgb'])
        assert parity['wrist_completion_max_abs'] == parity['third_completion_max_abs'] == 0
        # R7's denominator is STOCK full stage1 (.152), not per-request full.
        obs = Observation.from_dict(stage_io.stack_observation([inp], device=torch.device('cuda:0')))
        return dict(stock_full=lambda: stock(obs),
                    full=lambda: batcher.run_stage1_batch([CameraRequest(inp, 'full')]),
                    wrist_only=lambda: batcher.run_stage1_batch([CameraRequest(inp, 'wrist_only', {})]),
                    third_only=lambda: third(inp), completion_after_wrist=lambda: override.complete(model, wrist),
                    completion_after_third=lambda: complete_third(tp),
                    policy_remainder=lambda: model.run_stage3(model.run_stage2(full), num_steps=10)), parity
    return policy, paths, dict(checkpoint=checkpoint, steps=10, compile=False, attention='SDPA',
                               dtype='selected bfloat16 with float32 patch embedding/norms',
                               configured_dtype=cfg.model.dtype,
                               first_parameter_dtype=str(next(model.parameters()).dtype), dummy='cached exactly as R7')


def groot_setup(suite):
    import torch
    from gr00t.model.policy import Gr00tPolicy
    from custom_data_config import LiberoDataConfig
    from openpi.cache.groot.staged import GrootStagedRunner, GrootStage1Output
    from openpi.cache.groot.interceptor import _unsqueeze_values
    from exp.libero_groot.policy_adapter import build_groot_observation
    checkpoint = f'/data/ckpt/n15_libero_{"10" if suite == "l10" else "spatial"}'
    config = LiberoDataConfig()
    policy = Gr00tPolicy(model_path=checkpoint, embodiment_tag='new_embodiment',
                         modality_config=config.modality_config(), modality_transform=config.transform(),
                         denoising_steps=8, device='cuda')
    runner = GrootStagedRunner(policy.model, compile_vision=False)

    def half(norm, camera):
        runner._require_session('benchmark half')
        backbone, action = policy.model.prepare_input(norm)
        inp = {k.removeprefix('eagle_'): v for k, v in backbone.items() if k.startswith('eagle_')}
        inp.pop('image_sizes', None)
        assert set(inp) == {'input_ids', 'attention_mask', 'pixel_values'}
        assert inp['input_ids'].shape[0] == 1 and inp['pixel_values'].shape[0] == 2
        eagle = runner._eagle
        embeds = eagle.language_model.get_input_embeddings()(inp['input_ids'])
        mask = inp['input_ids'] == eagle.image_token_index
        positions = torch.nonzero(mask.reshape(-1)).reshape(-1)
        assert len(positions) == 512
        tokens = eagle.extract_feature(inp['pixel_values'][camera:camera+1])
        assert tokens.shape == (1, 256, embeds.shape[-1])
        flat = embeds.reshape(-1, embeds.shape[-1])
        flat[positions] = 0
        flat[positions[camera*256:(camera+1)*256]] = tokens.reshape(-1, tokens.shape[-1])
        out = GrootStage1Output(input_embeds=embeds, attention_mask=inp['attention_mask'],
                               image_token_mask=mask, action_inputs=action)
        missing = 1 - camera
        return out, inp['pixel_values'][missing:missing+1].detach().clone(), positions[missing*256:(missing+1)*256]

    def complete(partial):
        out, pixels, positions = partial
        embeds = out.input_embeds.clone()
        embeds.reshape(-1, embeds.shape[-1])[positions] = runner._eagle.extract_feature(pixels).reshape(-1, embeds.shape[-1])
        return dataclasses.replace(out, input_embeds=embeds)

    def paths(raw):
        inp = _unsqueeze_values(build_groot_observation(raw))
        inp = {k: v if isinstance(v, np.ndarray) else np.array(v) for k, v in inp.items()}
        norm = policy.apply_transforms(inp)
        with runner.session():
            full = runner.run_stage1(norm)
            wrist, third = half(norm, 1), half(norm, 0)
            cw, ct = complete(wrist), complete(third)
            parity = dict(wrist_completion_max_abs=float((full.input_embeds - cw.input_embeds).abs().max()),
                          third_completion_max_abs=float((full.input_embeds - ct.input_embeds).abs().max()),
                          image_mask_tokens=int(full.image_token_mask.sum()), camera_order=['image', 'wrist_image'])
        def call(fn):
            with runner.session():
                return fn()
        return dict(full=lambda: call(lambda: runner.run_stage1(norm)),
                    wrist_only=lambda: call(lambda: half(norm, 1)), third_only=lambda: call(lambda: half(norm, 0)),
                    completion_after_wrist=lambda: call(lambda: complete(wrist)),
                    completion_after_third=lambda: call(lambda: complete(third)),
                    policy_remainder=lambda: call(lambda: runner.run_stage3(runner.run_stage2_llm(full)))), parity
    return policy, paths, dict(checkpoint=checkpoint, steps=8, compile=False, dtype='bfloat16 + runner.session autocast',
                              attention=str(policy.model.backbone.eagle_model.language_model.config._attn_implementation),
                              full_vision_batch=2, half_vision_batch=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', choices=('pi05', 'groot'), required=True)
    ap.add_argument('--suite', choices=('l10', 'spatial'), default='l10')
    ap.add_argument('--reps', type=int, default=40); ap.add_argument('--warmup', type=int, default=8)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.reps < 30 or a.warmup < 5:
        raise ValueError('at least 30 repetitions and 5 warmups required')
    a.out.parent.mkdir(parents=True, exist_ok=True)
    checks, budget = memory_gate(a.model, a.out.with_suffix('.gpu_checks.json'))
    import torch
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    torch.cuda.set_per_process_memory_fraction(budget / torch.cuda.get_device_properties(0).total_memory * (1 << 20))
    policy, paths, stack = pi05_setup() if a.model == 'pi05' else groot_setup(a.suite)
    rows = []
    with torch.inference_mode():
        for raw, meta in observations(a.model):
            if a.model == 'groot' and meta['suite'] != a.suite:
                continue
            fns, parity = paths(raw)
            rec = dict(observation=meta, parity=parity, timings={})
            # Rotate path order per observation to avoid a fixed order bias.
            names = list(fns); shift = len(rows) % len(names); names = names[shift:] + names[:shift]
            for name in names:
                fn = fns[name]
                for _ in range(a.warmup): fn()
                torch.cuda.synchronize()
                values = []
                for _ in range(a.reps):
                    torch.cuda.synchronize(); start = time.perf_counter()
                    fn(); torch.cuda.synchronize()
                    values.append((time.perf_counter() - start) * 1000)
                rec['timings'][name] = dict(samples_ms=values, **summarize(values))
                print('TIMING', a.model, meta['suite'], meta['row'], name, json.dumps(summarize(values)), flush=True)
            rows.append(rec)
            a.out.write_text(json.dumps(dict(model=a.model, stack=stack, gpu_checks=checks, rows=rows), indent=1) + '\n')
    names = rows[0]['timings']
    pooled = {n: summarize([v for r in rows for v in r['timings'][n]['samples_ms']]) for n in names}
    basis = .152 if a.model == 'pi05' else .148
    denominator = pooled['stock_full' if a.model == 'pi05' else 'full']['median_ms']
    prices = {n: basis * v['median_ms'] / denominator for n, v in pooled.items() if n != 'policy_remainder'}
    result = dict(model=a.model, stack=stack, reps=a.reps, warmup=a.warmup, gpu_checks=checks, rows=rows,
                  pooled_timings=pooled, owner_basis=basis, normalization_full_ms=denominator,
                  owner_prices=prices, owner_policy_remainder=1-basis,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / (1 << 20),
                  peak_reserved_mib=torch.cuda.max_memory_reserved() / (1 << 20),
                  timestamp_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    a.out.write_text(json.dumps(result, indent=1) + '\n')
    print('BENCH_COMPLETE', json.dumps({k: result[k] for k in ('model', 'pooled_timings', 'owner_prices', 'peak_reserved_mib')}), flush=True)
    del policy, paths, fns, fn, raw
    gc.collect(); torch.cuda.empty_cache()
    print('MODEL_PROCESS_EXIT releases all CUDA memory', flush=True)


if __name__ == '__main__':
    main()
