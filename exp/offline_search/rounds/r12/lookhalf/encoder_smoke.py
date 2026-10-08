"""Bounded real-weight encoder smoke; no policy, server or evaluation launch."""
from __future__ import annotations
import argparse
import contextlib
import json
from pathlib import Path
import pickle
import subprocess
from types import SimpleNamespace as NS
import numpy as np

HERE = Path(__file__).resolve().parent
STORE = Path('/home/weiland/trace_runs/offline_search_store')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, choices=('pi05', 'groot'))
    args = ap.parse_args()
    import torch
    from safetensors import safe_open
    status = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,memory.total',
                                      '--format=csv,noheader,nounits'], text=True).strip()
    used, total = map(int, status.split(','))
    if used > 1000:
        raise RuntimeError(f'encoder smoke requires idle GPU: {status}')
    torch.cuda.set_per_process_memory_fraction(4*1024/total)
    torch.set_num_threads(1)
    # Exact stored raw B images, from one row in the selected pi05 library.
    # The GR00T smoke uses these same pixels and a synthetic text layout;
    # exact live GR00T preprocessing/policy deltas are recorded by B pilots.
    with (HERE/'artifacts/pi05_l10_50/off.pkl').open('rb') as f:
        method = pickle.load(f)['method']
    row = int(method.row_subset[0])
    parent = STORE/'library/pi05_l10'/method.library
    images = np.stack([np.load(parent/'tok'/f'img{i}.npy', mmap_mode='r')[row] for i in range(2)])
    pixels = torch.from_numpy(images.copy()).permute(0, 3, 1, 2).to('cuda', torch.float32)/127.5-1
    calls = []
    if args.model == 'pi05':
        from transformers import CONFIG_MAPPING
        from transformers.models.paligemma.modeling_paligemma import PaliGemmaMultiModalProjector
        from transformers.models.siglip.modeling_siglip import SiglipVisionModel
        from exp.offline_search.closed_loop.stage_overrides import Pi05Override
        from exp.offline_search.rounds.r07.c2_wrist.test_dispatch import Model, stock
        from openpi.models.model import Observation
        from openpi.cache.components.key_builder import CP1SpatialPool16KeyBuilder
        from openpi.cache.types import CheckpointID
        cfg = CONFIG_MAPPING['paligemma']()
        cfg.vision_config.intermediate_size = 4304
        cfg.vision_config.projection_dim = 2048
        cfg.vision_config.projector_hidden_act = 'gelu_fast'
        cfg.text_config.hidden_size = 2048
        vision = SiglipVisionModel(cfg.vision_config)
        connector = PaliGemmaMultiModalProjector(cfg)
        checkpoint = Path('/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch/model.safetensors')
        with safe_open(checkpoint, framework='pt', device='cpu') as f:
            for part, module in [('vision_tower', vision), ('multi_modal_projector', connector)]:
                prefix = f'paligemma_with_expert.paligemma.model.{part}.'
                module.load_state_dict({k[len(prefix):]: f.get_tensor(k) for k in f.keys() if k.startswith(prefix)}, strict=True)
        vision = vision.eval().to('cuda', torch.bfloat16)
        connector = connector.eval().to('cuda', torch.bfloat16)
        def encode(x):
            calls.append(int(len(x)))
            return connector(vision(pixel_values=x.to(torch.bfloat16)).last_hidden_state)
        model = Model()
        model.paligemma_with_expert = NS(paligemma=NS(vision_tower=vision, multi_modal_projector=connector),
            embed_image=encode, embed_language_tokens=lambda tokens: torch.zeros((*tokens.shape, 2048), device='cuda', dtype=torch.bfloat16))
        names = ['base_0_rgb', 'left_wrist_0_rgb', 'right_wrist_0_rgb']
        obs = Observation(images=dict(zip(names, [pixels[:1], pixels[1:], -torch.ones_like(pixels[:1])])),
            image_masks={name: torch.tensor([i<2], device='cuda') for i,name in enumerate(names)},
            state=torch.zeros(1,32,device='cuda'), tokenized_prompt=torch.tensor([[1,2]], device='cuda'),
            tokenized_prompt_mask=torch.tensor([[True,False]], device='cuda'))
        override = Pi05Override('per_request')
        with torch.inference_mode():
            full = stock(model, obs)
            override.seed_full_dummy(model, obs, full)
            calls.clear()
            half = override.stage1(model, obs, mode='wrist_only')
            assert calls == [1], calls
            completed = override.complete(model, half)
            assert calls == [1,1], calls
            torch.testing.assert_close(completed.prefix_embs, full.prefix_embs, atol=0, rtol=0)
            kb = CP1SpatialPool16KeyBuilder()
            kb.collect(CheckpointID.CP1, stage1=completed)
            keys = kb.build(CheckpointID.CP1)
            delta = float((completed.prefix_embs-full.prefix_embs).abs().max())
        detail = dict(wrist_and_completion_batches=calls, prefix_max_abs=delta,
                      key_shapes={k:list(v.shape) for k,v in keys.items()}, completion_matches_stock=True)
    else:
        from transformers import SiglipVisionConfig, SiglipVisionModel
        from exp.offline_search.closed_loop.lookhalf_cameras import GrootOverride
        from exp.offline_search.closed_loop.lookhalf_numerics import deltas
        cfgpath = Path('/home/weiland/projects/openpi_ext/third_party/gr00t_n15/gr00t/model/backbone/eagle2_hg_model/config.json')
        cfg = SiglipVisionConfig(**json.loads(cfgpath.read_text())['vision_config'])
        cfg._attn_implementation = 'flash_attention_2'
        vision = SiglipVisionModel(cfg)
        connector = torch.nn.Linear(1152, 2048)
        checkpoint = Path('/data/ckpt/n15_libero_10')
        index = json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
        for part, module in [('vision_model', vision), ('mlp1', torch.nn.Sequential(connector))]:
            prefix = f'backbone.eagle_model.{part}.'
            state = {}
            for shard in sorted({v for k,v in index.items() if k.startswith(prefix)}):
                with safe_open(checkpoint/shard, framework='pt', device='cpu') as f:
                    state.update({k[len(prefix):]: f.get_tensor(k) for k,v in index.items() if k.startswith(prefix) and v == shard})
            module.load_state_dict(state, strict=True)
        vision, connector = vision.eval().to('cuda'), connector.eval().to('cuda')
        def encode(x):
            calls.append(int(len(x)))
            return connector(vision(pixel_values=x, return_dict=True).last_hidden_state)
        ids = torch.cat([torch.ones((1,2),device='cuda',dtype=torch.long),
                         torch.full((1,256),7,device='cuda'),torch.ones((1,1),device='cuda',dtype=torch.long),
                         torch.full((1,256),7,device='cuda')], 1)
        inputs = dict(eagle_input_ids=ids,eagle_attention_mask=torch.ones_like(ids),eagle_pixel_values=pixels)
        action = dict(state=torch.zeros(1,1,32,device='cuda'),state_mask=torch.ones(1,1,32,device='cuda',dtype=torch.bool))
        eagle = NS(image_token_index=7,extract_feature=encode,
            language_model=NS(get_input_embeddings=lambda: lambda tokens: torch.zeros((*tokens.shape,2048),device='cuda',dtype=torch.bfloat16)))
        runner = NS(_eagle=eagle,_require_session=lambda stage:None,_model=NS(prepare_input=lambda obs:(inputs,action)))
        override = GrootOverride(); audit = {}
        with torch.inference_mode(), torch.autocast('cuda', torch.bfloat16):
            half = override.partial(runner, {}, audit)
            assert calls == [1], calls
            wrist = half.input_embeds[:,259:].clone()
            override.complete(runner, half); override.complete(runner, half)
            assert calls == [1,1] and audit['completions'] == 1, (calls,audit)
            assert torch.equal(wrist, half.input_embeds[:,259:])
            joint = encode(pixels)
            separate = torch.stack([half.input_embeds[0,2:258],half.input_embeds[0,259:]])
            difference = deltas(joint.float().cpu().numpy(),separate.float().cpu().numpy())
        detail = dict(wrist_and_completion_batches=calls[:2],joint_diagnostic_batch=calls[2],
            completions=audit['completions'],wrist_retained=True,joint_separate_token_delta=difference,
            policy_deltas='pending exact-input B pilot; no policy loaded by smoke',text_layout='synthetic')
    torch.cuda.synchronize()
    result = dict(model=args.model,gpu_before=status,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,
        allocator_cap_mib=4096,source=f'{parent}/tok/img[01].npy',selected_parent_row=row,
        checkpoint=str(checkpoint),torch=torch.__version__,encoder_only=True,details=detail)
    assert result['peak_reserved_mib'] < 4096
    (HERE/f'ENCODER_SMOKE_{args.model}.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
