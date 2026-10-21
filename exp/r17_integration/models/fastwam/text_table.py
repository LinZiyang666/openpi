"""Precomputed Fast-WAM prompt encodings, so a resident never loads umT5-xxl.

Every campaign prompt is fixed by its task, so the T5 context is computed once
(batch 1, exactly FastWAM.encode_prompt) and the server only looks rows up.

Build: streams the encoder through the GPU one block at a time (weights stay in
the mmapped checkpoint), so it runs beside live residents in ~1.5 GB:
  python -m exp.r17_integration.models.fastwam.text_table build --wan-root W \
      --benchmark metaworld:B.json --benchmark libero:B.json --out table.pt [--check-batch 16]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

TEMPLATE = "A video recorded from a robot's point of view executing the following instruction: {task}"


def libero_language(name):  # libero.benchmark.grab_language_from_filename, as formal_batch_server.cache_text
    task = name.split('/', 1)[1]
    if task[0].isupper():
        task = task[task.find('SCENE')+(8 if 'SCENE10' in task else 7):]
    return ' '.join(task.split('_'))


def prompts(domain, benchmark, template=TEMPLATE):
    if domain == 'metaworld':
        return sorted({template.format(task=row[1]) for row in benchmark['tasks']})
    return sorted({template.format(task=libero_language(n)) for n in benchmark['ordered_tasks']})


def _encoder(wan_root):
    import sys
    import torch
    sys.path.insert(0, '/data/mw_finetune/vendor/FastWAM/src')
    from fastwam.models.wan22.wan_video_text_encoder import WanTextEncoder
    from lerobot.policies.fastwam.wan.components import build_wan_tokenizer
    with torch.device('meta'):
        encoder = WanTextEncoder()
    state = torch.load(Path(wan_root)/'models_t5_umt5-xxl-enc-bf16.pth', map_location='cpu', weights_only=True, mmap=True)
    encoder.load_state_dict(state, strict=True, assign=True)
    encoder = encoder.to(dtype=torch.bfloat16).eval().requires_grad_(False)
    return encoder, build_wan_tokenizer(model_id=str(Path(wan_root)/'google/umt5-xxl'), tokenizer_max_len=128)


def encode(encoder, tokenizer, batches, device):
    """FastWAM.encode_prompt for each batch, with the encoder moved to the GPU one block at a time.

    Layer-major: every batch passes a block before the next block is loaded, so each block
    crosses PCIe once; each batch still runs exactly the kernels of an on-GPU encoder.
    """
    import torch
    tokens = [tokenizer(b, return_mask=True, add_special_tokens=True) for b in batches]
    masks = [m.to(device, dtype=torch.bool) for _, m in tokens]
    with torch.no_grad():
        xs = [encoder.token_embedding(ids).to(device) for ids, _ in tokens]  # row gather: identical on the CPU
        for module in [*encoder.blocks, encoder.norm]:
            module.to(device)
            xs = [module(x, m, pos_bias=None) if module is not encoder.norm else module(x) for x, m in zip(xs, masks)]
            module.to('cpu')
    out = []
    for x, m in zip(xs, masks):
        for i, v in enumerate(m.gt(0).sum(dim=1).long()):
            x[i, v:] = 0
        out.append((x, torch.ones_like(m)))
    return out


def build(args):
    import torch
    encoder, tokenizer = _encoder(args.wan_root)
    table, by_domain = {}, {}
    for item in args.benchmark:
        domain, path = item.split(':', 1)
        by_domain[domain] = prompts(domain, json.loads(Path(path).read_text()))
    names = sorted({p for v in by_domain.values() for p in v})
    rows = names[:args.check_batch] if args.check_batch else []
    # Batch 1 per prompt (the table), a repeat of the first prompt, and one live-like stage-2 batch.
    encoded = encode(encoder, tokenizer, [[p] for p in names] + [[names[0]]] + ([rows] if rows else []), args.device)
    for p, (emb, mask) in zip(names, encoded):
        table[p] = (emb.cpu(), mask.cpu())
    report = dict(prompts=len(table), domains={d: len(v) for d, v in by_domain.items()},
                  shape=list(next(iter(table.values()))[0].shape), dtype='bfloat16', gpu=torch.cuda.get_device_name(args.device))
    report['batch1_repeat_bitwise'] = bool(torch.equal(encoded[len(names)][0].cpu(), table[names[0]][0]))
    if rows:
        # Live serving encodes a stage-2 batch of up to 16 prompts together.
        emb = encoded[len(names)+1][0]
        diffs = [(emb[i:i+1].cpu().float() - table[p][0].float()).abs() for i, p in enumerate(rows)]
        scale = max(table[p][0].float().abs().max().item() for p in rows)
        report['batched'] = dict(batch=len(rows), bitwise_rows=sum(int(d.max().item() == 0) for d in diffs),
                                 max_abs=max(d.max().item() for d in diffs), mean_abs=sum(d.mean().item() for d in diffs)/len(rows),
                                 max_abs_value=scale)
    torch.save(dict(table=table, report=report, template=TEMPLATE), args.out)
    print(json.dumps(report, indent=1), flush=True)


def load(path, device):
    import torch
    doc = torch.load(path, map_location='cpu', weights_only=True)
    return {p: (e.to(device), m.to(device)) for p, (e, m) in doc['table'].items()}


def install(model, table):
    """Replace FastWAM.encode_prompt with a lookup; an unknown prompt raises (there is no T5 to fall back to)."""
    import torch
    def encode_prompt(prompt):
        rows = [table[p] for p in ([prompt] if isinstance(prompt, str) else prompt)]
        return torch.cat([r[0] for r in rows]), torch.cat([r[1] for r in rows])
    model.encode_prompt = encode_prompt


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='action', required=True)
    b = sub.add_parser('build')
    b.add_argument('--wan-root', default='/data/xwam/wan22_5b')
    b.add_argument('--benchmark', action='append', required=True, help='domain:benchmark.json')
    b.add_argument('--out', required=True)
    b.add_argument('--device', default='cuda:0')
    b.add_argument('--check-batch', type=int, default=0)
    args = parser.parse_args()
    build(args)


if __name__ == '__main__':
    main()
