"""CPU-only sealed standalone descriptor; the formal launcher seals it itself."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path


def main():
    from .loader import LoadSpec, cpu_manifest
    from .protocols import benchmark_profile
    from exp.r17_integration.ops.formal_eval import benchmark
    from exp.r17_integration.common.provider import AdapterProvider, register_provider
    from exp.r17_integration.common.qualification import config_for_probe
    from .manifest import make_sampler
    import yaml
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',default='/data/mw_finetune/models/cosmos_libero')
    p.add_argument('--vendor',default=LoadSpec.__dataclass_fields__['vendor_root'].default)
    p.add_argument('--output',required=True);p.add_argument('--official-50',action='store_true');a=p.parse_args()
    output=Path(a.output);output.mkdir(parents=True,exist_ok=False)
    profile=benchmark_profile(benchmark(),a.official_50)
    spec=LoadSpec(a.checkpoint,vendor_root=a.vendor,benchmark_protocols=(profile.identity,))
    manifest=cpu_manifest(spec)
    spec=replace(spec,expected_digests=dict(manifest.content_digests))
    provider=AdapterProvider(manifest.spec.model_id,manifest.spec,lambda:None,make_sampler)
    register_provider(provider)
    config=config_for_probe(provider,5,16);config.enabled=False;config.checkpoints={};config.recipe.enabled=False
    config.recipe.ordered_tasks=list(profile.ordered_tasks);config.recipe.task_table_identity=profile.task_table_identity
    for name,value in (('load.json',asdict(spec)),('model_manifest.json',asdict(manifest)),('benchmark.json',profile.manifest())):
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    (output/'full.yaml').write_text(yaml.safe_dump(asdict(config)))
    print(output)


if __name__=='__main__':main()
