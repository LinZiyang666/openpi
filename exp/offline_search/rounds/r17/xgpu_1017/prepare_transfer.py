"""CPU-only minimal DP replay file list; never connects or copies bulk assets.

Emits public-rsync file lists and path-relocated job/cache descriptors. No weights
or datasets are duplicated locally. All generated files remain under --output.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import yaml
from artifacts import write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign', default='/data/mw_finetune/eval/fa_dp_f_u_o')
    p.add_argument('--cache-yaml', default='/data/mw_finetune/eval/fa_dp_f_u_o/assets/cache_yamls/n50_sp_chrg_koff.yaml')
    p.add_argument('--output', required=True)
    p.add_argument('--destination', default='/home/exouser/xgpu_1017')
    p.add_argument('--dependency-receipt', default=str(Path(__file__).parent/'evidence/cache_cpu_integration_current.json'))
    a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    campaign=json.loads((Path(a.campaign)/'campaign.json').read_text())
    if campaign['model']!='dp':raise ValueError('minimal transfer currently implements DP only')
    desc=json.loads((Path(campaign['assets'])/'load.json').read_text())
    files=set()
    def add(path):
        path=Path(path).resolve(strict=True)
        if path.is_file():files.add(path)
        else:
            # The only recursive trees admitted here are code, a checkpoint,
            # and a tokenizer/model runtime. No dataset or parent teacher tree.
            for child in path.rglob('*'):
                if child.is_file() and '__pycache__' not in child.parts and child.suffix!='.pyc':files.add(child.resolve())
    add(campaign['runtime'])
    add(desc['checkpoint']);add(desc['text_encoder'])
    for name in ('model_spec.json','model_manifest.json','benchmark.json'):
        add(Path(campaign['assets'])/name)
    cache=yaml.safe_load(Path(a.cache_yaml).read_text())
    recipe=cache['recipe']
    for key in ('fitted_path','stall_path'):
        if recipe.get(key):add(recipe[key])
    store=Path(recipe['store_path'])
    while True:
        if (store/'VIEW.json').exists():
            view=json.loads((store/'VIEW.json').read_text())
            add(store/'VIEW.json');add(store/'row_indices.npy')
            store=(store/view['parent_path']).resolve()
        else:
            add(store/'READY.json');break
    root=Path(a.destination)/'rootfs'
    def relocated(path):return str(root/str(Path(path).resolve()).lstrip('/'))
    for key in ('checkpoint','text_encoder'):desc[key]=relocated(desc[key])
    for key in ('store_path','fitted_path','stall_path'):
        if recipe.get(key):recipe[key]=relocated(recipe[key])
    assets=out/'assets';assets.mkdir()
    write_json(assets/'load.json',desc)
    for name in ('model_spec.json','model_manifest.json','benchmark.json'):
        (assets/name).write_bytes((Path(campaign['assets'])/name).read_bytes())
    (out/'cache.yaml').write_text(yaml.safe_dump(cache))
    job=dict(campaign,assets=str(Path(a.destination)/'prepared/assets'))
    job['runtime']=relocated(campaign['runtime'])
    write_json(out/'job.json',job)
    (out/'rootfs.files').write_text(''.join(str(f).lstrip('/')+'\n' for f in sorted(files)))
    total=sum(f.stat().st_size for f in files)
    # Site-packages are shared through a .pth in DP's nominal venv. Do not copy
    # that empty wrapper. The coordinator can mirror the listed serving island
    # with --exclude __pycache__ / *.pyc; dependency pruning needs a tested lock.
    import importlib.metadata as md
    base=str(md.distribution('torch').locate_file(''))
    from packaging.requirements import Requirement
    from packaging.utils import canonicalize_name
    roots=['torch','torchvision','transformers','diffusers','numpy','scipy','scikit-learn',
           'PyYAML','safetensors','Pillow','einops','tqdm','packaging','psutil','msgpack',
           'websockets','jaxtyping','beartype','sentencepiece','pandas','imageio','rich',
           'jax','jaxlib']  # Also observed during the complete DP CPU load/replay.
    receipt=json.loads(Path(a.dependency_receipt).read_text())
    roots += [name for name in receipt['imported_distributions'] if name not in ('openpi','openpi-client')]
    pending=roots[:];packages={};package_files=set();missing=[]
    while pending:
        name=canonicalize_name(pending.pop())
        if name in packages:continue
        try:dist=md.distribution(name)
        except md.PackageNotFoundError:
            missing.append(name);continue
        packages[name]=dist.version
        for requirement in dist.requires or []:
            req=Requirement(requirement)
            if req.marker is None or req.marker.evaluate({'extra':''}):pending.append(req.name)
        for name_in_dist in dist.files or []:
            source=Path(dist.locate_file(name_in_dist))
            if source.is_file() and source.resolve().is_relative_to(Path(base).resolve()) and source.suffix!='.pyc' and '__pycache__' not in source.parts:
                package_files.add(source.relative_to(base).as_posix())
    (out/'site_packages.files').write_text(''.join(name+'\n' for name in sorted(package_files)))
    package_bytes=sum((Path(base)/name).stat().st_size for name in package_files)
    result=dict(schema='xgpu.dp.transfer.v1',rootfs_files=len(files),rootfs_bytes=total,
        destination=a.destination,runtime_identity=campaign['runtime_identity'],
        site_packages=base,model_and_cache_no_dataset=True,
        runtime=job['runtime'],source_campaign=a.campaign,
        excluded='teacher H5/RGB datasets; optimizer; logs; runtime.tar; pip/uv/HF caches',
        python_version=sys.version.split()[0], python_base_prefix=sys.base_prefix,
        package_versions=packages, package_bytes=package_bytes, missing_optional_roots=missing,
        runtime_environment_note='copy the used serving island and matching base interpreter once; recreate venv using that interpreter; do not copy nonportable python symlinks')
    write_json(out/'TRANSFER.json',result)
    print(json.dumps(result))


if __name__=='__main__':main()
