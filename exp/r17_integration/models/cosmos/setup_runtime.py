"""Copy existing pure Python LIBERO imports into a private model-island path."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil


def setup(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    copied=[]
    for name in ('libero','robosuite','bddl','easydict','robomimic','thop','nltk','future','past'):
        origin,target=source/name,destination/name
        if origin.exists() and not target.exists():
            shutil.copytree(origin,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            copied.append(name)
    config=destination/'config';config.mkdir(exist_ok=True)
    root=destination/'libero/libero'
    doc={'benchmark_root':str(root),'bddl_files':str(root/'bddl_files'),'assets':str(root/'assets'),
         'init_states':str(root/'init_files'),'datasets':str(destination/'libero/datasets')}
    import yaml
    path=config/'config.yaml'
    text=yaml.safe_dump(doc)
    if path.exists() and path.read_text()!=text:
        raise ValueError('private LIBERO config already differs; use another destination')
    path.write_text(text)
    return dict(private_dependencies=str(destination),copied=copied)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',default='/home/weiland/projects/openpi_ext/envs/libero_sim/lib/python3.8/site-packages')
    p.add_argument('--destination',default='/data/cosmos/r17_runtime_deps310');a=p.parse_args()
    print(json.dumps(setup(a.source,a.destination)))
