"""Validate a frozen PILOT v2 manifest and refit by config, with no runtime edits.

CPU command: python -m exp.offline_search.rounds.r15.qk.impl.swap --size 50
Add --refit after both final PILOT artifacts are frozen; rerun prepare/checks.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from .build import HERE, ASTRA, REPO, DERIVED, build, write
from .encoder import FrozenEncoder, sha


def configure(size):
    directory=ASTRA/f'round2/pilot/pi05_l10_{size}'
    manifest=directory/'manifest.json'
    data=json.loads(manifest.read_text())
    if (data['format']!='r15_qk_student_v2' or data['cell']!=f'pi05_l10_{size}'
            or data['cpu_only'] is not True or data['blindness_pass'] is not True):
        raise ValueError('PILOT is not a frozen image-driven CPU artifact for this cell')
    for name,digest in data['files'].items():
        if Path(name).name!=name or sha(directory/name)!=digest:
            raise ValueError('changed PILOT artifact file: '+name)
    subset=REPO/f'exp/offline_search/rounds/r10/subsets/pi05_l10_{size}.json'
    if data['provenance']['subset_sha256']!=sha(subset):
        raise ValueError('PILOT trained on another cell subset')
    with np.load(directory/'library_features.npz',allow_pickle=False) as z:
        rows=z['rows'].copy()
        assert z['features'].shape==(len(rows),128) and len(rows)==data['rows']
    if hashlib.sha256(rows.tobytes()).hexdigest()!=data['parent_row_sha256']:
        raise ValueError('PILOT parent row binding differs')
    spec=json.loads((directory/'encoder_config.json').read_text())
    if spec['completed_epochs']!=data['epochs'] or spec['completed_epochs']!=spec['declared_epochs']:
        raise ValueError('partial PILOT checkpoint')
    spec.update(cpu_affinity=list(range(2,10))+list(range(46,54)),candidate_name=data['name'],
        candidate_manifest=str(manifest),candidate_manifest_sha256=sha(manifest),
        library_features_reference=str(directory/'library_features.npz'),
        library_features_reference_sha256=data['files']['library_features.npz'])
    # Pinning is runtime scheduling metadata, not a different representation.
    spec.setdefault('representation_sha256',sha(directory/'student.ts'))
    output=HERE/f'candidate_configs/pi05_l10_{size}.json'
    write(output,spec)
    return output


def preserve_round1(size):
    archive=HERE/f'round1/artifacts/pi05_l10_{size}'
    if not archive.exists():
        source=HERE/f'artifacts/pi05_l10_{size}'
        shutil.copytree(source,archive)
        data=DERIVED/f'pi05_l10_{size}'
        saved=DERIVED/f'round1/pi05_l10_{size}'
        saved.mkdir(parents=True,exist_ok=True)
        for p in data.glob('*.npz'):shutil.copy2(p,saved/p.name)
        manifest=json.loads((archive/'manifest.json').read_text())
        manifest['library_features']=str(saved/'library.npz')
        for level in manifest['records']:
            manifest['records'][level]['path']=str(archive/f'{level}.pkl')
        write(archive/'manifest.json',manifest)
        for name in ('plugin_replay.json','verification.json','gate_report.json','HANDBACK.md'):
            path=HERE/name
            if path.is_file():shutil.copy2(path,HERE/'round1'/name)
        for p in (HERE/'evaluation').glob(f'*_{size}.json'):
            dest=HERE/'round1/evaluation'/p.name
            report=json.loads(p.read_text())
            if report.get('data') and Path(report['data']).is_relative_to(data):
                report['data']=str(saved/Path(report['data']).relative_to(data))
            write(dest,report)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--size',type=int,choices=(50,200),required=True)
    ap.add_argument('--refit',action='store_true')
    a=ap.parse_args();config=configure(a.size)
    print('PILOT_CONFIG_READY',config,flush=True)
    if a.refit:
        preserve_round1(a.size)
        build(a.size,config)
