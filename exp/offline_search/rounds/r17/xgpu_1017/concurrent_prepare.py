"""CPU-only pinning of deployed campaigns; no GPU, workers or lane changes."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import yaml
from artifacts import digest, sha, write_json

SOURCES = {
    'dp': 'fa_ls_dp_f_u_o_l',
    'fastwam': 'fa_ls_fw_f_u_o_l',
    'groot': 'fa_ls_gr_f_u_m',
    'pi05': 'pimk1_fuld',
    'cosmos': 'fa_co_r_u_l',
}


def include_list(files, base):
    """An rsync daemon allowlist; no wildcard directories or followed symlinks."""
    paths = set()
    for path in files:
        rel = Path(path).relative_to(base)
        if '..' in rel.parts:
            raise ValueError('unsafe path')
        paths.add('/' + rel.as_posix())
        for parent in rel.parents:
            if parent != Path('.'):
                paths.add('/' + parent.as_posix() + '/')
    return ''.join(x + '\n' for x in sorted(paths))


def prepare(source, output):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    c = json.loads((source / 'campaign.json').read_text())
    assets = Path(c['local_assets'])
    dest = output / 'assets'
    dest.mkdir()
    pinned = {}
    for name in ('load.json', 'model_spec.json', 'model_manifest.json', 'benchmark.json',
                 'full.yaml', 'manifest.json', 'batching.json', 'ffk_bootstrap.py'):
        path = assets / name
        if path.exists():
            shutil.copyfile(path, dest / name)
            pinned[name] = sha(path)
    seal = json.loads((assets / 'CACHE_STUDY.json').read_text())
    first = seal['groups'][0]
    groups = [r for r in seal['groups'] if r['suite'] == first['suite'] and
              r.get('library_size') == first.get('library_size')]
    # Native pure, fitted cache with knob off, and every sealed knob level for
    # this small library. No refit, YAML setting changes or invented thresholds.
    dependencies = set()
    def add_tree(path):
        path = Path(path).resolve(strict=True)
        for f in ([path] if path.is_file() else path.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts and f.suffix != '.pyc':
                if f.suffix.lower() in ('.h5', '.hdf5', '.tar', '.mp4'):
                    raise ValueError('bulk dataset not allowed in replay assets')
                dependencies.add(f.resolve())
    for row in groups:
        path = assets / row['file']
        target = dest / row['file']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        pinned[row['file']] = sha(path)
        recipe = yaml.safe_load(path.read_text())['recipe']
        for key in ('fitted_path', 'stall_path'):
            if recipe.get(key):
                add_tree(recipe[key])
        store = Path(recipe['store_path'])
        while (store / 'VIEW.json').exists():
            view = json.loads((store / 'VIEW.json').read_text())
            add_tree(store / 'VIEW.json')
            add_tree(store / 'row_indices.npy')
            store = (store / view['parent_path']).resolve()
        add_tree(store / 'READY.json')
    shutil.copyfile(source / 'campaign.json', output / 'source_campaign.json')
    write_json(dest / 'selected_groups.json', groups)
    pinned['selected_groups.json'] = sha(dest / 'selected_groups.json')
    dependencies = sorted(dependencies)
    (output / 'data.files').write_text(''.join(str(p) + '\n' for p in dependencies))
    (output / 'data.include').write_text(include_list(dependencies, Path('/data/mw_finetune')))
    batch = next((c[k]['max_batch'] for k in ('dp_batching', 'fastwam_batching', 'stage_batching')
                  if c.get(k)), 1)
    boot = (dest / 'ffk_bootstrap.py').read_text() if (dest / 'ffk_bootstrap.py').exists() else ''
    pin = dict(schema='xgpu.concurrent.pin.v1', model=c['model'], source=str(source),
        source_campaign_sha256=sha(source / 'campaign.json'), files=pinned,
        runtime_identity=c['runtime_identity'], replicas=c['replicas'], batch=batch,
        clients=c.get('execution_workers', c['workers']), memory_budget_gib=c['memory_budget_gib'],
        mps=c.get('mps'), bootstrap_sha256=c.get('ffk_bootstrap_sha256'),
        segment_clock_overlay='segment_decision' in boot,
        cache_groups=len(groups), data_bytes=sum(p.stat().st_size for p in dependencies),
        scope='one pinned model/domain/K and one library size; not all suites or recipes')
    pin['identity'] = digest(pin)
    write_json(output / 'PIN.json', pin)
    return pin


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=list(SOURCES), required=True)
    p.add_argument('--source')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    print(json.dumps(prepare(a.source or '/data/mw_finetune/eval/' + SOURCES[a.model], a.output)))


if __name__ == '__main__':
    main()
