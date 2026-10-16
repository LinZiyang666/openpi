from __future__ import annotations

from pathlib import Path
import hashlib
import json

from exp.offline_search.rounds.r17.filter_failed.audit_counts import REPO, read, summarize, LEGACY
from exp.offline_search.rounds.r17.filter_failed.build import write

HERE = Path(__file__).resolve().parent
EVAL = Path('/data/mw_finetune/eval')
MT = {
    'pi05': ('pimk1_redc', 'pimk1_fuld'),
    'groot': ('grmk1_red', 'grmk1_ful'),
    'dp': ('dpmk1_redx', 'dpmk1_fulx'),
    'fastwam': ('fwmk1_redx', 'fwmk1_fulx'),
}
LIBERO = {
    'dp': ('knobdevdpred2', 'knobdevdp2'),
    'fastwam': ('knobdevfwred8', 'knobdevfw4'),
    'cosmos': ('knobdevcosred', 'knobdevcos4'),
}
OFF = {'dp': 'fsreddp2', 'fastwam': 'fsredfw', 'cosmos': 'fsredcos'}
PYTHON = {
    ('local', 'pi05'): str(REPO/'.venv/bin/python'),
    ('local', 'dp'): str(REPO/'.venv/bin/python'),
    ('local', 'fastwam'): '/data/mw_finetune/envs/fastwam/bin/python',
    ('h100', 'dp'): '/data/mw_finetune/dp/env/bin/python',
    ('h100', 'fastwam'): '/data/mw_finetune/envs/fastwam/bin/python',
    ('h100', 'groot'): '/home/exouser/gr00t_n15_venv/.venv/bin/python',
    ('h100', 'cosmos'): '/data/cosmos/cosmos-policy/.venv/bin/python',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''): h.update(block)
    return h.hexdigest()


def cells():
    return {(c['model'], c['suite'], c['size']): c for c in read(HERE.parent/'filter_failed/CELLS.json')['cells']}


def dump(path, obj):
    write(path, obj)


def source_jobs():
    for benchmark, mapping in (('mt50', MT), ('libero', LIBERO)):
        for model, sources in mapping.items():
            for mode, source in zip(('reduced', 'full'), sources):
                yield dict(key=f'{model}_{benchmark}_{mode}', model=model, benchmark=benchmark,
                           mode=mode, source=str(EVAL/source), stage='descent' if benchmark=='mt50' else 'dev')


def frozen_inputs(root):
    root = Path(root)
    return {str(p): sha(p) for p in sorted(root.rglob('*')) if p.is_file() and
            '__pycache__' not in p.parts and p.suffix != '.pyc'}
