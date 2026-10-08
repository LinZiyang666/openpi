"""Hash-bound, fail-closed per-task acquisition table (stdlib only)."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

SPEC = 'exp.offline_search.closed_loop.half_everywhere:R12HalfEverywhere'
SELECTION = 'half_everywhere'
SCHEMA = 'offline_search.r12_half_camera.v1'


def load_table(binding, *, cell=None):
    if (not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}
            or not isinstance(binding['path'], str) or not binding['path']
            or not isinstance(binding['sha256'], str) or len(binding['sha256']) != 64):
        raise ValueError('camera_table requires path + SHA256')
    path = Path(binding['path'])
    if not path.is_file():
        raise ValueError(f'missing camera table: {path}')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != binding['sha256']:
        raise ValueError('camera table SHA256 mismatch')
    table = json.loads(data)
    if (table.get('schema') != SCHEMA or not isinstance(table.get('camera'), dict)
            or set(table['camera']) != set(map(str, range(10)))
            or any(v not in ('both', 'wrist') for v in table['camera'].values())):
        raise ValueError('camera table must specify exactly ten wrist/both tasks')
    if cell is not None and table.get('cell') != cell:
        raise ValueError('camera table cell mismatch')
    return table


def validate_kwargs(kwargs, *, check_file=True):
    if not isinstance(kwargs, dict) or 'camera_table' not in kwargs:
        raise ValueError('R12HalfEverywhere requires camera_table path + SHA256')
    if kwargs.get('knob_method', 'off') not in ('off', 'periodic'):
        raise ValueError('R12HalfEverywhere supports off/periodic only')
    if 'method' in kwargs:
        raise ValueError('R12HalfEverywhere uses knob_method, not method')
    binding = kwargs['camera_table']
    if (not isinstance(binding, dict) or set(binding) != {'path', 'sha256'}
            or not isinstance(binding['path'], str) or not binding['path']
            or not isinstance(binding['sha256'], str) or len(binding['sha256']) != 64
            or any(c not in '0123456789abcdef' for c in binding['sha256'])):
        raise ValueError('camera_table requires path + SHA256')
    if check_file:
        load_table(kwargs['camera_table'])


def validate_fitted(method, kwargs):
    table = load_table(kwargs['camera_table'], cell=method.table_cell)
    if (method.camera_table != kwargs['camera_table'] or method.camera != table['camera']
            or method.knob_method != kwargs.get('knob_method', 'off')
            or method.target != kwargs.get('target')):
        raise ValueError('fitted half-everywhere configuration mismatch')
