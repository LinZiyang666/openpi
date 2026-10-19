"""Portable, pickle-free tensor trees and content identities."""
from __future__ import annotations
import dataclasses
import hashlib
import json
from pathlib import Path
import numpy as np


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu()
        if str(value.dtype) == 'torch.bfloat16':
            value = value.float()
        value = value.numpy()
    return np.array(value, copy=True)


def pack(tree, arrays, dtypes=None):
    if hasattr(tree, 'shape') and hasattr(tree, 'dtype'):
        key = f'a{len(arrays):06d}'
        arrays[key] = array(tree)
        if arrays[key].dtype.hasobject:
            raise ValueError('object arrays are forbidden')
        if dtypes is not None:
            dtypes[key] = str(tree.dtype)
        return {'$array': key}
    if dataclasses.is_dataclass(tree):
        return {f.name: pack(getattr(tree, f.name), arrays, dtypes) for f in dataclasses.fields(tree)}
    if isinstance(tree, dict):
        if '$array' in tree:
            raise ValueError('reserved tree key')
        return {str(k): pack(tree[k], arrays, dtypes) for k in sorted(tree, key=str)}
    if isinstance(tree, (list, tuple)):
        return [pack(v, arrays, dtypes) for v in tree]
    if isinstance(tree, np.generic):
        return tree.item()
    if tree is None or isinstance(tree, (bool, int, float, str)):
        return tree
    raise TypeError(f'unsupported portable value: {type(tree)}')


def unpack(tree, arrays):
    if isinstance(tree, dict):
        if set(tree) == {'$array'}:
            return arrays[tree['$array']].copy()
        return {k: unpack(v, arrays) for k, v in tree.items()}
    if isinstance(tree, list):
        return [unpack(v, arrays) for v in tree]
    return tree


def save_tree(path, tree):
    path = Path(path)
    arrays, dtypes = {}, {}
    encoded = pack(tree, arrays, dtypes)
    np.savez(path.with_suffix('.npz'), **arrays)
    write_json(path.with_suffix('.json'), dict(tree=encoded, native_dtypes=dtypes,
                                             arrays_sha256=sha(path.with_suffix('.npz'))))
    return sha(path.with_suffix('.json'))


def load_tree(path):
    path = Path(path)
    meta = json.loads(path.with_suffix('.json').read_text())
    if sha(path.with_suffix('.npz')) != meta['arrays_sha256']:
        raise ValueError('tensor archive digest mismatch')
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as data:
        return unpack(meta['tree'], data)


def tensor_leaves(tree, prefix=''):
    if hasattr(tree, 'shape') and hasattr(tree, 'dtype'):
        yield prefix, array(tree)
    elif isinstance(tree, dict):
        for key, value in tree.items():
            yield from tensor_leaves(value, prefix + '/' + str(key))
    elif isinstance(tree, (list, tuple)):
        for key, value in enumerate(tree):
            yield from tensor_leaves(value, prefix + '/' + str(key))


def tree_identity(tree):
    arrays = {}
    encoded = pack(tree, arrays)
    return digest(dict(tree=encoded, arrays={k: dict(shape=list(v.shape), dtype=str(v.dtype),
        sha256=hashlib.sha256(v.tobytes()).hexdigest()) for k, v in arrays.items()}))
