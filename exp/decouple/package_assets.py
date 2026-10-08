"""Stage the gitignored pi05 norm stats locally; never write through a symlink."""
from pathlib import Path
import json
import shutil
import uuid

NORM_RELATIVE = Path('pi05_libero/physical-intelligence/libero/norm_stats.json')


def prepare_assets(repo, source):
    repo, source = Path(repo), Path(source)
    root = repo/'assets'
    if root.is_symlink():
        # Retain the owner's workaround as an artifact, then stage a regular
        # local directory. Copying into the old link would modify the main tree.
        if root.resolve() != source.resolve():
            raise ValueError(f'unexpected assets symlink: {root}')
        backup = repo/'exp/decouple/results'/('owner_assets_symlink_'+uuid.uuid4().hex[:8])
        backup.parent.mkdir(parents=True,exist_ok=True)
        root.rename(backup)
    target = root/NORM_RELATIVE
    if not target.exists():
        original = source/NORM_RELATIVE
        if not original.is_file():
            raise FileNotFoundError(f'required pi05 norm stats absent: {target}; read-only source also absent: {original}')
        target.parent.mkdir(parents=True,exist_ok=True)
        if not target.parent.resolve().is_relative_to(repo.resolve()):
            raise ValueError('assets path escapes the worktree')
        shutil.copy2(original,target)
    if not target.resolve().is_relative_to(repo.resolve()):
        raise ValueError('norm stats must be a local worktree file')
    validate_assets(target)
    return target


def validate_assets(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'required norm stats missing: {path}; regenerate package config to stage them before GPU launch')
    values = json.loads(path.read_text())
    values = values.get('norm_stats',values)
    if not all(key in values and 'mean' in values[key] and 'std' in values[key] for key in ('actions','state')):
        raise ValueError(f'invalid pi05 norm stats: {path}')
