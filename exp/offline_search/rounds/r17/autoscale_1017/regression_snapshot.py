"""Read-only dependency mirror with the 19 SHA-verified original sources.

Unchanged dependencies are linked, not omitted or bulk-copied. At directories
containing restored modules, Python sources are copied so __file__/ROOT stay
inside the mirror. No source in the working tree is replaced.
"""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
BACKUPS = Path('/home/weiland/.claude/jobs/35f2869b/tmp/autoscale_backup')


def create(target):
    target = Path(target).absolute()
    target.mkdir(exist_ok=False)
    pins = json.loads((HERE/'PATCH_BASE.json').read_text())
    # Copy all source files (including examples and transitive imports), but
    # keep datasets/model weights linked. A payload builder legitimately
    # rejects source symlinks, so linking entire code packages is insufficient.
    listing = subprocess.check_output(['rg', '--files', '--hidden', '-g', '*.py',
        '-g', '*.sh', '-g', '*.toml', '-g', '!**/autoscale_1017/tests/tmp*/**',
        '-g', '!**/autoscale_1017/baseline_*/**'], cwd=ROOT, text=True)
    sources = {Path(p) for p in listing.splitlines()}
    branches = {Path('.')}
    for name in set(map(Path, pins)) | sources:
        branches.update(name.parents)
    verified = {}
    for name, pin in pins.items():
        original = BACKUPS/name.replace('/', '__')
        digest = hashlib.sha256(original.read_bytes()).hexdigest()
        assert digest == pin['sha256'], name
        verified[name] = digest
    def mirror(relative):
        destination = target/relative
        destination.mkdir(exist_ok=True)
        for source in sorted((ROOT/relative).iterdir()):
            name = relative/source.name
            out = target/name
            if name in branches:
                mirror(name)
            elif str(name) in pins:
                shutil.copyfile(BACKUPS/str(name).replace('/', '__'), out)
            elif name in sources and source.is_file():
                shutil.copyfile(source, out)
            elif source.name != '__pycache__':
                out.symlink_to(source, target_is_directory=source.is_dir())
    mirror(Path('.'))
    (target/'BASELINE.json').write_text(json.dumps(dict(source=str(ROOT), backups=str(BACKUPS),
        hashes=verified, dependency_policy='full unchanged directory/file links; Python siblings copied'), indent=2)+'\n')
    return target


if __name__ == '__main__':
    import sys
    print(create(sys.argv[1]))
