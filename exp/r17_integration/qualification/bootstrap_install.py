"""Bootstrap the *actual P6 installer* from a whole-archive SHA-checked payload."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import types


def main():
    archive, destination, expected = sys.argv[1:4]
    immutable = sys.argv[4:] == ['--immutable']
    with open(archive, 'rb') as f:
        actual = hashlib.file_digest(f, 'sha256').hexdigest()
    if actual != expected:
        raise ValueError('bootstrap archive SHA mismatch')
    for name in ('exp', 'exp.r17_integration', 'exp.r17_integration.ops'):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    with tarfile.open(archive) as tar:
        modules = ('build_client_bundle', 'sync_back', 'deploy') + (('worker_installs',) if immutable else ())
        for short in modules:
            name = 'exp.r17_integration.ops.' + short
            module = types.ModuleType(name)
            module.__package__ = 'exp.r17_integration.ops'
            sys.modules[name] = module
            source = tar.extractfile('exp/r17_integration/ops/' + short + '.py').read()
            exec(compile(source, name, 'exec'), module.__dict__)
    from exp.r17_integration.ops.deploy import install_payload
    if immutable:
        from exp.r17_integration.ops.worker_installs import install_client
        manifest = install_client(archive, destination, expected)
    else:
        manifest = install_payload(archive, destination, expected)
    print(json.dumps(dict(installed=str(Path(destination)), **manifest), sort_keys=True))


if __name__ == '__main__':
    main()
