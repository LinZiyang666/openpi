"""Extract each LIBERO arm's sealed formal workers.tar (CPU, safe members only)."""
import hashlib, json, sys, tarfile
from pathlib import Path
OUT = Path(__file__).resolve().parents[1]
for model in sys.argv[1:]:
    for arm in ('pure', 'cache'):
        d = OUT / model / arm
        source = Path(json.loads((d / 'PREREGISTRATION.json').read_text())['source'])
        tar = source / 'workers.tar'
        digest = hashlib.sha256(tar.read_bytes()).hexdigest()
        target = d / 'worker_payload'
        files = {}
        with tarfile.open(tar) as tf:
            for m in tf.getmembers():
                rel = Path(m.name)
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError('unsafe member ' + m.name)
                if m.isdir():
                    continue
                if not m.isfile():
                    raise ValueError('non-regular member ' + m.name)
                raw = tf.extractfile(m).read()
                q = target / rel
                q.parent.mkdir(parents=True, exist_ok=True)
                if q.exists() and q.read_bytes() != raw:
                    raise ValueError('payload differs ' + m.name)
                q.write_bytes(raw)
                files[m.name] = hashlib.sha256(raw).hexdigest()
        (d / 'WORKER_PAYLOAD.json').write_text(json.dumps(dict(source=str(tar), archive_sha256=digest, files=files), indent=2, sort_keys=True) + '\n')
        print(model, arm, tar, digest[:16], len(files))
