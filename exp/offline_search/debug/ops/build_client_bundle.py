"""Deterministic R8 client payload; no model, simulator or remote imports."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile


ROOT = Path(__file__).resolve().parents[4]
PACKAGE = ROOT / "exp/offline_search/debug"
REMOTE = "/scratch/zixuans8/openpi_trace"
MARKER = b'"""Preserve installed namespace portions."""\nfrom pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n'


def build(out, run=None, arms=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    sources = [(p, str(p.relative_to(ROOT)), "replace_owned") for sub in ("client", "transport", "backfill")
               for p in sorted((PACKAGE / sub).rglob("*.py"))]
    sources.append((PACKAGE / "schema.py", "exp/offline_search/debug/schema.py", "replace_owned"))
    sources.append((PACKAGE / "ops/remote/run_arm_debug.sh", "os_cl/run_arm_debug.sh", "replace_owned"))
    for rel in ("exp", "exp/offline_search", "exp/offline_search/debug"):
        sources.append((None, rel + "/__init__.py", "marker_if_missing"))
    if run is not None:
        rows = json.loads((Path(run) / "arms.json").read_text())
        available = {r["arm"] for r in rows}
        selected = set(arms or available)
        if selected - available:
            raise ValueError("unknown bundle arms")
        from exp.offline_search.debug.transport.protocol import identifier
        for arm in sorted(selected):
            identifier(arm, arm=True)
            for filename in (arm + ".yaml", "matrix_" + arm + ".yaml"):
                sources.append((Path(run) / "config" / filename, "os_cl/cfg/" + filename, "replace_owned"))
    records = []
    for source, relative, operation in sources:
        data = source.read_bytes() if source is not None else MARKER
        dest = out / "payload" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        records.append(dict(relative_path=relative, remote_path=REMOTE + "/" + relative, operation=operation,
                            bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
    manifest = dict(schema="osdebug.client_bundle.v1", remote_root=REMOTE, files=records,
                    protected=["os_cl/run_arm.sh", "os_cl/run_gtp_subset.py", "os_cl/count.py"])
    (out / "client_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (out / "install_client_payload.py").write_bytes((PACKAGE / "ops/install_client_payload.py").read_bytes())
    archive = out / "client_bundle.tar"
    with tarfile.open(str(archive), "w") as tf:
        for path in sorted(out.rglob("*")):
            if not path.is_file() or path == archive:
                continue
            data = path.read_bytes()
            entry = tarfile.TarInfo(str(path.relative_to(out)))
            entry.size, entry.mtime, entry.mode = len(data), 0, 0o644
            tf.addfile(entry, io.BytesIO(data))
    (out / "client_bundle.sha256").write_text(hashlib.sha256(archive.read_bytes()).hexdigest() + "  client_bundle.tar\n")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-root", type=Path)
    ap.add_argument("--arms", nargs="*")
    a = ap.parse_args()
    manifest = build(a.out, a.run_root, a.arms)
    print(json.dumps(dict(files=len(manifest["files"]), bundle=str(a.out / "client_bundle.tar"))))


if __name__ == "__main__":
    main()
