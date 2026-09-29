"""Build a minimal, deterministic client tar; no remote access or model imports."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile

HERE = Path(__file__).resolve().parent
REMOTE = "/scratch/zixuans8/openpi_trace/"
PREFIX = "exp/offline_search/rounds/r06/p3_profiling/"
# Install new dependencies before publishing the driver entrypoint.
MODULES = ("stream_protocol", "dispatch_fence", "worker_v2", "telemetry", "snapshots", "client_compat", "client_preflight", "stream_sink", "run_gtp_v2")
MARKER = b'"""P3 client package; preserve other namespace portions."""\nfrom pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n'


def build(out, run=None, arms=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    entries = [(HERE/(x+".py"), PREFIX+x+".py", "replace_owned", None) for x in MODULES]
    entries.append((HERE/"run_arm_v2.sh", "os_cl/run_arm_v2.sh", "replace_owned", None))
    parts = PREFIX.rstrip("/").split("/")
    for i in range(1, len(parts)+1):
        rel = "/".join(parts[:i])+"/__init__.py"
        entries.append((None, rel, "marker_if_missing", MARKER))
    if run is not None:
        run = Path(run)
        rows = json.loads((run/"arms.json").read_text())
        selected = set(arms or [r["arm"] for r in rows])
        if selected - {r["arm"] for r in rows}:
            raise ValueError("unknown arm")
        for name in sorted(selected):
            if not re.fullmatch(r"[A-Za-z0-9_]+", name):
                raise ValueError("unsafe arm")
            for filename in (name+".yaml", "matrix_"+name+".yaml"):
                entries.append((run/"config"/filename, "os_cl/cfg/"+filename, "replace_owned", None))
    rows = []
    for source, rel, operation, generated in entries:
        data = source.read_bytes() if source is not None else generated
        target = out / "payload" / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append(dict(local_path=str(source) if source is not None else str(target.resolve()),
                         remote_path=REMOTE+rel, relative_path=rel, operation=operation,
                         sha256=hashlib.sha256(data).hexdigest(), bytes=len(data)))
    manifest = dict(schema="r6p3.client_bundle.v1", remote_root=REMOTE.rstrip("/"), files=rows,
                    protected=["os_cl/run_arm.sh", "os_cl/run_gtp_subset.py", "os_cl/count.py"],
                    stock_copy="os_cl/run_arm.sh -> os_cl/run_arm.stock.sh (on remote, unchanged source)")
    (out/"client_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    (out/"install_client_payload.py").write_bytes((HERE/"install_client_payload.py").read_bytes())
    tarpath = out/"client_bundle.tar"
    with tarfile.open(tarpath, "w") as tf:
        for f in sorted(out.rglob("*")):
            if not f.is_file() or f == tarpath:
                continue
            data = f.read_bytes()
            info = tarfile.TarInfo(str(f.relative_to(out)))
            info.size, info.mtime, info.mode = len(data), 0, 0o644
            tf.addfile(info, io.BytesIO(data))
    (out/"client_bundle.sha256").write_text(hashlib.sha256(tarpath.read_bytes()).hexdigest()+"  client_bundle.tar\n")
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-root", type=Path)
    ap.add_argument("--arms", nargs="*")
    a = ap.parse_args()
    report = build(a.out, a.run_root, a.arms)
    print(json.dumps(dict(files=len(report["files"]), bundle=str(a.out/"client_bundle.tar"))))


if __name__ == "__main__":
    main()
