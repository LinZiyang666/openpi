"""Resume-safe, lossless 224 RGB preparation of the pinned MT50 v3 dataset.

PNG blobs live in SQLite shards; 4-D arrays stay float32 without transforms.
The original dataset remains untouched. No full-resolution copy is needed on H100.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from .common import DATA_REVISION, disk_gate, resize_rgb


def convert_file(args):
    source, dest = map(Path, args)
    if dest.exists():
        with sqlite3.connect(dest) as db:
            size = db.execute("SELECT value FROM metadata WHERE key='source_size'").fetchone()
            if size and int(size[0]) == source.stat().st_size:
                return str(dest)
        raise RuntimeError(f"Existing shard {dest} does not match source; refusing replacement")
    disk_gate(dest.parent, 0.2, floor_gb=5)
    tmp = dest.with_suffix(".sqlite.tmp")
    # A prior interrupted attempt is replaced, never mistaken for a complete shard.
    with sqlite3.connect(tmp) as db:
        db.execute("DROP TABLE IF EXISTS images")
        db.execute("DROP TABLE IF EXISTS metadata")
        db.execute("CREATE TABLE images (id INTEGER PRIMARY KEY, png BLOB NOT NULL)")
        db.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        columns = ["index", "observation.image"]
        for batch in pq.ParquetFile(source).iter_batches(batch_size=64, columns=columns):
            records = []
            for row in batch.to_pylist():
                stored = row["observation.image"]
                if stored["bytes"] is None:
                    raise ValueError("MT50 images must be embedded in parquet")
                with Image.open(io.BytesIO(stored["bytes"])) as image:
                    rgb = resize_rgb(np.asarray(image.convert("RGB")))
                buf = io.BytesIO()
                Image.fromarray(rgb).save(buf, format="PNG", compress_level=1)
                records.append((row["index"], buf.getvalue()))
            db.executemany("INSERT INTO images VALUES (?,?)", records)
        db.execute("INSERT INTO metadata VALUES ('source_size',?)", (str(source.stat().st_size),))
        db.commit()
    tmp.replace(dest)
    return str(dest)


def prepare(source, destination, workers=8, limit_files=None, partial=False):
    source, dest = Path(source), Path(destination)
    dest.mkdir(parents=True, exist_ok=True)
    paths = sorted(source.glob("data/chunk-*/*.parquet"))
    if limit_files:
        paths = paths[:limit_files]
    if not paths:
        raise FileNotFoundError("No MT50 parquet files")
    jobs = [(p, dest / f"part-{p.parent.name}-{p.stem}.sqlite") for p in paths]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, path in enumerate(pool.map(convert_file, jobs), 1):
            print(json.dumps({"prepared_shards": i, "total": len(jobs), "path": path}), flush=True)
    cols = ["index", "episode_index", "frame_index", "task_index", "observation.state", "action"]
    tables = [pq.read_table(p, columns=cols).to_pydict() for p in paths]
    values = {k: np.asarray([v for t in tables for v in t[k]], dtype=np.float32 if k in ("action", "observation.state") else np.int64) for k in cols}
    part = np.concatenate([np.full(len(t["index"]), i, dtype=np.int32) for i, t in enumerate(tables)])
    order = np.argsort(values["index"])
    values = {k: v[order] for k, v in values.items()}
    part = part[order]
    if not np.array_equal(values["index"], np.arange(len(part))):
        raise ValueError("Input files are incomplete/noncontiguous; finish the pinned dataset download")
    info = json.loads((source / "meta/info.json").read_text())
    if not partial and len(part) != info["total_frames"]:
        raise ValueError(f"Expected {info['total_frames']} frames, found {len(part)}; use --partial only for smoke")
    states, actions = values.pop("observation.state"), values.pop("action")
    if states.shape != (len(part), 4) or actions.shape != (len(part), 4):
        raise ValueError("MT50 state/action must be 4-D")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite source state/action")
    np.savez(dest / "index.npz", **values, part=part, state=states, action=actions)
    task_rows = pq.read_table(source / "meta/tasks.parquet").to_pylist()
    tasks = {str(r["task_index"]): r.get("task", r.get("__index_level_0__")) for r in task_rows}
    manifest = {"format": "fastwam-mt50-png-v1", "revision": DATA_REVISION, "repo_id": "lerobot/metaworld_mt50",
        "frames": len(part), "episodes": len(np.unique(values["episode_index"])), "partial": partial,
        "fps": info["fps"], "image_size": [224, 224], "tasks": tasks,
        "stats": json.loads((source / "meta/stats.json").read_text()),
        "shards": [Path(p).name for _, p in jobs], "source_files": [str(p.relative_to(source)) for p in paths],
        "index_sha256": hashlib.sha256((dest / "index.npz").read_bytes()).hexdigest()}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("frames", "episodes", "partial", "index_sha256")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="/data/mw_finetune/datasets/metaworld_mt50")
    parser.add_argument("--destination", default="/data/mw_finetune/fastwam/mt50_224")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-files", type=int)
    parser.add_argument("--partial", action="store_true")
    a = parser.parse_args()
    prepare(a.source, a.destination, a.workers, a.limit_files, a.partial)
