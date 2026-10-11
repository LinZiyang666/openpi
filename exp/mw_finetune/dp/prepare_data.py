"""Build exact 480→96 MT50 images; reuse C's audited raw vector index."""
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


def convert(job):
    src, dest = map(Path, job)
    if dest.exists():
        with sqlite3.connect(f"file:{dest}?mode=ro", uri=True) as db:
            row = db.execute("SELECT value FROM metadata WHERE key='complete'").fetchone()
            if row and row[0] == str(src.stat().st_size):
                return dest.name
        raise RuntimeError(f"Unverified existing shard: {dest}")
    sums = np.zeros(96, np.float64)
    squares = np.zeros(96, np.float64)
    lo, hi = np.ones(96, np.float32), np.zeros(96, np.float32)
    count = 0
    tmp = dest.with_suffix(".partial")
    if tmp.exists():
        tmp.rename(tmp.with_suffix(f".interrupted-{tmp.stat().st_mtime_ns}"))
    with sqlite3.connect(tmp) as db:
        db.execute("CREATE TABLE images(id INTEGER PRIMARY KEY, png BLOB NOT NULL)")
        db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        for batch in pq.ParquetFile(src).iter_batches(batch_size=64, columns=["index", "observation.image"]):
            records = []
            for row in batch.to_pylist():
                with Image.open(io.BytesIO(row["observation.image"]["bytes"])) as im:
                    rgb = np.asarray(im.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR))
                arr = rgb.astype(np.float64) / 255
                sums += arr.sum((0, 2)); squares += (arr**2).sum((0, 2))
                lo = np.minimum(lo, arr.min((0, 2))); hi = np.maximum(hi, arr.max((0, 2)))
                count += 96 * 3
                buf = io.BytesIO(); Image.fromarray(rgb).save(buf, format="PNG", compress_level=1)
                records.append((int(row["index"]), buf.getvalue()))
            db.executemany("INSERT INTO images VALUES (?,?)", records)
        db.execute("INSERT INTO metadata VALUES('image_stats',?)", (json.dumps(dict(sum=sums.tolist(),
            squares=squares.tolist(), min=lo.tolist(), max=hi.tolist(), count=count)),))
        db.execute("INSERT INTO metadata VALUES('complete',?)", (str(src.stat().st_size),))
        db.commit()
    tmp.replace(dest)
    return dest.name


def stats(arr):
    arr = np.asarray(arr, np.float64)
    return {"mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(), "min": arr.min(0).tolist(),
            "max": arr.max(0).tolist(), "q01": np.quantile(arr, .01, axis=0).tolist(),
            "q99": np.quantile(arr, .99, axis=0).tolist()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("/data/mw_finetune/datasets/metaworld_mt50"))
    ap.add_argument("--index-source", type=Path, default=Path("/data/mw_finetune/fastwam/mt50_224"))
    ap.add_argument("--output", type=Path, default=Path("/data/mw_finetune/dp/mt50_96"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((a.index_source / "manifest.json").read_text())
    if manifest["partial"] or manifest["frames"] != 204806 or manifest["episodes"] != 2500 or len(manifest["tasks"]) != 50:
        raise ValueError("Complete audited C index required")
    jobs = []
    for name in manifest["shards"]:
        chunk, file = name.removeprefix("part-").removesuffix(".sqlite").split("-file-")
        jobs.append((a.source / "data" / chunk / f"file-{file}.parquet", a.output / name))
    with ProcessPoolExecutor(a.workers) as pool:
        for i, name in enumerate(pool.map(convert, jobs), 1):
            if i % 25 == 0 or i == len(jobs):
                print(json.dumps({"complete_shards": i, "total": len(jobs)}), flush=True)
    index_bytes = (a.index_source / "index.npz").read_bytes()
    (a.output / "index.npz").write_bytes(index_bytes)
    index = np.load(a.output / "index.npz", allow_pickle=False)
    sums, squares = np.zeros(96), np.zeros(96)
    lo, hi, count = np.ones(96), np.zeros(96), 0
    for _, path in jobs:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
            s = json.loads(db.execute("SELECT value FROM metadata WHERE key='image_stats'").fetchone()[0])
        sums += s["sum"]; squares += s["squares"]; count += s["count"]
        lo = np.minimum(lo, s["min"]); hi = np.maximum(hi, s["max"])
    mean = sums / count
    manifest.update(format="dp-mt50-96-png-v1", image_size=[96, 96],
        image_stats={"mean": mean.tolist(), "std": np.sqrt(np.maximum(squares / count - mean**2, 0)).tolist(),
                     "min": lo.tolist(), "max": hi.tolist()},
        index_sha256=hashlib.sha256(index_bytes).hexdigest(),
        external_stats={"state": stats(index["state"]), "actions": stats(index["action"])},
        resize="PIL bilinear 480 to96 directly; no flip/rotation", source=str(a.source))
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"prepared": str(a.output), "frames": manifest["frames"], "index_sha256": manifest["index_sha256"]}))


if __name__ == "__main__":
    main()
