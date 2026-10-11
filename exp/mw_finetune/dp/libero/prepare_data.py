"""Pinned physical-intelligence/libero -> paired96 PNGs, raw8/7 vectors.

Resumes completed per-episode SQLite shards. Partial probes never masquerade as
the full1693-episode mixed-suite dataset.
"""
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

from ..prepare_data import stats

REPO = "physical-intelligence/libero"
REVISION = "a4336d589d589045d1c56423ffdf3b88a0e19b1f"


def convert(job):
    source, target = map(Path, job)
    if target.exists():
        with sqlite3.connect(f"file:{target}?mode=ro", uri=True) as db:
            if db.execute("SELECT value FROM metadata WHERE key='source_bytes'").fetchone()[0] != str(source.stat().st_size):
                raise ValueError(f"Source changed: {target}")
            return target.name
    partial = target.with_suffix(".partial")
    if partial.exists():
        partial.rename(partial.with_name(partial.name + f".interrupted-{partial.stat().st_mtime_ns}"))
    summaries = []
    for _ in range(2):
        summaries.append(dict(sum=np.zeros(96), squares=np.zeros(96), min=np.ones(96), max=np.zeros(96), count=0))
    with sqlite3.connect(partial) as db:
        db.execute("CREATE TABLE images(id INTEGER PRIMARY KEY, png BLOB, wrist_png BLOB)")
        db.execute("CREATE TABLE vectors(id INTEGER PRIMARY KEY, episode INTEGER, task INTEGER, state BLOB, action BLOB)")
        db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT)")
        for batch in pq.ParquetFile(source).iter_batches(batch_size=32):
            for row in batch.to_pylist():
                images = []
                for key, summary in zip(("image", "wrist_image"), summaries):
                    with Image.open(io.BytesIO(row[key]["bytes"])) as image:
                        rgb = np.asarray(image.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR))
                    pixels = rgb.astype(np.float64) / 255
                    summary["sum"] += pixels.sum((0, 2)); summary["squares"] += (pixels ** 2).sum((0, 2))
                    summary["min"] = np.minimum(summary["min"], pixels.min((0, 2)))
                    summary["max"] = np.maximum(summary["max"], pixels.max((0, 2)))
                    summary["count"] += 96 * 3
                    buffer = io.BytesIO(); Image.fromarray(rgb).save(buffer, format="PNG", compress_level=1)
                    images.append(buffer.getvalue())
                state, action = np.asarray(row["state"], np.float32), np.asarray(row["actions"], np.float32)
                if state.shape != (8,) or action.shape != (7,) or not np.isfinite(state).all() or not np.isfinite(action).all():
                    raise ValueError("Invalid raw LIBERO vectors")
                db.execute("INSERT INTO images VALUES(?,?,?)", (row["index"], *images))
                db.execute("INSERT INTO vectors VALUES(?,?,?,?,?)", (row["index"], row["episode_index"], row["task_index"], state.tobytes(), action.tobytes()))
        for key, value in [("source_bytes", str(source.stat().st_size)), ("image_stats", json.dumps([
            {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in s.items()} for s in summaries]))]:
            db.execute("INSERT INTO metadata VALUES(?,?)", (key, value))
        db.commit()
    partial.rename(target)
    return target.name


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", type=Path, default=Path("/data/mw_finetune/dp/libero_source"))
    ap.add_argument("--output", type=Path, default=Path("/data/mw_finetune/dp/libero_96"))
    ap.add_argument("--download", action="store_true", help="Download missing pinned source files into E's local disk")
    ap.add_argument("--download-workers", type=int, default=2, help="Keep anonymous HF request load bounded")
    ap.add_argument("--probe-episodes", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.download:
        from huggingface_hub import snapshot_download
        patterns = ["meta/*", "README.md", "data/*/*.parquet"]
        if a.probe_episodes:
            patterns = ["meta/*", "README.md"] + [f"data/chunk-{i//1000:03d}/episode_{i:06d}.parquet" for i in range(a.probe_episodes)]
        snapshot_download(REPO, repo_type="dataset", revision=REVISION, local_dir=a.source, allow_patterns=patterns, max_workers=a.download_workers)
    info = json.loads((a.source / "meta/info.json").read_text())
    tasks = {str(row["task_index"]): row["task"] for row in map(json.loads, (a.source / "meta/tasks.jsonl").read_text().splitlines())}
    if (info["total_episodes"], info["total_frames"], len(tasks)) != (1693, 273465, 40):
        raise ValueError("Unexpected source revision/layout")
    count = a.probe_episodes or info["total_episodes"]
    if not 1 <= count <= info["total_episodes"]:
        ap.error("probe-episodes must be0 (full) or1..1693")
    a.output.mkdir(parents=True, exist_ok=True)
    jobs = [(a.source / info["data_path"].format(episode_chunk=i//1000, episode_index=i), a.output / f"episode-{i:06d}.sqlite") for i in range(count)]
    missing = [str(source) for source, _ in jobs if not source.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} source episodes absent; use --download. First: {missing[0]}")
    with ProcessPoolExecutor(a.workers) as pool:
        for i, name in enumerate(pool.map(convert, jobs), 1):
            if i % 50 == 0 or i == count:
                print(json.dumps({"converted": i, "total": count}), flush=True)
    index = {k: [] for k in ("index", "episode_index", "task_index", "part", "state", "action")}
    totals = [dict(sum=np.zeros(96), squares=np.zeros(96), min=np.ones(96), max=np.zeros(96), count=0) for _ in range(2)]
    for part, (_, target) in enumerate(jobs):
        with sqlite3.connect(f"file:{target}?mode=ro", uri=True) as db:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError(f"Broken SQLite: {target}")
            for frame, episode, task, state, action in db.execute("SELECT * FROM vectors ORDER BY id"):
                for key, value in dict(index=frame, episode_index=episode, task_index=task, part=part,
                                       state=np.frombuffer(state, np.float32).copy(), action=np.frombuffer(action, np.float32).copy()).items():
                    index[key].append(value)
            for total, summary in zip(totals, json.loads(db.execute("SELECT value FROM metadata WHERE key='image_stats'").fetchone()[0])):
                for key in ("sum", "squares", "count"):
                    total[key] += summary[key]
                total["min"] = np.minimum(total["min"], summary["min"])
                total["max"] = np.maximum(total["max"], summary["max"])
    index = {k: np.asarray(v, dtype=np.float32 if k in ("state", "action") else np.int64) for k, v in index.items()}
    frames = len(index["index"])
    if not a.probe_episodes and (frames != info["total_frames"] or len(np.unique(index["task_index"])) != 40):
        raise ValueError("Incomplete full dataset")
    np.savez_compressed(a.output / "index.npz", **index)
    image_stats = []
    for total in totals:
        mean = total["sum"] / total["count"]
        image_stats.append(dict(mean=mean.tolist(), std=np.sqrt(np.maximum(total["squares"] / total["count"] - mean**2, 0)).tolist(),
                                min=total["min"].tolist(), max=total["max"].tolist()))
    manifest = dict(format="dp-libero-96-png-v1", source_repo=REPO, source_revision=REVISION,
        source=str(a.source), frames=frames, episodes=count, tasks=tasks, fps=10, partial=bool(a.probe_episodes),
        shards=[target.name for _, target in jobs], external_stats=dict(state=stats(index["state"]), actions=stats(index["action"])),
        image_stats=image_stats[0], wrist_image_stats=image_stats[1],
        index_sha256=hashlib.sha256((a.output / "index.npz").read_bytes()).hexdigest(),
        resize="PIL bilinear256 to96; source orientation preserved; current two cameras")
    (a.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("frames", "episodes", "partial", "source_revision")}), flush=True)


if __name__ == "__main__":
    main()
