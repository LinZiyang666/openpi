"""Convert embedded-image LeRobot v2/v3 MT50 into GR00T N1.5's v2 layout.

Bounded memory, lossless RGB H.264, idempotent episode files, exact raw actions.
Source downloads may be resumed independently. Full conversion rejects a partial
download; --limit-episodes is explicitly a smoke-only dataset.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import shutil
import time

import numpy as np

from .conventions import EMBODIMENT, HORIZON, modality_metadata, statistics

SOURCE_REVISION = "b46a1c185a2368660650a82ac5a5083a56722f48"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def image_array(value, source):
    from PIL import Image
    if isinstance(value, dict):
        value = io.BytesIO(value["bytes"]) if value.get("bytes") else source / value["path"]
    elif isinstance(value, (bytes, bytearray)):
        value = io.BytesIO(value)
    array = np.asarray(Image.open(value).convert("RGB"))
    if array.shape != (480, 480, 3):
        raise ValueError(f"unexpected source image shape {array.shape}")
    return array


def float32_table(table):
    """Arrow's Python-list inference chooses float64; preserve LeRobot float32."""
    import pyarrow as pa
    for key in ["observation.state", "action", "observation.environment_state"]:
        if key in table.column_names:
            idx = table.schema.get_field_index(key)
            table = table.set_column(idx, key, table[key].cast(pa.list_(pa.float32())))
    for key in ["timestamp", "next.reward"]:
        if key in table.column_names:
            idx = table.schema.get_field_index(key)
            table = table.set_column(idx, key, table[key].cast(pa.float32()))
    if "task_id" in table.column_names:
        idx = table.schema.get_field_index("task_id")
        table = table.set_column(idx, "task_id", table["task_id"].cast(pa.int16()))
    return table


def encode_episode(rows, source, output, fps):
    import av
    import pyarrow as pa
    import pyarrow.parquet as pq
    episode = int(rows[0]["episode_index"])
    chunk = episode // 1000
    data = output / f"data/chunk-{chunk:03d}/episode_{episode:06d}.parquet"
    video = output / f"videos/chunk-{chunk:03d}/observation.image/episode_{episode:06d}.mp4"
    marker = output / f"conversion/episode_{episode:06d}.json"
    digest = hashlib.sha256()
    for row in rows:
        digest.update(np.asarray(row["observation.state"] + row["action"], dtype=np.float32).tobytes())
        digest.update(str((row["frame_index"], row["task_index"])).encode())
        image = row["observation.image"]
        digest.update(image.get("bytes") or str(image.get("path")).encode())
    identity = {"episode_index": episode, "length": len(rows), "sha256": digest.hexdigest()}
    if marker.exists():
        if json.loads(marker.read_text()) != identity:
            raise ValueError(f"existing episode has a different source: {marker}")
        if data.exists() and video.exists():
            table = pq.read_table(data)
            cast = float32_table(table)
            if table.schema != cast.schema:
                data_tmp = data.with_suffix(".partial.parquet")
                pq.write_table(cast, data_tmp, compression="zstd")
                data_tmp.replace(data)
            return identity
    elif data.exists() or video.exists():
        raise FileExistsError(f"untracked output for episode {episode}; use a new destination")
    # Claim this source identity before either atomic output publish. A resumed
    # conversion can then finish an episode interrupted between the two files.
    write_json(marker, identity)
    data.parent.mkdir(parents=True, exist_ok=True)
    video.parent.mkdir(parents=True, exist_ok=True)
    video_tmp = video.with_suffix(".partial.mp4")
    container = av.open(str(video_tmp), mode="w")
    stream = container.add_stream("libx264rgb", rate=Fraction(fps))
    stream.width = stream.height = 480
    stream.pix_fmt = "rgb24"
    stream.options = {"crf": "0", "preset": "veryfast", "threads": "1", "g": "16"}
    try:
        for i, row in enumerate(rows):
            if int(row["frame_index"]) != i or int(row["episode_index"]) != episode:
                raise ValueError(f"noncontiguous frames in episode {episode}")
            frame = av.VideoFrame.from_ndarray(image_array(row["observation.image"], source), format="rgb24")
            frame.pts = i
            frame.time_base = Fraction(1, fps)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    video_tmp.replace(video)
    table = float32_table(pa.Table.from_pylist([{key: value for key, value in row.items() if key != "observation.image"}
                                               for row in rows]))
    data_tmp = data.with_suffix(".partial.parquet")
    pq.write_table(table, data_tmp, compression="zstd")
    data_tmp.replace(data)
    write_json(marker, identity)
    return identity


def episode_metadata(source):
    import pyarrow.parquet as pq
    path = source / "meta/episodes.jsonl"
    if path.exists():
        rows = [json.loads(line) for line in path.read_text().splitlines()]
    else:
        rows = []
        for path in sorted((source / "meta/episodes").rglob("*.parquet")):
            rows.extend(pq.read_table(path, columns=["episode_index", "length", "tasks"]).to_pylist())
    return sorted(rows, key=lambda row: row["episode_index"])


def convert(source, output, workers=8, limit_episodes=None, wait_source=False, wait_timeout=7200):
    import pyarrow.parquet as pq
    started = time.perf_counter()
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("source and output must be different")
    info = json.loads((source / "meta/info.json").read_text())
    episodes = episode_metadata(source)
    if len(episodes) != info["total_episodes"]:
        raise ValueError("incomplete source episode metadata")
    selected = episodes if limit_episodes is None else episodes[:limit_episodes]
    lengths = {int(row["episode_index"]): int(row["length"]) for row in selected}
    paths = sorted((source / "data").rglob("*.parquet"))
    expected_sizes = {}
    if wait_source:
        from huggingface_hub import HfApi
        remote = HfApi().dataset_info("lerobot/metaworld_mt50", revision=SOURCE_REVISION, files_metadata=True)
        expected_sizes = {source / f.rfilename: f.size for f in remote.siblings
                          if f.rfilename.startswith("data/") and f.rfilename.endswith(".parquet")}
        paths = sorted(expected_sizes)
    frames_on_disk = sum(pq.ParquetFile(path).metadata.num_rows for path in paths if path.exists())
    if not wait_source and limit_episodes is None and frames_on_disk != info["total_frames"]:
        raise ValueError(f"source download incomplete: {frames_on_disk}/{info['total_frames']} frames; wait for download")
    existing = output
    while not existing.exists():
        existing = existing.parent
    free = shutil.disk_usage(existing).free / 1e9
    # Source PNGs expand less than this conservative converted-data allowance.
    if free < (30 if str(output).startswith("/data/") else 2) + (12 if limit_episodes is None else 1):
        raise RuntimeError(f"insufficient output free space: {free:.1f} GB")
    config = {"source": str(source), "source_revision": SOURCE_REVISION,
              "limit_episodes": limit_episodes, "codec": "libx264rgb crf=0", "fps": info["fps"]}
    config_path = output / "conversion/config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("destination belongs to a different conversion; choose another output")
    write_json(config_path, config)
    tasks_path = source / "meta/tasks.jsonl"
    if tasks_path.exists():
        tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()]
    else:
        tasks = [{"task_index": int(row["task_index"]), "task": row["__index_level_0__"]}
                 for row in pq.read_table(source / "meta/tasks.parquet").to_pylist()]
    states, actions, seen, pending = [], [], set(), []
    episode_task_counts = Counter()
    current_id, current_rows = None, []

    def submit(pool):
        nonlocal current_rows
        if not current_rows:
            return
        if len(current_rows) != lengths[current_id]:
            raise ValueError(f"incomplete episode {current_id}: {len(current_rows)}/{lengths[current_id]}")
        if current_id in seen:
            raise ValueError(f"duplicate/noncontiguous episode {current_id}")
        seen.add(current_id)
        task_ids = {int(row["task_index"]) for row in current_rows}
        if len(task_ids) != 1:
            raise ValueError(f"task changes inside episode {current_id}")
        episode_task_counts[next(iter(task_ids))] += 1
        states.extend(row["observation.state"] for row in current_rows)
        actions.extend(row["action"] for row in current_rows)
        pending.append(pool.submit(encode_episode, current_rows, source, output, int(info["fps"])))
        current_rows = []
        if len(pending) >= workers * 2:
            pending.pop(0).result()
        if len(seen) % 100 == 0:
            print(f"converted/submitted {len(seen)}/{len(selected)} episodes", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for path in paths:
            if wait_source:
                # HF snapshot_download atomically publishes completed files.
                # Conversion can proceed during download, but full metadata is
                # published only after every episode/frame has been checked.
                while not path.exists() or path.stat().st_size != expected_sizes[path]:
                    if time.perf_counter() - started > wait_timeout:
                        raise TimeoutError(f"source download did not finish in {wait_timeout} seconds")
                    print(f"waiting for source {path.relative_to(source)}", flush=True)
                    time.sleep(30)
            for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
                for row in batch.to_pylist():
                    ep = int(row["episode_index"])
                    if ep not in lengths:
                        continue
                    if ep != current_id:
                        submit(pool)
                        current_id = ep
                    current_rows.append(row)
            if limit_episodes is not None and len(seen) == len(selected) - 1 and current_rows and len(current_rows) == lengths[current_id]:
                break
        submit(pool)
        for future in pending:
            future.result()
    if seen != set(lengths):
        raise ValueError(f"missing source episodes: {sorted(set(lengths) - seen)[:20]}")
    if limit_episodes is None and len(states) != info["total_frames"]:
        raise ValueError(f"wrong final frame count: {len(states)}/{info['total_frames']}")
    stats = {"observation.state": statistics(states), "action": statistics(actions)}
    write_json(output / "meta/stats.json", stats)
    write_json(output / "meta/modality.json", modality_metadata())
    features = {key: value for key, value in info["features"].items()}
    features["observation.image"] = {"dtype": "video", "shape": [480, 480, 3],
                                     "names": ["height", "width", "channel"],
                                     "video_info": {"video.fps": info["fps"], "video.codec": "h264", "video.channels": 3}}
    output_info = dict(info, codebase_version="v2.1", robot_type=EMBODIMENT,
                       total_episodes=len(selected), total_frames=len(states), total_tasks=len(tasks),
                       chunks_size=1000, features=features, splits={"train": f"0:{len(selected)}"},
                       total_videos=len(selected), total_chunks=(max(lengths) // 1000 + 1),
                       data_path="data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
                       video_path="videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4")
    output_info["data_files_size_in_mb"] = sum(p.stat().st_size for p in (output / "data").rglob("*.parquet")) / 1e6
    output_info["video_files_size_in_mb"] = sum(p.stat().st_size for p in (output / "videos").rglob("*.mp4")) / 1e6
    write_json(output / "meta/info.json", output_info)
    for name, rows in [("tasks", tasks), ("episodes", selected)]:
        (output / f"meta/{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    report = {"source": str(source), "output": str(output), "source_revision": SOURCE_REVISION,
              "episodes": len(selected), "frames": len(states), "tasks": len(tasks),
              "fps": info["fps"], "action_horizon": HORIZON, "embodiment": EMBODIMENT,
              "smoke_only": limit_episodes is not None, "seconds": time.perf_counter() - started,
              "state_min": stats["observation.state"]["min"], "state_max": stats["observation.state"]["max"],
              "action_min": stats["action"]["min"], "action_max": stats["action"]["max"],
              "episodes_per_task_index": dict(sorted(episode_task_counts.items())),
              "codec": "lossless RGB H.264 (libx264rgb, crf=0)", "complete": True}
    write_json(output / "conversion/report.json", report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/data/mw_finetune/datasets/metaworld_mt50"))
    parser.add_argument("--output", type=Path, default=Path("/data/mw_finetune/datasets/metaworld_mt50_groot"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--wait-source", action="store_true", help="convert completed shards while the pinned download continues")
    parser.add_argument("--wait-timeout", type=int, default=7200)
    args = parser.parse_args()
    if args.workers < 1 or (args.limit_episodes is not None and args.limit_episodes < 1):
        parser.error("workers/limit-episodes must be positive")
    convert(**vars(args))


if __name__ == "__main__":
    main()
