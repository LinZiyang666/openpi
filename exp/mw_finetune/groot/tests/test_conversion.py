import io
import json

import av
import numpy as np
from PIL import Image
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from exp.mw_finetune.groot.convert_data import convert


def source_dataset(root):
    (root / "meta/episodes/chunk-000").mkdir(parents=True)
    (root / "data/chunk-000").mkdir(parents=True)
    info = {"codebase_version": "v3.0", "total_episodes": 2, "total_frames": 4,
            "total_tasks": 2, "fps": 80,
            "features": {"observation.state": {"dtype": "float32", "shape": [4]},
                         "action": {"dtype": "float32", "shape": [4]}}}
    (root / "meta/info.json").write_text(json.dumps(info))
    pq.write_table(pa.Table.from_pylist([{"episode_index": i, "length": 2, "tasks": [f"task {i}"]}
                                       for i in range(2)]), root / "meta/episodes/chunk-000/file-000.parquet")
    pq.write_table(pa.Table.from_pylist([{"task_index": i, "__index_level_0__": f"task {i}"}
                                       for i in range(2)]), root / "meta/tasks.parquet")
    rows = []
    for ep in range(2):
        for i in range(2):
            image = np.full((480, 480, 3), ep * 20 + i, dtype=np.uint8)
            image[0, 0] = [25, 150, 250]
            buffer = io.BytesIO()
            Image.fromarray(image).save(buffer, format="PNG")
            rows.append({"episode_index": ep, "frame_index": i, "task_index": ep,
                         "timestamp": i / 80, "index": ep * 2 + i,
                         "observation.state": [0., .7, .2, .6], "action": [2., -3., .5, -.6],
                         "observation.image": {"bytes": buffer.getvalue(), "path": None}})
    # Deliberately split an episode between shards.
    for idx, subset in enumerate([rows[:1], rows[1:]]):
        pq.write_table(pa.Table.from_pylist(subset), root / f"data/chunk-000/file-{idx:03d}.parquet")
    return rows


def test_embedded_images_shard_boundary_lossless_and_idempotent(tmp_path):
    src, dst = tmp_path / "source", tmp_path / "output"
    rows = source_dataset(src)
    result = convert(src, dst, workers=2)
    assert result["frames"] == 4 and result["episodes"] == 2 and not result["smoke_only"]
    for ep in range(2):
        table = pq.read_table(dst / f"data/chunk-000/episode_{ep:06d}.parquet").to_pylist()
        np.testing.assert_array_equal(np.asarray(table[0]["action"], np.float32),
                                      np.asarray(rows[ep * 2]["action"], np.float32))
        assert pq.read_schema(dst / f"data/chunk-000/episode_{ep:06d}.parquet").field("action").type == pa.list_(pa.float32())
        with av.open(str(dst / f"videos/chunk-000/observation.image/episode_{ep:06d}.mp4")) as video:
            decoded = list(video.decode(video=0))
        assert len(decoded) == 2
        for i, frame in enumerate(decoded):
            expected = np.asarray(Image.open(io.BytesIO(rows[ep * 2 + i]["observation.image"]["bytes"])))
            np.testing.assert_array_equal(frame.to_ndarray(format="rgb24"), expected)
    old_mtime = (dst / "data/chunk-000/episode_000000.parquet").stat().st_mtime_ns
    convert(src, dst, workers=2)
    assert (dst / "data/chunk-000/episode_000000.parquet").stat().st_mtime_ns == old_mtime


def test_partial_source_cannot_masquerade_as_mt50(tmp_path):
    src = tmp_path / "source"
    source_dataset(src)
    path = src / "meta/info.json"
    info = json.loads(path.read_text())
    info["total_frames"] = 999
    path.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="download incomplete"):
        convert(src, tmp_path / "output")
