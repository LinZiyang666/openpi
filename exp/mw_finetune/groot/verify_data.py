"""Verify source equality, lossless RGB decode, horizon padding and GR00T transforms on CPU."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from .conventions import CHANNELS, EMBODIMENT, HORIZON, action_chunk
from .embodiment import register_embodiment


def verify(dataset, source=None, samples=12):
    import av
    import pyarrow.parquet as pq
    register_embodiment()
    from .data_config import MetaWorldDataConfig
    from .convert_data import image_array, write_json
    from gr00t.data.dataset import LeRobotSingleDataset
    dataset = Path(dataset)
    cfg = MetaWorldDataConfig()
    ds = LeRobotSingleDataset(str(dataset), cfg.modality_config(), transforms=cfg.transform(),
                             embodiment_tag=EMBODIMENT, video_backend="decord")
    expected = json.loads((dataset / "meta/info.json").read_text())
    assert len(ds) == expected["total_frames"]
    # All numeric convention columns, not just sampled episodes.
    import pyarrow as pa
    from .convert_data import float32_table
    states, actions = [], []
    episode_tasks = Counter()
    for path in sorted((dataset / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        cast = float32_table(table)
        if table.schema != cast.schema:
            tmp = path.with_suffix(".partial.parquet")
            pq.write_table(cast, tmp, compression="zstd")
            tmp.replace(path)
        assert cast.schema.field("action").type == pa.list_(pa.float32())
        assert cast.schema.field("observation.state").type == pa.list_(pa.float32())
        states.extend(cast["observation.state"].to_pylist())
        actions.extend(cast["action"].to_pylist())
        task_ids = set(cast["task_index"].to_pylist())
        assert len(task_ids) == 1
        episode_tasks[next(iter(task_ids))] += 1
    from .conventions import statistics
    declared_stats = json.loads((dataset / "meta/stats.json").read_text())
    for key, values in [("observation.state", states), ("action", actions)]:
        for stat, numbers in statistics(values).items():
            np.testing.assert_allclose(declared_stats[key][stat], numbers, rtol=1e-7, atol=1e-8)
    if expected["total_episodes"] == 2500:
        assert len(episode_tasks) == 50 and set(episode_tasks.values()) == {50}
    source_rows = {}
    ids = [int(ds.trajectory_ids[i]) for i in np.linspace(0, len(ds.trajectory_ids) - 1, samples, dtype=int)]
    if source:
        wanted = set(ids)
        for path in sorted((Path(source) / "data").rglob("*.parquet")):
            cols = ["episode_index", "frame_index", "observation.state", "action", "task_index", "observation.image"]
            # Read images only for source shards containing sampled episodes.
            epi = pq.read_table(path, columns=["episode_index"])["episode_index"].to_pylist()
            if wanted.intersection(epi):
                for row in pq.read_table(path, columns=cols).to_pylist():
                    if int(row["episode_index"]) in wanted:
                        source_rows[(int(row["episode_index"]), int(row["frame_index"]))] = row
    checked = []
    for ep in ids:
        table = ds.get_trajectory_data(ep)
        for start in sorted({0, len(table) // 2, len(table) - 1}):
            raw = ds.get_step_data(ep, start)
            expected_action = action_chunk(np.stack(table["action"]), start)
            np.testing.assert_array_equal(np.concatenate([raw[f"action.{name}"] for name in CHANNELS], axis=1), expected_action)
            np.testing.assert_array_equal(np.concatenate([raw[f"state.{name}"] for name in CHANNELS], axis=1)[0], table["observation.state"].iloc[start])
            assert raw["video.image"].shape == (1, 480, 480, 3)
            assert raw["video.image"].dtype == np.uint8
            if source:
                row = source_rows[(ep, start)]
                np.testing.assert_array_equal(table["observation.state"].iloc[start], row["observation.state"])
                np.testing.assert_array_equal(table["action"].iloc[start], row["action"])
                np.testing.assert_array_equal(raw["video.image"][0], image_array(row["observation.image"], Path(source)))
            transformed = ds.transforms(raw)
            assert transformed["state"].shape == (1, 64)
            assert transformed["action"].shape == (HORIZON, 32)
            assert np.isfinite(transformed["action"]).all()
            import torch
            recovered = ds.transforms.unapply({"action": torch.from_numpy(transformed["action"])[None]})
            from .conventions import from_groot_actions
            np.testing.assert_allclose(from_groot_actions(recovered), expected_action, rtol=1e-5, atol=1e-6)
            checked.append({"episode": ep, "frame": start, "prompt": raw["annotation.human.action.task_description"][0]})
        with av.open(str(ds.get_video_path(ep, "image"))) as container:
            assert sum(1 for _ in container.decode(video=0)) == len(table)
    report = {"dataset": str(dataset), "frames": len(ds), "episodes": len(ds.trajectory_ids),
              "samples": checked, "lossless_source_equality": bool(source), "passed": True,
              "statistics_recomputed_and_equal": True, "parquet_vectors_float32": True,
              "normalization_roundtrip": True,
              "episodes_per_task_index": dict(sorted(episode_tasks.items()))}
    expected["data_files_size_in_mb"] = sum(p.stat().st_size for p in (dataset / "data").rglob("*.parquet")) / 1e6
    expected["video_files_size_in_mb"] = sum(p.stat().st_size for p in (dataset / "videos").rglob("*.mp4")) / 1e6
    write_json(dataset / "meta/info.json", expected)
    report["metadata_sha256"] = {name: hashlib.sha256((dataset / "meta" / name).read_bytes()).hexdigest()
                                 for name in ["info.json", "stats.json", "modality.json"]}
    write_json(dataset / "conversion/verification.json", report)
    print(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--samples", type=int, default=12)
    verify(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
