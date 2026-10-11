"""Reproducible reset captures and dataset/renderer pixel comparisons."""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

TASKS = ("assembly-v3", "basketball-v3", "button-press-v3", "push-v3", "reach-v3")


def stats(left, right, mask=None):
    diff = np.asarray(left, dtype=np.float32) - np.asarray(right, dtype=np.float32)
    if mask is not None:
        diff = diff[mask]
    absolute = np.abs(diff)
    return {"pixels": int(diff.size // 3), "mae_255": float(absolute.mean()),
            "rmse_255": float(np.sqrt(np.mean(diff ** 2))), "max_abs": float(absolute.max()),
            "channels_equal_fraction": float(np.mean(absolute == 0)),
            "p99_abs": float(np.percentile(absolute, 99))}


def side_by_side(path, images, labels):
    canvas = Image.new("RGB", (480 * len(images), 504), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (im, label) in enumerate(zip(images, labels, strict=True)):
        canvas.paste(Image.fromarray(im), (i * 480, 24))
        draw.text((i * 480 + 5, 5), label, fill="black")
    canvas.save(path)


def capture(out, renderer, device_id, tasks):
    from exp.mw_finetune.rendering import configure, make_env, STAMP
    configure(renderer, device_id)
    from exp.metaworld import env as E
    out.mkdir(parents=True, exist_ok=True)
    records = {}
    for name in tasks:
        e = make_env(name, 0, 7)
        try:
            state = E.reset_and_settle(e)
            image = E.render_image(e)
            Image.fromarray(image).save(out / f"{name}.png")
            Image.fromarray(e.render()).save(out / f"{name}_raw.png")
            records[name] = {"state": state[:4].tolist(), "worker": dict(STAMP)}
        finally:
            e.close()
    (out / "captures.json").write_text(json.dumps(records, indent=2) + "\n")


def compare(out, dataset, captures):
    import pyarrow.parquet as pq
    from exp.metaworld import tasks as T
    episodes = []
    for file in sorted((dataset / "meta/episodes").rglob("*.parquet")):
        episodes.extend(pq.read_table(file, columns=["episode_index", "tasks", "data/chunk_index",
                        "data/file_index", "dataset_from_index"]).to_pylist())
    task_text = {r["task_index"]: r["__index_level_0__"] for r in pq.read_table(dataset / "meta/tasks.parquet").to_pylist()}
    results = {}
    for name in TASKS:
        task_id = T.task_id_of(name)
        ep = next(e for e in episodes if e["tasks"] == [task_text[task_id]])
        file = dataset / f"data/chunk-{ep['data/chunk_index']:03d}/file-{ep['data/file_index']:03d}.parquet"
        import pyarrow.dataset as ds
        row = ds.dataset(file).to_table(filter=ds.field("index") == ep["dataset_from_index"],
                 columns=["observation.image", "observation.state", "task_index", "frame_index"]).to_pylist()[0]
        assert row["task_index"] == task_id and row["frame_index"] == 0
        image = np.asarray(Image.open(io.BytesIO(row["observation.image"]["bytes"])).convert("RGB"))
        assert image.shape == (480, 480, 3)
        Image.fromarray(image).save(out / f"dataset_{name}.png")
        # Fixed regions selected by scene geometry, independent of the render
        # difference: top-right static background; left arm/body, excluding
        # gripper/task objects. Episode resets differ, so full-frame equality
        # is not an expected or required result.
        background = np.zeros((480, 480), bool); background[:100, 320:] = True
        robot = np.zeros((480, 480), bool); robot[20:180, 40:180] = True
        tabletop = np.zeros((480, 480), bool); tabletop[340:430, 340:450] = True
        rim = np.zeros((480, 480), bool); rim[170:210, 360:440] = True
        regions = {"whole": None, "static_background": background, "static_tabletop": tabletop,
                   "static_table_rim": rim, "robot_body": robot}
        models = {tag: np.asarray(Image.open(root / f"{name}.png")) for tag, root in captures.items()}
        entry = {"episode_index": ep["episode_index"], "data_file": str(file), "frame_index": 0,
                 "dataset_state": row["observation.state"], "task_text": task_text[task_id]}
        entry["dataset_comparison"] = {tag: {region: stats(image, im, mask) for region, mask in regions.items()}
                                       for tag, im in models.items()}
        entry["orientation_mae"] = {}
        for tag, im in models.items():
            entry["orientation_mae"][tag] = {orientation: stats(image, candidate)["mae_255"] for orientation, candidate in
                (("historical_180", im), ("raw_unrotated", im[::-1, ::-1]),
                 ("vertical_only", im[:, ::-1]), ("horizontal_only", im[::-1]))}
        tags = list(models)
        states = {tag: json.loads((root / "captures.json").read_text())[name]["state"]
                  for tag, root in captures.items()}
        if not all(np.array_equal(states[tags[0]], value) for value in states.values()):
            raise ValueError(f"renderer captures have different reset states for {name}: {states}")
        entry["identical_bench_reset_states"] = states
        entry["renderer_comparison_same_reset"] = {f"{left}_vs_{right}": stats(models[left], models[right])
                for i, left in enumerate(tags) for right in tags[i + 1:]}
        results[name] = entry
        side_by_side(out / f"comparison_{name}.png", [image, *models.values()], ["dataset first frame", *tags])
    result = {"camera": "corner2", "camera_pos": [0.75, 0.075, 0.7], "resolution": [480, 480],
              "rotation_degrees": 180, "regions": {"static_background": "rows0:100 cols320:480",
              "static_tabletop": "rows340:430 cols340:450", "static_table_rim": "rows170:210 cols360:440",
              "robot_body": "rows20:180 cols40:180"}, "tasks": results,
              "note": "Dataset and bench reset seeds/poses differ; region/orientation comparison verifies convention, not identical resets."}
    (out / "pixel_stats.json").write_text(json.dumps(result, indent=2) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="mode", required=True)
    cap = sub.add_parser("capture")
    cap.add_argument("--out", type=Path, required=True)
    cap.add_argument("--renderer", choices=["egl", "osmesa"], required=True)
    cap.add_argument("--device-id", type=int, default=0)
    cap.add_argument("--tasks", nargs="+", default=list(TASKS))
    cmp = sub.add_parser("compare")
    cmp.add_argument("--out", type=Path, required=True)
    cmp.add_argument("--dataset", type=Path, default=Path("/data/mw_finetune/datasets/metaworld_mt50"))
    cmp.add_argument("--capture", action="append", required=True, help="LABEL=DIR")
    a = ap.parse_args()
    if a.mode == "capture":
        capture(a.out, a.renderer, a.device_id, a.tasks)
    else:
        a.out.mkdir(parents=True, exist_ok=True)
        captures = dict(item.split("=", 1) for item in a.capture)
        compare(a.out, a.dataset, {k: Path(v) for k, v in captures.items()})


if __name__ == "__main__":
    main()
