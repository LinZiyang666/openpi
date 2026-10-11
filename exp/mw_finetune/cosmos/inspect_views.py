"""CPU-only viewpoint evidence; render in the simulator venv, compare in PyArrow venv."""
import argparse
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from exp.metaworld import env as E
from exp.mw_finetune.cosmos.probe_evaluate import make_probe_env


def compare_frames(directory, dataset):
    import pyarrow.parquet as pq

    columns = ["observation.image", "observation.state", "task_index", "episode_index", "frame_index"]
    source = dataset / "data/chunk-000/file-000.parquet"
    row = next(pq.ParquetFile(source).iter_batches(batch_size=1, columns=columns)).to_pylist()[0]
    frame = Image.open(io.BytesIO(row.pop("observation.image")["bytes"])).convert("RGB")
    frame.save(directory / "dataset_frame0.png")
    (directory / "dataset_frame0.json").write_text(json.dumps(row, indent=2) + "\n")
    reference = np.asarray(frame, dtype=np.float32) / 255
    report = {"metric": "RGB [0,1], different episode initialization; pose is not controlled against training frame",
              "views": {}}
    for view in ("corner2_canonical480", "corner2_raw480", "corner3_raw256"):
        image = Image.open(directory / (view + ".png")).convert("RGB")
        if image.size != frame.size:
            image = image.resize(frame.size, Image.Resampling.BICUBIC)
        difference = np.asarray(image, dtype=np.float32) / 255 - reference
        report["views"][view] = {"mae": float(np.abs(difference).mean()),
                                  "rmse": float(np.sqrt(np.square(difference).mean()))}
    (directory / "image_comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "results/viewpoint_evidence")
    ap.add_argument("--compare-only", action="store_true", help="compare existing views with dataset frame0; no simulator")
    ap.add_argument("--dataset", type=Path, default=Path("/data/mw_finetune/datasets/metaworld_mt50"))
    a = ap.parse_args()
    directory = a.output
    directory.mkdir(parents=True, exist_ok=True)
    if a.compare_only:
        compare_frames(directory, a.dataset)
        return
    state = {}
    env = E.make_env("assembly-v3", 0, 7)
    obs = E.reset_and_settle(env)
    state["canonical"] = obs[:4].tolist()
    Image.fromarray(env.render()).save(directory / "corner2_raw480.png")
    Image.fromarray(E.render_image(env)).save(directory / "corner2_canonical480.png")
    env.close()
    env = make_probe_env("assembly-v3", 0, 7)
    obs = E.reset_and_settle(env)
    state["flowdagger"] = obs[:4].tolist()
    Image.fromarray(E.render_image(env)).save(directory / "corner3_raw256.png")
    env.close()
    (directory / "states.json").write_text(json.dumps(state, indent=2) + "\n")


if __name__ == "__main__":
    main()
