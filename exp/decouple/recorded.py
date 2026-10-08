"""Read only recorded B-pool tokens/images/states; never consult test-A runs."""
from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np

ROOT = Path("/home/weiland/trace_runs/offline_search_store")


def load(model, suite="spatial", limit=16, root=ROOT):
    pool = "bpool_cs" if model == "pi05" else "bpool_all"
    path = Path(root) / "library" / f"{model}_{suite}" / pool
    manifest = json.loads((path / "manifest.json").read_text())
    arrays = {name: np.load(path / f"{name}.npy", mmap_mode="r")
              for name in ("key_v0", "key_v1", "rs", "action", "task_id", "episode", "step")}
    token = {name: np.load(path / "tok" / f"{name}.npy", mmap_mode="r")
             for name in ("rows", "v0", "v1")}
    image_path = path
    if not (image_path / "tok/img0.npy").exists():
        # The GR00T B library was collected without camera images. Use recorded
        # π0.5 B images for CPU preprocessing only, never synthesize cameras or
        # describe them as paired GR00T observations.
        image_path = Path(root) / "library" / f"pi05_{suite}" / "bpool_cs"
    for name in ("img0", "img1"):
        token[name] = np.load(image_path / "tok" / f"{name}.npy", mmap_mode="r")
    rows = np.asarray(token["rows"][:limit], dtype=np.int64)
    return SimpleNamespace(path=path, manifest=manifest, arrays=arrays, token=token, rows=rows,
                           image_source=image_path)


def observation(data, index):
    row = int(data.rows[index])
    state = np.zeros(8, np.float32)
    state[:min(8, data.arrays["rs"].shape[1])] = data.arrays["rs"][row, :8]
    task = data.manifest["tasks"][str(int(data.arrays["task_id"][row]))]
    # Images here are the recorded model-input images. Raw simulator images
    # are not reconstructed; profiling must label this distinction.
    return {"observation/image": np.array(data.token["img0"][index]),
            "observation/wrist_image": np.array(data.token["img1"][index]),
            "observation/state": state, "prompt": task}
