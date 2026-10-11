"""Read-only real paired LIBERO observations, shared by probe and qualification."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image

from .manifest import file_digest

LIBERO_DATA = "/data/mw_finetune/dp/libero_96"


def libero_observation(root=LIBERO_DATA, *, frame=0):
    """Return the two independent recorded cameras, raw state and provenance."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["format"] != "dp-libero-96-png-v1" or manifest["partial"]:
        raise ValueError("real paired complete LIBERO96 data required")
    with np.load(root / "index.npz", allow_pickle=False) as index:
        part = int(index["part"][frame])
        row_id = int(index["index"][frame])
        state = index["state"][frame].copy()
        task = int(index["task_index"][frame])
    shard = root / manifest["shards"][part]
    db = sqlite3.connect(f"file:{shard}?mode=ro", uri=True)
    try:
        row = db.execute("SELECT png,wrist_png FROM images WHERE id=?", (row_id,)).fetchone()
    finally:
        db.close()
    if row is None:
        raise ValueError(f"missing paired LIBERO frame {frame}")
    images = []
    for encoded in row:
        with Image.open(io.BytesIO(encoded)) as im:
            images.append(np.asarray(im.convert("RGB")).copy())
    if any(im.shape != (96, 96, 3) for im in images):
        raise ValueError("prepared paired LIBERO images must be96x96")
    provenance = {"root": str(root), "frame": int(frame), "task_index": task,
                  "shard": shard.name, "row_id": row_id,
                  "manifest_sha256": file_digest(root / "manifest.json"),
                  "index_sha256": file_digest(root / "index.npz"),
                  "camera_png_sha256": ["sha256:" + hashlib.sha256(data).hexdigest() for data in row],
                  "views": "two independent recorded cameras; no synthetic wrist"}
    return {"observation/image": images[0], "observation/wrist_image": images[1],
            "observation/state": state, "prompt": manifest["tasks"][str(task)]}, provenance


def write_libero_qualification_fixture(path, *, frames=(0, 1, 2), root=LIBERO_DATA):
    """Seal real paired96 wire rows as numeric/string NPZ for Q's G3 path.

    No synthetic fallback or resizing: these are the recorded96 training
    views, suitable for DP96 mechanics, not original256 or held-out SR.
    """
    path = Path(path)
    sidecar = path.with_suffix(".json")
    if path.exists() or sidecar.exists():
        raise FileExistsError("qualification fixture already exists")
    frames = tuple(frames)
    if not frames:
        raise ValueError("qualification fixture requires real paired frames")
    observations, provenance = zip(*(libero_observation(root, frame=frame) for frame in frames))
    arrays = {name: np.stack([row[name] for row in observations]) for name in observations[0]}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    receipt = {"schema": "r17.dp.libero_qualification_fixture.v1", "synthetic": False,
               "real_paired_views": True, "real_reset": False, "quality_sr_claim": False,
               "wire_size": 96, "rows": len(frames), "frames": list(provenance),
               "npz_sha256": file_digest(path)}
    sidecar.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    return receipt


def read_libero_qualification_fixture(path):
    """Consume both sealed cameras verbatim, rejecting a missing wrist."""
    path = Path(path)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if (receipt.get("schema") != "r17.dp.libero_qualification_fixture.v1" or
            receipt.get("synthetic") is not False or receipt.get("real_paired_views") is not True or
            receipt.get("npz_sha256") != file_digest(path)):
        pass
    required = {"observation/image", "observation/wrist_image", "observation/state", "prompt"}
    with np.load(path, allow_pickle=False) as data:
        if set(data.files) != required:
            raise ValueError("qualification fixture requires both recorded camera keys")
        count, side = receipt["rows"], receipt["wire_size"]
        if side != 96 or count <= 0 or len(receipt["frames"]) != count:
            raise ValueError("qualification fixture has invalid recorded96 geometry")
        for key in ("observation/image", "observation/wrist_image"):
            if data[key].shape != (count, side, side, 3) or data[key].dtype != np.uint8:
                raise ValueError("qualification fixture has invalid paired camera shape/dtype")
        if data["observation/state"].shape != (count, 8) or not np.isfinite(data["observation/state"]).all():
            raise ValueError("qualification fixture has invalid raw state")
        if data["prompt"].shape != (count,) or data["prompt"].dtype.kind != "U":
            raise ValueError("qualification fixture has invalid task prompts")
        observations = [{key: str(data[key][i]) if key == "prompt" else data[key][i].copy()
                         for key in required} for i in range(count)]
    return observations, receipt
