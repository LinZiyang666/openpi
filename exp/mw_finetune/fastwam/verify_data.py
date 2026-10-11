"""Audit every prepared frame key and decode one image per MT50 task."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import io

import numpy as np
from PIL import Image


def verify(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    with np.load(root / "index.npz", allow_pickle=False) as stored:
        index = {key: stored[key] for key in stored.files}
    assert not manifest["partial"]
    assert manifest["frames"] == 204806 and manifest["episodes"] == 2500
    assert len(manifest["tasks"]) == 50 and manifest["fps"] == 80
    fingerprint = hashlib.sha256((root / "index.npz").read_bytes()).hexdigest()
    assert fingerprint == manifest["index_sha256"]
    for part, name in enumerate(manifest["shards"]):
        with sqlite3.connect(f"file:{root / name}?mode=ro", uri=True) as db:
            actual = np.asarray([row[0] for row in db.execute("SELECT id FROM images ORDER BY id")])
        expected = index["index"][index["part"] == part]
        np.testing.assert_array_equal(actual, expected)
    for task in np.unique(index["task_index"]):
        frame = int(np.flatnonzero(index["task_index"] == task)[0])
        name = manifest["shards"][int(index["part"][frame])]
        with sqlite3.connect(f"file:{root / name}?mode=ro", uri=True) as db:
            blob = db.execute("SELECT png FROM images WHERE id=?", (int(index["index"][frame]),)).fetchone()[0]
        with Image.open(io.BytesIO(blob)) as decoded:
            image = np.asarray(decoded.convert("RGB"))
        assert image.shape == (224, 224, 3) and image.dtype == np.uint8
    for key in ("state", "action"):
        assert index[key].shape == (204806, 4) and index[key].dtype == np.float32
        assert np.isfinite(index[key]).all()
    return {"passed": True, "frames": len(index["action"]), "episodes": 2500, "tasks": 50,
        "fps": 80, "image_size": [224, 224], "shards": len(manifest["shards"]),
        "state_action_dim": 4, "index_sha256": fingerprint,
        "bytes": sum(p.stat().st_size for p in root.iterdir() if p.is_file())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="/data/mw_finetune/fastwam/mt50_224")
    parser.add_argument("--output")
    args = parser.parse_args()
    report = verify(args.data)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
