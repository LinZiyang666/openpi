"""Audit all96 PNG shards and compare spread samples to original480 parquet."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import torch

from .data import Dataset


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("/data/mw_finetune/dp/mt50_96"))
    ap.add_argument("--source", type=Path, default=Path("/data/mw_finetune/datasets/metaworld_mt50"))
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--report", type=Path, default=Path("exp/mw_finetune/dp/reports/data_audit.json"))
    a = ap.parse_args(); data = Dataset(a.dataset, torch.zeros(50, 768)); m = data.manifest
    if hashlib.sha256((a.dataset / "index.npz").read_bytes()).hexdigest() != m["index_sha256"]:
        raise ValueError("Index fingerprint mismatch")
    count = 0
    for name in m["shards"]:
        with sqlite3.connect(f"file:{a.dataset / name}?mode=ro", uri=True) as db:
            if db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError(f"Corrupt {name}")
            count += db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
    assert count == len(data) == 204806
    assert len(np.unique(data.index["episode_index"])) == 2500
    assert len(np.unique(data.index["task_index"])) == 50
    rows = []
    if a.samples:
        for frame in np.linspace(0, len(data)-1, a.samples, dtype=int):
            name = m["shards"][int(data.index["part"][frame])]
            chunk, file = name.removeprefix("part-").removesuffix(".sqlite").split("-file-")
            table = pq.read_table(a.source / "data" / chunk / f"file-{file}.parquet",
                columns=["index", "observation.image", "observation.state", "action"])
            ids = np.asarray(table["index"])
            row = table.slice(int(np.flatnonzero(ids == frame)[0]), 1).to_pylist()[0]
            sample = data[int(frame)]
            with Image.open(io.BytesIO(row["observation.image"]["bytes"])) as im:
                expected = np.asarray(im.convert("RGB").resize((96, 96), Image.Resampling.BILINEAR))
            actual = (sample["obs"]["base_0_rgb"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            assert np.array_equal(expected, actual)
            assert np.array_equal(np.asarray(row["observation.state"], np.float32), data.index["state"][frame])
            assert np.array_equal(np.asarray(row["action"], np.float32), data.index["action"][frame])
            rows.append(int(frame))
    report = {"frames": count, "episodes": 2500, "tasks": 50, "shards": len(m["shards"]),
        "index_sha256": m["index_sha256"], "exact_source_samples": rows,
        "raw_state_action_preserved": True, "resize": "480 to96 bilinear", "fps": m["fps"]}
    a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
