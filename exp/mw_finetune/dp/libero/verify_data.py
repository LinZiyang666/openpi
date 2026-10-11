"""Independent spread-frame comparison against pinned raw paired LIBERO data."""
import argparse
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("/data/mw_finetune/dp/libero_96"))
    ap.add_argument("--output", type=Path, default=Path("exp/mw_finetune/dp/reports/libero_data_audit.json"))
    a = ap.parse_args(); manifest=json.loads((a.dataset/"manifest.json").read_text())
    index=np.load(a.dataset/"index.npz",allow_pickle=False)
    if manifest["partial"] or (manifest["frames"],manifest["episodes"],len(manifest["tasks"])) != (273465,1693,40):
        raise ValueError("Expected complete four-suite dataset")
    samples=[]; total=0
    for name in manifest["shards"]:
        with sqlite3.connect(f"file:{a.dataset/name}?mode=ro",uri=True) as db:
            assert db.execute("PRAGMA quick_check").fetchone()[0]=="ok"
            images=db.execute("SELECT COUNT(*) FROM images").fetchone()[0]
            assert images==db.execute("SELECT COUNT(*) FROM vectors").fetchone()[0]
            total+=images
    assert total==manifest["frames"]
    for frame in np.linspace(0,total-1,12,dtype=int):
        episode=int(index["episode_index"][frame]); ident=int(index["index"][frame])
        source=Path(manifest["source"])/f"data/chunk-{episode//1000:03d}/episode_{episode:06d}.parquet"
        row=next(row for row in pq.read_table(source).to_pylist() if row["index"]==ident)
        assert np.array_equal(index["state"][frame],np.asarray(row["state"],np.float32))
        assert np.array_equal(index["action"][frame],np.asarray(row["actions"],np.float32))
        part=int(index["part"][frame])
        with sqlite3.connect(f"file:{a.dataset/manifest['shards'][part]}?mode=ro",uri=True) as db:
            images=db.execute("SELECT png,wrist_png FROM images WHERE id=?",(ident,)).fetchone()
        for name,blob in zip(("image","wrist_image"),images):
            with Image.open(io.BytesIO(row[name]["bytes"])) as image:
                expected=np.asarray(image.convert("RGB").resize((96,96),Image.Resampling.BILINEAR))
            with Image.open(io.BytesIO(blob)) as image:
                assert np.array_equal(np.asarray(image),expected)
        samples.append(dict(frame=int(frame),episode=episode,task=int(index["task_index"][frame]),raw_vectors_exact=True,paired_pixels_exact=True))
    report=dict(frames=total,episodes=manifest["episodes"],tasks=len(manifest["tasks"]),sqlite_checks="all pass",samples=samples,
                source_revision=manifest["source_revision"])
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__ == "__main__":
    main()
