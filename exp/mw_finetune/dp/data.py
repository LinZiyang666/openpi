"""Single-observation training reader; horizon tail repeats final episode action."""
from collections import OrderedDict
import io
import json
from pathlib import Path
import sqlite3

import numpy as np
from PIL import Image
import torch

from .preprocess import quantile_normalize


class Dataset:
    def __init__(self, root, embeddings, *, allow_partial=False):
        self.root = Path(root)
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["format"] not in ("dp-mt50-96-png-v1", "dp-libero-96-png-v1") or (self.manifest["partial"] and not allow_partial):
            raise ValueError("Full prepared96 DP data required")
        index = np.load(self.root / "index.npz", allow_pickle=False)
        self.index = {k: index[k] for k in index.files}
        ep = self.index["episode_index"]
        starts = np.r_[0, np.flatnonzero(np.diff(ep)) + 1]
        ends = np.r_[starts[1:], len(ep)]
        self.end_for_frame = np.repeat(ends, ends - starts)
        self.embeddings = embeddings
        self.connections = OrderedDict()

    def __len__(self):
        return len(self.end_for_frame)

    def __getitem__(self, frame):
        part = int(self.index["part"][frame])
        if part not in self.connections:
            path = self.root / self.manifest["shards"][part]
            self.connections[part] = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            if len(self.connections) > 16:
                _, db = self.connections.popitem(last=False); db.close()
        self.connections.move_to_end(part)
        paired = self.manifest["format"] == "dp-libero-96-png-v1"
        columns = "png, wrist_png" if paired else "png"
        row = self.connections[part].execute(f"SELECT {columns} FROM images WHERE id=?", (int(self.index["index"][frame]),)).fetchone()
        if row is None:
            raise ValueError(f"Missing image {frame}")
        with Image.open(io.BytesIO(row[0])) as im:
            image = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float() / 255
        wrist = torch.zeros_like(image)
        if paired:
            with Image.open(io.BytesIO(row[1])) as im:
                wrist = torch.from_numpy(np.asarray(im.convert("RGB")).copy()).permute(2, 0, 1).float() / 255
        ids = np.minimum(np.arange(frame, frame + 16), self.end_for_frame[frame] - 1)
        stats = self.manifest["external_stats"]
        return {"obs": {"base_0_rgb": image, "left_wrist_0_rgb": wrist,
                        "state": torch.from_numpy(quantile_normalize(self.index["state"][frame], stats["state"])),
                        "lang_emb": self.embeddings[int(self.index["task_index"][frame])].clone()},
                "action": torch.from_numpy(quantile_normalize(self.index["action"][ids], stats["actions"]))}


class Batches:
    """Each microbatch's indices depend only on seed/update/microstep, even after resume."""
    def __init__(self, length, seed, start, stop, batch, accumulation):
        self.length, self.seed, self.start, self.stop = length, seed, start, stop
        self.batch, self.accumulation = batch, accumulation

    def __iter__(self):
        for update in range(self.start, self.stop):
            for micro in range(self.accumulation):
                yield np.random.default_rng([self.seed, update, micro]).integers(self.length, size=self.batch).tolist()

    def __len__(self):
        return (self.stop - self.start) * self.accumulation
