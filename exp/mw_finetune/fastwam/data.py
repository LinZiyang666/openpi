"""Small indexed reader; padding never crosses an episode boundary."""
import io
import json
from pathlib import Path
import sqlite3
from collections import OrderedDict

import numpy as np
from PIL import Image

from .common import normalize, window_indices


class Dataset:
    def __init__(self, root, allow_partial=False, mode="video_lora"):
        self.root = Path(root)
        self.mode = mode
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest["partial"] and not allow_partial:
            raise ValueError("Partial data is only allowed for smoke/probe")
        index = np.load(self.root / "index.npz", allow_pickle=False)
        self.index = {k: index[k] for k in index.files}
        episodes = self.index["episode_index"]
        split = np.flatnonzero(np.diff(episodes)) + 1
        self.starts = np.concatenate(([0], split))
        ends = np.concatenate((split, [len(episodes)]))
        self.end_for_frame = np.repeat(ends, ends - self.starts)
        self._connections = OrderedDict()

    def __len__(self):
        return len(self.index["action"])

    def _image(self, index):
        part = int(self.index["part"][index])
        if part not in self._connections:
            path = self.root / self.manifest["shards"][part]
            self._connections[part] = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
            if len(self._connections) > 16:
                _, old = self._connections.popitem(last=False)
                old.close()
        self._connections.move_to_end(part)
        blob = self._connections[part].execute("SELECT png FROM images WHERE id=?", (int(self.index["index"][index]),)).fetchone()
        if blob is None:
            raise ValueError(f"Missing prepared image {index}")
        return np.asarray(Image.open(io.BytesIO(blob[0])).convert("RGB"))

    def __getitem__(self, frame):
        import torch
        ai, pad, vi, vpad = window_indices(frame, int(self.end_for_frame[frame]))
        if self.mode == "frozen_video":
            vi, vpad = vi[:1], vpad[:1]
        video = np.stack([self._image(int(i)) for i in vi])
        task = str(int(self.index["task_index"][frame]))
        return {"video": torch.from_numpy(video.copy()).permute(3, 0, 1, 2).float() / 255,
            "action": torch.from_numpy(normalize(self.index["action"][ai], self.manifest["stats"]["action"])),
            "proprio": torch.from_numpy(normalize(self.index["state"][frame:frame + 1], self.manifest["stats"]["observation.state"])),
            "action_is_pad": torch.from_numpy(pad), "image_is_pad": torch.from_numpy(vpad),
            "prompt": self.manifest["tasks"][task]}
