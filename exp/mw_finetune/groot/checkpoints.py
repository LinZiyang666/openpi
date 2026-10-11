"""Validate resumable snapshots and preserve interrupted writes without deletion."""
from __future__ import annotations

import json
from pathlib import Path
import re
import time


def complete_checkpoint(path):
    path = Path(path)
    required = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth",
                "config.json", "experiment_cfg/metadata.json"]
    if not all((path / filename).is_file() for filename in required):
        return False
    try:
        state = json.loads((path / "trainer_state.json").read_text())
        if state["global_step"] != int(path.name.split("-")[-1]):
            return False
        index = path / "model.safetensors.index.json"
        if index.exists():
            weights = set(json.loads(index.read_text())["weight_map"].values())
            if not weights or not all((path / name).is_file() for name in weights):
                return False
        elif not (path / "model.safetensors").is_file():
            return False
    except (KeyError, ValueError, OSError):
        return False
    return True


def checkpoint_status(output):
    candidates = sorted((p for p in Path(output).glob("checkpoint-*")
                         if re.fullmatch(r"checkpoint-\d+", p.name)),
                        key=lambda p: int(p.name.split("-")[-1]))
    complete, incomplete = [], []
    for path in candidates:
        (complete if complete_checkpoint(path) else incomplete).append(path)
    return complete, incomplete


def preserve_incomplete(paths):
    renamed = []
    for path in paths:
        preserved = path.with_name(f"incomplete-{path.name}-{time.time_ns()}")
        path.rename(preserved)
        renamed.append(str(preserved))
    return renamed


def validate_resume_statistics(checkpoint, dataset):
    from .conventions import CHANNELS, EMBODIMENT
    saved = json.loads((Path(checkpoint) / "experiment_cfg/metadata.json").read_text())[EMBODIMENT]
    current = json.loads((Path(dataset) / "meta/stats.json").read_text())
    for group, original_key in [("state", "observation.state"), ("action", "action")]:
        for i, name in enumerate(CHANNELS):
            for statistic, numbers in current[original_key].items():
                if saved["statistics"][group][name][statistic] != [numbers[i]]:
                    raise ValueError(f"resume normalization differs: {group}.{name}/{statistic}")
