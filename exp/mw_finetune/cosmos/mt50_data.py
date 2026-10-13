"""MT50 samples for the official Cosmos loss, using the existing RGB224 cache."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import torch
from torch.utils.data import Sampler

from exp.metaworld.tasks import MT50_TASKS
from exp.mw_finetune.fastwam.data import Dataset as RGBDataset

CHUNK = 16
PREPROCESSING = dict(image_resize="bilinear", use_jpeg_compression=False, trained_with_image_aug=True)


def canonical_prompts(source_tasks):
    prompts = {prompt for _, prompt, _ in MT50_TASKS}
    push_back = next(prompt for task, prompt, _ in MT50_TASKS if task == "push-back-v3")
    aliases = {"Push the puck back toward the robot to the goal": push_back}
    result = {str(index): aliases.get(text, text) for index, text in source_tasks.items()}
    unknown = set(result.values()) - prompts
    if unknown:
        raise ValueError(f"Dataset prompts missing from our MT50 table: {unknown}")
    return result


def proprio(state):
    s = np.asarray(state, dtype=np.float32)
    result = np.zeros((*s.shape[:-1], 9), dtype=np.float32)
    result[..., 0], result[..., 1] = s[..., 3], -s[..., 3]
    result[..., 2:5], result[..., 8] = s[..., :3], 1
    return result


def actions(action):
    a = np.asarray(action, dtype=np.float32)
    result = np.zeros((*a.shape[:-1], 7), dtype=np.float32)
    result[..., :3], result[..., 6] = a[..., :3], a[..., 3]
    return result


def dataset_statistics(index):
    stats = {}
    for name, values in (("proprio", proprio(index["state"])), ("actions", actions(index["action"]))):
        values = values.astype(np.float64)
        low, high = values.min(0), values.max(0)
        constant = high == low
        # Fixed rotation / quaternion channels normalize to zero, without 0/0.
        low[constant] -= .5
        high[constant] += .5
        for key, value in (("min", low), ("max", high), ("mean", values.mean(0)), ("std", values.std(0))):
            stats[name + "_" + key] = value.tolist()
    return stats


def normalize(values, stats, name):
    low, high = np.asarray(stats[name + "_min"]), np.asarray(stats[name + "_max"])
    return (2 * (values - low) / (high - low) - 1).astype(np.float32)


def prepare_assets(prepared, source, embeddings, output):
    """CPU only. Recompute numeric statistics; reuse unmodified prepared images."""
    import pyarrow.parquet as pq
    prepared, source, output = Path(prepared), Path(source), Path(output)
    with np.load(prepared / "index.npz") as index:
        stats = dataset_statistics(index)
        episodes = index["episode_index"].copy()
        count = len(index["state"])
    success = np.zeros(int(episodes.max()) + 1, dtype=bool)
    for path in sorted(source.glob("data/chunk-*/*.parquet")):
        rows = pq.read_table(path, columns=["episode_index", "next.success"]).to_pydict()
        ep = np.asarray(rows["episode_index"], dtype=int)
        ok = np.asarray(rows["next.success"], dtype=bool).reshape(-1)
        np.logical_or.at(success, ep, ok)
    source_tasks = {int(r["task_index"]): r.get("task", r.get("__index_level_0__"))
                    for r in pq.read_table(source / "meta/tasks.parquet").to_pylist()}
    task_prompts = canonical_prompts(source_tasks)
    with Path(embeddings).open("rb") as f:
        cache = pickle.load(f)
    canonical = {prompt: cache[prompt].cpu() for _, prompt, _ in MT50_TASKS}
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_statistics.json").write_text(json.dumps(stats, indent=2) + "\n")
    with (output / "t5_embeddings.pkl").open("wb") as f:
        pickle.dump(canonical, f)
    np.save(output / "episode_success.npy", success)
    (output / "task_prompts.json").write_text(json.dumps(task_prompts, indent=2) + "\n")
    report = dict(frames=count, episodes=len(np.unique(episodes)), successful_episodes=int(success.sum()),
        prompt_entries=len(canonical), source_fps=80,
        prompt_rewrites=[dict(dataset_prompt_index=i, dataset=text, serving=task_prompts[str(i)])
                         for i, text in source_tasks.items() if text != task_prompts[str(i)]],
        statistics="recomputed from all prepared float32 state/action rows; constant channels centered +/-0.5",
        camera="corner2 RGB480 rotated 180; existing lossless bilinear RGB224 PNG cache",
        preprocessing=PREPROCESSING)
    (output / "data_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


class FrameSampler(Sampler):
    """Resume at an optimizer boundary; deterministic shuffled epochs, no padding."""
    def __init__(self, size, seed, start, count):
        self.size, self.seed, self.start, self.count = size, seed, start, count

    def __iter__(self):
        draw, stop = self.start, self.start + self.count
        while draw < stop:
            epoch, offset = divmod(draw, self.size)
            order = np.random.default_rng(self.seed + epoch).permutation(self.size)
            for frame in order[offset:min(self.size, offset + stop - draw)]:
                yield int(frame), draw
                draw += 1

    def __len__(self):
        return self.count


class MT50Dataset(RGBDataset):
    def __init__(self, prepared, assets, *, seed=7, augment=True):
        super().__init__(prepared)
        self.stats = json.loads((Path(assets) / "dataset_statistics.json").read_text())
        with (Path(assets) / "t5_embeddings.pkl").open("rb") as f:
            self.embeddings = pickle.load(f)
        self.success = np.load(Path(assets) / "episode_success.npy")
        self.prompts = json.loads((Path(assets) / "task_prompts.json").read_text())
        self.seed, self.augment = seed, augment
        # Policy uses successful demonstrations; world/value also see failures.
        self.policy_frames = np.flatnonzero(self.success[self.index["episode_index"]])
        if not len(self.policy_frames):
            raise ValueError("No successful demonstrations for policy training")

    def __getitem__(self, key):
        from cosmos_policy.datasets.dataset_utils import preprocess_image
        from cosmos_policy.experiments.robot.cosmos_utils import apply_image_transforms
        frame, draw = key if isinstance(key, tuple) else (int(key), int(key))
        rng = np.random.default_rng(self.seed + draw)
        kind = int(rng.integers(4))  # 0,1=policy; 2=world model; 3=value
        if kind < 2 and not self.success[self.index["episode_index"][frame]]:
            frame = int(rng.choice(self.policy_frames))
        end = int(self.end_for_frame[frame])
        future = min(frame + CHUNK, end - 1)
        offsets = np.minimum(frame + np.arange(CHUNK), end - 1)
        current_image, future_image = self._image(frame), self._image(future)
        pair = np.stack([current_image, future_image])
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.seed + draw)
            if self.augment:
                pair = preprocess_image(pair, 224, use_image_aug=True, stronger_image_aug=True)
                pair = pair.permute(1, 2, 3, 0).numpy()
            else:
                pair = apply_image_transforms(pair)
        blank = np.zeros_like(pair[0])
        slots = [blank, blank, pair[0], blank, blank, pair[1], blank]
        video = np.concatenate([blank[None]] + [np.repeat(s[None], 4, 0) for s in slots[1:]])
        ep = int(self.index["episode_index"][frame])
        value = 2 * float(self.success[ep]) * .99 ** (end - 1 - future) - 1
        prompt = self.prompts[str(int(self.index["task_index"][frame]))]
        sample = dict(dataset_name="video_data", video=torch.from_numpy(video.transpose(3, 0, 1, 2).copy()),
            actions=normalize(actions(self.index["action"][offsets]), self.stats, "actions"),
            proprio=normalize(proprio(self.index["state"][frame]), self.stats, "proprio"),
            future_proprio=normalize(proprio(self.index["state"][future]), self.stats, "proprio"),
            t5_text_embeddings=self.embeddings[prompt].squeeze(0), t5_text_mask=torch.ones(512, dtype=torch.int64),
            fps=16, padding_mask=torch.zeros(1, 224, 224), image_size=torch.full((4,), 224),
            num_conditional_frames=3, __key__=frame,
            rollout_data_mask=int(kind >= 2), rollout_data_success_mask=int(self.success[ep]),
            world_model_sample_mask=int(kind == 2), value_function_sample_mask=int(kind == 3),
            value_function_return=np.float32(value), global_rollout_idx=ep)
        indices = dict(current_proprio=1, current_image=2, action=3, future_proprio=4, future_image=5, value=6,
            current_wrist_image=-1, current_wrist_image2=-1, current_image2=-1,
            future_wrist_image=-1, future_wrist_image2=-1, future_image2=-1)
        sample.update({name + "_latent_idx": idx for name, idx in indices.items()})
        return sample


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", default="/data/mw_finetune/fastwam/mt50_224")
    p.add_argument("--source", default="/data/mw_finetune/datasets/metaworld_mt50")
    p.add_argument("--embeddings", default="/data/mw_finetune/models/cosmos_mw/t5_embeddings.pkl")
    p.add_argument("--output", default="/data/mw_finetune/cosmos/mt50_assets")
    a = p.parse_args()
    prepare_assets(a.prepared, a.source, a.embeddings, a.output)
