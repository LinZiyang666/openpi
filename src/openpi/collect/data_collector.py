from __future__ import annotations

import datetime
import logging
import pathlib
import threading
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class InferenceEmbeddings:
    """Embeddings captured for a single infer() call."""

    vision_embs: list[np.ndarray]
    prompt_emb: np.ndarray
    robot_state: np.ndarray
    noise_action_steps: list[np.ndarray]
    clean_action: np.ndarray
    # Post-transform model input images, mask=True slots only.
    # e.g. {"base_0_rgb": (224,224,3) uint8, "left_wrist_0_rgb": ...}
    input_images: dict[str, np.ndarray] | None = None


class EpisodeDataCollector:
    """Buffers per-step embeddings for one episode and flushes them to HDF5."""

    def __init__(self, base_dir: str) -> None:
        self._base_dir = pathlib.Path(base_dir)
        self._buffer: list[InferenceEmbeddings] = []
        self._experiment = "unknown"
        self._task = ""
        self._episode_id = -1
        # Optional client-provided filename stem (may contain subdirs such as
        # "task_3/episode_7"). Empty falls back to legacy timestamp naming.
        self._episode_name = ""
        # Free-form per-episode HDF5 attrs (prompt, init_state_idx, ...).
        # ``clear()`` in ``on_episode_start`` so the dict is reused, keeping
        # any external reference valid; rebinding would break that.
        self._episode_attrs: dict[str, Any] = {}
        self._lock = threading.Lock()

    # Identity keys persisted from ``extra_metadata`` into the episode's HDF5
    # attrs. An allowlist, not a passthrough: free-form client metadata must
    # not silently become schema (dispatch-surface cohort identity, G2-B4).
    _METADATA_ATTR_ALLOWLIST = (
        "task_id", "init_state_idx", "orig_init_state_idx",
        "subset_init_state_idx", "split",
    )

    def on_episode_start(
        self,
        experiment: str,
        task: str,
        episode_id: int,
        *,
        episode_name: str = "",
        extra_metadata: dict | None = None,
    ) -> None:
        with self._lock:
            self._buffer = []
            self._experiment = experiment
            self._task = task
            self._episode_id = episode_id
            self._episode_name = episode_name
            self._episode_attrs.clear()
            if extra_metadata:
                for key in self._METADATA_ATTR_ALLOWLIST:
                    if key in extra_metadata:
                        self._episode_attrs[key] = extra_metadata[key]
        logger.info("EpisodeDataCollector: episode %d started (%s / %s)", episode_id, experiment, task)

    def record_inference(self, embs: InferenceEmbeddings) -> None:
        with self._lock:
            self._buffer.append(embs)

    def set_episode_attr(self, key: str, value: Any) -> None:
        """Record an HDF5-bound attribute for the current episode.

        Calls before ``on_episode_start`` are accepted silently (the attr
        dict survives across episodes until ``on_episode_start`` clears it),
        which matches ``record_inference``'s tolerant contract. Overwriting
        an existing key is intentional — callers may refine values mid-episode
        (for example, extending a prompt string).
        """
        with self._lock:
            self._episode_attrs[key] = value

    def has_pending_data(self) -> bool:
        with self._lock:
            return bool(self._buffer)

    def on_episode_end(self, success: bool) -> None:
        with self._lock:
            if not self._buffer:
                logger.warning("EpisodeDataCollector: episode %d has no data, skipping write.", self._episode_id)
                return

            buffer = self._buffer
            experiment = self._experiment
            task = self._task
            episode_id = self._episode_id
            episode_name = self._episode_name
            episode_attrs = dict(self._episode_attrs)  # snapshot under lock
            self._buffer = []

        out_dir = self._base_dir / experiment
        out_dir.mkdir(parents=True, exist_ok=True)
        if episode_name:
            # ``episode_name`` may embed subdirs ("task_3/episode_7"); resolve
            # and assert containment so a hostile value like "../../etc/passwd"
            # cannot escape ``out_dir``. This is belt-and-braces: the wire
            # format is authenticated server-side, but path traversal is a
            # one-line defense with zero legitimate cost.
            candidate = (out_dir / f"{episode_name}.h5").resolve()
            out_dir_resolved = out_dir.resolve()
            if not candidate.is_relative_to(out_dir_resolved):
                raise ValueError(
                    f"episode_name {episode_name!r} escapes base directory "
                    f"{out_dir_resolved}; refusing to write."
                )
            path = candidate
            path.parent.mkdir(parents=True, exist_ok=True)
        else:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = out_dir / f"episode_{episode_id:04d}_{ts}.h5"
        # ``with_suffix`` on a ``.h5`` path yields ``.h5.tmp`` cleanly even when
        # ``path.name`` already ends in ``.h5``.
        tmp_path = path.with_suffix(".h5.tmp")

        try:
            with h5py.File(tmp_path, "w") as f:
                f.attrs["experiment_name"] = experiment
                f.attrs["task"] = task
                f.attrs["episode_id"] = episode_id
                f.attrs["num_steps"] = len(buffer)
                f.attrs["timestamp"] = datetime.datetime.now().isoformat()
                f.attrs["success"] = success
                # Merge caller-supplied attrs (prompt, init_state_idx, ...).
                # Standard keys above are written first so episode_attrs cannot
                # accidentally shadow bookkeeping fields; any overlap is still
                # allowed (last-write-wins) if an advanced caller really wants
                # to override, e.g. a canonicalised ``task`` value.
                for k, v in episode_attrs.items():
                    f.attrs[k] = v

                for step_idx, embs in enumerate(buffer):
                    grp = f.create_group(f"step_{step_idx:04d}")
                    for i, vision_emb in enumerate(embs.vision_embs):
                        grp.create_dataset(f"vision_{i}", data=vision_emb, compression="lzf")
                    grp.create_dataset("prompt_emb", data=embs.prompt_emb, compression="lzf")
                    grp.create_dataset("robot_state", data=embs.robot_state)
                    for i, noise_action in enumerate(embs.noise_action_steps, start=1):
                        grp.create_dataset(f"noise_action_{i}", data=noise_action)
                    grp.create_dataset("clean_action", data=embs.clean_action)
                    if embs.input_images:
                        img_grp = grp.create_group("input_images")
                        for key, img in embs.input_images.items():
                            img_grp.create_dataset(key, data=img, compression="lzf")

            tmp_path.rename(path)
            logger.info(
                "EpisodeDataCollector: episode %d written -> %s (%d steps, success=%s)",
                episode_id,
                path,
                len(buffer),
                success,
            )
        except Exception:
            logger.exception("EpisodeDataCollector: failed to write episode %d", episode_id)
            tmp_path.unlink(missing_ok=True)
