from __future__ import annotations

import datetime
import logging
import os
import pathlib
import threading
from dataclasses import dataclass
from typing import Any

import h5py
import numpy as np

logger = logging.getLogger(__name__)


# Identity keys persisted from ``extra_metadata`` into the episode's HDF5
# attrs. An allowlist, not a passthrough: free-form client metadata must
# not silently become schema (dispatch-surface cohort identity, G2-B4).
# Module-level so the trace writer (``openpi.cache.trace.h5_sink``) and the
# legacy collector persist exactly the same identity keys.
METADATA_ATTR_ALLOWLIST = (
    "task_id",
    "init_state_idx",
    "orig_init_state_idx",
    "subset_init_state_idx",
    "split",
    # Pinned-object provenance: the identity the episode claims plus the
    # slot->mesh map the scene actually realized (JSON). The auditor admits
    # an episode on the realized value, so it has to survive the allowlist
    # or the check would silently have nothing to read.
    "pin_id",
    "pin_task_id",
    "realized_objects",
)


def resolve_episode_path(
    base_dir: pathlib.Path,
    experiment: str,
    episode_id: int,
    episode_name: str,
    *,
    pid_suffix: bool = False,
) -> pathlib.Path:
    """Final ``.h5`` path for one episode, creating the parent directories.

    ``episode_name`` may embed subdirs ("task_3/episode_7"); the resolved
    candidate is asserted to stay inside ``base_dir/experiment`` so a hostile
    value like "../../etc/passwd" cannot escape. Empty ``episode_name`` falls
    back to the legacy timestamp naming; ``pid_suffix`` appends ``_p<pid>`` to
    that fallback so several server processes writing the same root cannot
    collide (the trace writer turns it on; the legacy collector keeps its
    exact historical names).
    """
    out_dir = pathlib.Path(base_dir) / experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    if episode_name:
        candidate = (out_dir / f"{episode_name}.h5").resolve()
        out_dir_resolved = out_dir.resolve()
        if not candidate.is_relative_to(out_dir_resolved):
            raise ValueError(
                f"episode_name {episode_name!r} escapes base directory "
                f"{out_dir_resolved}; refusing to write."
            )
        candidate.parent.mkdir(parents=True, exist_ok=True)
        return candidate
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    suffix = f"_p{os.getpid()}" if pid_suffix else ""
    return out_dir / f"episode_{episode_id:04d}_{ts}{suffix}.h5"


def write_episode_attrs(
    f: h5py.File,
    *,
    experiment: str,
    task: str,
    episode_id: int,
    num_steps: int,
    success: bool,
    episode_attrs: dict[str, Any],
) -> None:
    """Write the file-level attrs of one collected episode.

    Standard keys are written first so ``episode_attrs`` cannot accidentally
    shadow bookkeeping fields; any overlap is still allowed (last-write-wins)
    if an advanced caller really wants to override, e.g. a canonicalised
    ``task`` value.
    """
    f.attrs["experiment_name"] = experiment
    f.attrs["task"] = task
    f.attrs["episode_id"] = episode_id
    f.attrs["num_steps"] = num_steps
    f.attrs["timestamp"] = datetime.datetime.now().isoformat()
    f.attrs["success"] = success
    for k, v in episode_attrs.items():
        f.attrs[k] = v


def write_step_group(grp: h5py.Group, embs: "InferenceEmbeddings") -> None:
    """Write one step's legacy datasets into an already-created step group.

    This is the single definition of the collected-step schema: the legacy
    collector and the trace writer both call it, which is what makes a trace
    file a strict superset of a collected one.
    """
    for i, vision_emb in enumerate(embs.vision_embs):
        grp.create_dataset(f"vision_{i}", data=vision_emb, compression="lzf")
    grp.create_dataset("prompt_emb", data=embs.prompt_emb, compression="lzf")
    grp.create_dataset("robot_state", data=embs.robot_state)
    if embs.init_noise is not None:
        grp.create_dataset("noise_action_0", data=embs.init_noise)
    for i, noise_action in enumerate(embs.noise_action_steps, start=1):
        grp.create_dataset(f"noise_action_{i}", data=noise_action)
    grp.create_dataset("clean_action", data=embs.clean_action)
    if embs.input_images:
        img_grp = grp.create_group("input_images")
        for key, img in embs.input_images.items():
            img_grp.create_dataset(key, data=img, compression="lzf")


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
    # The pure-noise x_0 the denoise loop started from. Written as
    # ``noise_action_0`` so a teacher action can be reproduced bit-for-bit;
    # never a warm-start point, so it stays out of ``noise_action_steps`` and
    # the 1-based numbering below is untouched.
    init_noise: np.ndarray | None = None


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

    # Kept as a class attribute for callers that read it; the definition
    # lives at module level so the trace writer shares it.
    _METADATA_ATTR_ALLOWLIST = METADATA_ATTR_ALLOWLIST

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
        logger.info(
            "EpisodeDataCollector: episode %d started (%s / %s)",
            episode_id,
            experiment,
            task,
        )

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
                logger.warning(
                    "EpisodeDataCollector: episode %d has no data, skipping write.",
                    self._episode_id,
                )
                return

            buffer = self._buffer
            experiment = self._experiment
            task = self._task
            episode_id = self._episode_id
            episode_name = self._episode_name
            episode_attrs = dict(self._episode_attrs)  # snapshot under lock
            self._buffer = []

        path = resolve_episode_path(
            self._base_dir, experiment, episode_id, episode_name
        )
        # ``with_suffix`` on a ``.h5`` path yields ``.h5.tmp`` cleanly even when
        # ``path.name`` already ends in ``.h5``.
        tmp_path = path.with_suffix(".h5.tmp")

        try:
            with h5py.File(tmp_path, "w") as f:
                write_episode_attrs(
                    f,
                    experiment=experiment,
                    task=task,
                    episode_id=episode_id,
                    num_steps=len(buffer),
                    success=success,
                    episode_attrs=episode_attrs,
                )
                for step_idx, embs in enumerate(buffer):
                    grp = f.create_group(f"step_{step_idx:04d}")
                    write_step_group(grp, embs)

            tmp_path.rename(path)
            logger.info(
                "EpisodeDataCollector: episode %d written -> %s (%d steps, success=%s)",
                episode_id,
                path,
                len(buffer),
                success,
            )
        except Exception:
            logger.exception(
                "EpisodeDataCollector: failed to write episode %d", episode_id
            )
            tmp_path.unlink(missing_ok=True)
