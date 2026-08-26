"""Evidence-emitting episode runner for the ws2 (text-IVF) search round.

Overrides exactly one seam — ``RobocasaEpisodeRunner._episode_header_rows``,
the single-reset capture hook — to prepend one header row per episode carrying
the ground-truth ``prompt``/``seed`` the environment actually produced. The
row rides the ordinary ``EpisodeResult.per_step_rows`` channel next to the
``__hit_meta__`` rows, so the ws2 bucket-attribution join gets, per
``(task_uid, attempt)``, both the eval-side prompt and the winner ids without
any seed replay.

Public interface: ``Ws2EpisodeRunner`` (drop-in for ``RobocasaEpisodeRunner``;
constructed by ``worker_entry`` only under ``--episode-header-rows``).
"""

from __future__ import annotations

from typing import Any

from openpi.conductor import task as _task

from exp.robocasa365.episode_runner import RobocasaEpisodeRunner

# Sentinel step index for the header row: real decision rows use step_idx >= 0.
HEADER_STEP_IDX = -1


class Ws2EpisodeRunner(RobocasaEpisodeRunner):
    """RobocasaEpisodeRunner emitting one prompt/seed header row per episode."""

    def _episode_header_rows(
        self, task: _task.EpisodeTask, *, prompt: str, seed: int
    ) -> list[dict[str, Any]]:
        return [
            {
                "task_uid": task.task_uid,
                "yaml_id": task.yaml_id,
                "step_idx": HEADER_STEP_IDX,
                "attempt": task.attempt,
                "prompt": prompt,
                "seed": seed,
            }
        ]
