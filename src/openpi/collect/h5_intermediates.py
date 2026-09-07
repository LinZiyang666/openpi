"""Read denoise-loop snapshots out of collected HDF5 episodes.

Every offline library builder used to carry its own copy of the same block:
scan ``noise_action_*``, assume ten Pi0.5 steps, map index ``i`` to
``t = 1 - i/10``. That assumption is wrong twice for GR00T (different step
count, opposite time direction) and fails silently -- a four-step file's
``noise_action_1..3`` reads back as t = 0.9/0.8/0.7, every one a legal Pi0.5
timestep. This module is the single place that mapping lives now, and it takes
the step count and direction from the file's own ``denoise_schedule_id`` /
``denoising_num_steps`` attributes (``openpi.cache.types.DenoiseSchedule``).

Public interface: ``episode_schedule`` (file attrs -> schedule or ``None`` for
files written before schedules were stamped) and ``read_step_intermediates``
(one step group -> ``(intermediates, denoising_num_steps)`` ready for
``CachePayload``). Lives in the installed package rather than ``exp/common`` so
the builders keep working when run as plain scripts from that directory.
Depends only on numpy, torch and ``openpi.cache.types``.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import torch

from openpi.cache.types import PI05_V1, DenoiseSchedule, schedule_from_id

SCHEDULE_ID_ATTR = "denoise_schedule_id"
NUM_STEPS_ATTR = "denoising_num_steps"
_NOISE_ACTION_RE = re.compile(r"^noise_action_(\d+)$")


def _attr_str(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode()
    return str(value)


def episode_schedule(h5_file: Any) -> DenoiseSchedule | None:
    """Schedule the episode was collected under, from its file-level attrs.

    Returns ``None`` for files written before the collectors stamped one --
    every such file came from the Pi0.5 loop, and the caller decides whether
    that legacy reading is acceptable for the library it is building. A stamp
    whose step count disagrees with its own id is corrupt and raises.
    """
    attrs = h5_file.attrs
    if SCHEDULE_ID_ATTR not in attrs:
        return None
    schedule = schedule_from_id(_attr_str(attrs[SCHEDULE_ID_ATTR]))
    if NUM_STEPS_ATTR not in attrs:
        raise ValueError(
            f"{h5_file.filename}: {SCHEDULE_ID_ATTR}={schedule.schedule_id!r} but "
            f"no {NUM_STEPS_ATTR} attribute; the stamp is incomplete."
        )
    recorded = int(attrs[NUM_STEPS_ATTR])
    if recorded != schedule.num_steps:
        raise ValueError(
            f"{h5_file.filename}: {NUM_STEPS_ATTR}={recorded} disagrees with "
            f"{schedule.schedule_id!r} ({schedule.num_steps} steps)."
        )
    return schedule


def snapshot_indices(group: Any) -> list[int]:
    """Sorted ``i`` of every ``noise_action_i`` dataset with ``i >= 1``.

    ``noise_action_0`` is the pure-noise start (``InferenceEmbeddings.init_noise``);
    it is deliberately not a snapshot and is skipped here.
    """
    indices: list[int] = []
    for name in group.keys():
        match = _NOISE_ACTION_RE.match(name)
        if match is None:
            continue
        index = int(match.group(1))
        if index >= 1:
            indices.append(index)
    return sorted(indices)


def read_step_intermediates(
    group: Any, schedule: DenoiseSchedule | None
) -> tuple[dict[float, torch.Tensor] | None, int | None]:
    """Snapshots of one step group keyed by flow time, plus the loop length.

    ``schedule`` is the file's stamp (``episode_schedule``). A stamped file must
    carry exactly the ``num_steps - 1`` snapshots its loop produces -- a gap or
    an extra index means the collector and the stamp disagree about the loop,
    and the whole point of the stamp is that such a file never becomes a
    library. An unstamped file is read as the Pi0.5 legacy loop with the old
    tolerance for partial snapshot sets. A group with no snapshots at all
    yields ``(None, None)`` under either reading.
    """
    indices = snapshot_indices(group)
    if not indices:
        return None, None
    if schedule is None:
        schedule = PI05_V1
        too_high = [i for i in indices if i >= schedule.num_steps]
        if too_high:
            raise ValueError(
                f"{group.name}: unstamped file has noise_action_{too_high} but the "
                f"legacy {schedule.schedule_id} loop only has indices 1.."
                f"{schedule.num_steps - 1}; stamp the file with its real schedule."
            )
    else:
        expected = list(range(1, schedule.num_steps))
        if indices != expected:
            raise ValueError(
                f"{group.name}: snapshot indices {indices} do not match the "
                f"{schedule.schedule_id} loop, which writes exactly {expected}."
            )
    intermediates = {
        schedule.snapshot_t(i): torch.from_numpy(
            np.array(group[f"noise_action_{i}"])
        ).float()
        for i in indices
    }
    return intermediates, schedule.num_steps
