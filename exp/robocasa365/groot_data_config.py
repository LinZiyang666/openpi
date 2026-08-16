"""GR00T data config for the RoboCasa365 ``new_embodiment`` checkpoint.

``Gr00tPolicy`` does not carry its own modality configuration: the caller must
supply both ``modality_config()`` and ``transform()``.  Neither ships with the
checkpoint (its ``experiment_cfg/`` holds only ``metadata.json``, i.e. statistics
plus an alphabetically-sorted key listing), and ``gr00t``'s ``DATA_CONFIG_MAP``
has no RoboCasa entry -- hence this file.

Every ordered list below is sourced from ``exp.robocasa365.groot_keys``, which
transcribes the official RoboCasa365 configuration verbatim; see that module for
the citation and for why ordering is safety-critical.  The lists are restated
here (rather than inherited) so the contract is readable at the point of use and
so an upstream reordering in the parent class cannot silently change it.

What is inherited and still unproven
------------------------------------
The parent supplies the normalization modes (``min_max``), the rotation target
(``rotation_6d``) and the binary treatment of ``gripper_close`` / ``control_mode``.
The official config does *not* document these, so they remain an inference
guarded by the G0-B behavioural gate described in the plan.

Environment
-----------
Imports ``gr00t``; runnable only inside the GR00T island on ``weilandserver``:
``/home/weiland/gr00t_n15_venv/.venv``.
"""

from __future__ import annotations

from gr00t.experiment.data_config import SinglePandaGripperDataConfig

from exp.robocasa365 import groot_keys


class RoboCasa365DataConfig(SinglePandaGripperDataConfig):
    """Modality contract for the official RoboCasa365 GR00T N1.5 checkpoint.

    Differs from the parent in the camera names and -- critically -- in the
    language key: the parent declares N1.7's ``ROBOCASA_PANDA`` variant
    (``annotation.human.action.task_description``), whereas N1.5
    ``new_embodiment`` uses the env-native ``annotation.human.task_description``.
    """

    video_keys = list(groot_keys.VIDEO_KEYS)
    state_keys = list(groot_keys.STATE_KEYS)
    action_keys = list(groot_keys.ACTION_KEYS)
    language_keys = list(groot_keys.LANGUAGE_KEYS)

    observation_indices = [0]
    action_indices = list(range(groot_keys.ACTION_HORIZON))
