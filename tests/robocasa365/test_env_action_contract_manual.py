"""Manual test binding our action keys to robocasa's real ``convert_action``.

Why this is separate
--------------------
The non-manual suite can only compare our key list against a transcription of
the contract, which is a self-consistent check: it pins our own constants but
would not notice robocasa changing the shape of ``env.step``'s input.  Closing
that gap needs the actual ``robocasa`` package, which lives in the *simulation*
island -- a different interpreter from the GR00T island, hence a third file
rather than an addition to ``test_groot_data_config_manual.py``.

Run inside the simulation island::

    /home/weiland/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python \\
      -m pip install pytest          # not present there by default
    cd /home/weiland/openpi && \\
    PYTHONPATH=/home/weiland/openpi \\
      /home/weiland/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python \\
      -m pytest tests/robocasa365/test_env_action_contract_manual.py --run-manual -q

``--run-manual`` is required: conftest.py default-skips the manual marker, and
``-m manual`` alone only selects them.
"""

from __future__ import annotations

import numpy as np
import pytest

from exp.robocasa365 import groot_keys

pytest.importorskip("robocasa", reason="robocasa only exists in the simulation island")

pytestmark = pytest.mark.manual


def test_action_keys_match_real_convert_action():
    """Our ACTION_KEYS must be exactly what env.step consumes.

    This is the check the non-manual suite cannot make: it calls the genuine
    ``convert_action`` rather than comparing our constants to themselves.
    """
    from robocasa.utils.env_utils import convert_action

    flat = np.zeros(sum(groot_keys.ACTION_DIMS.values()))
    produced = convert_action(flat)
    assert set(produced) == set(groot_keys.ACTION_KEYS)


def test_action_dims_match_real_convert_action_slices():
    """Each key's width must match the slice convert_action carves out."""
    from robocasa.utils.env_utils import convert_action

    total = sum(groot_keys.ACTION_DIMS.values())
    # A ramp makes each slice identifiable by value, not just by width.
    produced = convert_action(np.arange(total, dtype=np.float64))
    for key, width in groot_keys.ACTION_DIMS.items():
        assert produced[key].shape == (width,), f"{key} width mismatch"


def test_total_action_width_is_12():
    """RoboCasa365's PandaOmron action space is 12-dimensional."""
    assert sum(groot_keys.ACTION_DIMS.values()) == 12
