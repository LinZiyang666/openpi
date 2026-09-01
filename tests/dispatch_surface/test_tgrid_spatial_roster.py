"""The frozen 29-cell threshold grid extends to libero_spatial without touching the l10 spec."""
from __future__ import annotations

import pytest

from exp.dispatch_surface.phase0_roster import (
    TGRID_SUITES,
    tgrid_cells,
    tgrid_roster_spec,
    tgrid_roster_spec_digest,
)

L10_FROZEN_DIGEST = "013468854bd94b47db4f5ab9fb9a79ec58065eba911e221e9dcac10c8e2a16c7"


def test_l10_spec_digest_is_byte_identical_to_the_freeze():
    assert tgrid_roster_spec_digest("libero_10") == L10_FROZEN_DIGEST


def test_spatial_spec_mirrors_the_grid_but_binds_its_own_suite():
    spec = tgrid_roster_spec("libero_spatial")
    l10 = tgrid_roster_spec("libero_10")
    assert spec["suite"] == "libero_spatial" and len(spec["arms"]) == len(tgrid_cells()) == 29
    assert spec["arms"] == l10["arms"] and spec["fh"] == l10["fh"] and spec["ws"] == l10["ws"]
    assert tgrid_roster_spec_digest("libero_spatial") != L10_FROZEN_DIGEST


def test_unknown_suites_are_still_refused():
    assert TGRID_SUITES == ("libero_10", "libero_spatial")
    with pytest.raises(SystemExit, match="frozen"):
        tgrid_roster_spec("libero_object")
