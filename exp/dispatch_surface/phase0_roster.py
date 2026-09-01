"""Frozen Phase 0 exploratory rosters (Rev 2 protocol section 7, plan section 3.3).

The roster is a code constant, not a command-line choice: the emitter refuses
any arm set that is not exactly this one, the runner refuses any matrix whose
roster spec digest differs, and the cost-map refuses sources it cannot map
back to these roles. Quantiles are quantiles of D_dev.y10 with NumPy's
``method="linear"`` -- the same call the Rev 1 delta grid uses.
"""

from __future__ import annotations

import hashlib
import json

LAYER_EXPLORATORY = "exploratory"
PROTOCOL_PHASE0 = "dispatch_surface_rev2_phase0"
ANCHOR_ARM = "always_full_inference"
ANCHOR_ROLE = "always_full_inference_anchor"
#: A cosine score (L2-normalised dot product) can never exceed 1.0, so a
#: threshold judge at this value returns MISS on every step while still
#: searching (the analytic cost is then exactly unit_cost("MISS")).
ANCHOR_THRESHOLD = 1.5

FAMILY_SV = "sv"
FAMILY_S0 = "s0"
FAMILY_ANCHOR = "anchor"
FAMILY_THRESHOLD = "threshold"

#: suite -> arm -> (family, quantile). ``quantile`` is None for the anchor.
ROSTERS: dict[str, dict[str, tuple[str, float | None]]] = {
    "libero_10": {
        ANCHOR_ARM: (FAMILY_ANCHOR, None),
        "dsp_sv_p85": (FAMILY_SV, 0.85),
        "dsp_s0_p80": (FAMILY_S0, 0.80),
        "dsp_s0_p95": (FAMILY_S0, 0.95),
    },
    "libero_spatial": {
        ANCHOR_ARM: (FAMILY_ANCHOR, None),
        "dsp_sv_p95": (FAMILY_SV, 0.95),
        "dsp_sv_p975": (FAMILY_SV, 0.975),
        "dsp_s0_p80": (FAMILY_S0, 0.80),
        "dsp_s0_p95": (FAMILY_S0, 0.95),
    },
}

#: The SV exploratory arm whose artifact supplies the launch contract
#: (h_exec, policy fingerprint). Frozen per suite; the runner never guesses.
CONTRACT_ANCHOR_ARM = {"libero_10": "dsp_sv_p85", "libero_spatial": "dsp_sv_p95"}

#: Rev 1 primary points that join the cost-map candidate set, with their
#: family and D_dev.y10 quantile (delta* is the p90 grid point in both suites;
#: SV- is the p80 neighbour). The threshold arms carry no delta.
REV1_CANDIDATES: dict[str, tuple[str, float | None]] = {
    "dsp_sv_minus": (FAMILY_SV, 0.80),
    "dsp_sv": (FAMILY_SV, 0.90),
    "dsp_s0": (FAMILY_S0, 0.90),
    "dsp_t_fh30_ws20": (FAMILY_THRESHOLD, None),
    "dsp_t_fh50_ws20": (FAMILY_THRESHOLD, None),
    "dsp_t_fh70_ws10": (FAMILY_THRESHOLD, None),
}
#: Pre-registered aggressiveness order of the GST (threshold) family: endpoints
#: are fh70 (cheapest) and fh30 (most expensive); the middle is fh50. They
#: never pass through the isotonic fit because they carry no delta.
THRESHOLD_ORDER = ("dsp_t_fh70_ws10", "dsp_t_fh50_ws20", "dsp_t_fh30_ws20")


def roster_spec(suite: str) -> dict:
    """JSON-serialisable roster spec for one suite."""
    if suite not in ROSTERS:
        raise SystemExit(f"no frozen Phase 0 roster for suite {suite!r}")
    arms = {
        arm: {"family": family, "quantile": quantile}
        for arm, (family, quantile) in ROSTERS[suite].items()
    }
    return {
        "protocol": PROTOCOL_PHASE0,
        "layer": LAYER_EXPLORATORY,
        "suite": suite,
        "arms": arms,
        "contract_anchor_arm": CONTRACT_ANCHOR_ARM[suite],
        "anchor_arm": ANCHOR_ARM,
        "anchor_role": ANCHOR_ROLE,
        "anchor_threshold": ANCHOR_THRESHOLD,
        "rev1_candidates": {
            arm: {"family": family, "quantile": quantile}
            for arm, (family, quantile) in REV1_CANDIDATES.items()
        },
        "threshold_order": list(THRESHOLD_ORDER),
    }


def roster_spec_digest(suite: str) -> str:
    blob = json.dumps(roster_spec(suite), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def assert_roster(suite: str, arms: dict) -> None:
    """Refuse an arm map (arm -> {family, quantile}) that is not exactly the roster."""
    want = roster_spec(suite)["arms"]
    got_ids, want_ids = set(arms), set(want)
    if got_ids != want_ids:
        raise SystemExit(
            f"{suite}: roster mismatch — missing {sorted(want_ids - got_ids)}, "
            f"unexpected {sorted(got_ids - want_ids)}"
        )
    for arm, spec in want.items():
        got = arms[arm]
        if got.get("family") != spec["family"] or got.get("quantile") != spec["quantile"]:
            raise SystemExit(
                f"{suite}: arm {arm} declares {got}, roster freezes {spec}"
            )
    seen = [spec["quantile"] for arm, spec in want.items() if spec["quantile"] is not None]
    if len(seen) != len(set((want[a]["family"], want[a]["quantile"]) for a in want if want[a]["quantile"] is not None)):
        raise SystemExit(f"{suite}: roster carries a duplicate (family, quantile)")


# ----------------------------------------------------------------------
# Rev 2 confirmation plan (plan section 3.2 / 3.3): dense GST threshold grid and
# the C-roster selector constants. Frozen at G1 Round 3; the freeze record in
# exp/dispatch_surface/config/confirmation_freeze_record.json pins them.
# ----------------------------------------------------------------------

LAYER_TGRID = "exploratory_tgrid"
PROTOCOL_TGRID = "dispatch_surface_rev2_tgrid_dev"
TGRID_SUITE = "libero_10"
#: Post-hoc exploratory extension (owner goal, 2026-08-30): the identical
#: 29-cell grid may also be emitted for libero_spatial. The l10 spec bytes and
#: digest are untouched -- the spec depends only on the suite string passed in.
TGRID_SUITES = (TGRID_SUITE, "libero_spatial")
THRESHOLD_GRID_FH = (20, 30, 40, 50, 60, 70, 80)
THRESHOLD_GRID_WS = (0, 10, 20, 30, 40)
#: Rev 1 cells already measured (see REV1_CANDIDATES); they are NOT re-emitted.
REV1_THRESHOLD_CELLS = ((30, 20), (50, 20), (70, 10))
#: C-roster selector (plan 3.3-2): bootstrap frequency floor and per-family cap.
F_MIN = 0.20
M_MAX = 6


def tgrid_cells(*, include_rev1: bool = False) -> list[tuple[int, int]]:
    """Legal grid cells (fh + ws <= 100) in row-major (fh, ws) order; 32 in
    total, 29 once the three Rev 1 cells are excluded."""
    cells = [(fh, ws) for fh in THRESHOLD_GRID_FH for ws in THRESHOLD_GRID_WS if fh + ws <= 100]
    if include_rev1:
        return cells
    return [c for c in cells if c not in REV1_THRESHOLD_CELLS]


def tgrid_arm_id(fh: int, ws: int) -> str:
    if (fh, ws) not in tgrid_cells(include_rev1=True):
        raise SystemExit(f"({fh}, {ws}) is not a legal threshold-grid cell")
    return f"dsp_tg_fh{fh}_ws{ws}"


def tgrid_roster_spec(suite: str) -> dict:
    if suite not in TGRID_SUITES:
        raise SystemExit(f"the dense threshold grid is frozen for {TGRID_SUITES!r} only, not {suite!r}")
    return {
        "protocol": PROTOCOL_TGRID,
        "layer": LAYER_TGRID,
        "suite": suite,
        "fh": list(THRESHOLD_GRID_FH),
        "ws": list(THRESHOLD_GRID_WS),
        "legality": "fh + ws <= 100",
        "rev1_cells_excluded": [list(c) for c in REV1_THRESHOLD_CELLS],
        "arms": {tgrid_arm_id(fh, ws): {"family": FAMILY_THRESHOLD, "fh": fh, "ws": ws}
                 for fh, ws in tgrid_cells()},
        "contract_source": "rev1_package:artifact.dsp_sv",
        "ws_zero_representation": "no warm_tiers key",
    }


def tgrid_roster_spec_digest(suite: str) -> str:
    blob = json.dumps(tgrid_roster_spec(suite), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()

#: RIT-PL (piecewise-linear risk curve, IR-addressed) exploratory family; it
#: never enters a frozen roster (``assert_roster`` and ``ROSTERS`` are unchanged).
FAMILY_S0_PL = "s0_pl"
