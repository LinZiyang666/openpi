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
#: Pre-registered aggressiveness order of the threshold family: endpoints
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
