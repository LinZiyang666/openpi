"""The plan §4.3.4 arm table: step_diag arm id -> ``warm_reset`` spec (jax-free).

Kept apart from ``_support`` so the GR00T island's manual test can import it
without the Pi0.5 model stack.
"""

from __future__ import annotations

from typing import Optional

from exp.step_diag import envs as E
from openpi.cache.warm_reset.types import WarmResetSpec

DEFAULT_SEED_KEYS = ("experiment", "task", "orig_init_state_idx", "attempt")


# ------------------------------------------------------------------
# Arm table (plan §4.3.4)
# ------------------------------------------------------------------

#: step_diag served mode (cache start) -> (point, kind, level per policy), Pi0.5 time.
BASE_MODES = {
    "warmreset": ("snapshot", "reset", {"pi05": 1.0, "groot": 1.0}),
    "resetfinal": ("final", "reset", {"pi05": 1.0, "groot": 1.0}),
    "midfinal": ("final", "reset", {"pi05": 0.9, "groot": 0.75}),
    "midfinal50": ("final", "reset", {"pi05": 0.5, "groot": 0.5}),
    "midreset": ("snapshot", "reset", {"pi05": 0.9, "groot": 0.75}),
    "midreset50": ("snapshot", "reset", {"pi05": 0.5, "groot": 0.5}),
    "warmshoot": ("snapshot", "shoot", {"pi05": 1.0, "groot": 1.0}),
    "midshoot": ("snapshot", "shoot", {"groot": 0.75}),
    "midshoot50": ("snapshot", "shoot", {"groot": 0.5}),
}


def split_mode(mode: str) -> tuple[str, str]:
    """``(source, cache mode)`` of a served mode (``selfmidreset`` -> ``("self", "midreset")``)."""
    if mode.startswith("self") and mode[4:] in BASE_MODES:
        return "self", mode[4:]
    return "cache", mode


def spec_for(
    policy: str, arm_id: str, *, evidence_dir: str = "/tmp/warm_reset_evidence", namespace: str = "ns"
) -> Optional[WarmResetSpec]:
    """The ``warm_reset`` spec of a step_diag warm-family arm; ``None`` for the exact resume."""
    mode = E.warm_mode_of(arm_id)
    if mode == "warm":
        return None
    source, base = split_mode(mode)
    point, kind, levels = BASE_MODES[base]
    self_start = source == "self"
    return WarmResetSpec(
        source=source,
        point=point,
        kind=kind,
        level=levels[policy],
        num_steps=E.warm_steps_of(arm_id),
        seed_namespace=namespace if self_start else None,
        seed_keys=DEFAULT_SEED_KEYS if self_start else (),
        evidence_dir=evidence_dir,
    )


def block_of(spec: WarmResetSpec) -> dict:
    """The yaml ``warm_reset`` block that freezes to ``spec``."""
    grid = {"kind": spec.kind, ("entry_t" if spec.kind == "reset" else "step_budget"): spec.level}
    block = {
        "start": {"source": spec.source, "point": spec.point},
        "grid": grid,
        "num_steps": "remaining" if spec.num_steps is None else spec.num_steps,
        "evidence_dir": spec.evidence_dir,
    }
    if spec.self_start:
        block["self_seed"] = {"namespace": spec.seed_namespace, "identity_keys": list(spec.seed_keys)}
    return block


def warm_arms() -> list[tuple[str, str]]:
    """Every (policy, arm) of the step_diag rounds that is a warm-family arm."""
    tables = [
        *[(p, a) for p, arms in E.MACRO13_ARMS_BY_POLICY.items() for a in arms],
        *[(p, a) for p, arms in E.SELF13_ARMS_BY_POLICY.items() for a in arms],
        *[(p, a) for p, arms in E.LIBERO_SELF_ARMS_BY_POLICY.items() for a in arms],
        *[("pi05", a) for a in E.VAR500_ARMS],
        *[(p, a) for p, arms in E.XSEED_ARMS_BY_POLICY.items() for a in arms],
    ]
    out = []
    for policy, arm in tables:
        try:
            E.warm_mode_of(arm)
        except ValueError:
            continue
        if (policy, arm) not in out:
            out.append((policy, arm))
    return out
