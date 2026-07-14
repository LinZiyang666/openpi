"""Projected-lane emit / filter / diagnostic consumers (TRACER Phase 6, §6.3-§6.6).

Implements the code that CONSUMES the trained artifacts (numeric weights need not be
fabricated to implement the code that reads them):

  * ``emit_projection_config`` fills a projected-lane YAML template's
    ``__FILL_AT_EXECUTION__`` placeholders (weights path, preload artifact, refit
    normalizers, solved gate betas) into a fully-loadable config;
  * ``ical_filter`` / ``ical_only_entries`` filter a dual artifact to the I_cal (even-init)
    library with a machine-readable drop manifest + fail-loud **zero-odd** guard proven
    against the source scan, not the already-filtered list (Pass-3 self-retrieval guard, §B3).

The Retrieval@K top-1-prefix diagnostic (§C) runs the REAL frozen ``DualRetrievalKnnStrategy``
at K=1 and K=5 -- see ``phase6_assemble.strategy_topk_prefix_ok`` (a re-sort of a caller-supplied
score list would be tautological, so it is not implemented here).
"""

from __future__ import annotations

import re

_PLACEHOLDER = "__FILL_AT_EXECUTION__"


# ------------------------------------------------------------------
# Config emitter (§6.6)
# ------------------------------------------------------------------
def emit_projection_config(
    template_text: str,
    *,
    weights_path: str,
    preload_path: str,
    normalizers: dict,
    betas: dict,
) -> str:
    """Return a fully-loadable projected-lane YAML with all placeholders filled.

    ``normalizers`` = {field: {"mu": .., "sigma": ..}} for vision_0/vision_1/robot_state;
    ``betas`` = {"b0": .., "b3": ..}. Raises if any ``__FILL_AT_EXECUTION__`` remains, so a
    half-filled (non-loadable) config can never be shipped.
    """
    import yaml

    cfg = yaml.safe_load(template_text)
    cfg["key_builder"]["projection"]["weights_path"] = weights_path
    cfg["backend"]["in_memory"]["preload_path"] = preload_path
    judge = cfg["checkpoints"]["cp1"]["judge"]
    judge["gate_betas"]["b0"] = float(betas["b0"])
    judge["gate_betas"]["b3"] = float(betas["b3"])
    norm = cfg["checkpoints"]["cp1"]["search_strategy"]["score_normalization"]["fields"]
    for field, mv in normalizers.items():
        norm[field]["params"]["mu"] = float(mv["mu"])
        norm[field]["params"]["sigma"] = float(mv["sigma"])
    out = yaml.safe_dump(cfg, sort_keys=False)
    if _PLACEHOLDER in out:
        remaining = re.findall(r"\S*" + _PLACEHOLDER + r"\S*", out)
        raise ValueError(f"emitted config still has unfilled placeholders: {remaining}")
    return out


# ------------------------------------------------------------------
# I_cal-only library filter (§B3)
# ------------------------------------------------------------------
def ical_filter(entries: list, ident_fn) -> tuple[list, dict]:
    """Partition ``entries`` into the I_cal (even-init) library + a machine-readable manifest.

    ``ident_fn(entry) -> (task_id, orig_init_state_idx) | None``. Returns ``(kept, manifest)``
    where ``kept`` is the even-init survivors and ``manifest`` records exact drop provenance.
    The old ``ical_only_entries`` was tautological in two ways this fixes: it re-checked the
    already-even ``kept`` list against itself, and it evaluated ``ident_fn`` TWICE per entry
    (partition then re-check), so a non-deterministic resolver could pass a stale check. Here
    each entry is resolved ONCE, routed affirmatively to kept / odd / unresolved, and the
    partition is asserted EXHAUSTIVE (kept+odd+unresolved == total) -- a conservation invariant
    that actually fails if any entry is silently lost. Unresolved rows are COUNTED in the
    manifest, not silently dropped (the second reviewer complaint).
    """
    kept, odd, unresolved = [], [], 0
    for e in entries:
        ident = ident_fn(e)  # resolved exactly once
        if ident is None:
            unresolved += 1
        elif ident[1] % 2 == 0:
            kept.append(e)
        else:
            odd.append(ident)
    if len(kept) + len(odd) + unresolved != len(entries):
        raise ValueError("I_cal partition is not exhaustive (kept+odd+unresolved != total)")
    manifest = {
        "n_total": len(entries),
        "n_kept": len(kept),
        "n_odd_dropped": len(odd),
        "n_unresolved_dropped": unresolved,
        "odd_idents": sorted(set(odd)),
    }
    return kept, manifest


def ical_only_entries(entries: list, ident_fn) -> list:
    """Even-init (I_cal) survivors only; fails loud on any odd-init leakage (see ``ical_filter``)."""
    kept, _ = ical_filter(entries, ident_fn)
    return kept
