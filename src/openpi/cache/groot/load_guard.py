"""Refuse, at config-load time, every cache recipe the GR00T split cannot serve.

Why this exists as a load-time check
------------------------------------
The generic validator was written for Pi0.5 and legitimately permits things a
two-stage model cannot do: warm-start judges and warm tiers on CP1, a CP3
checkpoint (it is even in ``CacheConfig``'s default), and any gate. None of
those can be caught later, because none of them fail loudly at run time:

* an unsatisfiable WARM_START is *silently downgraded to MISS* by the
  orchestrator when the payload has no intermediates — which is always true of
  a GR00T library — so a runtime guard would never fire and the only symptom
  is an inexplicably low hit rate;
* a CP3 checkpoint would be built and registered, and simply never consulted;
* a gate other than ``always_search`` would make ``searched`` mean something
  the downstream analysis does not expect.

The artifact identity check is here for a sharper reason: ``load_artifact``
only compares ``vector_dims``, and mean-pool and max-pool libraries are
dimensionally identical. Nothing else would ever notice the swap.

Coupling map:
  DEPENDS ON:  CacheConfig, CacheStorage.artifact_meta
  CONSUMED BY: exp/robocasa365/serve_groot_n15.py
  IF CHANGED:  the guard rejection matrix test must be updated
"""

from __future__ import annotations

from typing import Any

from openpi.cache.config import CacheConfig, ConfigValidationError

_ALLOWED_JUDGE_TYPES = frozenset({"threshold", "always_hit"})
_REQUIRED_GATE = "always_search"


def validate_groot_cache_config(config: CacheConfig) -> None:
    """Reject a recipe the two-stage GR00T split cannot honour.

    Collects every problem before raising so a mis-written YAML is fixed in one
    pass rather than one error at a time.

    Raises:
        ConfigValidationError: naming each offending field.
    """
    errors: list[str] = []

    enabled = {
        name.lower()
        for name, cp in config.checkpoints.items()
        if not name.startswith("_") and cp.enabled
    }
    if enabled != {"cp1"}:
        errors.append(
            f"enabled checkpoints must be exactly {{'cp1'}}, got {sorted(enabled)}. "
            "The GR00T split has no third stage, so CP3 would be built, "
            "registered and never consulted."
        )

    cp1 = config.checkpoints.get("cp1")
    if cp1 is not None:
        if cp1.judge.type not in _ALLOWED_JUDGE_TYPES:
            errors.append(
                f"cp1.judge.type={cp1.judge.type!r} is not serviceable; valid: "
                f"{sorted(_ALLOWED_JUDGE_TYPES)}. Verdicts must be binary — there "
                "is no partial third stage to warm-start into."
            )
        if cp1.judge.warm_tiers:
            errors.append(
                "cp1.judge.warm_tiers must be empty. A WARM_START verdict is "
                "silently downgraded to MISS when the payload carries no "
                "intermediates, which is every GR00T library entry, so this "
                "would present as a low hit rate rather than an error."
            )
        if cp1.gate.type != _REQUIRED_GATE:
            errors.append(
                f"cp1.gate.type={cp1.gate.type!r} but only {_REQUIRED_GATE!r} is "
                "supported. Other gates emit searched=False steps, which the "
                "downstream analysis would count as real verdicts."
            )

    if config.write_policy.type != "never":
        errors.append(
            f"write_policy.type={config.write_policy.type!r} is not allowed at "
            "serving time; use 'never' and build libraries offline. An online "
            "write path makes the library's contents depend on run order, which "
            "destroys the cross-scene experiment's only independent variable."
        )

    if errors:
        raise ConfigValidationError(
            "GR00T cache config rejected:\n  - " + "\n  - ".join(errors)
        )


def validate_artifact_identity(storage: Any, config: CacheConfig) -> None:
    """Refuse a library that was not built by the configured key builder.

    ``storage`` is a ``CacheStorage`` facade; its ``artifact_meta`` property is
    the sanctioned way to read the loaded artifact's identity — the backend is
    private and must not be reached into.

    Two distinct "unknown" shapes are handled separately because they mean
    different things to whoever has to fix it: a missing property means the
    backend never loaded an artifact at all, while present-but-empty fields
    mean the artifact predates identity recording.

    Raises:
        ConfigValidationError: on mismatch or on any unknown identity.
    """
    expected = config.key_builder.type
    meta = storage.artifact_meta

    if meta is None:
        raise ConfigValidationError(
            "The cache backend exposes no artifact identity, so the library "
            "cannot be matched against key_builder.type="
            f"{expected!r}. Either the backend never loaded an artifact "
            "(check backend.in_memory.preload_path) or it is of a type that "
            "does not support preloading."
        )

    actual = meta.get("key_builder_type")
    checkpoint_id = meta.get("checkpoint_id")

    if actual is None or checkpoint_id is None:
        raise ConfigValidationError(
            "The loaded artifact predates identity recording "
            f"(key_builder_type={actual!r}, checkpoint_id={checkpoint_id!r}). "
            "Rebuild it with exp/common/build_in_memory_cache_artifact.py so the "
            "library can be matched against the configured key builder; a "
            "same-dimension mismatch is otherwise undetectable."
        )

    if actual != expected:
        raise ConfigValidationError(
            f"Artifact was built by {actual!r} but key_builder.type is "
            f"{expected!r}. Dimensions alone cannot catch this — mean-pool and "
            "max-pool libraries have identical vector_dims — so the keys would "
            "silently mean something different from the queries."
        )

    if checkpoint_id != "CP1":
        raise ConfigValidationError(
            f"Artifact checkpoint_id={checkpoint_id!r}, expected 'CP1'."
        )
