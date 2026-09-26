"""Refuse, at config-load time, every cache recipe the GR00T split cannot serve.

Why this exists as a load-time check
------------------------------------
The generic validator was written for Pi0.5 and legitimately permits things
the GR00T split cannot do: a CP3 checkpoint (it is even in ``CacheConfig``'s
default), any gate, and -- the subtle one -- a warm-start recipe that does not
say which flow-matching loop its timesteps belong to. None of those can be
caught reliably by the generic configuration validator:

* an unsatisfiable WARM_START is rejected by the orchestrator when the payload
  lacks the requested ``start_t``; this guard catches the same defect before a recipe
  whose timesteps come from a different loop than the library's (Pi0.5's ten
  descending steps versus this head's ``num_inference_timesteps`` ascending
  ones, or a k=4 library under a k=8 server) reaches live traffic. Warm-start recipes are
  therefore admitted only when they name a GR00T ``denoise_schedule`` and, when
  the caller supplies the live step count, that schedule is the one the head
  is actually running;
* a CP3 checkpoint would be built and registered, and simply never consulted;
* a gate other than ``always_search`` emits ``searched=False`` steps, which a
  downstream analysis that assumes every step really searched would count as
  real verdicts.

That last restriction is the only one with a legitimate exception, and it is
opted into **per serving entry point** rather than relaxed globally
(``allow_hysteresis_gate``). The gate itself lives in the Orchestrator and is
model-agnostic, so ``score_hysteresis`` is mechanically serviceable here; what
is not portable is the *analysis* assumption above, which the cross-scene
RoboCasa365 line still relies on. Widening the default would silently carry one
experiment's exception into another subsystem, so the default stays
``always_search`` and callers that have taught their analysis to read
``searched`` say so explicitly.

The artifact identity check is here for a sharper reason: ``load_artifact``
only compares ``vector_dims``, and mean-pool and max-pool libraries are
dimensionally identical. Nothing else would ever notice the swap.

Library-free recipes (``is_library_free``: no enabled checkpoint plus a
``miss`` block or a ``trigger: always`` warm reset) retrieve nothing, so the
checkpoint-set, judge / gate and artifact identity rules do not apply; only
their loop identity is checked against the live head (the MISS step count,
the self start's ``denoise_schedule``).

Coupling map:
  DEPENDS ON:  CacheConfig, CacheStorage.artifact_meta
  CONSUMED BY: exp/robocasa365/serve_groot_n15.py (default),
               exp/libero_groot/serve_groot_libero.py (allow_hysteresis_gate)
  IF CHANGED:  the guard rejection matrix test must be updated
"""

from __future__ import annotations

from typing import Any

from openpi.cache.config import (
    CacheConfig,
    ConfigValidationError,
    is_library_free,
    required_warm_timesteps,
)
from openpi.cache.types import DIRECTION_ASC, groot_n15_schedule, schedule_from_id

_ALLOWED_JUDGE_TYPES = frozenset({"threshold", "always_hit", "always_warm_start", "online_rit"})
#: Gates every GR00T serving entry point may use, with no opt-in.
_BASE_ALLOWED_GATES = frozenset({"always_search"})
#: The single documented exception, admitted only via ``allow_hysteresis_gate``.
_HYSTERESIS_GATE = "score_hysteresis"
#: The two checkpoint sets the GR00T split can serve: the CP1 arm (key cut from
#: the stage-1 sequence) or the CP2 arm (ActionCache-style key cut from the
#: action head's encoded conditioning). Never both -- the interceptor is a
#: two-way switch -- and never CP3.
_SERVICEABLE_CHECKPOINT_SETS = (frozenset({"cp1"}), frozenset({"cp2"}))
#: The only key builder the CP2 arm may use on GR00T.
_CP2_GROOT_BUILDER = "cp2_groot_ternary"


def enabled_checkpoint_names(config: CacheConfig) -> frozenset[str]:
    """Lower-case names of the enabled checkpoints (``_``-prefixed entries ignored)."""
    return frozenset(
        name.lower()
        for name, cp in config.checkpoints.items()
        if not name.startswith("_") and cp.enabled
    )


def _library_free_errors(config: CacheConfig, num_inference_timesteps: int | None) -> list[str]:
    """A library-free recipe's loop identity: the MISS count / self-start schedule is the live head's."""
    errors: list[str] = []
    miss = getattr(config, "miss", None)
    if miss is not None and num_inference_timesteps is not None and miss.num_steps != num_inference_timesteps:
        errors.append(
            f"miss.num_steps={miss.num_steps} but the served action head runs "
            f"{num_inference_timesteps} steps; GR00T's MISS step count is process-level "
            "(--denoising-steps), serve this arm on a server started with that count."
        )
    if getattr(config, "warm_reset", None) is not None:
        if config.denoise_schedule is None:
            errors.append(
                "a trigger: always warm reset must name denoise_schedule (groot_n15_k<N>_v1 with "
                "N = the served head's num_inference_timesteps): its self start runs that loop."
            )
        else:
            try:
                schedule = schedule_from_id(config.denoise_schedule)
            except ValueError as exc:
                return errors + [f"denoise_schedule: {exc}"]
            if schedule.direction != DIRECTION_ASC:
                errors.append(f"denoise_schedule={schedule.schedule_id!r} is not a GR00T loop.")
            elif num_inference_timesteps is not None and groot_n15_schedule(num_inference_timesteps) != schedule:
                errors.append(
                    f"denoise_schedule={schedule.schedule_id!r} but the served action head runs "
                    f"{num_inference_timesteps} steps."
                )
    return errors


def single_enabled_checkpoint(config: CacheConfig) -> str | None:
    """The one enabled checkpoint name when the set is serviceable, else None."""
    enabled = enabled_checkpoint_names(config)
    if enabled in _SERVICEABLE_CHECKPOINT_SETS:
        return next(iter(enabled))
    return None


def _inspected_checkpoint(config: CacheConfig) -> str | None:
    """The checkpoint whose judge / gate / warm shape the guard reports on.

    The serviceable one when the set is valid; otherwise ``cp1`` if present,
    so an invalid set (e.g. cp1+cp3) still gets every other problem reported
    in the same pass, exactly as before the CP2 arm existed.
    """
    name = single_enabled_checkpoint(config)
    if name is not None:
        return name
    return "cp1" if "cp1" in config.checkpoints else None


def validate_groot_cache_config(
    config: CacheConfig,
    *,
    allow_hysteresis_gate: bool = False,
    num_inference_timesteps: int | None = None,
) -> None:
    """Reject a recipe the GR00T split cannot honour.

    Collects every problem before raising so a mis-written YAML is fixed in one
    pass rather than one error at a time.

    Args:
        config: the loaded recipe to check.
        num_inference_timesteps: the served action head's live step count.
            When given, a warm-start recipe's ``denoise_schedule`` must be the
            schedule of exactly this count; a serving entry point that has the
            policy in hand must pass it, because the count is a runtime
            property (baked into one checkpoint, overridden by CLI on another)
            and nothing else can tell a k=4 recipe from a k=8 head.
        allow_hysteresis_gate: admit ``score_hysteresis`` in addition to
            ``always_search``. Opt-in per entry point, and deliberately a
            boolean rather than a caller-supplied allow-set: a set parameter
            would let any caller smuggle in an arbitrary gate, and this module
            would no longer own the knowledge of which gates can ever be
            served. Callers passing ``True`` must read ``searched`` in their
            analysis, since the hysteresis gate emits gate-skipped steps.

    Raises:
        ConfigValidationError: naming each offending field.
    """
    errors: list[str] = []

    enabled = enabled_checkpoint_names(config)
    if is_library_free(config):
        # Plain / full arm or library-free self start: nothing is retrieved,
        # so no checkpoint set applies; only the loop identity is checked.
        errors.extend(_library_free_errors(config, num_inference_timesteps))
    elif enabled not in _SERVICEABLE_CHECKPOINT_SETS:
        errors.append(
            f"enabled checkpoints must be exactly {{'cp1'}} or {{'cp2'}}, got "
            f"{sorted(enabled)}. The GR00T split has no third stage, so CP3 would be "
            "built, registered and never consulted; CP1 and CP2 are two mutually "
            "exclusive arms of the same interceptor."
        )
    # A library-free recipe has no served checkpoint to inspect.
    cp_name = None if is_library_free(config) else _inspected_checkpoint(config)
    if cp_name == "cp2" and config.key_builder.type != _CP2_GROOT_BUILDER:
        errors.append(
            f"a CP2 recipe on GR00T must use key_builder.type={_CP2_GROOT_BUILDER!r} "
            f"(got {config.key_builder.type!r}); the Pi0.5 CP2 builder reads "
            "Stage2Output.prefix_out, which the GR00T split does not produce."
        )
    if cp_name == "cp2":
        # The CP2 arm names its teacher loop even when it never warm-starts: an
        # N_hit=0 arm's MISS is still the k=8 teacher its library and its cost
        # table describe, and only the schedule stamp can tell that head from a
        # k=4 one serving the same geometry (G2-B3).
        errors.extend(_cp2_schedule_errors(config, num_inference_timesteps))

    cp = config.checkpoints.get(cp_name) if cp_name is not None else None
    if cp is not None:
        if cp.judge.type not in _ALLOWED_JUDGE_TYPES:
            errors.append(
                f"{cp_name}.judge.type={cp.judge.type!r} is not serviceable; valid: "
                f"{sorted(_ALLOWED_JUDGE_TYPES)}. Composite / router judges are "
                "not enumerable for the warm-library completeness check."
            )
        errors.extend(_warm_start_schedule_errors(config, num_inference_timesteps))
        # The hysteresis opt-in is a CP1 experiment's exception; the CP2 arm is
        # the ActionCache baseline and searches every step by definition.
        allowed_gates = _BASE_ALLOWED_GATES | (
            {_HYSTERESIS_GATE} if allow_hysteresis_gate and cp_name == "cp1" else set()
        )
        if cp.gate.type not in allowed_gates:
            hint = (
                ""
                if allow_hysteresis_gate and cp_name == "cp1"
                else f" ({_HYSTERESIS_GATE!r} is available to CP1 entry points that "
                "pass allow_hysteresis_gate=True and read `searched`)"
            )
            errors.append(
                f"{cp_name}.gate.type={cp.gate.type!r} but only {sorted(allowed_gates)} "
                f"are supported here{hint}. Other gates emit searched=False steps, "
                "which a downstream analysis that assumes every step searched "
                "would count as real verdicts."
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


def _cp2_schedule_errors(
    config: CacheConfig, num_inference_timesteps: int | None
) -> list[str]:
    """The CP2 arm's loop identity, warm start or not: named, a GR00T loop, the live one."""
    if config.denoise_schedule is None:
        return [
            "a CP2 recipe on GR00T must name denoise_schedule (groot_n15_k<N>_v1 with "
            "N = the served head's num_inference_timesteps) for both N_hit tiers: the "
            "library, the cost table and the MISS fallback all belong to one loop."
        ]
    try:
        schedule = schedule_from_id(config.denoise_schedule)
    except ValueError as exc:
        return [f"denoise_schedule: {exc}"]
    errors: list[str] = []
    if schedule.direction != DIRECTION_ASC:
        errors.append(
            f"denoise_schedule={schedule.schedule_id!r} is not a GR00T loop; the CP2 "
            "arm's teacher runs time upward from 0."
        )
    if num_inference_timesteps is not None:
        live = groot_n15_schedule(num_inference_timesteps)
        if live != schedule:
            errors.append(
                f"denoise_schedule={schedule.schedule_id!r} but the served action head "
                f"runs {live.num_steps} steps ({live.schedule_id}); a CP2 library keyed "
                "and priced under one loop cannot be served by another, even for "
                "FULL_HIT-only arms whose MISS is that teacher."
            )
    return errors


def live_num_inference_timesteps(policy: Any) -> int:
    """The served head's step count, read from the policy that will run.

    Serving entry points pass this into ``validate_groot_cache_config`` so a
    recipe's ``denoise_schedule`` is checked against the loop that actually
    runs, not against a constant.
    """
    return int(policy.model.action_head.num_inference_timesteps)


def _warm_start_schedule_errors(
    config: CacheConfig, num_inference_timesteps: int | None
) -> list[str]:
    """Problems with the loop identity of a recipe that may resume mid-loop."""
    try:
        warm = required_warm_timesteps(config)
    except ConfigValidationError as exc:
        return [str(exc)]
    # ``required_warm_timesteps`` already walks every enabled checkpoint; only
    # the judge-shape probe below is local, and it reads the one checkpoint the
    # GR00T split serves (cp1 or cp2) rather than a fixed name.
    cp_name = _inspected_checkpoint(config)
    cp1 = config.checkpoints.get(cp_name) if cp_name is not None else None
    warm_judge = cp1 is not None and (
        cp1.judge.type in ("always_warm_start", "online_rit") or bool(cp1.judge.warm_tiers)
    )
    if not warm and not warm_judge:
        return []
    errors: list[str] = []
    if config.denoise_schedule is None:
        errors.append(
            "a warm-start recipe must name denoise_schedule "
            "(groot_n15_k<N>_v1 with N = the served head's num_inference_timesteps); "
            "without it the timesteps would be read as Pi0.5's ten descending steps "
            "and the runtime schedule guard would reject every WARM_START."
        )
        return errors
    try:
        schedule = schedule_from_id(config.denoise_schedule)
    except ValueError as exc:
        return [f"denoise_schedule: {exc}"]
    if schedule.direction != DIRECTION_ASC:
        errors.append(
            f"denoise_schedule={schedule.schedule_id!r} is not a GR00T loop; this "
            "head runs time upward from 0 and a Pi0.5 schedule would reverse "
            "every remaining-step count."
        )
    if num_inference_timesteps is not None:
        live = groot_n15_schedule(num_inference_timesteps)
        if live != schedule:
            errors.append(
                f"denoise_schedule={schedule.schedule_id!r} but the served action "
                f"head runs {live.num_steps} steps ({live.schedule_id}); a library "
                "keyed under one loop cannot be resumed by another."
            )
    return errors


def validate_artifact_identity(storage: Any, config: CacheConfig) -> None:
    """Refuse a library that was not built by the configured key builder.

    ``storage`` is a ``CacheStorage`` facade; its ``artifact_meta`` property is
    the sanctioned way to read the loaded artifact's identity — the backend is
    private and must not be reached into.

    Two distinct "unknown" shapes are handled separately because they mean
    different things to whoever has to fix it: a missing property means the
    backend never loaded an artifact at all, while present-but-empty fields
    mean the artifact predates identity recording.

    A library-free recipe (``is_library_free``) never consults its storage,
    so there is no identity to match and the check passes.

    Raises:
        ConfigValidationError: on mismatch or on any unknown identity.
    """
    if is_library_free(config):
        return
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

    expected_cp = (single_enabled_checkpoint(config) or "cp1").upper()
    if checkpoint_id != expected_cp:
        raise ConfigValidationError(
            f"Artifact checkpoint_id={checkpoint_id!r}, expected {expected_cp!r} "
            "(the recipe's enabled checkpoint)."
        )
    if expected_cp == "CP2":
        # The generic schedule binding is skipped for FULL_HIT-only recipes
        # (they never read intermediates); the CP2 arm binds the loop anyway
        # because its MISS is the library's teacher (G2-B3).
        recorded = meta.get("schedule_id")
        if config.denoise_schedule is None or recorded != config.denoise_schedule:
            raise ConfigValidationError(
                f"CP2 artifact denoise schedule {recorded!r} does not match the recipe's "
                f"{config.denoise_schedule!r}; the library's teacher loop and the served "
                "loop must be one and the same for every N_hit tier."
            )
        steps = meta.get("denoising_num_steps")
        want = schedule_from_id(config.denoise_schedule).num_steps
        if steps is not None and int(steps) != want:
            raise ConfigValidationError(
                f"CP2 artifact entries carry denoising_num_steps={steps} but "
                f"{config.denoise_schedule!r} has {want} steps."
            )

    # Warm-start libraries must be stamped: a GR00T library that predates
    # schedules carries no snapshots at all, so serving it under a warm recipe
    # would be one silent MISS per step. (The generic schedule binding already
    # compares ids; this is the GR00T-specific "no legacy fallback" rule.)
    if config.denoise_schedule is not None and meta.get("schedule_id") is None:
        raise ConfigValidationError(
            f"config names denoise_schedule={config.denoise_schedule!r} but the "
            "artifact records no schedule_id: it was built before schedules were "
            "stamped and holds no intermediates. Rebuild from a stamped collection."
        )
