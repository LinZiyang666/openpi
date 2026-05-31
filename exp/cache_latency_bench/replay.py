"""Model-free cache-latency replay harness.

``ReplayHarness`` assembles a real ``CacheOrchestrator`` from a cache YAML
exactly like the server does (``load_cache_config`` -> ``build_cache_components``
-> ``CacheOrchestrator(**components)``), then drives it through the same
lifecycle the server's ``InferenceInterceptor`` uses, replacing the model
forward with HDF5 replay. The orchestrator's own ``SystemTimer`` records the
CP1 six-segment latency, which the harness slices per step.

Three harness-owned fail-loud guards (the production stack does not raise on
these by itself, see plan §1 / §6):
  * ``_ensure_timing_enabled``    — raise monitor level >= BASIC *before* the
                                    timer is constructed, else it is a no-op.
  * ``_reject_warmup_calibration``— composite + ``samples_source=warmup`` has
                                    no WarmupPool link offline; reject up front.
  * ``_validate_artifact_enrichment`` — composite offline factors silently
                                    return NaN if the preload pkl lacks the
                                    required ``payload.factors`` keys; validate
                                    all entries up front.

Equivalence and the broadcast-action fidelity caveats (FULL_HIT exact / MISS
resample / WARM_START type-mismatch) are documented in the plan §6.4 / §7.
"""

from __future__ import annotations

from collections import Counter
import csv
import dataclasses
import os
import pickle
import time

from openpi.cache.components.judge import HitType
from openpi.cache.config import build_cache_components
from openpi.cache.config import load_cache_config
from openpi.cache.config import validate_cache_config
from openpi.cache.orchestrator import CacheOrchestrator
from openpi.cache.types import CheckpointID

# Reuse the server's single source of truth for the C2 write-policy contract
# (fail-fast unless write_policy=never). This pulls in the server import graph,
# but reusing the contract is safer than re-inlining the check here.
from scripts.serve_policy import _enforce_runtime_write_policy

from exp.cache_latency_bench.h5_episode import EpisodeInput
from exp.cache_latency_bench.h5_episode import H5EpisodeSource
from exp.cache_latency_bench.h5_episode import StepInput

# CP1 segment probes auto-registered by the orchestrator (orchestrator.py:175-177).
_CP1_PROBES = (
    "cp1_collect",
    "cp1_gate",
    "cp1_build",
    "cp1_search",
    "cp1_judge",
    "cp1_fetch",
)
_SELFCHECK_PROBE = "__timing_selfcheck__"

CSV_HEADER = [
    "repeat",
    "episode_id",
    "step_idx",
    "task",
    "hit_type",
    "top_score",
    "cp1_collect_ms",
    "cp1_gate_ms",
    "cp1_build_ms",
    "cp1_search_ms",
    "cp1_judge_ms",
    "cp1_fetch_ms",
    "cp1_total_ms",
    "warm_start_action_approx",
    "action_history_approx_active",
]

# ----------------------------------------------------------------------
# Harness-owned fail-loud guards
# ----------------------------------------------------------------------


def _ensure_timing_enabled() -> None:
    """Raise monitor level to >= BASIC before ``SystemTimer`` is constructed.

    ``SystemTimer.__init__`` force-disables itself when ``get_monitor_level()``
    is below BASIC (default OFF). The level must be lifted *before*
    ``build_cache_components`` constructs the timer, otherwise the timer is a
    permanent no-op and every ``summary()`` comes back empty. BASIC suffices
    because the cache probes use the CPU backend.
    """
    from openpi.serving import monitor

    if monitor.get_monitor_level() < monitor.MonitorLevel.BASIC:
        monitor.set_monitor_level(monitor.MonitorLevel.BASIC)


def _enabled_checkpoints(config):
    """Yield ``(name, CheckpointConfig)`` for enabled, non-anchor checkpoints."""
    for cp_name, cp in config.checkpoints.items():
        if cp_name.startswith("_") or not getattr(cp, "enabled", True):
            continue
        yield cp_name, cp


def _reject_warmup_calibration(config) -> None:
    """Reject composite judges whose Layer-3 calibration uses warmup samples.

    The offline assembly path (``yaml_id=None``) has no server ctrl link to
    fill the ``WarmupPool``; the production code would raise a generic error
    deep in ``_load_calibration_samples``. We reject earlier with a clear
    message so the operator switches to ``samples_source.type='offline'``.
    """
    for cp_name, cp in _enabled_checkpoints(config):
        cal = getattr(cp.judge, "calibration", None)
        src = getattr(cal, "samples_source", None) if cal is not None else None
        if src is not None and getattr(src, "type", None) == "warmup":
            raise RuntimeError(
                f"checkpoint {cp_name!r}: calibration.samples_source.type='warmup' "
                "is unsupported by the latency bench (no WarmupPool preload link "
                "offline). Use samples_source.type='offline'."
            )


def _reject_unsupported_keybuilder(config) -> None:
    """Reject keybuilders that need a real model (incompatible with model-free replay).

    ``cp1_llm_layer_extract`` runs a partial PaliGemma forward inside ``build()``
    and needs model attachment plus full Stage1 mask fields, which the H5 replay
    does not provide. Fail fast with a clear message up front (plan R7).
    """
    kb_type = getattr(config.key_builder, "type", None)
    if kb_type == "cp1_llm_layer_extract":
        raise NotImplementedError(
            "key_builder.type='cp1_llm_layer_extract' needs a real model (partial "
            "PaliGemma forward + Stage1 masks) and is unsupported by the model-free "
            "latency bench. Use a SigLIP-only keybuilder (mean_pool / max_pool / "
            "spatial_pool_* / placeholder)."
        )


def _required_offline_factor_keys(config) -> set[str]:
    """Declared ``payload.factors`` keys for every offline factor in the config.

    Derived from config (not a judge instance): for each composite judge's
    ``FactorConfig``, resolve the factor class via the registry, keep the
    offline-source ones (the ``compute_for_episode`` duck-type, the same
    discriminator ``config._collect_offline_writers_from_judges`` uses), and
    take the keys its ``describe(params)`` classmethod declares. Online factors
    do not read ``payload.factors`` and need no enrichment.
    """
    from openpi.cache.components.factors import registry

    keys: set[str] = set()
    # Validate every enabled checkpoint (incl. CP3 if configured): the harness
    # only drives CP1, but a shared pkl should carry all declared offline keys,
    # and over-requiring is conservative (never silently under-validates).
    for _, cp in _enabled_checkpoints(config):
        if cp.judge.type != "composite" or not getattr(cp.judge, "factors", None):
            continue
        for fcfg in cp.judge.factors:
            cls = registry.get_class(fcfg.type)
            if hasattr(cls, "compute_for_episode"):  # offline-source factor
                keys.update(cls.describe(dict(fcfg.params)).keys())
    return keys


def _validate_artifact_enrichment(config) -> None:
    """Fail loud if any preload entry lacks a required offline ``payload.factors`` key.

    The production backend does NOT error on un-enriched artifacts: a missing
    ``library_stats`` is recomputed from entries, and offline factors return NaN
    per-entry when ``payload.factors`` lacks a key. Verdicts would then silently
    degrade to NaN/MISS and the recorded latency mix would be wrong. So we read
    the preload pkl and traverse ALL entries (not a sample), since a partially
    enriched artifact only fails when a missing-key entry happens to win.
    """
    required = _required_offline_factor_keys(config)
    if not required:
        return
    if config.backend.type != "in_memory":
        raise RuntimeError(
            "composite/offline-factor judge requires an in-memory preload artifact; "
            f"got backend.type={config.backend.type!r} (unsupported)."
        )
    path = config.backend.in_memory.preload_path
    if not path:
        raise RuntimeError(
            "composite/offline-factor judge requires backend.in_memory.preload_path."
        )
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    for entry in data.get("entries", []):
        factors = getattr(entry.payload, "factors", None) or {}
        missing = sorted(k for k in required if k not in factors)
        if missing:
            preview = ", ".join(missing[:5]) + ("..." if len(missing) > 5 else "")
            raise RuntimeError(
                f"artifact {path} not enriched: entry {entry.id!r} missing offline "
                f"factor keys [{preview}]; rebuild with exp/common/factor_postprocess.py."
            )


def _judge_consumes_action_history(config) -> bool:
    """True if any enabled composite judge uses an online action-channel factor.

    Only those factors read ``ctx.history.actions`` (online.py); for them the
    replayed action (MISS resample / WARM_START type-mismatch) is not call-for-call
    equivalent to the real server. Online action factors are named
    ``<descriptor>_online_action`` (``topk_action_variance`` is excluded).
    """
    for _, cp in _enabled_checkpoints(config):
        if cp.judge.type != "composite" or not getattr(cp.judge, "factors", None):
            continue
        if any(f.type.endswith("_online_action") for f in cp.judge.factors):
            return True
    return False


def _replay_action(res, clean_action):
    """The action to broadcast/buffer for this step (plan §6.4).

    FULL_HIT returns the cached payload action (identical to what the real
    server broadcasts). MISS / WARM_START have no live model output here, so we
    replay the H5 ``clean_action``.
    """
    if res.hit_type == HitType.FULL_HIT and res.payload is not None:
        return res.payload.action_chunk
    return clean_action


# ----------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------


class ReplayHarness:
    """Assemble a real cache orchestrator from a YAML and replay H5 episodes.

    Drives ``CacheOrchestrator`` through the server's lifecycle with the model
    forward replaced by HDF5 replay, recording per-step CP1 segment latency.
    """

    def __init__(self, cache_config_path: str, *, device: str = "cpu") -> None:
        if device != "cpu":
            raise NotImplementedError(
                f"device={device!r} not supported; only CPU (plan D1). GPU mode "
                "(faithful build-segment D2H) is a future extension."
            )
        # Lift monitor level BEFORE the timer is constructed by build_cache_components.
        _ensure_timing_enabled()

        config = load_cache_config(cache_config_path)  # validates internally
        config = _enforce_runtime_write_policy(config)  # D2: fail-fast unless never
        config = dataclasses.replace(
            config,
            timer=dataclasses.replace(config.timer, enabled=True),
        )
        validate_cache_config(config)  # idempotent; defensive
        _reject_unsupported_keybuilder(config)
        _reject_warmup_calibration(config)
        _validate_artifact_enrichment(config)

        components = build_cache_components(config)  # same source as the server
        self._orchestrator = CacheOrchestrator(**components)
        self._timer = components["timer"]  # same instance the orchestrator times on
        self._judge_reads_action_history = _judge_consumes_action_history(config)

        # Timing self-check: SystemTimer has no public ``enabled``; verify the
        # timer actually records via one real probe (registered first to avoid
        # the unregistered-probe RuntimeWarning path).
        self._timer.register_probe(_SELFCHECK_PROBE, backend="cpu")
        with self._timer.measure(_SELFCHECK_PROBE):
            pass
        self._timing_ready = _SELFCHECK_PROBE in self._timer.summary(task_only=True)
        if not self._timing_ready:
            raise RuntimeError(
                "SystemTimer is a no-op (monitor level < BASIC or "
                "config.timer.enabled=False); cannot collect latency."
            )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, source: H5EpisodeSource, *, repeats: int = 1, out_csv: str) -> dict:
        """Replay every episode ``repeats`` times, writing per-step rows to ``out_csv``.

        Returns a run-level summary dict. The cache lifecycle mirrors the server:
        one ``on_task_begin`` for the whole run, ``on_episode_start`` per H5,
        ``check`` -> ``broadcast_action`` -> ``buffer_for_write`` -> ``clear`` per
        step (no step skipped), ``on_episode_end`` per H5, one ``on_task_end``.
        """
        assert (
            self._timing_ready
        )  # set in __init__ via a real probe (no SystemTimer.enabled)
        os.makedirs(os.path.dirname(os.path.abspath(out_csv)), exist_ok=True)

        n_steps = 0
        hit_counts: Counter = Counter()
        with open(out_csv, "w", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(CSV_HEADER)
            self._orchestrator.on_task_begin()
            try:
                for r in range(repeats):
                    for ep in source.iter_episodes():
                        n_steps += self._run_episode(writer, r, ep, hit_counts)
            finally:
                self._orchestrator.on_task_end()

        return {
            "n_episodes": len(source) * repeats,
            "n_steps": n_steps,
            "repeats": repeats,
            "hit_counts": dict(hit_counts),
            "hit_rate": (hit_counts.get("FULL_HIT", 0) / n_steps) if n_steps else 0.0,
            "judge_consumes_action_history": self._judge_reads_action_history,
        }

    def _run_episode(
        self, writer, repeat: int, ep: EpisodeInput, hit_counts: Counter
    ) -> int:
        """Replay one episode; return the number of steps written."""
        self._orchestrator.on_episode_start(ep.task_key, str(ep.episode_id))
        # Cumulative flag: set once a non-FULL_HIT action enters _action_history,
        # so later steps' action-history verdicts are not read as call-equivalent.
        approx_seen = False
        n = 0
        for step in ep.steps():
            res, cp1_total_ms, stats = self._check_one_step(step)

            action = _replay_action(res, step.clean_action)
            self._orchestrator.broadcast_action(action)
            if res.query_keys is not None:
                self._orchestrator.buffer_for_write(res.query_keys, action)
            self._orchestrator.clear()

            is_full = res.hit_type == HitType.FULL_HIT
            warm_approx = (
                res.hit_type == HitType.WARM_START
            ) and self._judge_reads_action_history
            # action_history_approx_active is read BEFORE updating approx_seen so it
            # reflects "a prior step in this episode already wrote a non-FULL_HIT action".
            active = approx_seen if self._judge_reads_action_history else False
            if self._judge_reads_action_history and not is_full:
                approx_seen = True

            writer.writerow(
                _row(repeat, ep, step, res, stats, cp1_total_ms, warm_approx, active)
            )
            hit_counts[res.hit_type.name] += 1
            n += 1
        self._orchestrator.on_episode_end()
        return n

    def _check_one_step(self, step: StepInput):
        """Run one CP1 check and slice this step's six-segment timing.

        ``timer.on_task_begin()`` moves the ring-buffer marker so
        ``summary(task_only=True)`` afterwards holds only this step's probe
        records (each ``count=1``). ``cp1_total`` is wall time around ``check``
        only (broadcast/buffer/clear are outside, matching the server's
        ``cp1_sum`` boundary).
        """
        self._timer.on_task_begin()
        t0 = time.perf_counter()
        res = self._orchestrator.check(CheckpointID.CP1, stage1=step.fake_stage1)
        cp1_total_ms = (time.perf_counter() - t0) * 1000.0
        stats = self._timer.summary(task_only=True)
        return res, cp1_total_ms, stats


# ----------------------------------------------------------------------
# Row formatting
# ----------------------------------------------------------------------


def _row(repeat, ep, step, res, stats, cp1_total_ms, warm_approx, active):
    def seg(name):
        s = stats.get(name)
        return f"{s.mean_ms:.6f}" if s is not None else ""

    return [
        repeat,
        ep.episode_id,
        step.step_idx,
        ep.task_key,
        res.hit_type.name,
        "" if res.score is None else f"{res.score:.6f}",
        seg("cp1_collect"),
        seg("cp1_gate"),
        seg("cp1_build"),
        seg("cp1_search"),
        seg("cp1_judge"),
        seg("cp1_fetch"),
        f"{cp1_total_ms:.6f}",
        int(bool(warm_approx)),
        int(bool(active)),
    ]
