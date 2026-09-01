"""Hysteretic Cumulative-Risk Re-anchoring judge (H-CRD) — stateful extension
of the dispatch surface (exploratory prototype, 2026-08-30; design in
logs/dispatch_surface_rev2_amendment_result.md sections 12-13).

Two calibrated quantities per cache tier a in {FULL_HIT, WARM_START}, both
read from grids fitted on the frozen dispatch table (no rollout outcome):

    u_a(s, v)  conservative local risk  (upper-quantile grid, level 1 - alpha)
    d_a(s, v)  approximation-debt increment (central grid, level 0.5),
               divided by the task's calibration-only discrepancy scale

State per episode: D (accumulated debt since the last fresh inference), mode in
{ACTIVE, RECOVERY}, bad_run (consecutive steps with no locally admissible
tier), fh_run (consecutive executed FULL_HITs), recovery_misses (executed
MISSes since RECOVERY was entered).

Decision (ACTIVE):   local_ok(a) := u_a <= delta ;  budget_ok(a) := gamma*D + d_a <= beta
                     cheapest tier with both -> execute it; none -> MISS (D = 0)
                       * no tier local_ok  -> "region" MISS, bad_run += 1, RECOVERY once bad_run >= J_bad
                       * some tier local_ok -> "debt" MISS, bad_run = 0
                     fh_run >= L_max -> one forced "fuse" MISS (D = 0), stays ACTIVE
Decision (RECOVERY): after entry, execute at least ``min_recovery_misses``
                     additional MISSes; then the cheapest tier with
                     u_a <= delta_reopen (<= delta) AND budget_ok(a) reopens.
Contract failure (empty results, non-finite s / v, v out of range, a cache
tier the orchestrator downgraded) -> MISS, D = 0, RECOVERY fail-closed.

propose / commit: ``__call__`` only PROPOSES (no state mutation) and remembers
a one-shot proposal; the orchestrator feeds the FINAL executed verdict through
``commit_verdict`` (same hook as ``_feed_verdict_to_gate``), which is the only
place state changes. Legal (proposed, executed) pairs are exactly
``proposed == executed`` and ``(WARM_START, MISS)`` (the orchestrator's
payload downgrade); everything else raises with the proposal left in place. A
commit without a proposal, a second commit, or a new proposal while one is
pending all raise. ``commit_verdict`` returns the step's diagnostics, which the
orchestrator merges into ``CheckResult.factor_outputs["crd"]``.

Degenerate relations: gamma=0, beta=inf, J_bad=inf, L_max=None -> the static
surface at ``delta`` (s-bin lookup mirrors ``fit_surface.export_boundaries``);
J_bad=inf, L_max=None -> pure CRD; beta=inf -> hysteresis/backstop only.
A CRD artifact requires the CP1 gate to be ``always_search`` (enforced by the
config validator and by the orchestrator at assembly).
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from typing import Optional

import numpy as np
import torch

from openpi.cache.components.judge import HitType
from openpi.cache.components.surface_judge import SurfaceJudge
from openpi.cache.types import CheckpointID

JUDGE_VARIANT = "cumulative_risk"
_LAYER_WARM = 0
_LAYER_FULL = 1
ACTIVE, RECOVERY = "ACTIVE", "RECOVERY"
#: Legal (proposed, executed) pairs besides identity: the orchestrator may
#: downgrade an incomplete WARM_START payload to MISS, nothing else.
LEGAL_DOWNGRADES = frozenset({(HitType.WARM_START, HitType.MISS)})


def grid_sha256(arr) -> str:
    """Canonical digest of a grid, identical to ``fit_surface._digest_obj``."""
    return hashlib.sha256(
        json.dumps(np.asarray(arr, dtype=np.float64).tolist(), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_crd_meta(path: str) -> Optional[dict]:
    """Return metadata when an artifact declares the CRD variant.

    Variant detection deliberately does *not* require the CRD arrays.  Once an
    artifact declares ``cumulative_risk`` it must route to the strict CRD
    loader, which will reject missing arrays.  Treating a damaged CRD artifact
    as an ordinary static surface would silently disable the controller.
    """
    with np.load(path, allow_pickle=False) as data:
        if "meta_json" not in data.files:
            return None
        meta = json.loads(bytes(data["meta_json"]).decode("utf-8"))
    return meta if meta.get("judge_variant") == JUDGE_VARIANT else None


def is_crd_artifact(path: str) -> bool:
    return read_crd_meta(path) is not None


class CumulativeRiskJudge(SurfaceJudge):
    """Stateful dispatch-surface judge with a calibrated debt budget, a
    hysteretic recovery mode and a matched FULL_HIT run fuse."""

    def __init__(self, artifact_path: str, *, export_factor_outputs: bool = False) -> None:
        super().__init__(artifact_path, export_factor_outputs=export_factor_outputs)
        with np.load(artifact_path, allow_pickle=False) as data:
            if not {"q_hat", "q_hat_central", "s_edges"} <= set(data.files):
                raise ValueError("CRD artifact lacks q_hat / q_hat_central / s_edges")
            self._u = np.asarray(data["q_hat"], dtype=np.float64)
            self._d = np.asarray(data["q_hat_central"], dtype=np.float64)
            self._s_edges = np.asarray(data["s_edges"], dtype=np.float64)
        meta = self.artifact.meta
        crd = meta.get("crd")
        if meta.get("judge_variant") != JUDGE_VARIANT or not isinstance(crd, dict):
            raise ValueError("artifact does not declare judge_variant=cumulative_risk with a crd block")
        self._validate_grids(crd)
        gamma_raw = crd.get("gamma")
        beta_raw = crd.get("beta")
        if isinstance(gamma_raw, bool) or not isinstance(gamma_raw, (int, float)) or not math.isfinite(gamma_raw):
            raise ValueError(f"CRD gamma must be a finite number, got {gamma_raw!r}")
        if beta_raw is not None and (
            isinstance(beta_raw, bool) or not isinstance(beta_raw, (int, float)) or not math.isfinite(beta_raw)
        ):
            raise ValueError(f"CRD beta must be null (infinity) or a finite number, got {beta_raw!r}")
        self.gamma = float(gamma_raw)
        self.beta = float(beta_raw) if beta_raw is not None else math.inf

        def _optional_positive_int(key: str) -> Optional[int]:
            value = crd.get(key)
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"CRD {key} must be null or a positive integer, got {value!r}")
            return value

        self.j_bad = _optional_positive_int("j_bad")
        self.l_max = _optional_positive_int("l_max")
        recovery_raw = crd.get("min_recovery_misses", 0)
        if isinstance(recovery_raw, bool) or not isinstance(recovery_raw, int) or recovery_raw < 0:
            raise ValueError(
                "CRD min_recovery_misses must be a non-negative integer, "
                f"got {recovery_raw!r}"
            )
        self.min_recovery_misses = recovery_raw
        self.delta = float(self.artifact.delta)
        reopen_raw = crd.get("delta_reopen", self.delta)
        if isinstance(reopen_raw, bool) or not isinstance(reopen_raw, (int, float)) or not math.isfinite(reopen_raw):
            raise ValueError(f"CRD delta_reopen must be a finite number, got {reopen_raw!r}")
        self.delta_reopen = float(reopen_raw)
        scale_raw = crd.get("task_scale")
        if not isinstance(scale_raw, dict) or not scale_raw:
            raise ValueError("CRD task_scale must be a non-empty task-id mapping")
        self.task_scale: dict[int, float] = {}
        for k, v in scale_raw.items():
            try:
                key = int(k)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"CRD task_scale has a non-integer task id {k!r}") from exc
            if not isinstance(k, str) or str(key) != k or key < 0 or key in self.task_scale:
                raise ValueError(f"CRD task_scale has a non-canonical, negative, or duplicate task id {k!r}")
            if isinstance(v, bool) or not (isinstance(v, (int, float)) and math.isfinite(v) and v > 0):
                raise ValueError(f"CRD task_scale[{k}] must be finite and > 0, got {v!r}")
            self.task_scale[key] = float(v)
        if sorted(self.task_scale) != list(range(len(self.task_scale))):
            raise ValueError("CRD task_scale ids must be the complete contiguous range 0..N-1")
        if not (0.0 <= self.gamma <= 1.0) or not (self.beta > 0) \
                or not (0 < self.delta_reopen <= self.delta):
            raise ValueError(f"CRD parameters out of domain: {crd}")
        self._tokens = itertools.count(1)
        self._pending: Optional[dict] = None
        self._scale: Optional[float] = None
        self._reset_state()

    def _validate_grids(self, crd: dict) -> None:
        n_v = len(self.artifact.v_bin_edges) - 1
        for name, grid in (("q_hat", self._u), ("q_hat_central", self._d)):
            if grid.ndim != 3 or grid.shape[0] != 2 or grid.shape[1] + 1 != len(self._s_edges) or grid.shape[2] != n_v:
                raise ValueError(f"CRD {name} shape {grid.shape} inconsistent with edges")
            if not np.all(np.isfinite(grid)) or np.any(grid < 0):
                raise ValueError(f"CRD {name} must be finite and non-negative")
        if self._s_edges.ndim != 1 or not np.all(np.isfinite(self._s_edges)) or np.any(np.diff(self._s_edges) <= 0):
            raise ValueError("CRD s_edges must be finite and strictly increasing")
        if np.any(self._d > self._u + 1e-12):
            raise ValueError("CRD central (debt) grid exceeds the upper (admission) grid somewhere")
        for key, grid in (("upper_grid_sha256", self._u), ("central_grid_sha256", self._d)):
            if crd.get(key) != grid_sha256(grid):
                raise ValueError(f"CRD {key} does not match the stored grid")

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        self.D = 0.0
        self.mode = ACTIVE
        self.bad_run = 0
        self.fh_run = 0
        self.recovery_misses = 0
        self._pending = None
        # Identity is episode-scoped.  Clear it before validating the next
        # episode so a rejected/malformed start cannot retain the prior task's
        # normalization scale.
        self._scale = None

    def on_episode_start(self, extra_metadata: Optional[dict] = None, *, provisional: bool = False) -> None:
        self._reset_state()
        # CacheOrchestrator broadcasts a lifecycle reset from on_task_begin()
        # when a websocket connection opens, before the client's identified
        # episode_start frame arrives.  Reset state at that boundary, but do not
        # invent a task identity and do not permit a proposal yet.
        if provisional:
            return
        task_id = (extra_metadata or {}).get("task_id")
        if isinstance(task_id, bool) or not isinstance(task_id, int) or task_id not in self.task_scale:
            raise ValueError(f"CRD judge: episode task_id {task_id!r} has no frozen task scale")
        self._scale = self.task_scale[task_id]

    @property
    def state(self) -> dict:
        return {"D": self.D, "mode": self.mode, "bad_run": self.bad_run, "fh_run": self.fh_run,
                "recovery_misses": self.recovery_misses, "task_scale": self._scale, "pending": self._pending is not None}

    # ------------------------------------------------------------------
    # risk lookup (mirrors export_boundaries: the bin BELOW the one holding s)
    # ------------------------------------------------------------------
    def _cell(self, grid: np.ndarray, layer: int, s: float, v_bin: int) -> float:
        j = int(np.searchsorted(self._s_edges, s, side="right")) - 1
        idx = min(j - 1, grid.shape[1] - 1)
        if idx < 0:
            return math.inf
        return float(grid[layer, idx, v_bin])

    # ------------------------------------------------------------------
    # propose
    # ------------------------------------------------------------------
    def _propose(self, hit: HitType, winner_id, *, s, v, v_bin, src, start_t=None, **diag):
        if self._pending is not None:
            raise RuntimeError("CRD judge: new proposal while a proposal is still uncommitted")
        if self._scale is None:
            raise RuntimeError("CRD judge: proposal before on_episode_start")
        self._pending = {"token": next(self._tokens), "hit": hit, "src": src, **diag}
        return self._emit(hit, winner_id, s=s, v=v, v_bin=v_bin, src=src, start_t=start_t)

    def __call__(self, results, checkpoint_id: CheckpointID, cached_data: dict[str, torch.Tensor], *,
                 view=None, history=None, retrieval_signals=None):
        art = self.artifact
        if self._scale is None:
            raise RuntimeError("CRD judge: decision requested before on_episode_start")
        base = {"D_before": self.D, "mode_before": self.mode, "local_any": False}
        if checkpoint_id != CheckpointID.CP1:
            # Not this judge's checkpoint: a defensive MISS with no proposal (no commit will follow).
            return self._emit(HitType.MISS, None, s=None, v=None, v_bin=None, src="cp_guard")
        if not results:
            return self._propose(HitType.MISS, None, s=None, v=None, v_bin=None, src="empty", contract=True, **base)
        s = float(results[0].score)
        if not math.isfinite(s):
            return self._propose(HitType.MISS, None, s=s, v=None, v_bin=None, src="s_nonfinite", contract=True, **base)
        winner_id = results[0].id
        if art.uses_disagreement:
            v = self._compute_v(results, view)
            if not math.isfinite(v):
                return self._propose(HitType.MISS, None, s=s, v=v, v_bin=None, src="v_failclosed", contract=True, **base)
            if v > art.v_bin_edges[-1]:
                return self._propose(HitType.MISS, None, s=s, v=v, v_bin=None, src="v_oob_right", contract=True, **base)
            v_bin = int(np.clip(np.searchsorted(art.v_bin_edges, v, side="right") - 1, 0, len(art.v_bin_edges) - 2))
        else:
            v, v_bin = None, 0
        u_f, u_w = self._cell(self._u, _LAYER_FULL, s, v_bin), self._cell(self._u, _LAYER_WARM, s, v_bin)
        d_f = self._cell(self._d, _LAYER_FULL, s, v_bin) / self._scale
        d_w = self._cell(self._d, _LAYER_WARM, s, v_bin) / self._scale
        D_f, D_w = self.gamma * self.D + d_f, self.gamma * self.D + d_w
        local_f, local_w = u_f <= self.delta, u_w <= self.delta
        budget_f, budget_w = D_f <= self.beta, D_w <= self.beta
        diag = dict(base, contract=False, u_full=u_f, u_warm=u_w, d_full=d_f, d_warm=d_w, D_full=D_f, D_warm=D_w,
                    local_any=bool(local_f or local_w))

        if self.mode == RECOVERY:
            if self.recovery_misses >= self.min_recovery_misses:
                if u_f <= self.delta_reopen and budget_f:
                    return self._propose(HitType.FULL_HIT, winner_id, s=s, v=v, v_bin=v_bin, src="crd_reopen_full", D_next=D_f, **diag)
                if u_w <= self.delta_reopen and budget_w:
                    return self._propose(HitType.WARM_START, winner_id, s=s, v=v, v_bin=v_bin, src="crd_reopen_warm",
                                         start_t=art.start_t_ws, D_next=D_w, **diag)
            return self._propose(HitType.MISS, None, s=s, v=v, v_bin=v_bin, src="crd_recovery", **diag)

        if self.l_max is not None and self.fh_run >= self.l_max:
            return self._propose(HitType.MISS, None, s=s, v=v, v_bin=v_bin, src="crd_fuse", **diag)
        if local_f and budget_f:
            return self._propose(HitType.FULL_HIT, winner_id, s=s, v=v, v_bin=v_bin, src="crd_full", D_next=D_f, **diag)
        if local_w and budget_w:
            return self._propose(HitType.WARM_START, winner_id, s=s, v=v, v_bin=v_bin, src="crd_warm",
                                 start_t=art.start_t_ws, D_next=D_w, **diag)
        src = "crd_debt" if (local_f or local_w) else "crd_region"
        return self._propose(HitType.MISS, None, s=s, v=v, v_bin=v_bin, src=src, **diag)

    # ------------------------------------------------------------------
    # commit (the orchestrator's final executed verdict)
    # ------------------------------------------------------------------
    def commit_verdict(self, checkpoint_id: CheckpointID, *, hit_type: HitType, **_ignored) -> dict:
        if checkpoint_id != CheckpointID.CP1:
            return {}
        p = self._pending
        if p is None:
            raise RuntimeError("CRD judge: commit without a pending proposal")
        proposed = p["hit"]
        if not (hit_type == proposed or (proposed, hit_type) in LEGAL_DOWNGRADES):
            raise RuntimeError(f"CRD judge: illegal transition proposed={proposed.name} executed={hit_type.name}")
        self._pending = None
        outcome = {"token": p["token"], "proposed": proposed.name, "executed": hit_type.name, "src": p["src"],
                   "D_before": self.D, "mode_before": self.mode, "bad_run_before": self.bad_run, "fh_run_before": self.fh_run}
        for key in ("u_full", "u_warm", "d_full", "d_warm", "D_full", "D_warm"):
            if key in p:
                outcome[key] = p[key]
        if hit_type == HitType.FULL_HIT:
            self.D = float(p["D_next"])
            self.fh_run += 1
            self.bad_run = 0
            self.mode = ACTIVE
            self.recovery_misses = 0
            outcome["reason"] = "full"
        elif hit_type == HitType.WARM_START:
            self.D = float(p["D_next"])
            self.fh_run = 0
            self.bad_run = 0
            self.mode = ACTIVE
            self.recovery_misses = 0
            outcome["reason"] = "warm"
        else:
            self.D = 0.0
            self.fh_run = 0
            downgraded = proposed != HitType.MISS
            if downgraded or p.get("contract"):
                self.mode = RECOVERY
                self.bad_run = 0
                # This MISS caused entry; it was not executed *in* RECOVERY.
                # The dwell counter starts at zero so min_recovery_misses=N
                # means N additional MISSes after entry, with no off-by-one.
                self.recovery_misses = 0
                outcome["reason"] = "downgrade" if downgraded else "contract"
            elif self.mode == RECOVERY:
                self.recovery_misses += 1
                outcome["reason"] = "recovery"
            elif p["src"] == "crd_region":
                self.bad_run += 1
                outcome["reason"] = "region"
                if self.j_bad is not None and self.bad_run >= self.j_bad:
                    self.mode = RECOVERY
                    self.bad_run = 0
                    self.recovery_misses = 0
            else:
                self.bad_run = 0
                outcome["reason"] = "fuse" if p["src"] == "crd_fuse" else "debt"
        outcome.update({"D_after": self.D, "mode_after": self.mode, "bad_run": self.bad_run, "fh_run": self.fh_run,
                        "recovery_misses": self.recovery_misses})
        return outcome
