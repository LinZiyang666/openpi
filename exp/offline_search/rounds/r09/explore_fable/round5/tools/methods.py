"""Round 5 (fable): look-cost variants on top of the round-3c stacks.  New classes only; every r3c class is untouched.

The stacks pay a full look (.152 pi0.5 / .148 GR00T) at every 10-control anchor, about .075 of their ~.16-.18 IR.
Two task-agnostic levers (no task id enters any gate):

Lever 1 -- wrist-only looks (pi0.5 only; the plugin's per-request camera path, ``--os-request-cameras --os-tokens off``):
  ``wrist_gate="all"``   every method-planned look is wrist-only (lifecycle / forced looks stay full, the plugin's rule);
  ``wrist_gate="easy"``  R7's stage-wrist plan: gripper-mode unanimity along the chain to the next anchor, no gripper
                         event near, state within the library valve -- the one look lever R8 found lossless on pure cache;
  ``wrist_gate="pace"``  the episode keeps the demonstration's pace (opus's lag <= ``pace_lag`` decisions) and the
                         chain/valve conditions hold; no gripper-stage condition.
  A wrist-only look retrieves in R7's refitted wrist metric (72-D: wrist PCA + state) and skips the corrector, whose
  features need the third-camera key that a wrist look does not encode.

Lever 2 -- gated follow (both models): at the look due after 10 controls, when R7's extension gates pass (stage gate
  optional, state valve always), serve the stage-table successor block instead of looking; capped at one extra block.
  The no-progress guard keeps its blind veto (it fires before the extension is considered).

Fit-time asserts (as in round 3c): judge.burst unchanged, single judge family, CorrectedCacheJW base with blend .5,
stage table fitted to the same frozen retrieval (fingerprint), wrist metric refitted in its own 72-D space, no attribute
of the frozen judge shadowed.
"""
from __future__ import annotations

import dataclasses
import json
import pickle

import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult, LookReason
from exp.offline_search.harness import api
from exp.offline_search.rounds.r04.k3_cost.method import WristView
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler, fingerprint
from exp.offline_search.rounds.r07.c1_follow.methods import FollowExtension
from exp.offline_search.rounds.r07.c2_wrist.method import METRIC_FIELDS
from exp.offline_search.rounds.r07.stages.stages import StageTable
from exp.offline_search.rounds.r09.explore_fable.round3.tools.methods import (CorrectedCacheJ, NpGraspEsc3,
                                                                              NpGraspStackGroot3)

WRIST_GATES = ("off", "all", "easy", "pace")
WRIST_KWARG_KEYS = ("features", "lib", "fit_data", "kref", "k", "codes", "lam", "state_scale", "early", "step0_joint",
                    "lam_c", "norm_cap", "hyst", "nn", "serving", "budget", "gates", "residual_threshold")
LC_FULL = {"os_lc_cam": 0., "os_lc_reason": 1., "os_lc_lag": float("nan"), "os_lc_fgrant": 0.}


class CorrectedCacheJW(CorrectedCacheJ):
    """CorrectedCacheJ whose correction is skipped while ``wrist_pass`` is set (wrist-only look: no third-camera key)."""
    wrist_pass = False

    def os_synth(self, q, rows, w):
        if self.wrist_pass:
            return super(CorrectedCacheJ, self).os_synth(q, rows, w)      # plain kernel chunk + anchor memo
        return super().os_synth(q, rows, w)


class _LookCost:
    """Mixin over a fitted r3c stack (NpGraspEsc3 / NpGraspStackGroot3).  All state under ``lc_`` / camera hooks."""
    _LC_OWN = ("lc_wrist_gate", "lc_pace_lag", "lc_follow_blocks", "lc_follow_stage_gate", "lc_stage_fit", "lc_wrist_fit",
               "lc_wrist", "lc_stages", "lc_follow", "lc_follow_plan", "lc_plan_extras", "lc_last_lag", "lc_block_controls",
               "lc_state_width", "lc_gripper_dim", "lc_reference_blocks", "next_camera_mode", "_lc_camera_mode")
    camera_mode = "per_request"

    def _lc_init(self, wrist_gate, pace_lag, follow_blocks, follow_stage_gate, stage_fit, wrist_fit):
        if wrist_gate not in WRIST_GATES:
            raise ValueError(f"wrist_gate must be one of {WRIST_GATES}")
        if type(follow_blocks) is not int or follow_blocks not in (0, 1):
            raise ValueError("follow_blocks must be 0 or 1")
        if type(follow_stage_gate) is not bool:
            raise ValueError("follow_stage_gate must be bool")
        if isinstance(pace_lag, bool) or not np.isfinite(pace_lag) or pace_lag < 0:
            raise ValueError("pace_lag must be a finite nonnegative number of decisions")
        if wrist_gate != "off" and not wrist_fit:
            raise ValueError("wrist looks need the R7 wrist artifact (wrist_fit)")
        self.lc_wrist_gate, self.lc_pace_lag = str(wrist_gate), float(pace_lag)
        self.lc_follow_blocks, self.lc_follow_stage_gate = int(follow_blocks), bool(follow_stage_gate)
        self.lc_stage_fit, self.lc_wrist_fit = str(stage_fit), str(wrist_fit)
        self.lc_wrist = self.lc_stages = self.lc_follow = self.lc_follow_plan = None
        self.lc_plan_extras, self.lc_last_lag = dict(LC_FULL), float("nan")
        self.lc_block_controls = self.lc_state_width = self.lc_gripper_dim = self.lc_reference_blocks = 0
        self.next_camera_mode = self._lc_camera_mode = "full"

    # ----------------------------------------------------------------------------------------------- fit
    def fit(self, lib, ctx):
        with open(self.gm_onlynp_fit, "rb") as f:
            src = FitUnpickler(f).load()["method"]
        clash = sorted(k for k in self._LC_OWN if k in vars(src))
        if clash:
            raise api.ContractError("look-cost attributes shadow frozen judge attributes: %s" % clash)
        own = {k: getattr(self, k) for k in self._LC_OWN}
        self._gm_fit(ctx)                      # frozen judge + CorrectedCacheJ base + r3c asserts
        vars(self).update(own)
        self._lc_fit(lib, ctx)

    def _lc_fit(self, lib, ctx):
        base = self.base
        if type(base) is not CorrectedCacheJ or float(base.blend) != 0.5:
            raise api.ContractError("look-cost stack expects the r3c half-strength CorrectedCacheJ base")
        base.__class__ = CorrectedCacheJW
        base.wrist_pass = False
        if base.serving != "anchor_tail" or base.gates != "budget_only" or int(base.budget) != 1:
            raise api.ContractError("look-cost stack requires today's A commitment (anchor_tail, budget=1, budget_only)")
        deployed = lib if base.cand_name == "current" else ctx.open_library(base.cand_name)
        manifest = dict(deployed.meta)
        if self.lc_stage_fit:
            self.lc_stages = StageTable.load(self.lc_stage_fit, library=deployed, manifest=manifest)
        else:
            self.lc_stages = StageTable.fit(deployed, manifest={**manifest, "_retrieval": base})
        if self.lc_stages.retrieval_fingerprint != fingerprint(base):
            raise ValueError("stage calibration belongs to a different frozen A retrieval")
        self.lc_block_controls = int(manifest["exec_steps"])
        self.lc_state_width = int(manifest["rs_valid_dims"])
        self.lc_gripper_dim = int(manifest["gripper_dim"])
        self.lc_reference_blocks = int(base.budget) + 1
        if self.lc_stages.reference_blocks != self.lc_reference_blocks:
            raise api.ContractError("stage table reference commitment differs from the base commitment")
        if self.lc_wrist_gate != "off":
            if ctx.model != "pi05":
                raise api.ContractError("wrist-only looks are pi0.5 only (no validated one-camera GR00T path)")
            if manifest.get("img_source_keys", {}).get("img1", "").split("/")[-1] != "left_wrist_0_rgb":
                raise api.ContractError("wrist looks require the validated pi0.5 wrist camera mapping")
            wrist_kwargs = {k: getattr(base, k) for k in WRIST_KWARG_KEYS}
            with open(self.lc_wrist_fit, "rb") as f:
                blob = pickle.load(f)
            if blob["kwargs"] != wrist_kwargs or blob["cell"] != ctx.cell:
                raise ValueError("wrist artifact kwargs/cell mismatch")
            self.lc_wrist = blob["method"]
            self.lc_wrist.prof = api.NULL_PROFILER
            width = self.lc_wrist.B1T.shape[0] + self.lc_state_width
            if any(t.Wf.shape[0] != width for t in self.lc_wrist.tasks.values()):
                raise api.ContractError("wrist metric was not refitted in its own PCA+state space")
            if set(METRIC_FIELDS) - set(vars(self.lc_wrist)):
                raise api.ContractError("wrist artifact lacks metric fields")
        if self.lc_follow_blocks:
            self.lc_follow = FollowExtension(self.lc_stages, extend_blocks=self.lc_follow_blocks,
                                             stage_gate=self.lc_follow_stage_gate, state_valve=True)
        self.fit_info = dict(getattr(self, "fit_info", {}) or {}, lc_wrist_gate=self.lc_wrist_gate, lc_pace_lag=self.lc_pace_lag,
                             lc_follow_blocks=self.lc_follow_blocks, lc_follow_stage_gate=self.lc_follow_stage_gate,
                             lc_stage_fingerprint=self.lc_stages.fingerprint,
                             lc_stage_retrieval_fingerprint=self.lc_stages.retrieval_fingerprint,
                             lc_wrist_metric_width=(self.lc_wrist.B1T.shape[0] + self.lc_state_width) if self.lc_wrist else 0,
                             corrector_on_wrist_look="skipped", judge_burst=int(self.burst), base_class=type(base).__name__)

    # ----------------------------------------------------------------------------------------------- plugin hooks
    def set_camera_mode(self, mode):
        if mode not in ("full", "wrist_only") or (mode == "wrist_only" and self.lc_wrist is None):
            raise ValueError("unsupported camera request")
        self._lc_camera_mode = mode

    def invalidate_anchor(self):
        super().invalidate_anchor()
        self.lc_follow_plan = None
        self.next_camera_mode = "full"
        self.lc_plan_extras = dict(LC_FULL)

    def reset(self, episode):
        super().reset(episode)
        self.lc_follow_plan = None
        self.next_camera_mode = self._lc_camera_mode = "full"
        self.lc_plan_extras, self.lc_last_lag = dict(LC_FULL), float("nan")

    # ----------------------------------------------------------------------------------------------- planning
    def _lc_camera_plan(self, current, elapsed, target, command_action, lag):
        """R7's stage-wrist plan (reasons 2-5) gated by ``lc_wrist_gate``; sets ``next_camera_mode`` for the next look."""
        self.next_camera_mode = "full"
        ex = dict(LC_FULL, os_lc_lag=float(lag), os_lc_fgrant=self.lc_plan_extras.get("os_lc_fgrant", 0.))
        a = self.base._anchor
        if self.lc_wrist is None or a is None:
            self.lc_plan_extras = ex
            return
        rows, weights = a["rows"], a["weights"]
        command_action = a["action"] if command_action is None else command_action
        command = float(np.median(np.asarray(command_action)[:self.lc_block_controls, self.lc_gripper_dim]))
        cmd_mode = int(command > self.lc_stages.gripper_threshold) if np.isfinite(command) else -1
        online = self.lc_stages.online(rows, weights, cmd_mode=cmd_mode)
        reason = 0
        if not online["unanimous"] or online["unknown_mass"] > 0:
            reason = 2
        for depth in range(int(target) + 1):
            advanced = self.lc_stages.advance(rows, depth)
            if np.any(advanced < 0):
                reason = reason or 3
                break
            info = self.lc_stages.online(advanced, weights, cmd_mode=cmd_mode)
            if not info["unanimous"] or np.any(self.lc_stages.event_near[advanced]):
                reason = reason or 4
                break
        deviation, supported = self.lc_stages.displacement(current, a["rs"], rows, weights, elapsed)
        if not supported or not np.isfinite(deviation) or deviation > self.lc_stages.valve_radius:
            reason = reason or 5
        structural_ok = reason not in (3, 5)
        if self.lc_wrist_gate == "all":
            wrist = True
        elif self.lc_wrist_gate == "easy":
            wrist = reason == 0
        else:                                   # pace
            wrist = structural_ok and np.isfinite(lag) and lag <= self.lc_pace_lag
        ex.update(os_lc_cam=float(wrist), os_lc_reason=float(reason), os_lc_valve=float(deviation))
        if wrist:
            self.next_camera_mode = "wrist_only"
        self.lc_plan_extras = ex

    # ----------------------------------------------------------------------------------------------- decisions
    def query(self, q):
        wrist = self._lc_camera_mode == "wrist_only"
        if wrist:
            if int(q.step) == 0:
                raise api.ContractError("episode start must encode all cameras")
            base, adapter = self.base, self.lc_wrist
            saved = {k: getattr(base, k) for k in METRIC_FIELDS}
            base.wrist_pass = True
            try:
                for k in METRIC_FIELDS:
                    setattr(base, k, getattr(adapter, k))
                # The judge scores through the G3 contract; A's os_score_all splits the projection per camera, so the
                # wrist adapter's own os_score_all (72-D metric, vis aux from the wrist camera) answers this look.
                base.os_score_all = adapter.os_score_all
                res = super().query(WristView(q))
            finally:
                vars(base).pop("os_score_all", None)
                for k, v in saved.items():
                    setattr(base, k, v)
                base.wrist_pass = False
        else:
            res = super().query(q)
        a = self.base._anchor
        ex = dict(res.extras or {})
        if a is not None and a.get("step") == int(q.step) and len(res.topk):
            self.lc_last_lag = float(int(q.step) - int(self.base.lib_step[int(res.topk[0])]))
            self.lc_follow_plan = self.lc_follow.plan(a) if self.lc_follow is not None else None
            self.lc_plan_extras = dict(LC_FULL, os_lc_fgrant=float(self.lc_follow_plan.blocks) if self.lc_follow_plan else 0.)
            self._lc_camera_plan(q.rs, 0, self.lc_reference_blocks, a["action"], self.lc_last_lag)
        else:
            self.lc_follow_plan, self.lc_plan_extras = None, dict(LC_FULL)
            self.next_camera_mode = "full"
        ex = {**self.lc_plan_extras, "os_lc_wrist_look": float(wrist), **ex}
        return api.Result(res.topk, res.scores, res.confidence, action=res.action, library=res.library, extras=ex)

    def blind_step(self, bq):
        res = super().blind_step(bq)
        a = self.base._anchor
        plan = self.lc_follow_plan
        valid = (a is not None and bool(bq.prev_hit) and np.isfinite(np.asarray(bq.rs)[:self.lc_state_width]).all())
        if (self.lc_follow is not None and valid and plan is not None and plan.blocks
                and isinstance(res, LookReason) and res.code == 1):
            age = int(bq.step) - a["step"]
            if self.lc_reference_blocks <= age < self.lc_reference_blocks + plan.blocks:
                reason, ex = self.lc_follow.check(a, bq, age, plan)
                if reason is None:
                    res = self.lc_follow.serve(a, age, self.base.act, self.base.cand_name, ex)
                    a["phase"] = self.lc_stages.advance(a["rows"], age)
                    a["last_step"] = int(bq.step)
                    self.base.last_blind_extras = ex
                else:
                    res = reason
        if self.lc_wrist is not None:
            if valid:
                elapsed = int(bq.step) - a["step"]
                target = elapsed if isinstance(res, LookReason) else elapsed + 1
                command_action = res.action if isinstance(res, BlindResult) else bq.prev_a_exec
                self._lc_camera_plan(bq.rs, elapsed, target, command_action, self.lc_last_lag)
                if isinstance(res, LookReason) and res.code != 1:
                    self.next_camera_mode = "full"
                    self.lc_plan_extras.update(os_lc_cam=0., os_lc_reason=6.)
            else:
                self.next_camera_mode = "full"
                self.lc_plan_extras = dict(LC_FULL)
        if isinstance(res, BlindResult):
            res = dataclasses.replace(res, extras={**self.lc_plan_extras, **(res.extras or {})})
        return res


class LookCostEsc(_LookCost, NpGraspEsc3):
    """pi0.5: r3c guard + half corrector + escalation stack with wrist-only looks and/or gated follow."""
    family = "r9f5_lookcost_esc"

    def __init__(self, wrist_gate="off", pace_lag=1, follow_blocks=0, follow_stage_gate=True, stage_fit="", wrist_fit="", **kw):
        NpGraspEsc3.__init__(self, **kw)
        self._lc_init(wrist_gate, pace_lag, follow_blocks, follow_stage_gate, stage_fit, wrist_fit)
        self.gm_name = f"R9F5_lookcost_esc_W{wrist_gate}_F{follow_blocks}{'s' if follow_stage_gate else 'v'}"


class LookCostGroot(_LookCost, NpGraspStackGroot3):
    """GR00T: r3c guard + half corrector stack with gated follow (wrist looks unsupported for GR00T)."""
    family = "r9f5_lookcost_groot"

    def __init__(self, wrist_gate="off", pace_lag=1, follow_blocks=0, follow_stage_gate=True, stage_fit="", wrist_fit="", **kw):
        if wrist_gate != "off":
            raise ValueError("GR00T has no validated one-camera path; wrist_gate must be 'off'")
        NpGraspStackGroot3.__init__(self, **kw)
        self._lc_init(wrist_gate, pace_lag, follow_blocks, follow_stage_gate, stage_fit, wrist_fit)
        self.gm_name = f"R9F5_lookcost_groot_F{follow_blocks}{'s' if follow_stage_gate else 'v'}"
