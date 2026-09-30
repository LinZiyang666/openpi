"""Stage wrist: request-owned camera plans and A's independently refitted wrist metric.

The base may be A or C1's StageFollow. Only its retrieval fields are switched
for a wrist anchor; its anchor/tail/extension lifecycle remains authoritative.
Numbers come from the deployed A recipe, the library manifest and StageTable.
"""
from __future__ import annotations

import dataclasses
import pathlib
import pickle

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.closed_loop.blind import BlindResult, LookReason
from exp.offline_search.closed_loop.plugin import load_method_class
from exp.offline_search.rounds.r04.k1_blind.wrist import BlindWristAWM
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r04.k3_cost.method import WristView
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler, fingerprint
from exp.offline_search.rounds.r07.stages.stages import StageTable

BASE = "exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM"
METRIC_FIELDS = ("tasks", "B0T", "B1T", "mu0", "mu1", "muB0", "muB1", "zmu", "zsd", "zs_sd", "s_a")


class StageWrist(api.Method):
    tier = "T1"
    family = "r7_stage_wrist"
    camera_mode = "per_request"

    def __init__(self, *, enabled=True, base_spec=BASE, base_kwargs=None, base_fit="", wrist_fit="", stage_fit=""):
        self.enabled = bool(enabled)
        self.base_spec, self.base_kwargs = base_spec, dict(base_kwargs or {})
        self.base_fit, self.wrist_fit, self.stage_fit = str(base_fit), str(wrist_fit), str(stage_fit)
        self.name = "SW" if self.enabled else "SW_OFF"
        self.base = None
        self.next_camera_mode = self._camera_mode = "full"
        self.last_blind_extras = {}
        self._plan_extras = {}

    @property
    def _anchor(self):
        return None if self.base is None else self.base._anchor

    def fit(self, lib, ctx):
        cls, _ = load_method_class(self.base_spec)
        if self.base_fit:
            with open(self.base_fit, "rb") as f:
                blob = FitUnpickler(f).load()
            follow_keys = {"extend_blocks", "stage_gate", "state_valve", "stages_path"}
            source_kwargs = {k: v for k, v in self.base_kwargs.items() if k not in follow_keys}
            if blob["kwargs"] != source_kwargs or blob["cell"] != ctx.cell:
                raise ValueError("SW base artifact kwargs/cell mismatch")
            if isinstance(blob["method"], cls):
                self.base = blob["method"]
            elif isinstance(blob["method"], BlindAWM) and issubclass(cls, BlindAWM):
                self.base = cls(**self.base_kwargs)
                if not callable(getattr(self.base, "finish_follow_fit", None)):
                    raise ValueError("SW can compose only the validated C1 follow subclass")
                config = {k: v for k, v in vars(self.base).items() if k.startswith("follow_") or k == "name"}
                vars(self.base).update(vars(blob["method"]))
                vars(self.base).update(config)
                self.base.finish_follow_fit(lib, ctx)
            else:
                raise ValueError("SW base artifact class mismatch")
        else:
            self.base = cls(**self.base_kwargs)
            self.base.prof = self.prof
            self.base.fit(lib, ctx)
        self.base.prof = self.prof
        # SF is an A subclass, so these fields remain its authoritative metric.
        self.name += "__" + self.base.name
        self.os_library = self.base.cand_name
        self.H = self.base.H
        if not self.enabled:
            return
        if ctx.model != "pi05":
            raise api.SkipCell("SW unavailable: GR00T has no validated one-camera encoder path")
        deployed = lib if self.base.cand_name == "current" else ctx.open_library(self.base.cand_name)
        self.manifest = dict(deployed.meta)
        camera_sources = self.manifest.get("img_source_keys", {})
        if camera_sources.get("img1", "").split("/")[-1] != "left_wrist_0_rgb":
            raise api.ContractError("SW requires the validated pi05 wrist camera mapping")
        self.block_controls = int(self.manifest["exec_steps"])
        self.state_width = int(self.manifest["rs_valid_dims"])
        self.gripper_dim = int(self.manifest["gripper_dim"])
        self.reference_blocks = int(self.base.budget) + 1
        if self.base.serving != "anchor_tail" or self.base.gates != "budget_only" or self.base.budget != 1:
            raise api.ContractError("SW requires today's A commitment (anchor_tail, budget=1, budget_only)")
        wrist_kwargs = {k: getattr(self.base, k) for k in (
            "features", "lib", "fit_data", "kref", "k", "codes", "lam", "state_scale", "early",
            "step0_joint", "lam_c", "norm_cap", "hyst", "nn", "serving", "budget", "gates", "residual_threshold")}
        if self.wrist_fit:
            with open(self.wrist_fit, "rb") as f:
                blob = pickle.load(f)
            if blob["kwargs"] != wrist_kwargs or blob["cell"] != ctx.cell:
                raise ValueError("SW wrist artifact kwargs/cell mismatch")
            self.wrist = blob["method"]
        else:
            self.wrist = BlindWristAWM(**wrist_kwargs)
            self.wrist.prof = self.prof
            self.wrist.fit(lib, ctx)
        self.wrist.prof = self.prof
        expected_width = self.wrist.B1T.shape[0] + self.state_width
        if any(t.Wf.shape[0] != expected_width for t in self.wrist.tasks.values()):
            raise api.ContractError("wrist metric was not refitted in its own PCA+state space")
        if self.stage_fit:
            self.stages = StageTable.load(self.stage_fit, library=deployed, manifest=self.manifest)
        elif isinstance(getattr(self.base, "follow_table", None), StageTable):
            self.stages = self.base.follow_table
        else:
            self.stages = StageTable.fit(deployed, manifest={**self.manifest, "_retrieval": self.base})
        if self.stages.retrieval_fingerprint != fingerprint(self.base):
            raise ValueError("SW stage calibration belongs to another frozen A retrieval")
        self.fit_info = {"metric_width": expected_width, "wrist_pca_width": self.wrist.B1T.shape[0],
                         "library": self.base.cand_name, "stage_fingerprint": self.stages.fingerprint,
                         "stage_retrieval_fingerprint": self.stages.retrieval_fingerprint,
                         "calibration": self.stages.calibration,
                         "cost_assumption": "R4 proportional-latency: wrist=.055198, completion=.049890"}

    def reset(self, episode):
        self.base.reset(episode)
        if self.enabled:
            self.wrist.reset(episode)
        self.next_camera_mode = self._camera_mode = "full"
        self._plan_extras = {}
        self.last_blind_extras = {}

    def set_camera_mode(self, mode):
        if mode not in ("full", "wrist_only") or (not self.enabled and mode != "full"):
            raise ValueError("unsupported SW camera request")
        self._camera_mode = mode

    def invalidate_anchor(self):
        self.base.invalidate_anchor()
        self.next_camera_mode = "full"
        self._plan_extras = {"os_sw_next_camera": 0., "os_sw_reason": 1.}

    def _plan(self, current, elapsed, target, command_action=None):
        """All members stay in the current mode through the next anchor's head.

        StageTable owns segmentation, chain validation and the library-p95
        normal-state valve. A missing/failed member remains a hard stage even
        when its kernel weight underflows to zero.
        """
        self.next_camera_mode = "full"
        ex = {"os_sw_next_camera": 0., "os_sw_reason": 1.}
        a = self._anchor
        if a is None:
            self._plan_extras = ex
            return
        rows, weights = a["rows"], a["weights"]
        command_action = a["action"] if command_action is None else command_action
        command = float(np.median(command_action[:self.block_controls, self.gripper_dim]))
        cmd_mode = int(command > self.stages.gripper_threshold) if np.isfinite(command) else -1
        online = self.stages.online(rows, weights, cmd_mode=cmd_mode)
        ex.update(os_sw_mode_mass=max(online["mode_mass"]), os_sw_unanimous=float(online["unanimous"]),
                  os_sw_event_mass=online["event_mass"], os_sw_unknown_mass=online["unknown_mass"],
                  os_sw_rows_to_event=float(online["min_rows_to_event"]))
        reason = 0
        if not online["unanimous"] or online["unknown_mass"] > 0:
            reason = 2
        # Check every chain position, including the next anchor's head. The
        # command mode cannot change on the way to that anchor.
        for depth in range(target + 1):
            advanced = self.stages.advance(rows, depth)
            if np.any(advanced < 0):
                reason = reason or 3
                break
            info = self.stages.online(advanced, weights, cmd_mode=cmd_mode)
            if not info["unanimous"] or np.any(self.stages.event_near[advanced]):
                reason = reason or 4
                break
        deviation, supported = self.stages.displacement(current, a["rs"], rows, weights, elapsed)
        ex.update(os_sw_valve=deviation, os_sw_radius=self.stages.valve_radius,
                  os_sw_target_blocks=float(target))
        if not supported or not np.isfinite(deviation) or deviation > self.stages.valve_radius:
            reason = reason or 5
        if reason == 0:
            self.next_camera_mode = "wrist_only"
            ex["os_sw_next_camera"] = 1.
        ex["os_sw_reason"] = float(reason)
        self._plan_extras = ex

    def query(self, q):
        if not self.enabled:
            return self.base.query(q)
        chosen = self._camera_mode
        if int(q.step) == 0:
            if chosen != "full":
                raise api.ContractError("episode start must encode all cameras")
        if chosen == "wrist_only":
            # These references are changed only on this connection's cloned
            # method and restored even after failure. SF executes its ordinary
            # query/anchor bookkeeping with the new 72-D metric and same kernel.
            saved = {key: getattr(self.base, key) for key in METRIC_FIELDS}
            try:
                for key in METRIC_FIELDS:
                    setattr(self.base, key, getattr(self.wrist, key))
                result = self.base.query(WristView(q))
            finally:
                for key, value in saved.items():
                    setattr(self.base, key, value)
        else:
            result = self.base.query(q)
        self._plan(q.rs, 0, self.reference_blocks)
        # Keep camera/stage decisions before inherited diagnostics within cap 24.
        return dataclasses.replace(result, extras=self._extras(result.extras))

    def _extras(self, extras):
        # Essential composite decisions precede the scalar log budget; inherited
        # high-dimensional blind diagnostics may still live in blind_extras.
        extras = extras or {}
        follow = {k: extras[k] for k in ("os_sf_granted", "os_sf_structural", "os_sf_stage_ok",
                  "os_sf_cap", "os_sf_delta", "os_sf_radius", "os_sf_valve_fire", "os_sf_look") if k in extras}
        return {**follow, **self._plan_extras, **extras}

    def blind_step(self, bq):
        result = self.base.blind_step(bq)
        if not self.enabled:
            self.last_blind_extras = self.base.last_blind_extras
            return result
        a = self._anchor
        if a is not None and bq.prev_hit and np.isfinite(np.asarray(bq.rs)[:self.state_width]).all():
            elapsed = int(bq.step) - a["step"]
            target = elapsed if isinstance(result, LookReason) else elapsed + 1
            command_action = result.action if isinstance(result, BlindResult) else bq.prev_a_exec
            self._plan(bq.rs, elapsed, target, command_action=command_action)
            # C1's valve/event/lifecycle veto remains hard. Its cap LOOK permits
            # a normal camera plan for the new anchor.
            if isinstance(result, LookReason) and result.code not in (1,):
                self.next_camera_mode = "full"
                self._plan_extras.update(os_sw_next_camera=0., os_sw_reason=6.)
        else:
            self.next_camera_mode = "full"
            self._plan_extras = {"os_sw_next_camera": 0., "os_sw_reason": 1.}
        self.last_blind_extras = self._extras(getattr(self.base, "last_blind_extras", {}))
        if isinstance(result, BlindResult):
            result = dataclasses.replace(result, extras=self._extras(result.extras))
        return result

    def bytes_per_entry(self):
        return self.base.bytes_per_entry() + (self.wrist.bytes_per_entry() if self.enabled else 0.)
