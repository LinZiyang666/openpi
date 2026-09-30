"""R7 SF/UF and a reusable cache-only continuation component."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult, LookReason
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r07.stages.stages import StageTable


def _telemetry(extras, legacy=None):
    """All R7 scalars fit the pure-cache plugin's 24-scalar budget, first."""
    out = dict(extras)
    for key, value in (legacy or {}).items():
        if len(out) >= 24:
            break
        out.setdefault(key, value)
    return out


@dataclass(frozen=True)
class ExtensionPlan:
    blocks: int
    structural: bool
    stage_ok: bool
    extras: dict


class FollowExtension:
    """Stateless cache extension: plan at anchor, check current state at blind time.

    Composers retain the original cache anchor (rows/weights/action/rs), call
    plan once, check at every blind decision, and serve only cache provenance.
    A policy MISS clears the cache anchor; policy-tail hooks bypass this object.
    """

    def __init__(self, table, *, extend_blocks=1, stage_gate=True, state_valve=True):
        if type(extend_blocks) is not int or extend_blocks not in (0, 1, 2):
            raise ValueError("extend_blocks must be 0, 1 or 2")
        if type(stage_gate) is not bool or type(state_valve) is not bool:
            raise ValueError("stage_gate and state_valve must be bool")
        self.table, self.extend_blocks = table, extend_blocks
        self.stage_gate, self.state_valve = stage_gate, state_valve

    def plan(self, anchor):
        t, rows, w = self.table, anchor["rows"], anchor["weights"]
        info = t.online(rows, w)
        ex = {"os_sf_mode0": info["mode0_mass"], "os_sf_mode1": info["mode1_mass"],
              "os_sf_unanimous": float(info["unanimous"]), "os_sf_event_mass": info["event_mass"],
              "os_sf_rows_to_event": float(info["min_rows_to_event"]), "os_sf_unknown": info["unknown_mass"],
              "os_sf_stage_gate": float(self.stage_gate), "os_sf_state_valve": float(self.state_valve),
              "os_sf_cap": float(self.extend_blocks)}
        structural, stage_ok = True, bool(info["unanimous"])
        if self.extend_blocks:
            # Fixed cap: grant the requested E only if its entire extension is
            # supported; E=2 does not silently degrade to E=1.
            for age in range(1, t.reference_blocks + self.extend_blocks):
                future = t.advance(rows, age)
                if (future < 0).any():
                    structural = False; stage_ok = False; break
                stage_ok &= bool(np.all(t.mode[future] == info["mode"]))
        blocks = self.extend_blocks if structural and (stage_ok or not self.stage_gate) else 0
        ex.update(os_sf_structural=float(structural), os_sf_stage_ok=float(stage_ok), os_sf_granted=float(blocks))
        return ExtensionPlan(blocks, structural, stage_ok, ex)

    def check(self, anchor, bq, age, plan):
        ex = dict(plan.extras)
        ex.update(os_sf_age=float(age), os_sf_extension=float(age >= self.table.reference_blocks),
                  os_sf_valve_checked=0., os_sf_valve_fire=0., os_sf_delta=0.,
                  os_sf_radius=float(self.table.valve_radius), os_sf_look=0.)
        reason = None
        if age >= self.table.reference_blocks + plan.blocks:
            reason = LookReason(1, "budget")
        elif self.state_valve and plan.blocks:
            d, supported = self.table.displacement(bq.rs, anchor["rs"], anchor["rows"], anchor["weights"], age)
            ex.update(os_sf_valve_checked=1., os_sf_delta=d, os_sf_valve_fire=float(not supported or d > self.table.valve_radius))
            if ex["os_sf_valve_fire"]:
                reason = LookReason(11, "follow_state_valve")
        if reason is not None:
            ex["os_sf_look"] = float(reason.code)
        return reason, ex

    def serve(self, anchor, age, actions, library, extras):
        t, rows, w = self.table, anchor["rows"], anchor["weights"]
        stride, valid = t.exec_steps, int(t.manifest["act_valid_dims"])
        offset = age * stride
        action = np.zeros_like(anchor["action"], dtype=np.float32)
        if offset + stride <= len(anchor["action"]):
            # Native complete block: GR00T E=1 uses anchor controls 10:15.
            tail = anchor["action"][offset:, :valid]
            action[:len(tail), :valid] = tail
            action[len(tail):, :valid] = tail[-1]  # wire padding, not eligibility
            selected = rows
            extras["os_sf_source"] = 1.
        else:
            # π0.5 at ten controls and GR00T beyond its last complete block:
            # next^age member heads, original float32 weights, no renormalizing.
            selected = t.advance(rows, age)
            if (selected < 0).any():
                raise ValueError("unsupported extension reached serve")
            action[:stride, :valid] = np.tensordot(w, actions[selected, :stride, :valid], 1)
            action[stride:, :valid] = action[stride-1, :valid]
            extras["os_sf_source"] = 2.
        return BlindResult(action, np.asarray(selected, np.int64), w.copy(), library, extras)


class FollowAWM(BlindAWM):
    """A plus bounded cache continuation. Both gates disabled is uniform follow."""
    family = "r7_follow"

    def __init__(self, *, extend_blocks=1, stage_gate=True, state_valve=True, stages_path=None, **kwargs):
        # Public library names accepted; historical A's constructor calls dense
        # libraries 'big' and resolves the actual manifest name at fit time.
        explicit_lib = kwargs.get("lib")
        if explicit_lib in ("bpool_cs", "bpool_all"):
            kwargs["lib"] = "big"
        kwargs.setdefault("serving", "anchor_tail")
        kwargs.setdefault("budget", 1)
        kwargs.setdefault("gates", "budget_only")
        if kwargs["serving"] != "anchor_tail" or kwargs["budget"] != 1 or kwargs["gates"] != "budget_only":
            raise ValueError("SF/UF require selected A: anchor_tail, budget=1, budget_only")
        super().__init__(**kwargs)
        # Validate without fitting or reading a stage artifact.
        FollowExtension(None, extend_blocks=extend_blocks, stage_gate=stage_gate, state_valve=state_valve)
        self.follow_extend_blocks, self.follow_stage_gate, self.follow_state_valve = extend_blocks, stage_gate, state_valve
        self.follow_stages_path = str(stages_path) if stages_path is not None else None
        self.follow_explicit_lib = explicit_lib
        self._follow_plan = None
        self.name = f"SF_E{extend_blocks}_S{int(stage_gate)}_V{int(state_valve)}__{self.name}"

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        self.finish_follow_fit(lib, ctx)

    def finish_follow_fit(self, lib, ctx):
        if not self.follow_extend_blocks:
            # Disabled A must not acquire a new calibration/library dependency.
            self.follow_component = None
            return
        deployed = lib if self.cand_name == "current" else ctx.open_library(self.cand_name)
        if self.follow_explicit_lib in ("bpool_cs", "bpool_all") and self.cand_name != self.follow_explicit_lib:
            raise ValueError("explicit candidate library does not match model/library fit")
        manifest = json.loads((deployed.dir / "manifest.json").read_text())
        self.follow_table = StageTable.load(self.follow_stages_path, library=deployed, manifest=manifest) if self.follow_stages_path else StageTable.fit(deployed, manifest={**manifest, "_retrieval": self})
        from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import fingerprint
        if self.follow_table.retrieval_fingerprint != fingerprint(self):
            raise ValueError("stage calibration belongs to a different frozen A retrieval")
        self.follow_component = FollowExtension(self.follow_table, extend_blocks=self.follow_extend_blocks,
                                               stage_gate=self.follow_stage_gate, state_valve=self.follow_state_valve)
        self.fit_info = {**getattr(self, "fit_info", {}), "r7_stage_fingerprint": self.follow_table.fingerprint,
                         "r7_stage_calibration": self.follow_table.calibration}

    def invalidate_anchor(self):
        super().invalidate_anchor()
        self._follow_plan = None

    def _remember_anchor(self, q, rows, weights, action):
        super()._remember_anchor(q, rows, weights, action)
        self._follow_plan = self.follow_component.plan(self._anchor) if self.follow_extend_blocks else None

    def query(self, q):
        res = super().query(q)
        if self._follow_plan is not None:
            res.extras.update(self._follow_plan.extras)
        return res

    def blind_step(self, bq):
        if not self.follow_extend_blocks:
            return super().blind_step(bq)  # exact disabled path, including extras
        reason = self.lifecycle_reason(bq)
        if reason is not None:
            self.last_blind_extras = {"look_reason": 6., "os_sf_look": 6.}
            return reason
        a, plan = self._anchor, self._follow_plan
        age = int(bq.step) - a["step"]
        if plan is None or not plan.blocks:
            # Hard/unsupported anchor executes exactly A, including its first
            # blind block. Stage checks never cut an unextended commitment.
            result = super().blind_step(bq)
            if plan is not None:
                self.last_blind_extras.update(plan.extras)
            return result
        reason, ex = self.follow_component.check(a, bq, age, plan)
        if reason is not None:
            self.last_blind_extras = ex
            return reason
        if age < self.follow_table.reference_blocks:
            result = super().blind_step(bq)
            legacy = dict(self.last_blind_extras)
            result.extras.clear()
            result.extras.update(_telemetry(ex, legacy))
            self.last_blind_extras = result.extras
            return result
        result = self.follow_component.serve(a, age, self.act, self.cand_name, ex)
        a["last_step"] = int(bq.step)
        a["phase"] = self.follow_table.advance(a["rows"], age)
        self.last_blind_extras = ex
        return result


class StageFollow(FollowAWM):
    """SF defaults; kwargs may independently ablate either gate."""


class UniformFollow(FollowAWM):
    """UF: structural support only."""
    def __init__(self, **kwargs):
        kwargs.setdefault("stage_gate", False)
        kwargs.setdefault("state_valve", False)
        super().__init__(**kwargs)
