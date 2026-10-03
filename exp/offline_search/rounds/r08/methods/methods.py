"""Small R8 variants of frozen A, R7 follow/wrist, and CU's policy tail.

All coins use episode identity, never process/worker/arrival order or global RNG.
Diagnostics are saved while deciding; debug_record only returns detached JSON.
"""
import copy
import hashlib
import math
from types import SimpleNamespace

import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult, LookReason
from exp.offline_search.harness import api
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r05.q1_commit.judge import CommitJudge
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r07.c1_follow.methods import FollowAWM, FollowExtension
from exp.offline_search.rounds.r07.c2_wrist.method import StageWrist

BASE = "exp.offline_search.rounds.r04.k1_blind.blind_awm:BlindAWM"


def coin(seed, task_id, init, decision_seq, domain):
    """Uniform [0,1) with a named domain and 53 exactly representable bits."""
    if task_id < 0 or init < 0 or decision_seq < 0:
        raise api.ContractError("R8 coins require original nonnegative task/init/decision identity")
    key = "{}|{}|{}|{}|{}".format(domain, seed, task_id, init, decision_seq)
    return (int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big") >> 11) / float(2**53)


def _random_config(seed, domain):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("random_seed must be an integer in [0, 2**32)")
    if not isinstance(domain, str) or not domain:
        raise ValueError("coin_domain must be a nonempty string")


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return copy.deepcopy(value)


class Diagnostics:
    def debug_record(self):
        return _json(getattr(self, "_r8_diag", {}))


class FollowLottery(Diagnostics, FollowAWM):
    """At each cache anchor choose uniformly from structurally supported E."""
    family = "r8_follow_lottery"

    def __init__(self, random_seed=26093005, coin_domain="R8/follow", force_e=None, **kwargs):
        _random_config(random_seed, coin_domain)
        if force_e is not None and (type(force_e) is not int or force_e not in (0, 1, 2)):
            raise ValueError("force_e must be None, 0, 1 or 2")
        for k, v in (("extend_blocks", 2), ("stage_gate", False), ("state_valve", False)):
            if k in kwargs and kwargs[k] != v:
                raise ValueError("lottery requires {}={!r}".format(k, v))
            kwargs[k] = v
        super().__init__(**kwargs)
        self.lottery_seed, self.lottery_domain, self.lottery_force_e = random_seed, coin_domain, force_e
        self._r8_diag = {}
        self._r8_assignment = {}
        self.name = "R8_FL__" + self.name

    def reset(self, episode):
        super().reset(episode)
        self._r8_diag, self._r8_assignment = {}, {}

    def _remember_anchor(self, q, rows, weights, action):
        BlindAWM._remember_anchor(self, q, rows, weights, action)
        t, a = self.follow_table, self._anchor
        plans = [FollowExtension(t, extend_blocks=e, stage_gate=False, state_valve=False).plan(a)
                 for e in (0, 1, 2)]
        support = [e for e, plan in enumerate(plans) if e == 0 or plan.structural]
        u = coin(self.lottery_seed, q.task_id, q.episode.init, q.step, self.lottery_domain)
        chosen = support[min(int(u * len(support)), len(support) - 1)]
        if self.lottery_force_e is not None:
            if self.lottery_force_e not in support:
                raise api.ContractError("forced E is outside structural support")
            chosen = self.lottery_force_e
        self._follow_plan = plans[chosen]
        probabilities = [float(e == chosen) if self.lottery_force_e is not None else
                         (1. / len(support) if e in support else 0.) for e in (0, 1, 2)]
        self._r8_assignment = dict(support=support, propensities=probabilities, drawn_e=chosen,
            e_max=max(support), coin=u, coin_domain=self.lottery_domain, random_seed=self.lottery_seed,
            eligible=len(support) > 1, treatment=chosen, override="force_e" if self.lottery_force_e is not None else None,
            anchor_step=int(q.step), stage_ok_by_e=[bool(p.stage_ok) for p in plans],
            shadow_gates_enforced=False)
        self._record_shadow(q.rs, 0, "cache")

    def query(self, q):
        # Keep A's Result/extras exact, including the force_e=0 identity path.
        return BlindAWM.query(self, q)

    def _record_shadow(self, state, age, src):
        diag = dict(self._r8_assignment, src=src, blind_age_controls=age * self.follow_table.exec_steps)
        a, t = self._anchor, self.follow_table
        if a is not None:
            try:
                delta, supported = t.displacement(state, a["rs"], a["rows"], a["weights"], age)
                future = t.advance(a["rows"], age)
                absolute = t.deviation(state, future, a["weights"]) if (future >= 0).all() else None
                diag.update(shadow_valve_status="available", shadow_delta=delta, shadow_absolute=absolute,
                            shadow_radius=t.valve_radius, shadow_valve_supported=bool(supported),
                            shadow_valve_fire=bool(not supported or delta > t.valve_radius))
            except Exception as exc:
                diag.update(shadow_valve_status="error", shadow_valve_reason=str(exc))
        self._r8_diag = diag

    def blind_step(self, bq):
        # R7's structural-only continuation; neither its stage gate nor valve acts.
        self._record_shadow(bq.rs, int(bq.step) - self._r8_assignment.get("anchor_step", int(bq.step)), "blind")
        if self._follow_plan is not None and self._follow_plan.blocks == 0:
            result = BlindAWM.blind_step(self, bq)
        else:
            result = FollowAWM.blind_step(self, bq)
        self._r8_diag.update(src="look" if isinstance(result, LookReason) else "follow" if
            self._r8_diag["blind_age_controls"] >= self.follow_table.reference_blocks * self.follow_table.exec_steps else "cache_tail",
            look_reason=result.name if isinstance(result, LookReason) else None)
        if isinstance(result, BlindResult):
            age = int(bq.step) - self._anchor["step"]
            self._r8_diag.update(successor_rows=self.follow_table.advance(self._anchor["rows"], age).tolist(),
                follow_source="native_tail" if result.extras.get("os_sf_source", 1) == 1 else "successor_heads")
        return result


class EveryFiveAWM(Diagnostics, BlindAWM):
    """A's existing budget=0 path: a full look at every five-control request."""
    family = "r8_cadence"

    def __init__(self, **kwargs):
        kwargs.setdefault("serving", "anchor_tail")
        kwargs.setdefault("gates", "budget_only")
        kwargs.setdefault("budget", 0)
        if kwargs["budget"] != 0:
            raise ValueError("EveryFiveAWM requires budget=0")
        super().__init__(**kwargs)
        self._r8_diag = {}

    def query(self, q):
        result = super().query(q)
        self._r8_diag = dict(src="cache", cadence_controls=5, anchor_step=int(q.step))
        return result

    def blind_step(self, bq):
        result = super().blind_step(bq)
        self._r8_diag = dict(src="look", cadence_controls=5, look_reason=result.name)
        return result

    def reset(self, episode):
        super().reset(episode)
        self._r8_diag = {}


class ShiftedAWM(Diagnostics, BlindAWM):
    """Shorten only the first commitment to five controls, then ordinary A."""
    family = "r8_shifted_placebo"

    def __init__(self, shifted=True, **kwargs):
        if type(shifted) is not bool:
            raise ValueError("shifted must be bool")
        kwargs.setdefault("serving", "anchor_tail")
        kwargs.setdefault("budget", 1)
        kwargs.setdefault("gates", "budget_only")
        if (kwargs["serving"], kwargs["budget"], kwargs["gates"]) != ("anchor_tail", 1, "budget_only"):
            raise ValueError("ShiftedAWM requires selected A")
        super().__init__(**kwargs)
        self._r8_shifted, self._r8_diag = shifted, {}

    def reset(self, episode):
        super().reset(episode)
        self._r8_diag = {}

    def query(self, q):
        result = super().query(q)
        self._r8_diag = dict(src="cache", anchor_step=int(q.step),
                            commit_controls=5 if self._r8_shifted and int(q.step) == 0 else 10)
        return result

    def blind_step(self, bq):
        if self._r8_shifted and int(bq.step) == 1 and self._anchor is not None and self._anchor["step"] == 0:
            result = self.lifecycle_reason(bq) or LookReason(1, "shifted_first_anchor")
            self.last_blind_extras = {"look_reason": float(result.code)}
        else:
            result = super().blind_step(bq)
        self._r8_diag = dict(src="look" if isinstance(result, LookReason) else "cache_tail",
                            look_reason=result.name if isinstance(result, LookReason) else None)
        return result


class WristEveryLook(Diagnostics, StageWrist):
    """R7's wrist metric/encoder path with the stage gate removed."""
    family = "r8_wrist_every_look"

    def __init__(self, every_controls=10, **kwargs):
        if type(every_controls) is not int or every_controls not in (5, 10):
            raise ValueError("every_controls must be 5 or 10")
        super().__init__(**kwargs)
        self._r8_every_controls, self._r8_diag = every_controls, {}
        self.name = "R8_W{}__".format(every_controls) + self.name

    def finish_cadence(self):
        self.base.budget = self._r8_every_controls // 5 - 1
        if self.enabled:
            self.wrist.budget = self.base.budget
        self.reference_blocks = self.base.budget + 1

    def fit(self, lib, ctx):
        super().fit(lib, ctx)
        self.finish_cadence()

    def reset(self, episode):
        super().reset(episode)
        self._r8_diag = {}

    def _plan(self, current, elapsed, target, command_action=None):
        # StageWrist.blind_step still forces full on lifecycle/forced LOOKs.
        self.next_camera_mode = "wrist_only" if self._anchor is not None else "full"
        self._plan_extras = dict(os_sw_next_camera=float(self.next_camera_mode == "wrist_only"), os_sw_reason=0.)

    def query(self, q):
        chosen = self._camera_mode
        result = super().query(q)
        self._r8_diag = dict(src="cache", camera_mode=chosen, next_camera_mode=self.next_camera_mode,
                            cadence_controls=self._r8_every_controls, stage_gate_enforced=False)
        return result

    def blind_step(self, bq):
        result = super().blind_step(bq)
        self._r8_diag = dict(src="look" if isinstance(result, LookReason) else "cache_tail",
            next_camera_mode=self.next_camera_mode, cadence_controls=self._r8_every_controls,
            look_reason=result.name if isinstance(result, LookReason) else None, stage_gate_enforced=False)
        return result


class AnchorCalls(Diagnostics, api.Method):
    """A plus an independent fresh-anchor coin; CU's lifecycle-only policy tail."""
    tier = "T1"
    family = "r8_anchor_calls"

    def __init__(self, p=.25, base_kwargs=None, base_fit="", random_seed=26093005, coin_domain="R8/call"):
        _random_config(random_seed, coin_domain)
        if isinstance(p, bool) or not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("p must be finite in [0,1]")
        self.p, self.base_kwargs, self.base_fit = float(p), dict(base_kwargs or {}), str(base_fit)
        self.random_seed, self.coin_domain = random_seed, coin_domain
        self.base, self._episode_identity, self._r8_diag = None, None, {}
        self._policy_gate_anchor = None
        self.name = "R8_IP_p{:g}".format(self.p)

    def fit(self, lib, ctx):
        if self.base_fit:
            with open(self.base_fit, "rb") as f:
                blob = FitUnpickler(f).load()
            if (blob["spec"] != BASE or blob["kwargs"] != self.base_kwargs or blob["cell"] != ctx.cell):
                raise ValueError("R8 call base artifact spec/kwargs/cell mismatch")
            self.base = blob["method"]
        else:
            self.base = BlindAWM(**self.base_kwargs)
            self.base.prof = self.prof
            self.base.fit(lib, ctx)
        if (self.base.serving, self.base.budget, self.base.gates) != ("anchor_tail", 1, "budget_only"):
            raise api.ContractError("R8 anchor calls require selected A")
        self.base.prof = self.prof
        self.fit_info = dict(base_fit=self.base_fit, fixed_p=self.p, stall=False, cooldown=False,
                             policy_commit_controls=10, policy_tail_blocks=1)

    def reset(self, episode):
        if self.base is None:
            raise api.ContractError("fit before reset")
        self.base.reset(episode)
        self._episode_identity = (str(episode.uid), int(episode.task_id), int(episode.init))
        self._policy_gate_anchor, self._r8_diag = None, {}

    def _assignment(self, q):
        u = coin(self.random_seed, q.task_id, q.episode.init, q.step, self.coin_domain)
        return self.p, u, dict(coin_domain=self.coin_domain, random_seed=self.random_seed)

    def query(self, q):
        identity = (str(q.episode.uid), int(q.task_id), int(q.episode.init))
        if identity != self._episode_identity:
            self.reset(q.episode)
        result = self.base.query(q)
        p, u, details = self._assignment(q)
        call = u < p
        a = self.base._anchor
        self._policy_gate_anchor = dict(step=a["step"], task=a["task"], episode=a["episode"],
            rows=a["rows"].copy(), weights=a["weights"].copy()) if call else None
        self._r8_diag = dict(src="policy" if call else "cache", fresh=True, eligible=True,
            p_nominal=self.p, p_effective=p, coin=u, treatment=bool(call), stall_state="disabled",
            cooldown=False, budget_state="fixed_probability", anchor_step=int(q.step), **details)
        # The p=0 identity path leaves A's extras unchanged byte for byte.
        if self.p != 0 or getattr(self, "diagnostic_oracle", False):
            result.extras = dict(result.extras or {}, os_force_miss=float(call), os_reason=float(82 if call else 0),
                os_c_fresh=1., os_c_p=p, os_c_nominal_p=self.p, os_c_coin=u, os_c_call=float(call),
                os_c_stall_call=0., os_c_cooldown=0., os_c_extra_look=0., os_c_commit_controls=10.)
        return result

    def invalidate_anchor(self):
        # Keep the CU gate provenance; the plugin clears A after a policy MISS.
        self.base.invalidate_anchor()

    @property
    def last_blind_extras(self):
        return self.base.last_blind_extras

    def blind_step(self, bq):
        result = self.base.blind_step(bq)
        self._r8_diag = dict(self._r8_diag, fresh=False, eligible=False, p_effective=0., treatment=False,
                            src="look" if isinstance(result, LookReason) else "cache_tail",
                            look_reason=result.name if isinstance(result, LookReason) else None)
        return result

    def policy_tail_step(self, bq):
        facade = SimpleNamespace(policy_tail_gate="lifecycle", monitor="off", base=self.base,
                                 _policy_gate_anchor=self._policy_gate_anchor)
        try:
            result = CommitJudge.policy_tail_step(facade, bq)
        finally:
            self._policy_gate_anchor = facade._policy_gate_anchor
        self._r8_diag = dict(self._r8_diag, src="policy_tail" if isinstance(result, BlindResult) else "look",
                            fresh=False, eligible=False, p_effective=0., treatment=False,
                            look_reason=result.name if isinstance(result, LookReason) else None)
        return result

    def bytes_per_entry(self):
        return self.base.bytes_per_entry() if self.base is not None else 0.


class IdentificationProbe(AnchorCalls):
    def __init__(self, **kwargs):
        kwargs.setdefault("p", .25)
        super().__init__(**kwargs)


class PolicyEveryTen(AnchorCalls):
    """Fresh policy every ten controls; the intervening decision serves its tail."""
    def __init__(self, **kwargs):
        if "p" in kwargs and kwargs["p"] != 1:
            raise ValueError("PolicyEveryTen requires p=1")
        kwargs["p"] = 1.
        super().__init__(**kwargs)
        self.name = "R8_P10_D5"


class OracleGraspCalls(AnchorCalls):
    """Privileged diagnostic upper bound; payload arrives through q.oracle.

    Canonical payload: {status: available, objects: [{object_id, in_window,
    distance_m, lifted, satisfied}, ...]}. A single-object payload with the same
    fields is accepted. Tight calls require distance_m <= .05 and cap two per
    object, for the entire episode (no regrasp/reset of the object allowance).
    """
    family = "r8_diagnostic_oracle"
    diagnostic_oracle = True

    def __init__(self, tight=False, **kwargs):
        if type(tight) is not bool:
            raise ValueError("tight must be bool")
        kwargs["p"] = 0.
        super().__init__(**kwargs)
        self.tight = tight
        self._oracle, self._oracle_counts = None, {}
        self.name = "R8_ORACLE_5b" if tight else "R8_ORACLE_5a"

    def reset(self, episode):
        # set_oracle may precede the plugin's first reset on the same request.
        pending = self._oracle
        super().reset(episode)
        self._oracle, self._oracle_counts = pending, {}

    def set_oracle(self, payload):
        self._oracle = copy.deepcopy(payload)

    def _assignment(self, q):
        # S1's --os-oracle facade is request-local and always supplies this
        # property. set_oracle is an alternative for standalone replay callers.
        payload = getattr(q, "oracle", self._oracle)
        self._oracle = None
        details = dict(oracle=True, diagnostic=True, oracle_tight=self.tight,
                       oracle_status="available", coin_domain="deterministic_oracle", oracle_objects=[])
        if not isinstance(payload, dict):
            details.update(oracle_status="error", override="missing_or_unavailable_oracle")
            return 0., 0., details
        status = payload.get("status", "available")
        details["oracle_status"] = status
        if status not in ("available", "partial"):
            details.update(override="missing_or_unavailable_oracle")
            return 0., 0., details
        objects = payload.get("objects", [payload])
        if not isinstance(objects, list):
            details.update(oracle_status="error", override="malformed_oracle")
            return 0., 0., details
        candidates = []
        for obj in objects:
            if not isinstance(obj, dict):
                details["oracle_objects"].append(dict(object_id=None, status="error",
                    reason="malformed oracle object", admissible=False))
                continue
            oid = obj.get("object_id", obj.get("goal_object"))
            oid = str(oid) if oid is not None else None
            count = self._oracle_counts.get(oid, 0)
            # S2 reports unresolved objects alongside usable truth. Legacy
            # available payloads omit object status, so retain that spelling.
            record = dict(object_id=oid, status=obj.get("status", "available"),
                          reason=obj.get("reason", ""), calls=count, admissible=False)
            details["oracle_objects"].append(record)
            if record["status"] != "available":
                continue
            distance = obj.get("distance_m", obj.get("distance"))
            try:
                distance = float(distance)
            except (ValueError, TypeError):
                record.update(status="error", reason="missing or invalid distance")
                continue
            if oid is None or not math.isfinite(distance) or distance < 0:
                record.update(status="error", reason="missing object ID or invalid distance")
                continue
            inside = obj.get("in_window", obj.get("grasp_window", False)) is True
            admissible = (inside and obj.get("predicate_known", True) and
                          not obj.get("lifted", False) and not obj.get("satisfied", False))
            if self.tight:
                admissible = admissible and distance <= .05 and count < 2
            record.update(distance_m=distance, in_window=inside,
                predicate_known=bool(obj.get("predicate_known", True)),
                lifted=bool(obj.get("lifted", False)), satisfied=bool(obj.get("satisfied", False)),
                admissible=bool(admissible))
            if admissible:
                candidates.append((distance, oid))
        if candidates:
            _distance, oid = min(candidates)
            self._oracle_counts[oid] = self._oracle_counts.get(oid, 0) + 1
            details.update(oracle_selected_object=oid, override="oracle_window")
            return 1., 0., details
        details["override"] = "oracle_outside_window_or_cap"
        return 0., 0., details
