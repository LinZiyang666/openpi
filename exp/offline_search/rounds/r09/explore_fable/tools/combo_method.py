"""Residual-corrected cache plus per-task policy calls (R9 fable x astra combination).

``ResidualTaskCalls`` serves astra's half-residual cache (frozen ``ResidualCache``
artifact: cache retrieval + motion-only ridge correction, gripper and 10-control
commitment unchanged) on every task, and adds an independent per-anchor call coin
whose probability depends on the task: ``task_p[t]`` (0 = never call, 1 = pure
policy with 10-control commitment). The call machinery (coin, forced MISS, policy
tail lifecycle) is the R8 ``AnchorCalls`` one; only the base and the per-task
probability differ. Deployed without touching any existing serving file.
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r08.methods.methods import AnchorCalls, coin

RESIDUAL_SPEC = "exp.offline_search.rounds.r09.explore_astra.methods:ResidualCache"


class ResidualTaskCalls(AnchorCalls):
    """Half-residual cache everywhere + per-task anchor-call probability."""
    family = "r9_fable_residual_task_calls"

    def __init__(self, task_p, residual_fit, residual_kwargs, random_seed=26100201, coin_domain="R9F/combo"):
        p = np.asarray(task_p, float)
        if p.shape != (10,) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
            raise ValueError("task_p must be ten probabilities in [0, 1]")
        self.task_p = [float(v) for v in p]
        self.residual_fit, self.residual_kwargs = str(residual_fit), dict(residual_kwargs)
        self.res = None
        super().__init__(p=float(p.max()), base_kwargs=dict(residual_kwargs.get("base_kwargs", {})), base_fit="",
                         random_seed=random_seed, coin_domain=coin_domain)
        self.name = "R9F_combo_" + "".join("1" if v >= 1 else ("h" if v > 0 else "0") for v in self.task_p)

    def fit(self, lib, ctx):
        with open(self.residual_fit, "rb") as f:
            blob = FitUnpickler(f).load()
        if blob["spec"] != RESIDUAL_SPEC or blob["kwargs"] != self.residual_kwargs or blob["cell"] != ctx.cell:
            raise ValueError("residual artifact spec/kwargs/cell mismatch")
        self.res = blob["method"]
        self.res.prof = self.prof
        self.base = self.res.base                      # inner frozen BlindAWM: anchors, blind tail, policy-tail facade
        if (self.base.serving, self.base.budget, self.base.gates) != ("anchor_tail", 1, "budget_only"):
            raise api.ContractError("combo requires the frozen ten-control cache")
        self.base.prof = self.prof
        self.fit_info = dict(residual_fit=self.residual_fit, blend=self.res.blend, task_p=self.task_p, stall=False,
                             cooldown=False, policy_commit_controls=10, policy_tail_blocks=1)

    def reset(self, episode):
        if self.res is None:
            raise api.ContractError("fit before reset")
        self.res.reset(episode)
        self._episode_identity = (str(episode.uid), int(episode.task_id), int(episode.init))
        self._policy_gate_anchor, self._r8_diag = None, {}

    def _assignment(self, q):
        p = self.task_p[int(q.task_id)]
        u = coin(self.random_seed, q.task_id, q.episode.init, q.step, self.coin_domain)
        return p, u, dict(coin_domain=self.coin_domain, random_seed=self.random_seed, task_p=p)

    def query(self, q):
        identity = (str(q.episode.uid), int(q.task_id), int(q.episode.init))
        if identity != self._episode_identity:
            self.reset(q.episode)
        result = self.res.query(q)                     # corrected cache chunk; anchor already updated by ResidualCache
        p, u, details = self._assignment(q)
        call = u < p
        a = self.base._anchor
        self._policy_gate_anchor = dict(step=a["step"], task=a["task"], episode=a["episode"],
                                        rows=a["rows"].copy(), weights=a["weights"].copy()) if call else None
        self._r8_diag = dict(src="policy" if call else "cache", fresh=True, eligible=True, p_nominal=p, p_effective=p,
                             coin=u, treatment=bool(call), stall_state="disabled", cooldown=False,
                             budget_state="per_task_fixed_probability", anchor_step=int(q.step), **details)
        result.extras = dict(result.extras or {}, os_force_miss=float(call), os_reason=float(82 if call else 0),
                             os_c_fresh=1., os_c_p=p, os_c_nominal_p=p, os_c_coin=u, os_c_call=float(call),
                             os_c_stall_call=0., os_c_cooldown=0., os_c_extra_look=0., os_c_commit_controls=10.)
        return result

    def blind_step(self, bq):
        return super().blind_step(bq)                  # inner base tail (already the corrected anchor)

    def bytes_per_entry(self):
        return self.res.bytes_per_entry() if self.res is not None else 0.
