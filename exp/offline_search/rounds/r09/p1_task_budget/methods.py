"""Task switch over unchanged, fitted R8 A/CU/P10 controllers.

rho=0 bypasses C completely: its calibrated stall floor and ambiguous LOOKs
must not leak into pure-cache tasks. Nonzero rates select an independently
calibrated CU artifact; P10 is the existing ten-control policy controller.
No decisions, result extras, random coins, or action arrays are rewritten.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from exp.offline_search.closed_loop.blind import LookReason
from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import sha


class TaskBudgetController(api.Method):
    tier = "T1"
    family = "r9_p1_task_budget"
    uses_gt = False
    uses_nonlibrary_action = False

    def __init__(self, task_rates, sources, policy_tasks=()):
        if not isinstance(task_rates, dict) or not task_rates:
            raise ValueError("task_rates must explicitly cover every task")
        self.task_rates = {}
        for key, value in task_rates.items():
            task = int(key)
            if str(task) != str(key) or task < 0 or task in self.task_rates:
                raise ValueError("task IDs must be unique nonnegative integers")
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError("task rate must be finite in [0,1]")
            self.task_rates[task] = float(value)
        if any(type(t) is not int for t in policy_tasks) or len(set(policy_tasks)) != len(policy_tasks):
            raise ValueError("policy_tasks must contain distinct integer IDs")
        self.policy_tasks = tuple(sorted(policy_tasks))
        if not set(self.policy_tasks) <= self.task_rates.keys():
            raise ValueError("policy task missing from task_rates")
        if any(self.task_rates[t] != 1 for t in self.policy_tasks):
            raise ValueError("P10 tasks must be explicitly assigned rate 1")
        self.sources = dict(sources)
        self.controllers = {}
        self.active = None
        self._episode_identity = None
        self.name = "R9_P1_TASK_BUDGET"

    def branch(self, task):
        task = int(task)
        if task not in self.task_rates:
            raise api.ContractError(f"unregistered task {task}; no implicit fallback")
        if task in self.policy_tasks:
            return "P10"
        rate = self.task_rates[task]
        return "A" if rate == 0 else "CU:" + format(rate, ".12g")

    def fit(self, lib, ctx):
        """Load verified fits only; never refit or resolve budgets from test data."""
        needed = {self.branch(t) for t in self.task_rates}
        if set(self.sources) != needed:
            raise ValueError("sources must exactly match the selected branches")
        for branch in sorted(needed):
            source = self.sources[branch]
            path = Path(source["artifact"])
            if sha(path) != source["sha256"]:
                raise ValueError("source SHA mismatch: " + str(path))
            with path.open("rb") as f:
                blob = pickle.load(f)
            want = dict(spec=source["spec"], kwargs=source["kwargs"], cell=ctx.cell)
            if any(blob.get(k) != v for k, v in want.items()):
                raise ValueError("source metadata mismatch: " + branch)
            method, _ = clone_method(blob["method"], strict=True)
            if branch == "A":
                from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
                if type(method) is not BlindAWM or (method.serving, method.budget, method.gates) != (
                        "anchor_tail", 1, "budget_only"):
                    raise ValueError("A must be the selected R8 pure-cache controller")
            elif branch == "P10":
                from exp.offline_search.rounds.r08.methods.methods import PolicyEveryTen
                if type(method) is not PolicyEveryTen or method.p != 1:
                    raise ValueError("P10 must be the R8 ten-control policy controller")
            else:
                from exp.offline_search.rounds.r07.c3_calls.methods import CallController
                if (type(method) is not CallController or method.tilt or method.placement != "uniform"
                        or method.cooldown_scope != "stall" or method.stall_model is None
                        or method.rho != float(branch.split(":")[1])):
                    raise ValueError("CU must be unchanged uniform calls with calibrated stall")
            if blob.get("registered"):
                raise ValueError("these R8 fits must use the existing current library")
            self.controllers[branch] = method
        for branch, controller in self.controllers.items():
            base = controller if branch == "A" else controller.base
            if set(self.task_rates) != set(base.tasks):
                raise ValueError("task table does not exactly cover fitted library tasks")
        self.fit_info = dict(source_sha256={k: v["sha256"] for k, v in self.sources.items()},
                             task_rates=self.task_rates, policy_tasks=self.policy_tasks,
                             rule="dispatch unchanged R8 fitted controller at reset")

    def reset(self, episode):
        branch = self.branch(episode.task_id)
        if branch not in self.controllers:
            raise api.ContractError("fit before reset")
        self.active = self.controllers[branch]
        self.active.reset(episode)
        self._episode_identity = (str(episode.uid), int(episode.task_id), int(episode.init))

    def _activate(self, q):
        identity = (str(q.episode.uid), int(q.task_id), int(q.episode.init))
        if int(q.task_id) != int(q.episode.task_id):
            raise api.ContractError("query/episode task mismatch")
        if identity != self._episode_identity:
            self.reset(q.episode)
        return self.active

    def query(self, q):
        return self._activate(q).query(q)

    def blind_step(self, q):
        return self._activate(q).blind_step(q)

    def policy_tail_step(self, q):
        controller = self._activate(q)
        hook = getattr(controller, "policy_tail_step", None)
        return hook(q) if hook else LookReason(8, "A has no policy tail")

    def invalidate_anchor(self):
        if self.active is not None:
            self.active.invalidate_anchor()

    @property
    def last_blind_extras(self):
        return self.active.last_blind_extras if self.active is not None else {}

    def bytes_per_entry(self):
        return sum(c.bytes_per_entry() for c in self.controllers.values())
