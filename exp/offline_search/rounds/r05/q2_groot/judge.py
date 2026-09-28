"""GR00T CycleTail: periodic real-vision anchors and bounded same-chunk tails.

No MixedJudge, stuck/terminal guards, or fabricated visual keys. Normalized
GR00T gripper values below zero mean closed; its wire adapter flips openness
once to LIBERO's positive-close convention. Actions are never sign-modified here.
"""
from __future__ import annotations

import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult, LookReason, policy_tail_chunk
from exp.offline_search.harness import api
from exp.offline_search.rounds.r04.k1_blind.blind_awm import BlindAWM
from exp.offline_search.rounds.r04.k4_eval.seeded_inference import seed_process


class CycleTail(BlindAWM):
    family = "r5_cycle_tail"

    def __init__(self, cycle_k=4, tail_blocks=1, **kwargs):
        if isinstance(cycle_k, bool) or int(cycle_k) != cycle_k or cycle_k < 1:
            raise ValueError("cycle_k must be a positive integer")
        if isinstance(tail_blocks, bool) or tail_blocks not in (1, 2):
            raise ValueError("tail_blocks must be 1 or 2")
        for key, value in dict(serving="anchor_tail", budget=tail_blocks, gates="budget_only").items():
            if key in kwargs and kwargs[key] != value:
                raise ValueError(f"CycleTail requires {key}={value!r}")
            kwargs[key] = value
        super().__init__(**kwargs)
        self.cycle_k, self.tail_blocks = int(cycle_k), int(tail_blocks)
        self.name = f"CycleTail_k{cycle_k}_T{tail_blocks}__{self.name}"
        self._cycle_episode = None
        self._anchor_count = 0
        self._policy_proposal = None

    def fit(self, lib, ctx):
        if lib.model != "groot":
            raise api.SkipCell("CycleTail requires GR00T's H=16 and gripper convention")
        super().fit(lib, ctx)

    def reset(self, episode):
        super().reset(episode)
        self._cycle_episode = episode
        self._anchor_count = 0
        self._policy_proposal = None

    def query(self, q):
        if q.episode is not self._cycle_episode:
            self.reset(q.episode)
        seed_process()
        result = super().query(q)
        # Keep only provenance for the policy hook. invalidate_anchor() still
        # invalidates the cache source on MISS; it cannot become a policy source.
        self._policy_proposal = dict(self._anchor, episode_object=q.episode)
        ordinal = self._anchor_count
        self._anchor_count += 1
        result.extras.update(os_force_miss=float(ordinal % self.cycle_k == 0),
                             os_reason=51., cycle_anchor=float(ordinal),
                             source_anchor_step=float(q.step), tail_offset=0.,
                             tail_valid_rows=float(self.H))
        if q.step:
            result.extras["previous_gripper_closed"] = float(q.prev_a_exec[4, 6] < 0)
        return result

    def blind_step(self, bq):
        result = super().blind_step(bq)
        if isinstance(result, BlindResult):
            offset = 5 * (bq.step - self._anchor["step"])
            result.extras.update(source_anchor_step=float(self._anchor["step"]),
                                 tail_offset=float(offset), tail_valid_rows=float(self.H - offset))
        return result

    def policy_tail_step(self, bq):
        a = self._policy_proposal
        if a is None:
            return LookReason(6, "policy_tail_no_anchor")
        step = a["step"]
        h = bq.step - step
        if (bq.episode is not a["episode_object"] or bq.task_id != a["task"]
                or not 1 <= h <= self.tail_blocks or bq.blind_age != h - 1
                or len(bq.hist_has_vision) != bq.step or len(bq.hist_hit) != bq.step
                or len(bq.hist_a_exec) != bq.step or not bq.hist_has_vision[step]
                or bq.hist_hit[step] != 0 or np.any(bq.hist_has_vision[step + 1:])
                or np.any(bq.hist_hit[step + 1:] != 1)
                or not np.isfinite(bq.rs).all() or not np.isfinite(bq.raw_state).all()):
            return LookReason(6, "policy_tail_lifecycle")
        try:
            action = policy_tail_chunk(bq.hist_a_exec[step], 5 * h)
        except ValueError:
            return LookReason(6, "policy_tail_invalid_chunk")
        return BlindResult(action, a["rows"].copy(), a["weights"].copy(), self.cand_name,
                           dict(policy_tail=1., policy_anchor_step=float(step),
                                source_anchor_step=float(step), tail_offset=float(5 * h),
                                tail_valid_rows=float(self.H - 5 * h)))
