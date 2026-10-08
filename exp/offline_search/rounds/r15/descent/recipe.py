"""Opt-in query-init episode exclusion; frozen R14 task descent arithmetic."""
from __future__ import annotations
import copy
import numpy as np
from exp.offline_search.harness import api, dims
from exp.offline_search.rounds.r02.g3_recovery.g3_core import Tables, hash_chunk
from exp.offline_search.rounds.r13.knob.recipe import R13Knob
from exp.offline_search.rounds.r14.descent.recipe import TaskDescentKnob as FrozenTaskDescentKnob, LEVELS, validate_levels
from exp.offline_search.rounds.r14.knob.recipe import reference_diagnostics

ROW_FIELDS = ('rows', 'Z', 'z2', 'HD', 'h2', 'V0', 'V1', 'RS', 'rs2', 'Z0', 'n20')


class EpisodeExcludedTables(Tables):
    """Guard's exact action lookup excludes the same episode as retrieval."""
    def find_chunk(self, a7, task):
        excluded = getattr(self, 'excluded_episode', -1)
        if excluded < 0:
            return super().find_chunk(a7, task)
        if self.hsorted is None:
            return -1  # Deployed guard uses anchor_mode=self, with no hash index.
        h = np.uint64(hash_chunk(a7))
        i = int(np.searchsorted(self.hsorted, h))
        best = -1
        while i < self.L and self.hsorted[i] == h:
            r = int(self.horder[i])
            if (self.ep[r] != excluded and self.task[r] == task
                    and np.array_equal(dims.valid_action_chunk(self.act[r]), a7)):
                best = r if best < 0 else min(best, r)
            i += 1
        return best


class LeaveOwnEpisodeMixin:
    def enable_leave_own_episode_out(self, identities):
        self.leave_own_episode_out = True
        self.episode_by_init = {tuple(k): int(v) for k, v in identities.items()}
        self._full_tasks = dict(self.inner.base.tasks)
        self._excluded_episode = -1
        self.inner.C.__class__ = EpisodeExcludedTables
        self.inner.C.excluded_episode = -1

    def validate_leave_own_episode_out(self, subset):
        if not getattr(self, 'leave_own_episode_out', False):
            raise api.ContractError('leave-own-episode-out is disabled')
        ids = subset['episode_ids_by_task']
        inits = subset['excluded_inits_by_task']
        for t in range(10):
            actual = {i: e for (task, i), e in self.episode_by_init.items() if task == t}
            if set(actual) != set(inits[str(t)]) or set(actual.values()) != set(ids[str(t)]):
                raise api.ContractError('frozen init identity map differs from actual served subset')
        if not np.array_equal(self.inner.base.lib_ep, self.inner.C.ep):
            raise api.ContractError('retrieval and guard episode indices differ')
        if not isinstance(self.inner.C, EpisodeExcludedTables):
            raise api.ContractError('guard exact-match lookup is not excluded')
        if self.inner.recover or self.inner.blend:
            raise api.ContractError('episode exclusion supports the deployed non-recovery recipe only')

    def reset(self, episode):
        if getattr(self, 'leave_own_episode_out', False):
            task, init = int(episode.task_id), int(episode.init)
            if task not in range(10) or init < 0:
                raise api.ContractError('leave-own-out needs original task/init metadata')
            uid = str(episode.uid)
            if not uid.startswith(('A:', 'B:', 'r15d_')):
                raise api.ContractError('leave-own-out needs explicit A/B init pool in task_uid')
            # A and B index numbers refer to different physical states.
            own = self.episode_by_init.get((task, init), -1) if uid.startswith(('B:', 'r15d_')) else -1
            base = self.inner.base
            base.tasks = dict(self._full_tasks)
            if own >= 0:
                original = self._full_tasks[task]
                keep = base.lib_ep[original.rows] != own
                if keep.sum() < 16:
                    raise api.ContractError('insufficient non-own retrieval donors')
                filtered = copy.copy(original)
                for name in ROW_FIELDS:
                    value = getattr(original, name)
                    if value is not None:
                        value = np.ascontiguousarray(value[keep])
                        value.flags.writeable = False
                    setattr(filtered, name, value)
                base.tasks[task] = filtered
            self._excluded_episode = own
            self.inner.C.excluded_episode = own
        return super().reset(episode)

    def _audit_result(self, result):
        rows = getattr(result, 'topk', getattr(result, 'rows', None))
        own = getattr(self, '_excluded_episode', -1)
        if rows is not None and own >= 0:
            local = np.searchsorted(self.row_subset, rows)
            if (np.any(local >= len(self.row_subset))
                    or not np.array_equal(self.row_subset[local], rows)
                    or np.any(self.inner.base.lib_ep[local] == own)):
                raise api.ContractError('own episode appeared in served donor rows')
        return result

    def query(self, q):
        result = self._audit_result(super().query(q))
        if getattr(self, 'leave_own_episode_out', False):
            audit = (getattr(self, 'r14_diagnostics', None) or {}) if self.knob_method == 'task_periodic' else reference_diagnostics(self, q, result)
            anchor = self.inner.base._anchor
            audit.update(leave_own_episode_out=True, excluded_episode=self._excluded_episode,
                         query_init=int(q.episode.init), query_pool='A' if str(q.episode.uid).startswith('A:') else 'B',
                         parent_donor_rows=self.row_subset[anchor['rows']].tolist(),
                         donor_episode_ids=self.inner.base.lib_ep[anchor['rows']].astype(int).tolist(),
                         candidate_episode_ids=np.unique(self.inner.base.lib_ep[
                             self.inner.base.tasks[int(q.task_id)].rows]).astype(int).tolist())
            self.r14_diagnostics = audit
        return result

    def blind_step(self, q):
        return self._audit_result(super().blind_step(q))

    def policy_tail_step(self, q):
        return self._audit_result(super().policy_tail_step(q))


class R15Knob(LeaveOwnEpisodeMixin, R13Knob):
    family = 'r15_leave_own_episode_out'
    def __init__(self, *args, leave_own_episode_out=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.leave_own_episode_out = bool(leave_own_episode_out)


class TaskDescentKnob(LeaveOwnEpisodeMixin, FrozenTaskDescentKnob):
    family = 'r15_task_descent'
    def __init__(self, *args, leave_own_episode_out=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.leave_own_episode_out = bool(leave_own_episode_out)
