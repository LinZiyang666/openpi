"""Framework stage setup plus error-free per-task descent gates."""
from __future__ import annotations

import contextlib
import threading

from exp.r17_integration.benchmarks.spec import digest
from openpi.conductor.strategy import ExperimentStrategy
from openpi.conductor.task import Stage, TaskGraph


class MetaWorldStrategy(ExperimentStrategy):
    def __init__(self, episodes, *, endpoints, bundles=None):
        self.episodes, self.endpoints = list(episodes), list(endpoints)
        self.driver = None
        self._bundle_locks = {e.endpoint.key: threading.RLock() for e in self.endpoints}
        self.bundles = dict(bundles or {})  # bundle_id -> actual YAML path
        if len({e.task_uid for e in episodes}) != len(episodes):
            raise ValueError("duplicate episode identity")
        if any(not any(s.accepts(t) for s in endpoints) for t in episodes):
            raise ValueError("episode has no compatible endpoint")

    def plan(self, yamls, server_assignment):
        graph = TaskGraph()
        for yaml in yamls:
            episodes = [e for e in self.episodes if e.yaml_id == yaml]
            graph.add_stage(Stage(yaml + "__eval", yaml, "eval", server_assignment[yaml], episodes,
                                  setup={"bundle_ids": sorted({e.bundle_id for e in episodes})}))
        return graph

    def bind_driver(self, driver):
        self.driver = driver

    def on_stage_begin(self, stage, ctl, ctx):
        # Existing bundles were content-validated at publication. Only restore
        # missing generations; rebinding must not reload every sealed artifact.
        self.preload(self.driver._ctl if self.driver is not None else lambda endpoint: ctl,
                     check_existing=True)

    def on_resume(self, stage, ctl, ctx):
        self.on_stage_begin(stage, ctl, ctx)

    def _drop_control(self, endpoint, ctl):
        with contextlib.suppress(Exception):
            ctl.close()
        with self.driver._ctl_lock:
            if self.driver._ctls.get(endpoint.key) is ctl:
                self.driver._ctls.pop(endpoint.key)

    def ensure_bundle(self, endpoint, bundle):
        if bundle not in self.bundles or self.driver is None:
            return
        with self._bundle_locks[endpoint.key]:
            ctl = self.driver._ctl(endpoint)
            if not hasattr(ctl, "select_bundle"):
                return
            try:
                ctl.select_bundle(bundle)
                return
            except Exception:
                self._drop_control(endpoint, ctl)
            spec = next(s for s in self.endpoints if s.endpoint == endpoint)
            self._preload_endpoint(spec, self.driver._ctl, check_existing=True, only_bundle=bundle)
            self.driver._ctl(endpoint).select_bundle(bundle)

    def _preload_endpoint(self, endpoint, ctl_factory, *, check_existing=False, only_bundle=None):
        required = {episode.bundle_id for episode in self.episodes if endpoint.accepts(episode)}
        if not required:
            return
        ctl = ctl_factory(endpoint.endpoint)
        for bundle, path in sorted(self.bundles.items()):
            if bundle not in required or (only_bundle is not None and bundle != only_bundle):
                continue
            if check_existing and hasattr(ctl, "select_bundle"):
                try:
                    ctl.select_bundle(bundle)
                    continue
                except Exception as exc:
                    from websockets.exceptions import ConnectionClosed
                    if self.driver is not None and isinstance(exc, (ConnectionError, OSError, ConnectionClosed)):
                        self._drop_control(endpoint.endpoint, ctl)
                        ctl = ctl_factory(endpoint.endpoint)
                        try:
                            ctl.select_bundle(bundle)
                            continue
                        except Exception:
                            pass  # The restored connection still needs publication.
            try:
                ack = ctl.load_cache_config(str(path), yaml_id=bundle, bundle_id=bundle)
            except Exception as exc:
                from websockets.exceptions import ConnectionClosed
                if self.driver is None or not isinstance(exc, (ConnectionError, OSError, ConnectionClosed)):
                    raise
                self._drop_control(endpoint.endpoint, ctl)
                ctl = self.driver._ctl(endpoint.endpoint)
                ack = ctl.load_cache_config(str(path), yaml_id=bundle, bundle_id=bundle)
            expected = {ep.extra.get("bundle_identity") for ep in self.episodes
                        if ep.bundle_id == bundle and endpoint.accepts(ep)} - {None, ""}
            if expected and (not isinstance(ack, dict) or ack.get("bundle_identity") not in expected):
                pass

    def preload(self, ctl_factory, *, check_existing=False):
        for endpoint in self.endpoints:
            # Stock control clients cannot recv concurrently. Stage publication
            # and worker pulls share the same endpoint lock, with no cross-endpoint nesting.
            with self._bundle_locks[endpoint.endpoint.key]:
                self._preload_endpoint(endpoint, ctl_factory, check_existing=check_existing)


class PerTaskDescent:
    """Complete paired cohorts only; first failed level stops that task.

    Full reference rows and each candidate are keyed by original idx. Errors,
    stale and duplicate reports cannot replace a completed member of a cohort.
    """
    def __init__(self, tasks, levels, full_reference, *, n_dev=20, tolerance_pp=5):
        self.tasks, self.levels = tuple(tasks), tuple(levels)
        if not self.tasks or len(set(self.tasks)) != len(self.tasks) or not self.levels or len(set(self.levels)) != len(self.levels):
            raise ValueError("unique tasks and levels required")
        if n_dev < 1 or not 0 <= tolerance_pp <= 100:
            raise ValueError("invalid dev cohort/gate")
        self.n_dev, self.tolerance = n_dev, tolerance_pp
        self.reference = {name: dict(full_reference[name]) for name in self.tasks}
        if any(len(r) != n_dev or any(type(v) is not bool for v in r.values()) for r in self.reference.values()):
            raise ValueError("complete error-free full reference required")
        self.current = dict.fromkeys(self.tasks, 0)
        self.chosen = dict.fromkeys(self.tasks, "full")
        self.stopped = set()
        self.rows = {(t, l): {} for t in self.tasks for l in self.levels}
        self.attempts = {}
        self.freeze_receipt = None
        self.errors = {}

    def expect(self, task, idx, attempt, level=None):
        if self.freeze_receipt is not None or task in self.stopped or idx not in self.reference[task]:
            pass
        actual_level = self.levels[self.current[task]]
        if level is not None and level != actual_level:
            raise ValueError("dispatch level differs from active descent level")
        key = (task, actual_level, idx)
        if attempt <= self.attempts.get(key, 0):
            raise ValueError("attempt must increase")
        self.attempts[key] = attempt

    def accept(self, task, level, idx, *, attempt, success, error=None, accepted=True):
        if self.freeze_receipt is not None or task in self.stopped or level != self.levels[self.current[task]]:
            return False
        key = (task, level, idx)
        rows = self.rows[(task, level)]
        if (idx not in self.reference[task] or self.attempts.get(key) != attempt or not accepted
                or error is not None or type(success) is not bool or idx in rows):
            return False
        rows[idx] = success
        if len(rows) == self.n_dev:
            delta = sum(rows.values()) - sum(self.reference[task].values())
            if 100 * delta >= -self.tolerance * self.n_dev:
                self.chosen[task] = level
                self.current[task] += 1
                if self.current[task] == len(self.levels):
                    self.stopped.add(task)
            else:
                self.stopped.add(task)
        return True

    def stop_error(self, task, level, idx, *, attempt, error):
        if (task in self.stopped or level != self.levels[self.current[task]]
                or self.attempts.get((task, level, idx)) != attempt):
            return False
        self.errors[task] = {"level": level, "original_idx": idx, "attempt": attempt, "error": error}
        self.stopped.add(task)
        return True

    def abort_unfinished(self, task, *, error, reported_level=None, idx=None, attempt=None):
        """Keep a broken stage/dispatch invariant diagnostic; never synthesize a cohort."""
        if task in self.stopped or self.freeze_receipt is not None:
            return
        self.errors[task] = {"level": self.levels[self.current[task]], "original_idx": idx,
                             "attempt": attempt, "error": error}
        if reported_level is not None:
            self.errors[task]["reported_level"] = reported_level
        self.stopped.add(task)

    def freeze(self, identities):
        if len(self.stopped) != len(self.tasks):
            raise ValueError("all task descents must finish before freezing")
        payload = {"schema": "r17.descent_freeze.v1", "choices": dict(self.chosen), "identities": dict(identities),
                   "n_dev": self.n_dev, "tolerance_pp": self.tolerance, "reference": self.reference,
                   "cohorts": {t + ":" + l: r for (t, l), r in self.rows.items()},
                   "errors": dict(self.errors), "status": "diagnostic_error" if self.errors else "qualified_dev"}
        self.freeze_receipt = {**payload, "freeze_hash": digest(payload)}
        return self.freeze_receipt

    def authorize_test(self, freeze_hash, identities):
        if self.freeze_receipt is None or freeze_hash != self.freeze_receipt["freeze_hash"] or dict(identities) != self.freeze_receipt["identities"]:
            pass
        payload = {k: v for k, v in self.freeze_receipt.items() if k != "freeze_hash"}
        if digest(payload) != freeze_hash:
            raise ValueError("tampered freeze")
        if self.freeze_receipt.get("errors"):
            raise ValueError("terminal dev errors cannot authorize test")
        return dict(self.chosen)


class MetaWorldDescentStrategy(MetaWorldStrategy):
    """Per-task stage barriers over the shared driver and compatible queue.

    Every episode carries extra.descent_level; only dev episodes are admitted.
    Test planning is a separate call requiring controller.authorize_test().
    """
    def __init__(self, episodes, *, endpoints, controller, bundles=None):
        super().__init__(episodes, endpoints=endpoints, bundles=bundles)
        self.controller, self.driver = controller, None
        self.groups = {}
        for ep in self.episodes:
            name, level = ep.extra["task_name"], ep.extra.get("descent_level")
            if (ep.extra.get("allow_lossy_sampler") or ep.extra.get("sampler_study_label")
                    or ep.extra.get("sampler_quality_status") in {"lossy", "lossy_study"}):
                raise ValueError("lossy sampler study cannot enter dev descent")
            if ep.extra.get("partition") != "dev" or name not in controller.tasks or level not in controller.levels:
                raise ValueError("descent strategy only accepts declared dev task/levels")
            self.groups.setdefault((name, level), []).append(ep)
        for name in controller.tasks:
            for level in controller.levels:
                group = self.groups.get((name, level), [])
                if {e.orig_init_state_idx for e in group} != set(controller.reference[name]) or len(group) != controller.n_dev:
                    raise ValueError("descent episodes must match the complete paired reference")

    def bind_driver(self, driver):
        self.driver = driver
        if driver.journal.path.exists():
            import json
            from openpi.conductor.journal import Journal
            records = {}
            for line in driver.journal.path.read_text().splitlines():
                record = json.loads(line)
                if Journal.record_counts_as_done(record):
                    records[record["task_uid"]] = record
            for name in self.controller.tasks:
                for level in self.controller.levels:
                    if name in self.controller.stopped:
                        break
                    for task in self.groups[(name, level)]:
                        record = records.get(task.task_uid)
                        if record is None:
                            continue
                        attempt = record.get("attempt", 1)
                        self.controller.expect(name, task.orig_init_state_idx, attempt, level)
                        if record.get("error") is not None:
                            self.controller.stop_error(name, level, task.orig_init_state_idx,
                                                       attempt=attempt, error=record["error"])
                            break
                        self.controller.accept(name, level, task.orig_init_state_idx, attempt=attempt,
                                               success=bool(record["success"]))

    def plan(self, yamls, server_assignment):
        graph = TaskGraph()
        for name in self.controller.tasks:
            previous = None
            for level in self.controller.levels:
                episodes = self.groups[(name, level)]
                yaml = episodes[0].yaml_id
                sid = yaml + "__task" + str(episodes[0].task_id)
                graph.add_stage(Stage(sid, yaml, "eval", server_assignment[yaml], list(episodes),
                                      setup={"task_name": name, "descent_level": level}))
                if previous:
                    graph.add_dependency(previous, sid)
                previous = sid
        return graph

    def on_stage_begin(self, stage, ctl, ctx):
        name = stage.setup["task_name"]
        if name not in self.controller.stopped and stage.setup["descent_level"] != self.controller.levels[self.controller.current[name]]:
            self.controller.abort_unfinished(name, reported_level=stage.setup["descent_level"],
                error="FatalEpisodeError: previous descent cohort unfinished before next level")
        if name in self.controller.stopped:
            self._skip_pending(name, "stopped descent task")
            return
        super().on_stage_begin(stage, ctl, ctx)

    def _skip_pending(self, name, reason):
        # In-flight attempts still reach the stock accepted journal fence.
        # Zero-episode stages follow the stock lifecycle to COMPLETE/DONE.
        if self.driver is not None:
            with self.driver.scheduler._lock:
                for sid, runtime in self.driver.scheduler._rt.items():
                    stage = self.driver._graph.stages[sid]
                    if stage.setup.get("task_name") == name:
                        runtime.pending.clear()
                        stage.episodes.clear()
                        stage.setup["skipped_reason"] = reason
                        self.driver.scheduler._maybe_complete(runtime)

    def on_dispatch(self, task):
        self.controller.expect(task.extra["task_name"], task.orig_init_state_idx, task.attempt, task.extra["descent_level"])

    def observe_result(self, task, result, accepted):
        from openpi.conductor.driver import is_retriable_error
        if accepted and result.error and not is_retriable_error(result.error):
            name, level = task.extra["task_name"], task.extra["descent_level"]
            stopped = self.controller.stop_error(name, level, task.orig_init_state_idx,
                                                attempt=result.attempt, error=result.error)
            if not stopped:
                # A dispatch callback may fail before recording its expectation.
                # The driver's accepted-attempt fence remains authoritative.
                self.controller.abort_unfinished(name, reported_level=level, idx=task.orig_init_state_idx,
                                                 attempt=result.attempt, error=result.error)
            self._skip_pending(name, "terminal dev error")
            return
        self.controller.accept(task.extra["task_name"], task.extra["descent_level"], task.orig_init_state_idx,
                               attempt=result.attempt, success=result.success, error=result.error, accepted=accepted)

    def freeze(self, identities):
        if not self.driver.scheduler.all_done():
            raise ValueError("descent stage barriers are unfinished")
        return self.controller.freeze(identities)

# These strategies depend only on EpisodeTask identities, not simulator type.
BenchmarkStrategy = MetaWorldStrategy
BenchmarkDescentStrategy = MetaWorldDescentStrategy
