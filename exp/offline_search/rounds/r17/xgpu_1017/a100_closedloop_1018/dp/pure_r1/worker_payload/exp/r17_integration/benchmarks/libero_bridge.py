"""R17 LIBERO contract and acknowledgement adapter around the original runner.

No simulator/model imports until an environment is explicitly requested.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import hashlib
from pathlib import Path
import numpy as np

from examples.libero.episode_runner import LiberoEpisodeRunner, default_client_factory, _default_run_episode
from .spec import digest, canonical_table_identity

SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')
LIMITS = (220, 280, 300, 520)


@dataclass(frozen=True)
class LiberoBenchmarkSpec:
    # Same rows as MW: (canonical suite/task, original language, suite).
    tasks: tuple[tuple[str, str, str], ...]
    seed: int = 7
    seed_mode: str = "paired_semantic"
    settle_steps: int = 10
    replan_steps: int = 5
    wire_size: int = 256
    max_steps_by_suite: tuple[int, ...] = LIMITS
    formal_indices: tuple[int, ...] = tuple(range(10))
    library_indices: tuple[int, ...] = tuple(range(10, 30))
    dev_indices: tuple[int, ...] = tuple(range(30, 50))
    benchmark: str = 'libero'
    protocol_id: str = 'libero_four_suites_r17_v1'
    interface_version: str = 'r17.v1'

    def __post_init__(self):
        if self.seed_mode not in ("paired_semantic", "fixed"):
            raise ValueError("unknown LIBERO request seed rule")
        if self.wire_size not in (96, 256) or self.replan_steps < 1 or self.settle_steps < 0:
            raise ValueError('LIBERO requires explicit native wire96/256 and a positive replan')
        if len(self.tasks) != 40 or len({r[0] for r in self.tasks}) != 40:
            raise ValueError('complete canonical four-suite task map required')
        for i, (name, prompt, suite) in enumerate(self.tasks):
            if suite != SUITES[i // 10] or not name.startswith(suite + '/') or not prompt:
                raise ValueError('task map must preserve official suite order and canonical names')
        if len(self.max_steps_by_suite) != 4 or any(x < 1 for x in self.max_steps_by_suite):
            raise ValueError('explicit four-suite step limits required')
        groups = (self.formal_indices, self.library_indices, self.dev_indices)
        if any(len(g) != len(set(g)) or any(type(i) is not int or not 0 <= i < 50 for i in g) for g in groups):
            raise ValueError('invalid original init-state partition')
        if any(set(groups[i]) & set(groups[j]) for i in range(3) for j in range(i + 1, 3)):
            raise ValueError('init partitions overlap')

    @property
    def max_policy_controls(self):
        return max(self.max_steps_by_suite)

    @property
    def ordered_tasks(self):
        return tuple(row[0] for row in self.tasks)

    @property
    def task_table_identity(self):
        return canonical_table_identity(self)

    @property
    def identity(self):
        return self.to_contract().identity

    def to_contract(self):
        from exp.r17_integration.common.contracts import BenchmarkSpec
        source = Path(__file__).resolve().parents[3] / 'examples/libero'
        return BenchmarkSpec(self.benchmark, self.interface_version, self.protocol_id,
            tuple(r[0] for r in self.tasks), tuple(r[1] for r in self.tasks),
            self.replan_steps, self.settle_steps, self.max_policy_controls,
            camera_digest=digest(('agentview', 'robot0_eye_in_hand', 256, 180, self.wire_size)),
            group_digest=digest(asdict(self)),
            env_source_digest=digest({n: hashlib.sha256((source / n).read_bytes()).hexdigest()
                                     for n in ('main.py', 'episode_runner.py', 'worker_entry.py')}),
            success_semantics='original LIBERO env.step done',
            termination_semantics='done | per-suite step cap; exceptions are errors')

    def manifest(self):
        return {**asdict(self), 'protocol_identity': self.identity,
                'ordered_tasks': self.ordered_tasks, 'task_table_identity': self.task_table_identity,
                'expected_formal_episodes': 40 * len(self.formal_indices),
                'global_task_ids': 'suite offset 10 * index in spatial/object/goal/10 + official suite-local id',
                'init_source': 'original suite init pool[original_idx], never reset seed',
                'policy_rng': ('fixed explicit seed at every decision' if self.seed_mode == 'fixed' else
                               'sha256(libero:seed:suite/task:original_idx:decision)[:4] little-endian'),
                'execution_ack': 'r17.execution.v1; actual controls; partial final tails supported',
                'quality_sr_claim': False}

    @classmethod
    def from_task_maps(cls, maps, **kwargs):
        rows = []
        for suite in SUITES:
            doc = maps[suite]
            if doc.get('suite') != suite:
                raise ValueError('suite task map mismatch')
            tasks = sorted(doc['tasks'], key=lambda r: r['task_id'])
            if [r['task_id'] for r in tasks] != list(range(10)):
                raise ValueError('official suite-local IDs must be 0..9')
            rows.extend((suite + '/' + r['name'], r['language'], suite) for r in tasks)
        return cls(tuple(rows), **kwargs)

    @classmethod
    def from_official(cls, **kwargs):
        from exp.libero_groot.emit_task_map import task_map_from_benchmark
        return cls.from_task_maps({s: task_map_from_benchmark(s) for s in SUITES}, **kwargs)


def decision_seed(task_name, original_idx, decision_seq, seed=7):
    return int.from_bytes(hashlib.sha256(f'libero:{seed}:{task_name}:{original_idx}:{decision_seq}'.encode()).digest()[:4], 'little')


class _ArgsProxy:
    def __init__(self, args):
        self.args = args
        self.replan_steps = args.replan_steps
    def __getattr__(self, key):
        return getattr(self.args, key)


class _ObservedEnv:
    def __init__(self, env, runner):
        self.env, self.runner = env, runner
        self.steps = self.controls = 0
        self.decision = None
        self.count = self.cursor = 0
        self.source = None
        self.error = None
        self.last_raw_obs = None
    def __getattr__(self, name):
        return getattr(self.env, name)
    def reset(self, *args, **kwargs):
        raw = self.env.reset(*args, **kwargs)
        self._remember(raw)
        return raw
    def set_init_state(self, *args, **kwargs):
        raw = self.env.set_init_state(*args, **kwargs)
        self._remember(raw)
        return raw
    def _remember(self, raw):
        # The legacy axis-angle converter may clamp quaternion w in-place.
        # Preserve the current raw pose before that conversion touches it.
        self.last_raw_obs = None if raw is None else dict(raw)
        if self.last_raw_obs is not None:
            for key in ('robot0_gripper_qpos','robot0_eef_pos','robot0_eef_quat'):
                if key in raw:
                    self.last_raw_obs[key] = np.asarray(raw[key]).copy()
    def flush(self, terminal=False, reason=None):
        if self.decision is None:
            return
        self.runner.receipts.append({**self.runner.identity(), 'decision_seq': self.decision,
            'executed_count': self.count, 'last_cursor': self.cursor + self.count - 1,
            'cursor_end': self.cursor + self.count, 'source': self.source,
            'terminal': terminal, 'end_reason': reason})
        self.decision = None
    def step(self, action):
        try:
            result = self.env.step(action)
        except Exception as exc:
            self.error = exc
            raise RuntimeError('LIBERO native env step failed') from exc
        self.steps += 1
        self._remember(result[0])
        if self.steps <= self.runner.spec.settle_steps:
            return result
        self.controls += 1
        self.count += 1
        if result[2] or self.count == self.runner.active_args.replan_steps:
            self.flush(bool(result[2]), 'success' if result[2] else None)
        return result


class R17LiberoEpisodeRunner(LiberoEpisodeRunner):
    def __init__(self, args, episode_setup, *, spec, client_factory=default_client_factory,
                 run_episode_fn=_default_run_episode, framework_context=True):
        self.spec, self.framework_context, self.kernel = spec, framework_context, run_episode_fn
        self.task = None
        self.receipts, self.cost_rows = [], []
        self.final_ack_error = None
        self.task_data_hash = None
        self.active_environment = None
        if (args.resize_size, args.replan_steps, args.num_steps_wait, args.seed) != (spec.wire_size, spec.replan_steps, spec.settle_steps, spec.seed):
            raise ValueError('LIBERO worker Args differ from sealed protocol')
        def checked_setup(task):
            env, state, prompt, maximum = episode_setup(task)
            if spec.protocol_id.startswith('libero_cosmos_official_'):
                # Official libero_utils.get_libero_env fixes environment RNG0;
                # spec.seed195 independently binds policy diffusion noise.
                env.seed(0)
            raw = state.detach().cpu().numpy() if hasattr(state, 'detach') else np.asarray(state)
            actual = digest({'dtype': str(raw.dtype), 'shape': raw.shape,
                             'state_sha256': hashlib.sha256(raw.tobytes()).hexdigest(),
                             'task_name': spec.tasks[task.task_id][0]})
            expected = task.extra.get('task_data_hash')
            if (expected and expected != actual) or prompt != spec.tasks[task.task_id][1]:
                env.close()
                raise ValueError('LIBERO original init/task language differs from sealed episode')
            self.task_data_hash = actual
            self.active_environment = env
            return env, state, prompt, maximum
        super().__init__(args, checked_setup, client_factory=client_factory,
                         run_episode_fn=self._kernel if framework_context else run_episode_fn)

    def identity(self):
        t = self.task
        return {**t.extra, 'task_uid': t.task_uid, 'attempt': t.attempt, 'yaml_id': t.yaml_id,
                'bundle_id': t.bundle_id, 'bundle_generation': t.extra.get('bundle_generation', t.bundle_id),
                'task_id': t.task_id, 'episode_id': t.task_id * 50 + t.orig_init_state_idx, 'task_name': self.spec.tasks[t.task_id][0],
                'original_init_idx': t.orig_init_state_idx, 'orig_init_state_idx': t.orig_init_state_idx,
                'task_data_hash': self.task_data_hash or t.extra.get('task_data_hash'),
                'task_table_identity': self.spec.task_table_identity,
                'benchmark_protocol': self.spec.identity, 'ack_protocol': 'r17.execution.v1',
                'partial_tail_protocol': 'r17.partial_tail.v1', 'seed': self.spec.seed}

    def _ensure_client(self, task):
        client = super()._ensure_client(task)
        if not self.framework_context:
            return client
        runner = self
        class ContextClient:
            def __getattr__(self, name):
                return getattr(client, name)
            def episode_start(self, **kwargs):
                kwargs['extra_metadata'] = {**kwargs.get('extra_metadata', {}), **runner.identity()}
                return client.episode_start(**kwargs)
            def episode_end(self, **kwargs):
                from .execution import end_episode
                try:
                    return end_episode(client, kwargs, runner.receipts)
                except Exception as exc:
                    runner.final_ack_error = exc
                    raise
        return ContextClient()

    def run(self, task, report):
        if not 0 <= task.task_id < 40 or task.experiment != SUITES[task.task_id // 10]:
            raise ValueError('LIBERO suite/global task identity mismatch')
        if task.extra.get('benchmark_protocol') != self.spec.identity or not 0 <= task.orig_init_state_idx < 50:
            raise ValueError('LIBERO protocol/original init identity mismatch')
        if task.extra.get('task_table_identity', self.spec.task_table_identity) != self.spec.task_table_identity:
            raise ValueError('LIBERO canonical task table identity mismatch')
        self.task, self.receipts, self.cost_rows = task, [], []
        self.task_data_hash = None
        self.active_environment = None
        self.final_ack_error = None
        try:
            result = super().run(task, report)
        finally:
            if self.framework_context and self.active_environment is not None:
                self.active_environment.close()
                self.active_environment = None
        if not self.framework_context:
            return replace(result, attempt=task.attempt)
        if self.final_ack_error:
            raise RuntimeError('final execution acknowledgment failed') from self.final_ack_error
        verdicts = [r for r in result.per_step_rows if not r.get('_kind')]
        for row, cost in zip(verdicts, self.cost_rows, strict=True):
            row.update(cost)
        result.per_step_rows.extend({'_kind': 'execution_ack', **r} for r in self.receipts)
        return replace(result, attempt=task.attempt, n_steps=self.observed.controls)

    def _kernel(self, env, client, initial_state, prompt, args, max_steps, **kwargs):
        from .execution import cost_row, ack_envelope
        self.active_args = _ArgsProxy(args)
        observed = self.observed = _ObservedEnv(env, self)
        runner = self
        class ContextInfer:
            def infer(self, obs):
                seq = len(runner.cost_rows)
                t = runner.task
                request = {'campaign': t.extra.get('campaign', 'r17'), 'benchmark_protocol': runner.spec.identity,
                    'task_table_identity': runner.spec.task_table_identity,
                    'task_name': runner.spec.tasks[t.task_id][0], 'task_id': t.task_id,
                    'original_init_idx': t.orig_init_state_idx, 'decision_seq': seq, 'seed': runner.spec.seed,
                    'explicit_seed': (runner.spec.seed if runner.spec.seed_mode == 'fixed' else
                                      decision_seed(runner.spec.tasks[t.task_id][0], t.orig_init_state_idx, seq, runner.spec.seed)),
                    'attempt': t.attempt, 'arm': t.extra.get('arm', ''),
                    'bundle_generation': t.extra.get('bundle_generation', t.bundle_id)}
                wire = dict(obs, __request__=request)
                if runner.spec.protocol_id.startswith('libero_cosmos_official_'):
                    raw = observed.last_raw_obs
                    if raw is None:
                        raise ValueError('Cosmos requires the current raw simulator observation')
                    wire['observation/cosmos_proprio'] = np.concatenate((
                        raw['robot0_gripper_qpos'], raw['robot0_eef_pos'], raw['robot0_eef_quat']))
                if runner.receipts:
                    wire['__execution__'] = ack_envelope(runner.receipts[-1])
                try:
                    out = client.infer(wire)
                    actions = np.asarray(out['actions'])
                    if actions.ndim != 2 or actions.shape[1] != 7 or not len(actions) or not np.isfinite(actions).all():
                        raise ValueError('LIBERO requires finite nonempty Hx7 actions')
                except Exception as exc:
                    raise RuntimeError('LIBERO inference failed') from exc
                # Existing loop receives the actual partial tail length. No new simulator loop.
                runner.active_args.replan_steps = min(runner.spec.replan_steps, len(actions))
                hit = out.get('__hit_meta__', {})
                observed.decision, observed.count = seq, 0
                observed.cursor, observed.source = int(hit.get('cursor', 0)), hit.get('source')
                runner.cost_rows.append(cost_row(out, runner.identity(), seq))
                return out
        try:
            result = self.kernel(observed, ContextInfer(), initial_state, prompt, self.active_args, max_steps, **kwargs)
            if observed.error:
                raise RuntimeError('LIBERO kernel swallowed environment error') from observed.error
            timing = kwargs.get('client_timing', {})
            if timing.get('termination_reason') == 'exception':
                raise RuntimeError('LIBERO kernel reported exception')
            observed.flush(True, 'success' if result[0] else 'step_cap')
            if self.receipts:
                self.receipts[-1]['terminal'] = True
                self.receipts[-1]['end_reason'] = 'success' if result[0] else 'step_cap'
            return result
        except Exception:
            observed.flush(True, 'error')
            raise


class LiberoBenchmarkAdapter:
    def __init__(self, spec):
        self.spec = spec
    def tasks(self):
        return self.spec.tasks
    def make_episode_runner(self, args, episode_setup=None, **kwargs):
        if episode_setup is None:
            from examples.libero.worker_entry import _build_episode_setup
            from types import SimpleNamespace
            native = _build_episode_setup(SimpleNamespace(task_suite_name=SUITES[0]), self.spec.seed, '')
            def episode_setup(task):
                # Existing setup indexes the official suite with a LOCAL id; the conductor retains GLOBAL ids.
                environment, state, prompt, _ = native(replace(task, task_id=task.task_id % 10))
                return environment, state, prompt, self.spec.max_steps_by_suite[task.task_id // 10]
        return R17LiberoEpisodeRunner(args, episode_setup, spec=self.spec, **kwargs)
    def worker_probe(self, native_probe):
        return {**native_probe, 'schema': 'r17.worker_probe.v1', 'benchmark_protocol': self.spec.identity,
                'task_table_identity': self.spec.task_table_identity, 'settle_steps': self.spec.settle_steps,
                'replan_steps': self.spec.replan_steps, 'resize_size': self.spec.wire_size,
                'max_steps': dict(zip(SUITES, self.spec.max_steps_by_suite, strict=True))}


def make_episode_runner(*args, framework_context=False, **kwargs):
    if framework_context:
        return R17LiberoEpisodeRunner(*args, framework_context=True, **kwargs)
    return LiberoEpisodeRunner(*args, **kwargs)


def main(argv=None):
    """Emit a sealed official task map/profile; no policy or GPU calls."""
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wire-size", type=int, choices=(96, 256), default=256)
    parser.add_argument("--released-reference", action="store_true", help="separate seed42/replan32/spatial280 profile, unqualified")
    a = parser.parse_args(argv)
    kwargs = {"wire_size": a.wire_size}
    if a.released_reference:
        kwargs.update(seed=42, seed_mode="fixed", replan_steps=32, max_steps_by_suite=(280, 280, 300, 520),
                      protocol_id="libero_released_reference_r17_v1")
    spec = LiberoBenchmarkSpec.from_official(**kwargs)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(spec.manifest(), sort_keys=True, indent=2) + "\n"
    if a.output.exists() and json.loads(a.output.read_text()) != json.loads(text):
        raise ValueError("refusing to replace a sealed benchmark profile")
    a.output.write_text(text)


if __name__ == "__main__":
    main()
