"""Coordinator real-env admission: identical controls, capture off/on, both suites.

No policy or network. Use a recorded contact-rich action tape when available.
Default tape is deterministic and independent of simulator/global RNG state.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .capture import Client, EnvTap, FileSink


class TapeClient:
    def episode_start(self, **kw):
        pass

    def infer(self, obs):
        return {"actions": np.empty((0, 7)), "__debug__": {"v": 1, "decision_id": obs["__debug__"]["decision_id"], "status": "available"}}


def parse_pool_map(spec, suites):
    """Non-default pools must be explicitly bound for every requested suite."""
    if not spec:
        return {}
    pools = {}
    for item in spec.split(","):
        suite, sep, directory = item.partition("=")
        if not sep or not suite or not directory or suite in pools or suite not in suites:
            raise ValueError("use an explicit, unique suite=directory binding for every suite")
        pools[suite] = directory
    if set(pools) != set(suites):
        raise ValueError("non-default pools require a binding for every requested suite")
    return pools


def replay(env, initial, actions, directory=None, identity=None, seed=7, settle=10):
    client = None
    tap = env
    if directory is not None:
        client = Client(TapeClient(), directory, campaign="capture_parity", arm="capture", env_seed=seed, sink_factory=FileSink)
        client.episode_start(experiment=identity["suite"], task=identity["task"], episode_id=identity["init"],
                             extra_metadata=dict(task_uid=identity["uid"], task_id=identity["task_id"], attempt=1,
                                                 orig_init_state_idx=identity["init"]))
        client.configure(env, SimpleNamespace(seed=seed, num_steps_wait=settle, replan_steps=5), len(actions))
        tap = EnvTap(env, client)
    sequence = []
    env.reset()
    obs = tap.set_init_state(initial)
    for i, action in enumerate(actions):
        if client and i >= settle and (i - settle) % 5 == 0:
            client.infer({"observation/state": np.zeros(8), "prompt": identity["task"]})
        obs, reward, done, info = tap.step(action.tolist())
        data = env.env.sim.data if hasattr(env, "env") else env.sim.data
        sequence.append((np.array(data.qpos, dtype=np.float64, copy=True), np.array(data.qvel, dtype=np.float64, copy=True)))
    if client:
        client.finish(done, "parity_tape_end")
        if client.errors:
            raise RuntimeError("capture failed during parity: " + str(client.errors))
    return sequence


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suites", nargs="+", default=["libero_10", "libero_spatial"])
    ap.add_argument("--task-id", type=int, default=0)
    ap.add_argument("--inits", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--init-states-dir", default="", help="explicit suite=directory map for every suite; empty uses benchmark pools")
    ap.add_argument("--action-tape", type=Path, help="NPZ action float64 (n,7), including settling")
    ap.add_argument("--controls", type=int, default=120)
    ap.add_argument("--settle-controls", type=int, default=10)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if len(a.inits) < 3 or len(a.suites) < 2:
        ap.error("admission requires >=3 episodes per suite and >=2 suites")
    try:
        pools = parse_pool_map(a.init_states_dir, a.suites)
    except ValueError as exc:
        ap.error(str(exc))
    from .compat import install_import_compat
    install_import_compat()
    from examples.libero import main as m
    from libero.libero import benchmark
    from exp.offline_search.debug.transport.receiver import atomic_json
    if a.action_tape:
        with np.load(str(a.action_tape), allow_pickle=False) as payload:
            actions = payload["action"].astype(np.float64)
    else:
        actions = np.tile(np.asarray(m.LIBERO_DUMMY_ACTION, dtype=np.float64), (a.controls, 1))
        t = np.arange(max(0, a.controls - a.settle_controls))
        actions[a.settle_controls:, :3] = .05 * np.column_stack((np.sin(t / 11.), np.cos(t / 17.), np.sin(t / 23.)))
        actions[a.settle_controls:, -1] = np.where((t // 20) % 2 == 0, 1., -1.)
    results = []
    for suite_name in a.suites:
        suite = benchmark.get_benchmark_dict()[suite_name]()
        task = suite.get_task(a.task_id)
        states = m._load_init_states(task, suite, a.task_id, pools.get(suite_name, ""))
        if any(init < 0 or init >= len(states) for init in a.inits):
            ap.error("init index outside {} pool of {} states: {}".format(suite_name, len(states), a.inits))
        for init in a.inits:
            identity = dict(suite=suite_name, task=task.language, task_id=a.task_id, init=init,
                            uid="capture_parity:" + suite_name + ":" + str(a.task_id) + ":" + str(init))
            pair = []
            for enabled in (False, True):
                env, text = m._get_libero_env(task, m.LIBERO_ENV_RESOLUTION, a.seed)
                identity["task"] = text
                try:
                    pair.append(replay(env, states[init], actions, a.out / "client" if enabled else None,
                                       identity, a.seed, a.settle_controls))
                finally:
                    env.close()
            first = next((i for i, (left, right) in enumerate(zip(*pair))
                          if any(x.dtype != y.dtype or x.shape != y.shape or x.tobytes() != y.tobytes() for x, y in zip(left, right))), None)
            results.append(dict(suite=suite_name, init=init, seed=a.seed, controls=len(actions),
                                PASS=first is None and len(pair[0]) == len(pair[1]), first_different_control=first))
            a.out.mkdir(parents=True, exist_ok=True)
            atomic_json(a.out / "parity.json", results)
    print(json.dumps(dict(PASS=all(r["PASS"] for r in results), episodes=len(results), results=results)))
    if not all(r["PASS"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
