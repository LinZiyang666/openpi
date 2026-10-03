"""Exact B-pool save_trajectory records and CPU-only state normalization."""
import json
from pathlib import Path
import re

import numpy as np

from exp.offline_search.debug.transport.receiver import digest


def discover(library, trajectory_root, suite):
    """Match the source catalog's task_N/episode_M identities, never task text."""
    episodes = json.loads((Path(library) / "episodes.json").read_text())
    sources = {}
    for i, episode in enumerate(episodes):
        stem = episode["stem"]
        if not re.fullmatch(r"task_\d+/episode_\d+", stem) or episode.get("orig_init_state_idx") is None:
            continue
        path = Path(trajectory_root) / suite / (stem + ".h5")
        if path.is_file():
            sources[str(i)] = dict(h5=str(path.resolve()), suite=suite, rs_space="model",
                                   provenance="B-pool source episodes.json task/init stem -> same collection save_traj; validated rs/success before admission")
    return sources


def normalization(path, dim=32, quantiles=True):
    """Same arithmetic as transforms.Normalize, followed by state zero padding.

    No model/encoder import. The provided artifact is hashed into provenance.
    LIBERO pi05 uses quantiles; other model adapters can use --state-transform.
    """
    raw = json.loads(Path(path).read_text())
    stats = raw.get("norm_stats", raw)["state"]
    def transform(wire):
        wire = np.asarray(wire, dtype=np.float64)
        n = len(wire)
        if quantiles:
            lo, hi = np.asarray(stats["q01"], np.float64)[:n], np.asarray(stats["q99"], np.float64)[:n]
            state = (wire - lo) / (hi - lo + 1e-6) * 2. - 1.
        else:
            mean, std = np.asarray(stats["mean"], np.float64)[:n], np.asarray(stats["std"], np.float64)[:n]
            state = (wire - mean) / (std + 1e-6)
        if n > dim:
            raise ValueError("state has more valid dimensions than model manifest")
        return np.pad(state.astype(np.float32), (0, dim - n))
    return transform


def initial_from_pool(meta, episode, init_directory):
    from exp.offline_search.debug.client.compat import install_import_compat
    install_import_compat()
    from libero.libero import benchmark
    from examples.libero import main
    suite = benchmark.get_benchmark_dict()[meta["suite"]]()
    task = suite.get_task(int(episode["task_id"]))
    if not init_directory:
        raise ValueError("explicit original B-pool --init-states-dir required for save_traj replay")
    base = Path(init_directory)
    if list(base.glob("*.pruned_init")):
        raise ValueError("B-pool directory contains pruned A-pool overrides")
    path = base / (task.name + ".init")
    states = main._load_init_states(task, suite, episode["task_id"], str(base))
    index = int(episode["orig_init_state_idx"])
    if not 0 <= index < len(states):
        raise ValueError("library original init outside bound B-pool")
    meta.update(init_pool_file=str(path), init_pool_sha256=digest(path), orig_init_state_idx=index)
    return np.asarray(states[index], dtype=np.float64)


def load(path, initial_state, episode, suite):
    """Read only simulator/robot/executed-action datasets, not images/tokens."""
    import h5py
    path = Path(path)
    with h5py.File(str(path), "r") as handle:
        attrs = handle.attrs
        for key, expected in (("task_id", episode["task_id"]), ("orig_init_state_idx", episode["orig_init_state_idx"]),
                              ("success", episode["success"]), ("num_cycles", episode["n_rows"])):
            if attrs.get(key) != expected:
                raise ValueError("save_traj source identity/count mismatch: " + key)
        names = sorted(k for k in handle if k.startswith("step_"))
        if names != ["step_{:04d}".format(i) for i in range(len(names))]:
            raise ValueError("save_traj decision gap")
        actions, seq, robot, sim = [], [], [], []
        nwait = int(attrs["num_steps_wait"])
        # These are the stock collector's settling controls, not model proposals.
        wait = np.tile([0., 0., 0., 0., 0., 0., -1.], (nwait, 1))
        actions.append(wait)
        seq.extend([-1] * nwait)
        for i, name in enumerate(names):
            step = handle[name]
            issued = np.asarray(step["executed_actions"], dtype=np.float64)
            if issued.ndim != 2 or issued.shape[1] != 7 or len(issued) != int(step.attrs["executed_action_count"]) or len(issued) < 1:
                raise ValueError("invalid executed control dataset")
            actions.append(issued)
            seq.extend([i] * len(issued))
            robot.append(np.asarray(step["robot_state"], dtype=np.float64))
            sim.append(np.asarray(step["sim_state"], dtype=np.float64))
        count = sum(len(x) for x in actions) - nwait
        if count != int(attrs["num_steps"]) or int(attrs["final_env_timestep"]) != count + nwait:
            raise ValueError("save_traj executed controls/final timestep mismatch")
        arrays = dict(initial_state=np.array(initial_state, dtype=np.float64, copy=True), action=np.concatenate(actions),
                      decision_seq=np.asarray(seq, np.int32), expected_robot_state_wire=np.asarray(robot),
                      expected_sim_state=np.asarray(sim))
        meta = dict(env_seed=int(attrs["seed"]), suite=suite, source_h5=str(path.resolve()), sha256=digest(path),
                    settle_provenance="stock LIBERO_DUMMY_ACTION × recorded num_steps_wait",
                    provenance="exact save_trajectory executed_actions + bound original B-pool reset")
    return arrays, meta
