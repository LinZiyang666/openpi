"""CPU LIBERO reset proof, with the collection's ten dummy control steps.

The library keys are normalized model state, so report both normalized-space
and physical-space errors. GR00T keys also round through bfloat16. No renderer,
policy calls, fitting, or evaluation rollouts are created.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys

from exp.offline_search.closed_loop.devset import REPO, STORE, PARENT, sha, library_episodes

HERE = Path(__file__).resolve().parent


def cpu_libero(site, work):
    import numpy
    import torch
    import mujoco
    import numba
    import yaml
    site, work = Path(site), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    lib = site / 'libero/libero'
    assets = Path('/home/weiland/.cache/libero/assets')
    if not assets.is_dir():
        raise FileNotFoundError('existing local LIBERO assets required; no downloads')
    (work / 'config.yaml').write_text(yaml.safe_dump(dict(benchmark_root=str(lib), bddl_files=str(lib / 'bddl_files'),
        init_states=str(lib / 'init_files'), datasets=str(lib.parent / 'datasets'), assets=str(assets))))
    os.environ.update(LIBERO_CONFIG_PATH=str(work), NUMBA_CACHE_DIR=str(work / 'numba'), MPLCONFIGDIR=str(work / 'mpl'),
                      CUDA_VISIBLE_DEVICES='')
    os.environ.pop('MUJOCO_GL', None)  # No render context is created by ControlEnv.
    # Append the existing simulator's pure Python packages; all native modules
    # above come from the mandated Python 3.11 .venv, never its py3.8 wheels.
    sys.path.append(str(site))
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs.env_wrapper import ControlEnv
    # Avoid the assets downloader entirely (even its cache-hit branch).
    import libero.libero
    libero.libero._assets_path_cache = str(assets)
    return benchmark, get_libero_path, ControlEnv


def robot_state(obs):
    import numpy as np
    q = np.array(obs['robot0_eef_quat'], copy=True)
    q[3] = np.clip(q[3], -1, 1)
    den = np.sqrt(1 - q[3] * q[3])
    aa = np.zeros(3) if math.isclose(den, 0.) else q[:3] * 2 * math.acos(q[3]) / den
    return np.concatenate((obs['robot0_eef_pos'], aa, obs['robot0_gripper_qpos']))


def state_stats(model, suite):
    import numpy as np
    if model == 'pi05':
        path = REPO / 'assets/pi05_libero/physical-intelligence/libero/norm_stats.json'
        stats = json.loads(path.read_text())['norm_stats']['state']
        return path, np.asarray(stats['q01']), np.asarray(stats['q99']), 1e-6
    path = Path('/data/ckpt') / ('n15_libero_10' if suite == 'l10' else 'n15_libero_spatial') / 'experiment_cfg/metadata.json'
    stats = json.loads(path.read_text())['new_embodiment']['statistics']['state']
    a = [v for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper') for v in stats[key]['min']]
    b = [v for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'gripper') for v in stats[key]['max']]
    return path, np.asarray(a), np.asarray(b), 0.


def run(site, tasks, indices, output):
    import numpy as np
    import torch
    import h5py
    benchmark, get_path, Env = cpu_libero(site, HERE / 'evidence/libero')
    rows = []
    for suite, long_suite in (('l10', 'libero_10'), ('spatial', 'libero_spatial')):
        bs = benchmark.get_benchmark_dict()[long_suite]()
        libraries = {}
        for model in PARENT:
            ep_path, eps, _ = library_episodes(model, suite, PARENT[model])
            libraries[model] = (ep_path, eps, np.load(ep_path.parent / 'rs.npy', mmap_mode='r'))
        for task_id in tasks:
            task = bs.get_task(task_id)
            init_path = REPO / 'exp/common/data/db_init/libero' / long_suite / (task.name + '.init')
            inits = torch.load(init_path, weights_only=False)
            env = Env(bddl_file_name=str(Path(get_path('bddl_files')) / task.problem_folder / task.bddl_file),
                      use_camera_obs=False, has_offscreen_renderer=False, has_renderer=False)
            env.seed(7)
            try:
                for i in indices:
                    env.reset()
                    initial = env.set_init_state(inits[i])
                    # Check object-state is actually present in this CPU reset.
                    object_count = int(np.size(initial.get('object-state', [])))
                    obs = initial
                    for _ in range(10):
                        obs, _, _, _ = env.step([0.] * 6 + [-1.])
                    raw = robot_state(obs)
                    for model, (ep_path, eps, rs) in libraries.items():
                        candidates = [e for e in eps if e['task_id'] == task_id and e['orig_init_state_idx'] == i]
                        if len(candidates) != 1:
                            raise ValueError(f'{model}/{suite} {task_id}/{i}: not one parent episode')
                        e = candidates[0]
                        stored = np.asarray(rs[e['start'], :8], np.float64)
                        stats_path, a, b, epsilon = state_stats(model, suite)
                        # Quantile/min-max arithmetic of the exact collection.
                        normalized = (raw - a) / (b - a + epsilon) * 2 - 1
                        if model == 'groot':
                            normalized = np.clip(normalized, -1, 1)
                            compared = torch.tensor(normalized, dtype=torch.float32).to(torch.bfloat16).float().numpy()
                        else:
                            compared = normalized.astype(np.float32)
                        restored = (stored + 1) / 2 * (b - a + epsilon) + a
                        delta = float(np.max(np.abs(compared - stored)))
                        physical_delta = float(np.max(np.abs(raw - restored)))
                        object_keys = []
                        with h5py.File(e['file']) as h:
                            group = h[sorted(k for k in h if k.startswith('step_'))[0]]
                            recorded = np.asarray(group['robot_state'])[:8]
                            if not np.array_equal(recorded.astype(np.float32), rs[e['start'], :8]):
                                raise ValueError('library first row differs from source H5')
                            object_keys = [k for k in group if 'object' in k or 'sim_state' in k]
                        row = dict(model=model, suite=suite, task_id=task_id, orig_init_state_idx=i, library_row=e['start'],
                            init_sha256=sha(init_path), episodes_sha256=sha(ep_path), stats_sha256=sha(stats_path),
                            max_abs_delta=delta, max_abs_delta_physical=physical_delta,
                            normalized_unrounded_delta=float(np.max(np.abs(normalized - stored))),
                            raw_robot_state=raw.tolist(), recorded_robot_state=stored.tolist(),
                            objects_in_reset=object_count, recorded_object_keys=object_keys,
                            object_comparison='unavailable: library H5 stores no object/simulator state',
                            tolerance=1e-6 if model == 'pi05' else 0., passed=delta <= (1e-6 if model == 'pi05' else 0.))
                        rows.append(row)
                        print(f'MAPPING {model}/{suite} task={task_id} init={i} max_delta={delta:.9g} physical={physical_delta:.9g} pass={row["passed"]}', flush=True)
            finally:
                env.close()
    result = dict(passed=all(r['passed'] for r in rows), comparisons=len(rows), tasks=tasks, indices=indices,
                  dummy_control_steps=10, camera_observations=False, gpu=False, rows=rows,
                  max_abs_delta=max(r['max_abs_delta'] for r in rows),
                  max_abs_delta_physical=max(r['max_abs_delta_physical'] for r in rows))
    Path(output).write_text(json.dumps(result, indent=1) + '\n')
    print('MAPPING_PROOF', json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)
    if not result['passed']:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sim-site', default='/home/weiland/projects/openpi_ext/envs/libero_sim/lib/python3.8/site-packages')
    p.add_argument('--tasks', default='0,4,9')
    p.add_argument('--indices', default='0,17,49')
    p.add_argument('--output', default=str(HERE / 'mapping_proof.json'))
    a = p.parse_args()
    run(a.sim_site, list(map(int, a.tasks.split(','))), list(map(int, a.indices.split(','))), a.output)


if __name__ == '__main__':
    main()
