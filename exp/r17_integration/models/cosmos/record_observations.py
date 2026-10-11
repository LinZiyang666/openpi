"""CPU-only capture of raw LIBERO frames, original init0, after ten settles."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np


def record(output):
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    from examples.libero.main import _quat2axisangle
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    suite = benchmark.get_benchmark_dict()['libero_spatial']()
    task = suite.get_task(0)
    initial = suite.get_task_init_states(0)[0]
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path('bddl_files'), task.problem_folder, task.bddl_file),
                             camera_heights=256, camera_widths=256)
    env.seed(0)
    rows = []
    try:
        env.reset()
        raw = env.set_init_state(initial)
        for _ in range(10):
            raw, *_ = env.step([0,0,0,0,0,0,-1])
        for i in range(3):
            rows.append({'agent_raw': raw['agentview_image'].copy(), 'wrist_raw': raw['robot0_eye_in_hand_image'].copy(),
                         'proprio': np.concatenate((raw['robot0_gripper_qpos'],raw['robot0_eef_pos'],raw['robot0_eef_quat'])),
                         'state': np.concatenate((raw['robot0_eef_pos'],_quat2axisangle(raw['robot0_eef_quat'].copy()),raw['robot0_gripper_qpos'])),
                         'prompt': task.language})
            if i != 2:
                raw, *_ = env.step([0,0,0,0,0,0,-1])
    finally:
        env.close()
    arrays = {key: np.asarray([r[key] for r in rows]) for key in rows[0]}
    np.savez_compressed(output/'raw_observations.npz', **arrays)
    digest = hashlib.sha256((output/'raw_observations.npz').read_bytes()).hexdigest()
    receipt = dict(schema='r17.cosmos.recorded_observations.v1', recorded=True, synthetic=False,
                   rows=3, suite='libero_spatial', task_id=0, original_init_idx=0, env_seed=0,
                   settle_steps=10, render_resolution=256, camera_storage='unmodified raw simulator pixels',
                   proprio_order='g0,g1,eef_xyz,quat_xyzw', prompt=task.language,
                   files={'raw_observations.npz':digest})
    (output/'OBSERVATIONS.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def validate(directory):
    path = Path(directory)
    receipt = json.loads((path/'OBSERVATIONS.json').read_text())
    if receipt['synthetic'] or not receipt['recorded'] or receipt['rows'] != 3:
        raise ValueError('three real recorded observations required')
    for name, digest in receipt['files'].items():
        if hashlib.sha256((path/name).read_bytes()).hexdigest() != digest:
            raise ValueError('recorded observations changed')
    return receipt


if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True)
    print(json.dumps(record(p.parse_args().output)))
