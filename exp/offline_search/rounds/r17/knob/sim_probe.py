"""Same LIBERO task + init state on two worker machines: hash the first rendered frames and the
physics state along a fixed action sequence. usage: python render_probe.py OUT.npz"""
import hashlib
import os
import sys

import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

out = {}
lines = []
for suite_name, task_id in (("libero_10", 1), ("libero_spatial", 0)):
    suite = benchmark.get_benchmark_dict()[suite_name]()
    task = suite.get_task(task_id)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    env.seed(0)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(task_id)[0])
    rng = np.random.default_rng(0)
    states = []
    for step in range(210):
        action = [0, 0, 0, 0, 0, 0, -1] if step < 10 else list(rng.uniform(-0.5, 0.5, 6)) + [rng.choice([-1, 1])]
        obs, _, _, _ = env.step(action)
        if step == 9:
            for cam in ("agentview_image", "robot0_eye_in_hand_image"):
                out[f"{suite_name}_{cam}"] = obs[cam]
                lines.append(f"{suite_name} {task.name[:40]} {cam} sha={hashlib.sha256(obs[cam].tobytes()).hexdigest()[:16]}")
        states.append(env.sim.get_state().flatten())
    states = np.stack(states)
    out[f"{suite_name}_states"] = states
    lines.append(f"{suite_name} state@10 sha={hashlib.sha256(states[10].tobytes()).hexdigest()[:16]} "
                 f"state@209 sha={hashlib.sha256(states[209].tobytes()).hexdigest()[:16]}")
    env.close()
np.savez_compressed(sys.argv[1], **out)
print("\n".join(lines))
