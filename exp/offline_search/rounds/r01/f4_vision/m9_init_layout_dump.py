"""M9 step 1: identify which dims of a LIBERO init state are object poses (run in the libero_sim venv, CPU, no render).

A LIBERO ``.init`` file is torch.save of float64 [50, D]; each row is robosuite ``sim.get_state().flatten()`` =
[time (1), qpos (nq), qvel (nv)] (no act / udd state in LIBERO scenes). This script builds each task's env WITHOUT a
renderer, reads the MuJoCo joint table (name, type, qpos address) and writes, per suite x task, the index map into the
flattened state:

    robot joints (robot0_*, gripper0_*), free-joint objects (7 qpos: xyz + quat wxyz), articulated fixture joints
    (hinge / slide of drawers, doors, knobs), plus consistency checks (D == 1 + nq + nv; set_init_state round-trip).

Usage (from the repo root):
    LIBERO_CONFIG_PATH=<dir with config.yaml> MUJOCO_GL=osmesa taskset -c ... \
        /home/weiland/projects/openpi_ext/envs/libero_sim/bin/python \
        exp/offline_search/rounds/r01/f4_vision/m9_init_layout_dump.py --out <derived>/init_layout.json
"""
import argparse
import glob
import json
import os

import numpy as np
import torch

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 5))
def _tload(p):
    try:
        return torch.load(p, weights_only=False)
    except TypeError:  # old torch (libero_sim venv)
        return torch.load(p)


POOLS = {"libero_spatial": ("libero_spatial", "libero_spatial_apool"), "libero_10": ("libero_10", "libero_10_apool")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401  (registers problems)
    from libero.libero.envs.env_wrapper import ControlEnv
    import mujoco

    bd = benchmark.get_benchmark_dict()
    out = {}
    for suite, (bdir, adir) in POOLS.items():
        ts = bd[suite]()
        out[suite] = {}
        for tid in range(ts.n_tasks):
            task = ts.get_task(tid)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = ControlEnv(bddl_file_name=bddl, has_offscreen_renderer=False, use_camera_obs=False,
                             has_renderer=False)
            env.reset()
            sim = env.env.sim
            m = sim.model
            nq, nv = int(m.nq), int(m.nv)
            joints = []
            for j in range(m.njnt):
                name = m.joint_id2name(j)
                jt = int(m.jnt_type[j])  # 0 free, 1 ball, 2 slide, 3 hinge
                qadr = int(m.jnt_qposadr[j])
                width = {0: 7, 1: 4, 2: 1, 3: 1}[jt]
                body = m.body_id2name(int(m.jnt_bodyid[j]))
                kind = ("robot" if name.startswith(("robot0_", "gripper0_", "mount0_")) else
                        "object_free" if jt == 0 else "fixture_joint")
                # flattened state index = 1 (time) + qpos address
                joints.append({"name": name, "type": jt, "qadr": qadr, "width": width, "body": body, "kind": kind,
                               "state_idx": list(range(1 + qadr, 1 + qadr + width))})
            name = os.path.basename(bddl).replace(".bddl", "")
            fb = sorted(glob.glob(f"{REPO}/exp/common/data/db_init/libero/{bdir}/{task.name}.init"))
            fa = sorted(glob.glob(f"{REPO}/exp/common/data/db_init/libero/{adir}/{task.name}.init"))
            B = np.asarray(_tload(fb[0])) if fb else None
            A = np.asarray(_tload(fa[0])) if fa else None
            D = int(B.shape[1]) if B is not None else None
            # round-trip check: set the first A init and read it back
            rt = None
            if A is not None:
                env.set_init_state(A[0])
                s = sim.get_state().flatten()
                rt = float(np.abs(s[: A.shape[1]] - A[0]).max()) if s.shape[0] >= A.shape[1] else None
            rec = {"task_id": tid, "task": task.language, "name": task.name, "nq": nq, "nv": nv, "D": D,
                   "D_expected": 1 + nq + nv, "roundtrip_maxabs": rt, "joints": joints,
                   "bpool_file": fb[0] if fb else None, "apool_file": fa[0] if fa else None}
            # per-dim spread over the 100 inits (A+B) for the object free joints (sanity: robot ~const, objects vary)
            if A is not None and B is not None:
                AB = np.concatenate([A, B])
                rec["std_by_kind"] = {k: float(np.mean([AB[:, j["state_idx"][:3]].std(0).mean()
                                                        for j in joints if j["kind"] == k] or [np.nan]))
                                      for k in ("robot", "object_free", "fixture_joint")}
            out[suite][str(tid)] = rec
            print(suite, tid, task.name[:60], "nq", nq, "nv", nv, "D", D, "exp", 1 + nq + nv, "rt", rt,
                  "objs", [j["name"] for j in joints if j["kind"] == "object_free"],
                  "fix", [j["name"] for j in joints if j["kind"] == "fixture_joint"], rec.get("std_by_kind"),
                  flush=True)
            env.close()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
