"""Dump agentview images for each init state in db_init.

Usage (must run in libero_sim conda env):
    conda activate libero_sim
    cd /path/to/openpi
    MUJOCO_GL=egl python scripts/dump_libero_init_images.py [--suites libero_spatial libero_10 libero_90] [--workers 10]
"""

import argparse
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from libero.libero import benchmark, get_libero_path

RESOLUTION = 256
SEED = 7

INIT_SRC_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "db_init", "libero",
)
ORIG_INIT_DIR = get_libero_path("init_states")
OUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "data", "db_init", "libero_image",
)


def find_original_indices(diff_states, orig_states):
    """For each row in diff_states, find its index in orig_states."""
    diff_np = np.array(diff_states)
    orig_np = np.array(orig_states)
    indices = []
    for row in diff_np:
        matches = np.where(np.all(np.isclose(orig_np, row, atol=1e-7), axis=1))[0]
        assert len(matches) == 1, f"Expected exactly 1 match, got {len(matches)}"
        indices.append(int(matches[0]))
    return indices


def process_task(suite_name, task_id):
    """Process a single task: create env, render all diff states, save images."""
    from PIL import Image
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    suite = benchmark_dict[suite_name]()
    task = suite.get_task(task_id)
    task_name = task.name

    diff_path = os.path.join(INIT_SRC_DIR, suite_name, f"{task_name}.init")
    if not os.path.exists(diff_path):
        return f"SKIP {suite_name}/{task_name}: no diff file"

    diff_states = torch.load(diff_path, weights_only=False)
    orig_states = torch.load(
        os.path.join(ORIG_INIT_DIR, suite_name, f"{task_name}.init"),
        weights_only=False,
    )
    orig_indices = find_original_indices(diff_states, orig_states)

    out_dir = os.path.join(OUT_DIR, suite_name, task_name)
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    task_bddl_file = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=RESOLUTION,
        camera_widths=RESOLUTION,
    )
    env.seed(SEED)

    generated = 0
    skipped = 0
    for state, orig_idx in zip(diff_states, orig_indices):
        out_path = os.path.join(out_dir, f"{orig_idx}.png")
        if os.path.exists(out_path):
            skipped += 1
            continue
        env.reset()
        obs = env.set_init_state(state)
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        Image.fromarray(img).save(out_path)
        generated += 1

    env.close()
    return f"DONE {suite_name}/{task_name}: {generated} new, {skipped} skipped -> {out_dir}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suites", nargs="+",
        default=["libero_spatial", "libero_10", "libero_90"],
    )
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()

    benchmark_dict = benchmark.get_benchmark_dict()

    # Collect all (suite, task_id) pairs
    jobs = []
    for suite_name in args.suites:
        suite = benchmark_dict[suite_name]()
        for task_id in range(suite.n_tasks):
            jobs.append((suite_name, task_id))

    print(f"Total tasks: {len(jobs)}, workers: {args.workers}")

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(process_task, suite_name, task_id): (suite_name, task_id)
            for suite_name, task_id in jobs
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            print(f"[{done_count}/{len(jobs)}] {future.result()}")

    print("All done.")


if __name__ == "__main__":
    main()
