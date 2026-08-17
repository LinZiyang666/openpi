"""Roll out the GR00T N1.5 teacher on RoboCasa365 across two pinned kitchens.

Runs teacher-only episodes (no cache) on the same task set in scene A and scene
B and reports the paired success rates, mirroring the procedure already used for
the pi0.5 teacher so the two are directly comparable.

No format translation happens here beyond selection and downsampling: the
RoboCasa365 gym wrapper already emits GR00T's own ``video.*`` / ``state.*`` /
``annotation.*`` vocabulary, and the action dict the server returns is keyed
exactly as ``env.step`` expects.  In particular the language key is passed
through untouched -- N1.5 ``new_embodiment`` wants the env-native
``annotation.human.task_description``, not N1.7's ``.action.`` variant.

Where everything lives (all paths verified on the host named below)
------------------------------------------------------------------
host          weilandserver
this venv     /home/weiland/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv
              py3.12, numpy 2.2.5.  Anything installed here needs --no-deps:
              openpi-client pins numpy<2.0.0 and would otherwise downgrade numpy,
              breaking robocasa's import assertions and scipy.
working dir   must be /home/weiland/Isaac-GR00T/external_dependencies/robocasa365
              (robocasa resolves its assets relative to it)
server        exp/robocasa365/serve_groot_n15.py, in the *other* island, port 8020
EGL           this host has no system EGL, so all three must be exported:
                LD_LIBRARY_PATH=/home/weiland/nvidia-gl/root/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
                __EGL_VENDOR_LIBRARY_DIRS=/home/weiland/nvidia-gl/root/usr/share/glvnd/egl_vendor.d
                MUJOCO_GL=egl
results       exp/robocasa365/data/  (written incrementally, one flush per task,
              so a mid-run crash still leaves the completed tasks on disk)

Launch::

    tmux new-session -d -s grootrollout "export HOME=/home/weiland; \\
      export LD_LIBRARY_PATH=/home/weiland/nvidia-gl/root/usr/lib/x86_64-linux-gnu:\\$LD_LIBRARY_PATH; \\
      export __EGL_VENDOR_LIBRARY_DIRS=/home/weiland/nvidia-gl/root/usr/share/glvnd/egl_vendor.d; \\
      export MUJOCO_GL=egl PYTHONPATH=/home/weiland/openpi; \\
      cd /home/weiland/Isaac-GR00T/external_dependencies/robocasa365 && \\
      /home/weiland/Isaac-GR00T/gr00t/eval/sim/robocasa365/robocasa365_uv/.venv/bin/python \\
      /home/weiland/openpi/exp/robocasa365/groot_rollout_client.py --n-trials 5"
"""

from __future__ import annotations

import argparse
import collections
import json
import logging
import os
import pathlib
import time
import traceback
from typing import Any

import numpy as np

from exp.robocasa365 import groot_keys
from exp.robocasa365.groot_policy_adapter import iter_step_actions

# INFO, not WARNING: openpi_client's connect loop reports "still waiting for
# server" at INFO, and silencing it turns a missing server into a process
# that prints nothing at all and blocks forever.
logging.basicConfig(level=logging.INFO)

DEFAULT_OUTPUT = pathlib.Path(__file__).resolve().parent / "data" / "groot_rollout.json"


def _select_and_downsample(obs: dict[str, Any]) -> dict[str, Any]:
    """Pull the contract keys out of a raw env observation.

    Images are rendered at 512 and reduced to 256 with INTER_AREA, matching what
    the official RoboCasa365 wrapper does; rendering straight at 256 would skip
    the anti-aliasing the model was trained against.  Orientation is left alone:
    the gym wrapper already flips the frames.
    """
    import cv2

    element: dict[str, Any] = {}
    for key in groot_keys.VIDEO_KEYS:
        image = np.ascontiguousarray(obs[key])
        # Assert rather than cast: casting a float frame to uint8 would truncate
        # a [0, 1] image to all zeros *and* satisfy the server's dtype check,
        # disarming the one guard that could have caught it.
        if image.dtype != np.uint8:
            raise ValueError(f"{key} must be uint8 from the env, got {image.dtype}")
        if image.ndim != 3 or image.shape[0] != image.shape[1]:
            # cv2's dsize is (W, H) while shape is (H, W); they only coincide
            # while frames are square, which is all this pipeline renders.
            raise ValueError(f"{key} must be a square HxWx3 frame, got {image.shape}")
        target = (groot_keys.MODEL_IMAGE_RESOLUTION, groot_keys.MODEL_IMAGE_RESOLUTION)
        if image.shape[:2] != target:
            image = cv2.resize(image, target, interpolation=cv2.INTER_AREA)
        element[key] = np.ascontiguousarray(image)

    for key in groot_keys.STATE_KEYS:
        element[key] = np.asarray(obs[key], dtype=np.float64)

    for key in groot_keys.LANGUAGE_KEYS:
        element[key] = str(obs[key])
    return element


def parse_scene(text: str) -> tuple[int, int]:
    """Parse a ``layout,style`` pair, failing loudly on malformed input.

    Without this, a typo like ``--scene-a 1`` unpacks badly deep inside the
    rollout, where main()'s per-arm handler records it as "this task failed" and
    happily works through the whole task list before anyone notices.
    """
    parts = text.split(",")
    if len(parts) != 2:
        raise ValueError(f"scene must be 'layout,style', got {text!r}")
    return int(parts[0]), int(parts[1])


def summarize_exception(exc: BaseException) -> str:
    """Condense an exception into something worth reading in the results file.

    The inference server returns a full traceback as the message, and its
    informative part -- the actual exception type and text -- is at the *end*.
    Truncating from the front yields the same useless websocket-server preamble
    for every failure.
    """
    text = str(exc).strip()
    tail = text.splitlines()[-1] if text else ""
    return f"{type(exc).__name__}: {tail[:200]}"


def write_results_atomically(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Write via a temp file + rename so a crash cannot destroy earlier tasks.

    ``write_text`` truncates before writing; dying in that window would take the
    already-completed tasks with it, which defeats the point of flushing after
    every task in an unattended run.
    """
    encoded = json.dumps(payload, indent=1)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(encoded)
    os.replace(tmp, path)


def episode_seeds(n_trials: int, base_seed: int) -> list[int]:
    """Seeds replayed identically in both scenes.

    ``main()`` catches per-arm failures and moves on, so scene A and scene B are
    separate ``gym.make`` calls.  Reusing one seed list across both is what makes
    episode *i* of A and episode *i* of B comparable as a pair; without it the
    object placements and initial states are two independent random draws and
    the A/B numbers cannot be treated as paired samples.
    """
    return [base_seed + index for index in range(n_trials)]


def _send_ctrl(client: Any, ctrl: str, **fields: Any) -> None:
    """Send one ``__ctrl__`` frame, tolerating a server that has no cache wired.

    Field names are dunder-wrapped because the server reads them straight off
    the observation dict and must not confuse them with modality keys.
    """
    frame: dict[str, Any] = {"__ctrl__": ctrl}
    for key, value in fields.items():
        frame[f"__{key}__"] = value
    client.infer(frame)


def _record_hit(
    hit_log: Any,
    response: dict,
    task: str,
    episode: int,
    step: int,
) -> None:
    """Append one cache verdict to the JSONL log, if one is being written.

    Every field is read with ``.get`` so a server without the cache -- whose
    responses carry no ``__hit_meta__`` at all -- costs nothing and logs
    nothing, rather than needing a parallel code path.
    """
    if hit_log is None:
        return
    meta = response.get("__hit_meta__")
    if meta is None:
        return
    hit_log.write(
        json.dumps(
            {
                "task": task,
                "episode": episode,
                "step_idx": step,
                "hit_type": meta.get("hit_type"),
                "winner_id": meta.get("winner_id"),
                "cp1_score": meta.get("cp1_score"),
                "searched": meta.get("searched"),
            }
        )
        + "\n"
    )
    hit_log.flush()


def run_one(
    client: Any,
    env_name: str,
    scene: tuple[int, int],
    n_trials: int,
    replan: int,
    base_seed: int,
    hit_log: Any = None,
) -> dict:
    """Run ``n_trials`` episodes of one task in one pinned kitchen."""
    import gymnasium as gym
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    horizon = get_task_horizon(env_name)
    layout, style = scene
    seeds = episode_seeds(n_trials, base_seed)
    env = gym.make(
        f"robocasa/{env_name}",
        # split=None bypasses the branch that would also swap the object pool;
        # only the kitchen may vary between A and B.
        split=None,
        obj_instance_split="target",
        layout_and_style_ids=[(layout, style)],
        camera_heights=groot_keys.RENDER_RESOLUTION,
        camera_widths=groot_keys.RENDER_RESOLUTION,
    )

    successes = 0
    started = time.time()
    # Any failure below must still release the MuJoCo context: main() keeps
    # going after a failed arm, so a leaked env would accumulate across a batch
    # until the GPU runs out.
    try:
        for episode, seed in enumerate(seeds):
            obs, _ = env.reset(seed=seed)
            plan: collections.deque = collections.deque()
            step = 0
            done = False
            _send_ctrl(client, "episode_start", task=env_name, episode_id=episode)
            try:
                while step < horizon:
                    if not plan:
                        response = client.infer(_select_and_downsample(obs))
                        plan.extend(iter_step_actions(response["actions"], replan))
                        _record_hit(hit_log, response, env_name, episode, step)
                    obs, _, _, _, info = env.step(plan.popleft())
                    if info.get("success"):
                        done = True
                        break
                    step += 1
            finally:
                # Always sent, including on an exception mid-episode: the
                # server closes its search session inside this handler, and a
                # session left open leaks backend state into the next episode.
                _send_ctrl(client, "episode_end", success=done)
            successes += int(done)
            print(
                f"    ep{episode}(seed={seed}): {'OK' if done else 'fail'} t={step}/{horizon}",
                flush=True,
            )
    finally:
        env.close()

    return {
        "succ": successes,
        "n": n_trials,
        "sr": successes / n_trials,
        "wall_s": round(time.time() - started, 1),
        "seeds": seeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--n-trials", type=int, default=5)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--scene-a", default="1,1")
    parser.add_argument("--scene-b", default="7,7")
    parser.add_argument(
        "--tasks", default="", help="comma-separated; default = atomic_seen"
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=0,
        help="Episode seeds are base_seed..base_seed+n_trials-1, replayed "
        "identically in both scenes so the two arms are paired.",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=3,
        help="Abort once this many arms fail back to back; a dead connection "
        "would otherwise fail every remaining arm and waste the whole run.",
    )
    parser.add_argument(
        "--allow-diagnostic-server",
        action="store_true",
        help="Permit running against a server started with --diagnostic-seed. "
        "Success rates from such a server are biased; for wiring checks only.",
    )
    parser.add_argument("--output", type=pathlib.Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--hit-log",
        type=pathlib.Path,
        default=None,
        help="JSONL file for per-step cache verdicts. Only meaningful against a "
             "server started with --cache-config; otherwise nothing is written.",
    )
    args = parser.parse_args()

    # Validate through the same helper the rollout uses, so the CLI cannot admit
    # a bound the tested slicing logic would reject (e.g. zero or negative).
    probe = {
        key: np.zeros((groot_keys.ACTION_HORIZON, dim))
        for key, dim in groot_keys.ACTION_DIMS.items()
    }
    iter_step_actions(probe, args.replan_steps)

    if args.n_trials < 1:
        raise ValueError(f"--n-trials must be >= 1, got {args.n_trials}")
    scene_a = parse_scene(args.scene_a)
    scene_b = parse_scene(args.scene_b)

    from openpi_client import websocket_client_policy
    from robocasa.utils.dataset_registry import TASK_SET_REGISTRY

    tasks = (
        args.tasks.split(",") if args.tasks else list(TASK_SET_REGISTRY["atomic_seen"])
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    hit_log = None
    if args.hit_log is not None:
        args.hit_log.parent.mkdir(parents=True, exist_ok=True)
        hit_log = args.hit_log.open("w")
    print(
        f"tasks={len(tasks)} A={scene_a} B={scene_b} trials={args.n_trials} "
        f"-> {args.output}",
        flush=True,
    )
    # Announced before connecting: the client blocks until the server answers,
    # so without this line a server that never came up looks like a hung process
    # with no output at all.
    print(f"connecting to {args.host}:{args.port}", flush=True)
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    server_metadata = client.get_server_metadata() or {}
    print(f"server metadata: {server_metadata}", flush=True)
    server_horizon = server_metadata.get("action_horizon")
    if server_horizon is not None and server_horizon != groot_keys.ACTION_HORIZON:
        # Fail at startup rather than at every chunk, 300 tasks in.
        raise RuntimeError(
            f"server action_horizon {server_horizon} != client "
            f"{groot_keys.ACTION_HORIZON}; server and client are out of sync"
        )
    if (
        server_metadata.get("diagnostic_seed") is not None
        and not args.allow_diagnostic_server
    ):
        raise RuntimeError(
            "server is running with --diagnostic-seed, which pins the flow-matching "
            "noise and makes success rates a biased estimate of the teacher. "
            "Restart the server without it, or pass --allow-diagnostic-server."
        )

    results: dict[str, Any] = {
        "sceneA": list(scene_a),
        "sceneB": list(scene_b),
        "n_trials": args.n_trials,
        "replan_steps": args.replan_steps,
        "base_seed": args.base_seed,
        "server_metadata": server_metadata,
        "tasks": {},
    }
    consecutive_failures = 0
    for index, task in enumerate(tasks):
        row = {}
        for tag, scene in (("A", scene_a), ("B", scene_b)):
            try:
                row[tag] = run_one(
                    client,
                    task,
                    scene,
                    args.n_trials,
                    args.replan_steps,
                    args.base_seed,
                    hit_log=hit_log,
                )
                consecutive_failures = 0
            except Exception as exc:  # noqa: BLE001 - one bad task must not sink the run
                traceback.print_exc()
                row[tag] = {"error": summarize_exception(exc)}
                consecutive_failures += 1
                # A dead socket or a mis-specified run would otherwise fail every
                # remaining arm the same way, burning hours to produce a results
                # file that looks like "the model is bad".
                if consecutive_failures >= args.max_consecutive_failures:
                    results["tasks"][task] = row
                    write_results_atomically(args.output, results)
                    raise RuntimeError(
                        f"aborting after {consecutive_failures} consecutive failed arms; "
                        f"partial results written to {args.output}"
                    ) from exc
            print(f"[{index + 1}/{len(tasks)}] {task} {tag}: {row[tag]}", flush=True)
        results["tasks"][task] = row
        write_results_atomically(args.output, results)

    print(f"written {args.output}", flush=True)


if __name__ == "__main__":
    main()
