"""Step 0b — admission gate: does the teacher still work in a held-out kitchen?

Preprocessing is copied verbatim from robocasa-benchmark/openpi
examples/robocasa/main.py:114-152 (state concat order, resize_with_pad ->
convert_to_uint8, prompt source). The ONLY deviation is pinning the kitchen via
layout_and_style_ids while holding obj_instance_split fixed, so that scene is
the single varied factor between arm A and arm B.
"""
import collections, json, os, pathlib, time
import numpy as np
import gymnasium as gym
import robocasa  # noqa: F401
from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon
from openpi_client import websocket_client_policy as _wcp
from openpi_client import image_tools
from robocasa.utils.env_utils import convert_action

HOST = os.environ.get("PI_HOST", "127.0.0.1")
PORT = int(os.environ.get("PI_PORT", "8010"))
N = int(os.environ.get("N_TRIALS", "5"))
RESIZE = 224
REPLAN = int(os.environ.get("REPLAN_STEPS", "5"))
SEED = int(os.environ.get("SEED", "7"))
OUT = os.environ.get("OUT_JSON", "/tmp/step0b_result.json")
A = tuple(int(x) for x in os.environ.get("SCENE_A", "1,1").split(","))
B = tuple(int(x) for x in os.environ.get("SCENE_B", "7,7").split(","))
TASKS = os.environ["TASKS"].split(",") if os.environ.get("TASKS") else list(TASK_SET_REGISTRY["atomic_seen"])

print(f"tasks={len(TASKS)} A={A} B={B} trials={N} replan={REPLAN}", flush=True)
client = _wcp.WebsocketClientPolicy(HOST, PORT)
print(f"connected {HOST}:{PORT}", flush=True)


def rollout(env_name, scene, n):
    horizon = get_task_horizon(env_name)
    env = gym.make(f"robocasa/{env_name}", split=None, obj_instance_split="target",
                   layout_and_style_ids=[scene], seed=SEED)
    succ = 0
    for ep in range(n):
        obs, _ = env.reset()
        task_lang = obs["annotation.human.task_description"]
        plan = collections.deque()
        t, done = 0, False
        while t < horizon:
            img = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                np.ascontiguousarray(obs["video.robot0_agentview_left"]), RESIZE, RESIZE))
            wri = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                np.ascontiguousarray(obs["video.robot0_eye_in_hand"]), RESIZE, RESIZE))
            rig = image_tools.convert_to_uint8(image_tools.resize_with_pad(
                np.ascontiguousarray(obs["video.robot0_agentview_right"]), RESIZE, RESIZE))
            if not plan:
                state = np.concatenate((
                    obs["state.end_effector_position_relative"],
                    obs["state.end_effector_rotation_relative"],
                    obs["state.base_position"],
                    obs["state.base_rotation"],
                    obs["state.gripper_qpos"],
                ), axis=0)
                chunk = client.infer({
                    "observation/image": img, "observation/wrist_image": wri,
                    "observation/right_image": rig, "observation/state": state,
                    "prompt": task_lang,
                })["actions"]
                plan.extend(chunk[:REPLAN])
            obs, _, _, _, info = env.step(convert_action(plan.popleft()))
            if info.get("success"):
                done = True
                break
            t += 1
        succ += int(done)
        print(f"    ep{ep}: {'OK' if done else 'fail'} t={t}/{horizon}", flush=True)
    env.close()
    return succ


res = {"sceneA": list(A), "sceneB": list(B), "n_trials": N, "tasks": {}}
for i, task in enumerate(TASKS):
    row = {}
    for tag, sc in (("A", A), ("B", B)):
        t0 = time.time()
        try:
            s = rollout(task, sc, N)
            row[tag] = {"succ": s, "n": N, "sr": s / N, "wall_s": round(time.time() - t0, 1)}
        except Exception as e:  # noqa: BLE001
            import traceback; traceback.print_exc()
            row[tag] = {"error": f"{type(e).__name__}: {str(e)[:150]}"}
        print(f"[{i+1}/{len(TASKS)}] {task} {tag}: {row[tag]}", flush=True)
    res["tasks"][task] = row
    pathlib.Path(OUT).write_text(json.dumps(res, indent=1))

ok = [(t, r) for t, r in res["tasks"].items() if "sr" in r.get("A", {}) and "sr" in r.get("B", {})]
if ok:
    a = sum(r["A"]["succ"] for _, r in ok); b = sum(r["B"]["succ"] for _, r in ok); n = len(ok) * N
    res["summary"] = {"n_tasks": len(ok), "sr_A": a/n, "sr_B": b/n, "gap_pp": (a-b)/n*100}
    print(f"\n=== SUMMARY tasks={len(ok)} SR_A={a/n:.3f} SR_B={b/n:.3f} gap={100*(a-b)/n:+.1f}pp ===")
pathlib.Path(OUT).write_text(json.dumps(res, indent=1))
print("written", OUT)
