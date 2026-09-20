"""Aggregate one k of the X-WAM RoboTwin 2.0 ladder: <out>/<task>/_result.txt (last line = success rate) -> <out>/summary.json.
usage: python3 agg_robotwin.py <out dir> <k> <n per task> [video steps=50]"""
import glob, json, os, sys

out, k, ne = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
vk = int(sys.argv[4]) if len(sys.argv) > 4 else 50
r = {}
for f in sorted(glob.glob(os.path.join(out, "*", "_result.txt"))):
    t = os.path.basename(os.path.dirname(f))
    lines = [l.strip() for l in open(f) if l.strip()]
    sr = float(lines[-1])
    r[t] = {"success_rate": sr, "n": ne, "n_success": round(sr * ne)}
m = sum(v["success_rate"] for v in r.values()) / max(len(r), 1)
d = {"policy": "X-WAM robotwin_sft", "benchmark": "RoboTwin 2.0 demo_randomized (unseen instructions, seed 0)", "k": k,
     "action_denoise_steps": k, "video_denoise_steps": vk, "n_per_task": ne, "n_tasks": len(r), "macro_success_rate": m, "per_task": r}
json.dump(d, open(os.path.join(out, "summary.json"), "w"), indent=1)
print(f"XWAM-RT k={k} macro={m:.4f} tasks={len(r)} n_per_task={ne}")
