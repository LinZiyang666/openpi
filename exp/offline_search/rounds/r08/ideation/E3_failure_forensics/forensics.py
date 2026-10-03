"""E3 preliminary: automatic failure forensics on the compact R6 P3 v2 physical series (CPU, read-only).

Input: /tmp/r8_E3_failure_forensics/compact/*.npz from extract_compact.py (arms A = pure cache, P10 = pure policy
with 10-control commitment; test inits 0-1; 3 replicates). Output: per-episode events/labels CSV, paired A-vs-P10
divergence CSV, determinism probe JSON, and printed summaries.

Labels use kinematics + goal predicates only (the P3 v2 contact names are not index-aligned, see extract_compact).
Thresholds are explicit constants; carry distance is calibrated per task from successful episodes.
"""
import collections
import csv
import glob
import json
import pathlib
import re

import numpy as np

SRC = pathlib.Path("/tmp/r8_E3_failure_forensics/compact")
OUT = pathlib.Path("/tmp/r8_E3_failure_forensics/out")
LIFT = 0.03          # m above initial z counts as lifted
MOVED = 0.05         # m displacement of a non-target object counts as disturbed
TAIL = 100           # controls examined for stall/oscillation before the step cap
STALL_PATH = 0.03    # m eef path length over the tail => stall
OSC_RATIO = 0.15     # net/path over the tail below this with path > 0.15 m => oscillation
DIVERGE = 0.03       # m eef distance for paired divergence


def load(f):
    z = np.load(f, allow_pickle=False)
    name = pathlib.Path(f).stem
    arm, uid = name.split("__")
    m = re.match(r"r6p3v2_(\w+?)_(l10|sp)_(50|500)_(\w+)_r(\d)", arm)
    d = {k: z[k] for k in z.files}
    d.update(arm_full=name, model=m.group(1), suite=m.group(2), lib=m.group(3), arm=m.group(4), rep=int(m.group(5)),
             cell=f"{m.group(1)}_{m.group(2)}_{m.group(3)}", uid=uid, attempt=uid.split("_a")[-1])
    d["end"] = json.loads(str(d["end"]))
    d["det"] = json.loads(str(d["det"]))
    d["decisions"] = json.loads(str(d["decisions"]))
    d["objects"] = [str(x) for x in d["objects"]]
    d["preds"] = [json.loads(str(p)) for p in d["preds"]]
    d["fixtures"] = [str(x) for x in d["fixtures"]]
    return d


def dest_pos(d, region):
    base = re.match(r"(.+?_\d+)", region)
    base = base.group(1) if base else region
    if base in d["objects"]:
        return d["obj"][:, d["objects"].index(base)]
    for fb in d["fixtures"]:
        if fb.startswith(base):
            return d["fix"][:, d["fixtures"].index(fb)]
    return None


def build_taskmap(path=pathlib.Path("/tmp/r8_E3_failure_forensics/task_map.json")):
    """uid -> task_id/init/seed from the first line (attempt_start) of each controls.jsonl."""
    if not path.exists():
        runs = pathlib.Path("/home/weiland/trace_runs/os_closed_loop/r06_p3_pilot/runs")
        m = {}
        for f in glob.glob(str(SRC / "*.npz")):
            arm, uid = pathlib.Path(f).stem.split("__")
            with open(runs / arm / "client_telemetry" / uid / "controls.jsonl") as fh:
                j = json.loads(fh.readline())
            m[pathlib.Path(f).stem] = dict(task_id=j["task_id"], init=j["orig_init_state_idx"],
                                           seed=j["environment_seed"], task=j["task"])
        path.write_text(json.dumps(m))
    return json.load(open(path))


TASKMAP = build_taskmap()


def task_key(d):
    return (d["suite"], d["task_id"])


def ref_index(d):
    """First non-wait control: LIBERO objects spawn above the table and settle during the 10 wait controls."""
    w = d["wait"].astype(bool)
    return int(np.argmax(~w)) if (~w).any() else 0


def carry_series(d, k, carry_d):
    z0 = d["obj"][ref_index(d), k, 2]
    lifted = d["obj"][:, k, 2] - z0 > LIFT
    near = np.linalg.norm(d["eef"] - d["obj"][:, k], axis=1) < carry_d
    closed = d["act"][:, 6] > 0
    return lifted & near & closed, lifted


def calibrate(eps):
    """Per task: p95 eef-object distance while a goal object is lifted, over successful episodes."""
    acc = collections.defaultdict(list)
    for d in eps:
        if not d["end"]["success"]:
            continue
        for p in d["preds"]:
            if p[0] in ("in", "on") and p[1] in d["objects"]:
                k = d["objects"].index(p[1])
                t0 = ref_index(d)
                lifted = d["obj"][:, k, 2] - d["obj"][t0, k, 2] > LIFT
                moving = np.r_[False, np.linalg.norm(np.diff(d["obj"][:, k], axis=0), axis=1) > 0.002]
                sel = lifted & moving & (d["act"][:, 6] > 0)
                if sel.any():
                    acc[task_key(d)].extend(np.linalg.norm(d["eef"] - d["obj"][:, k], axis=1)[sel].tolist())
    return {t: float(np.clip(np.percentile(v, 95) + 0.02, 0.08, 0.15)) for t, v in acc.items()}


def analyse(d, carry_d):
    n = len(d["control"])
    wait = d["wait"].astype(bool)
    t0 = int(np.argmax(~wait)) if (~wait).any() else 0
    res = dict(cell=d["cell"], arm=d["arm"], rep=d["rep"], task=json.dumps(d["preds"]), uid=d["uid"],
               task_id=d["task_id"], init=d["init"], attempt=d["attempt"],
               success=bool(d["end"]["success"]), n_controls=n, reason=d["end"]["reason"])
    pred_end = d["pred"][-1]
    obj_labels, onsets = [], []
    first_near_all = []
    for i, p in enumerate(d["preds"]):
        ps = d["pred"][:, i]
        info = dict(pred=" ".join(p), final=bool(ps[-1]), ever=bool(ps.any()),
                    first_true=int(np.argmax(ps)) if ps.any() else None)
        if p[0] in ("in", "on") and p[1] in d["objects"]:
            k = d["objects"].index(p[1])
            dist = np.linalg.norm(d["eef"] - d["obj"][:, k], axis=1)
            carried, lifted = carry_series(d, k, carry_d)
            near = dist < carry_d
            info.update(dmin=float(dist.min()), first_near=int(np.argmax(near)) if near.any() else None,
                        first_lift=int(np.argmax(carried)) if carried.any() else None,
                        carried_frac=float(carried.mean()))
            if near.any():
                first_near_all.append(int(np.argmax(near)))
            # carry segments
            segs, start = [], None
            for t in range(n):
                if carried[t] and start is None:
                    start = t
                if not carried[t] and start is not None:
                    segs.append((start, t))
                    start = None
            if start is not None:
                segs.append((start, n))
            info["carry_segments"] = len(segs)
            dp = dest_pos(d, p[2])
            dxy_end = float(np.linalg.norm(d["obj"][-1, k, :2] - dp[-1, :2])) if dp is not None else None
            info["dest_xy_end"] = dxy_end
            # grasp attempts: close command onsets while near the object
            close = d["act"][:, 6] > 0
            onsets_close = np.flatnonzero(close[1:] & ~close[:-1]) + 1
            info["close_near"] = int(sum(1 for t in onsets_close if dist[t] < carry_d + 0.03))
            if ps[-1]:
                lab, onset = "ok", None
            elif ps.any():
                lab, onset = "undone", int(len(ps) - 1 - np.argmax(ps[::-1]))
            elif not near.any():
                lab, onset = "never_reached", None
            elif not carried.any():
                lab, onset = "grasp_miss", info["first_near"]
            elif segs[-1][1] == n:
                lab, onset = "held_not_placed", segs[-1][0]
            else:
                rel = segs[-1][1]
                # released while carried: near destination => misplacement, else drop
                dxy_rel = float(np.linalg.norm(d["obj"][rel, k, :2] - dp[rel, :2])) if dp is not None else None
                info["dest_xy_release"] = dxy_rel
                if dxy_rel is not None and dxy_rel < 0.10:
                    lab, onset = "misplace", rel
                else:
                    lab, onset = ("drop_regrasp_fail" if len(segs) > 1 else "drop"), segs[0][1]
            info["label"], info["onset"] = lab, onset
        else:
            info["label"] = "ok" if ps[-1] else ("fixture_undone" if ps.any() else "fixture_not_done")
            info["onset"] = None
        obj_labels.append(info)
    # non-target disturbance / wrong object
    goal_objs = {p[1] for p in d["preds"] if len(p) > 1}
    goal_objs |= {m.group(1) for p in d["preds"] if len(p) > 2 for m in [re.match(r"(.+?_\d+)", p[2])] if m}
    wrong, disturbed = [], []
    for k, o in enumerate(d["objects"]):
        if o in goal_objs:
            continue
        t0 = ref_index(d)
        lift = d["obj"][:, k, 2] - d["obj"][t0, k, 2] > LIFT
        dist = np.linalg.norm(d["eef"] - d["obj"][:, k], axis=1)
        if (lift & (dist < 0.12)).any():
            wrong.append((o, int(np.argmax(lift & (dist < 0.12)))))
        if np.linalg.norm(d["obj"][-1, k] - d["obj"][t0, k]) > MOVED:
            disturbed.append(o)
    # tail motion
    tail = d["eef"][-TAIL:]
    path = float(np.linalg.norm(np.diff(tail, axis=0), axis=1).sum())
    net = float(np.linalg.norm(tail[-1] - tail[0]))
    grip_toggles = int((np.diff((d["act"][-TAIL:, 6] > 0).astype(int)) != 0).sum())
    res.update(tail_path=path, tail_net=net, tail_grip_toggles=grip_toggles,
               wrong_object=";".join(f"{o}@{t}" for o, t in wrong), disturbed=";".join(disturbed),
               t0=t0)
    # episode label: first failing predicate in attempt order
    failing = [x for x in obj_labels if x["label"] != "ok"]
    if res["success"]:
        res["label"] = "success"
    elif not failing:
        res["label"] = "all_preds_true_but_fail"
    else:
        failing.sort(key=lambda x: (x.get("first_near") is None, x.get("first_near") or 0))
        lab = failing[0]["label"]
        if lab == "never_reached" and wrong:
            lab = "wrong_object"
        res["label"] = lab
        res["onset"] = failing[0]["onset"]
        res["subgoals_done"] = sum(1 for x in obj_labels if x["label"] == "ok")
        res["subgoals"] = len(obj_labels)
    res["tail"] = ("stall" if path < STALL_PATH else
                   "oscillation" if (path > 0.15 and net / max(path, 1e-9) < OSC_RATIO) else "moving")
    res["preds_detail"] = json.dumps(obj_labels)
    return res


def paired(eps):
    """Per (cell, task, init): divergence time of eef paths, A-vs-P10 and within-arm."""
    groups = collections.defaultdict(list)
    for d in eps:
        if d["end"]["reason"] == "exception":
            continue
        init = d["uid"]
        groups[(d["cell"], d["task_id"], d["init"])].append(d)
    rows = []
    for (cell, task, init), g in groups.items():
        for i in range(len(g)):
            for j in range(i + 1, len(g)):
                a, b = g[i], g[j]
                n = min(len(a["eef"]), len(b["eef"]))
                dist = np.linalg.norm(a["eef"][:n] - b["eef"][:n], axis=1)
                over = np.flatnonzero(dist > DIVERGE)
                act_diff = np.abs(a["act"][:n] - b["act"][:n]).max(axis=1)
                dig_diff = np.flatnonzero(a["digest"][:n] != b["digest"][:n])
                act_first = np.flatnonzero(act_diff > 0)
                pair = "".join(sorted([a["arm"][0], b["arm"][0]]))  # AA, AP, PP
                rows.append(dict(cell=cell, task=task, init=init, pair=pair, arm_a=a["arm"], rep_a=a["rep"],
                                 arm_b=b["arm"], rep_b=b["rep"], succ_a=a["end"]["success"], succ_b=b["end"]["success"],
                                 diverge=int(over[0]) if len(over) else None,
                                 first_state_diff=int(dig_diff[0]) if len(dig_diff) else None,
                                 first_action_diff=int(act_first[0]) if len(act_first) else None,
                                 n=n))
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    eps = []
    for f in sorted(glob.glob(str(SRC / "*.npz"))):
        d = load(f)
        eps.append(d)
    # init id: the reset digest identifies (task, init) physically
    for d in eps:
        d["init_id"] = str(d["init_digest"])
        tm = TASKMAP[d["arm_full"]]
        d["task_id"], d["init"] = tm["task_id"], tm["init"]
    by_task = collections.defaultdict(list)
    for d in eps:
        by_task[task_key(d)].append(d)
    carry = calibrate(eps)
    rows = []
    for d in eps:
        cd = carry.get(task_key(d), 0.10)
        rows.append(analyse(d, cd))
    keys = sorted({k for r in rows for k in r})
    with open(OUT / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    prow = paired(eps)
    with open(OUT / "pairs.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prow[0].keys()))
        w.writeheader()
        w.writerows(prow)
    json.dump({json.dumps(k): v for k, v in carry.items()}, open(OUT / "carry_calibration.json", "w"), indent=1)
    det = collections.Counter()
    for d in eps:
        det.update(d["det"])
    json.dump(dict(det), open(OUT / "determinism.json", "w"))
    print("episodes", len(rows), "before==prev_after", dict(det))


if __name__ == "__main__":
    main()
