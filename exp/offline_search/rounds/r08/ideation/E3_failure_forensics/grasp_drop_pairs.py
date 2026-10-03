"""E3 preliminary: grasp-attempt alignment, drop mechanism, onset timing and paired divergence (CPU, read-only).

Reads /tmp/r8_E3_failure_forensics/{compact/*.npz, out/episodes.csv, out/pairs.csv} written by
extract_compact.py and forensics.py. Prints summary tables used in the E3 proposal section 5.
"""
import collections
import csv
import glob
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import forensics as F  # noqa: E402

OUT = F.OUT


def attempts(d, k, carry_d):
    """Close-command onsets near object k; success = object lifted with gripper closed within 40 controls."""
    close = d["act"][:, 6] > 0
    on = np.flatnonzero(close[1:] & ~close[:-1]) + 1
    t0 = F.ref_index(d)
    carried, _ = F.carry_series(d, k, carry_d)
    rows = []
    for t in on:
        rel = d["eef"][t] - d["obj"][t, k]
        if np.linalg.norm(rel) > carry_d + 0.03:
            continue
        ok = bool(carried[t:t + 40].any())
        rows.append(dict(t=int(t), xy=float(np.linalg.norm(rel[:2])), dz=float(rel[2]), ok=ok,
                         src=str(d["source"][t])))
    return rows


def main():
    eps = [F.load(f) for f in sorted(glob.glob(str(F.SRC / "*.npz")))]
    for d in eps:
        tm = F.TASKMAP[d["arm_full"]]
        d["task_id"], d["init"] = tm["task_id"], tm["init"]
    eps = [d for d in eps if d["end"]["reason"] != "exception"]
    carry = {tuple(json.loads(k)): v for k, v in json.load(open(OUT / "carry_calibration.json")).items()}
    lab = {(r["cell"], r["arm"], r["rep"], r["uid"]): r for r in csv.DictReader(open(OUT / "episodes.csv"))}

    # 1. grasp attempts
    att = collections.defaultdict(list)
    for d in eps:
        cd = carry.get(F.task_key(d), 0.10)
        for p in d["preds"]:
            if p[0] in ("in", "on") and p[1] in d["objects"]:
                for a in attempts(d, d["objects"].index(p[1]), cd):
                    a.update(arm=d["arm"], model=d["model"], suite=d["suite"], succ=d["end"]["success"])
                    att[(d["arm"])].append(a)
    print("== grasp attempts near a goal object (close-command onsets) ==")
    for arm, rows in att.items():
        ok = [r for r in rows if r["ok"]]
        bad = [r for r in rows if not r["ok"]]
        print(arm, "attempts", len(rows), "first-try-ok share", round(len(ok) / len(rows), 3),
              "| xy offset median ok/bad", round(np.median([r["xy"] for r in ok]) * 100, 2),
              round(np.median([r["xy"] for r in bad]) * 100, 2) if bad else None, "cm",
              "| dz median ok/bad", round(np.median([r["dz"] for r in ok]) * 100, 2),
              round(np.median([r["dz"] for r in bad]) * 100, 2) if bad else None, "cm")
        for suite in ("l10", "sp"):
            s = [r for r in rows if r["suite"] == suite]
            print("   ", suite, "attempts", len(s), "fail share", round(1 - np.mean([r["ok"] for r in s]), 3))
    # AUROC of xy offset for failed attempts (pooled per arm)
    for arm, rows in att.items():
        y = np.array([not r["ok"] for r in rows])
        x = np.array([r["xy"] for r in rows])
        if y.any() and (~y).any():
            from itertools import product
            pos, neg = x[y], x[~y]
            auc = np.mean([(p > n) + 0.5 * (p == n) for p, n in product(pos, neg)])
            print("   ", arm, "AUROC(xy offset -> failed attempt)", round(float(auc), 3), "n_fail", int(y.sum()))

    # 2. drop mechanism: gripper command at the release control
    print("== drops: command at release ==")
    mech = collections.Counter()
    for d in eps:
        r = lab.get((d["cell"], d["arm"], str(d["rep"]), d["uid"]))
        if r is None or r["label"] not in ("drop", "drop_regrasp_fail"):
            continue
        det = json.loads(r["preds_detail"])
        cd = carry.get(F.task_key(d), 0.10)
        for x in det:
            if x["label"] not in ("drop", "drop_regrasp_fail"):
                continue
            k = d["objects"].index(x["pred"].split()[1])
            carried, _ = F.carry_series(d, k, cd)
            # end of first carry segment
            t = int(np.flatnonzero(carried)[0])
            while t < len(carried) and carried[t]:
                t += 1
            cmd = d["act"][max(t - 3, 0):t + 2, 6]
            opened = bool((cmd < 0).any())
            w = d["grip"][t - 1, 0] - d["grip"][t - 1, 1]
            z_above = d["obj"][t - 1, k, 2] - d["obj"][F.ref_index(d), k, 2]
            src = str(d["source"][t - 1])
            mech[(d["arm"], "open_cmd" if opened else "closed_cmd_slip")] += 1
            print(f"  {d['cell']:<14} {d['arm']:<4} task{d['task_id']} t={t}/{len(carried)} cmd={np.round(cmd, 2).tolist()} "
                  f"width={w * 100:.1f}cm z_above={z_above * 100:.1f}cm src={src}")
    print(dict(mech))

    # 3. onset timing
    print("== onset (control index of the decisive event) ==")
    for arm in ("A", "P10"):
        rows = [r for r in lab.values() if r["arm"] == arm and r["success"] == "False" and r["reason"] != "exception"
                and r.get("onset") not in (None, "", "None")]
        on = np.array([int(r["onset"]) for r in rows])
        n = np.array([int(r["n_controls"]) for r in rows])
        print(arm, "n", len(rows), "onset median", int(np.median(on)), "IQR", np.percentile(on, [25, 75]).astype(int).tolist(),
              "controls wasted after onset median", int(np.median(n - on)), "share of episode", round(float(np.median((n - on) / n)), 2))
    succ_len = collections.defaultdict(list)
    for r in lab.values():
        if r["success"] == "True":
            succ_len[r["arm"]].append(int(r["n_controls"]))
    print("success length median", {k: int(np.median(v)) for k, v in succ_len.items()})

    # 4. paired divergence
    print("== paired eef divergence (first control with eef distance > 3 cm) ==")
    pr = list(csv.DictReader(open(OUT / "pairs.csv")))
    for pair in ("AA", "AP", "PP"):
        rows = [r for r in pr if r["pair"] == pair]
        dv = np.array([int(r["diverge"]) if r["diverge"] not in ("", "None") else 10**6 for r in rows])
        sd = np.array([int(r["first_state_diff"]) if r["first_state_diff"] not in ("", "None") else 10**6 for r in rows])
        ad = np.array([int(r["first_action_diff"]) if r["first_action_diff"] not in ("", "None") else 10**6 for r in rows])
        print(pair, "pairs", len(rows), "diverge median", int(np.median(dv)), "p25/p75",
              np.percentile(dv, [25, 75]).astype(int).tolist(),
              "| first sim-state difference median", int(np.median(sd)),
              "| first issued-action difference median", int(np.median(ad)),
              "| state differs before action", int(np.sum(sd < ad)), "/", len(rows),
              "| never differ", int(np.sum(sd == 10**6)))
        for suite in ("l10", "sp"):
            s = dv[[r["cell"].split("_")[1] == suite for r in rows]]
            print("    ", suite, "diverge median", int(np.median(s)))
    # outcome discordance vs divergence for AP pairs of the same replicate seed
    rows = [r for r in pr if r["pair"] == "AP"]
    disc = [r for r in rows if r["succ_a"] != r["succ_b"]]
    print("AP pairs discordant outcome", len(disc), "/", len(rows))


if __name__ == "__main__":
    main()


def window_share():
    """Share of post-wait controls/decisions inside the sim-truth grasp window (eef near an un-lifted, unsatisfied
    goal object). Sizes a privileged 'oracle grasp-window call' diagnostic arm."""
    carry = {tuple(json.loads(k)): v for k, v in json.load(open(OUT / "carry_calibration.json")).items()}
    acc = collections.defaultdict(list)
    for f in sorted(glob.glob(str(F.SRC / "*.npz"))):
        d = F.load(f)
        if d["end"]["reason"] == "exception":
            continue
        tm = F.TASKMAP[d["arm_full"]]
        d["task_id"] = tm["task_id"]
        cd = carry.get(F.task_key(d), 0.10)
        t0 = F.ref_index(d)
        win = np.zeros(len(d["eef"]), bool)
        for i, p in enumerate(d["preds"]):
            if p[0] in ("in", "on") and p[1] in d["objects"]:
                k = d["objects"].index(p[1])
                carried, lifted = F.carry_series(d, k, cd)
                near = np.linalg.norm(d["eef"] - d["obj"][:, k], axis=1) < cd
                win |= near & ~lifted & ~d["pred"][:, i].astype(bool)
        w = win[t0:]
        dec = w[: len(w) // 5 * 5].reshape(-1, 5).any(axis=1)
        acc[(d["suite"], d["arm"], bool(d["end"]["success"]))].append((w.mean(), dec.mean(), len(w)))
    for k, v in sorted(acc.items()):
        v = np.array(v)
        print(k, "n", len(v), "control share", round(float(v[:, 0].mean()), 3), "decision share",
              round(float(v[:, 1].mean()), 3), "controls", int(np.median(v[:, 2])))
