"""Served-block analysis of the round-5 gated-follow arms (inits 20-29 only; every record filtered by init at parse time).

Why does GR00T lose 9-15 pp with gated follow while pi0.5 does not?  For the follow arms and their same-batch controls:
  * dose: follow blocks per episode and paired outcome flips vs the control conditioned on the dose;
  * where: step, gripper stage, rows-to-event and valve margin at the follow blocks;
  * seam: action jump between the last executed control of the blind block and the first control of the follow block,
    against the ordinary seams (anchor->blind, blind->fresh look) of the same arm;
  * aftermath: judge diagnostics at the look that follows a follow block (distance to library ``dnn``, pace ``lag``,
    forced-miss rate) vs looks after an ordinary blind block, and the no-progress / escalation reaction.
Nothing task-indexed is used; tasks appear only as pairing keys.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

R5 = Path("/home/weiland/trace_runs/os_closed_loop/r09_fable_r5")
OUT = Path(__file__).resolve().parents[1] / "out"
ARMS = {"pi05": ("r9f5_pi05_l10_50_esc", ["r9f5_pi05_l10_50_esc_fg"]),
        "groot": ("r9f5_groot_l10_50_np", ["r9f5_groot_l10_50_np_fg", "r9f5_groot_l10_50_np_fv"])}
INITS = range(20, 30)


def journal(run):
    ok = {}
    for line in (run / "client" / "journal.jsonl").open():
        r = json.loads(line)
        if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
            t, i = int(r["task_uid"].split(":")[-2]), int(r["task_uid"].split(":")[-1])
            if i in INITS:
                ok[(t, i)] = bool(r.get("success"))
    return ok


def decisions(run):
    eps = defaultdict(list)
    for f in sorted(run.glob("server_*/decisions_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            if r.get("ev") != "dec":
                continue
            t, i = int(r["uid"].split(":")[-2]), int(r["uid"].split(":")[-1])
            if i not in INITS:
                continue
            ex = r.get("extras") or {}
            head = np.asarray(r.get("served_head") or np.full((5, 7), np.nan), np.float32)
            eps[(t, i)].append(dict(step=int(r["step"]), vision=bool(r.get("vision")), hit=bool(r.get("hit", True)), src=r.get("src"),
                                    follow=float(ex.get("os_sf_source", 0) or 0) > 0, delta=ex.get("os_sf_delta", np.nan),
                                    radius=ex.get("os_sf_radius", np.nan), mode1=ex.get("os_sf_mode1", np.nan),
                                    rte=ex.get("os_sf_rows_to_event", np.nan), dnn=ex.get("dnn", np.nan), lag=ex.get("lag", np.nan),
                                    force=ex.get("os_force_miss", np.nan), reason=ex.get("os_reason", np.nan),
                                    esc=ex.get("r9o_escalated", np.nan), head=head[:5, :7] if head.ndim == 2 else np.full((5, 7), np.nan)))
    for k in eps:
        eps[k].sort(key=lambda d: d["step"])
    return eps


def seam(prev, cur):
    return float(np.sqrt(np.mean((cur["head"][0, :6] - prev["head"][4, :6]) ** 2)))


def analyse(model):
    ctrl_name, follow_arms = ARMS[model]
    ctrl_ok = journal(R5 / "runs" / ctrl_name)
    ctrl_dec = decisions(R5 / "runs" / ctrl_name)
    report = {}
    for arm in follow_arms:
        ok, dec = journal(R5 / "runs" / arm), decisions(R5 / "runs" / arm)
        rows = []
        for key, ds in dec.items():
            n_follow = sum(d["follow"] for d in ds)
            steps = [d["step"] for d in ds if d["follow"]]
            rows.append(dict(task=key[0], init=key[1], n_dec=len(ds), n_follow=n_follow, first_follow=min(steps) if steps else -1,
                             success=ok.get(key), ctrl_success=ctrl_ok.get(key),
                             follow_frac_pos=np.mean([s / len(ds) for s in steps]) if steps else np.nan))
        ep = pd.DataFrame(rows)
        # --- dose-response: paired flips vs control by follow count
        bins = pd.cut(ep.n_follow, [-1, 0, 2, 5, 1000], labels=["0", "1-2", "3-5", "6+"])
        dose = ep.groupby(bins, observed=False).apply(lambda g: pd.Series(dict(n=len(g), sr=g.success.mean(), ctrl_sr=g.ctrl_success.mean(),
                                                                                 lost=int((g.ctrl_success & ~g.success).sum()), gained=int((~g.ctrl_success & g.success).sum()))))
        # --- where the follow blocks are
        fdec = [d for ds in dec.values() for d in ds if d["follow"]]
        where = dict(n_follow=len(fdec), share=len(fdec) / sum(len(ds) for ds in dec.values()),
                     median_step=float(np.median([d["step"] for d in fdec])), mode_closed_share=float(np.nanmean([d["mode1"] for d in fdec])),
                     rows_to_event_median=float(np.nanmedian([d["rte"] for d in fdec])), rows_to_event_le2=float(np.nanmean([d["rte"] <= 2 for d in fdec])),
                     valve_margin_median=float(np.nanmedian([d["delta"] / d["radius"] for d in fdec if d["radius"]])),
                     valve_margin_gt_half=float(np.nanmean([d["delta"] / d["radius"] > .5 for d in fdec if d["radius"]])))
        # --- seams and aftermath
        seams = defaultdict(list)
        after = defaultdict(lambda: defaultdict(list))
        for key, ds in dec.items():
            for p, c in zip(ds[:-1], ds[1:]):
                if not (np.isfinite(p["head"]).all() and np.isfinite(c["head"]).all()):
                    continue
                if c["follow"]:
                    kind = "blind->follow"
                elif c["src"] == "cache_blind":
                    kind = "anchor->blind"
                elif c["vision"] and c["hit"] and p["follow"]:
                    kind = "follow->look(hit)"
                elif c["vision"] and c["hit"] and p["src"] == "cache_blind":
                    kind = "blind->look(hit)"
                else:
                    continue
                seams[kind].append(seam(p, c))
                if c["vision"]:
                    tag = "after follow" if p["follow"] else ("after blind" if p["src"] == "cache_blind" else None)
                    if tag:
                        after[tag]["dnn"].append(c["dnn"]); after[tag]["lag"].append(c["lag"]); after[tag]["miss"].append(float(not c["hit"]))
                        after[tag]["reason4"].append(float(c["reason"] == 4)); after[tag]["reason91"].append(float(c["reason"] == 91))
        seam_tab = {k: dict(n=len(v), mean=float(np.mean(v)), p90=float(np.quantile(v, .9))) for k, v in seams.items()}
        after_tab = {k: {m: float(np.nanmean(v)) for m, v in d.items()} | {"n": len(d["miss"])} for k, d in after.items()}
        # control seams for reference
        cseams = defaultdict(list)
        for ds in ctrl_dec.values():
            for p, c in zip(ds[:-1], ds[1:]):
                if not (np.isfinite(p["head"]).all() and np.isfinite(c["head"]).all()):
                    continue
                if c["src"] == "cache_blind":
                    cseams["anchor->blind"].append(seam(p, c))
                elif c["vision"] and c["hit"] and p["src"] == "cache_blind":
                    cseams["blind->look(hit)"].append(seam(p, c))
        report[arm] = dict(episodes=ep.to_dict("records"), dose=dose.reset_index().rename(columns={"n_follow": "dose"}).to_dict("records"),
                           where=where, seams=seam_tab, control_seams={k: dict(n=len(v), mean=float(np.mean(v))) for k, v in cseams.items()},
                           aftermath=after_tab, sr=float(ep.success.mean()), ctrl_sr=float(ep.ctrl_success.mean()))
        print(f"\n== {arm}  SR {report[arm]['sr']:.3f} vs control {report[arm]['ctrl_sr']:.3f}")
        print("dose-response (paired vs control):"); print(dose.to_string())
        print("where:", json.dumps({k: round(v, 3) for k, v in where.items()}))
        print("seams (RMS jump, motion dims):", json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in seam_tab.items()}))
        print("control seams:", json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in report[arm]["control_seams"].items()}))
        print("look after follow vs after blind:", json.dumps({k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in after_tab.items()}))
    return report


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = {m: analyse(m) for m in ("pi05", "groot")}
    (OUT / "follow_analysis.json").write_text(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
