"""Load closed-loop arms (server decision logs + client journal) into per-decision / per-episode tables.
Reconstructs the served action head ([:5,:7]) from the library for each decision:
  CL0 native: action[top1]; CL1 M4 mean-5: mean(action[topk[:5]]); CL2 AWM kr5: kernel mean over the logged top-10.
Writes <scratch>/tables/<arm>_dec.pkl and <arm>_ep.pkl (pandas).
"""
import glob, json, os, sys, pathlib
import numpy as np, pandas as pd

RUN = pathlib.Path("/home/weiland/trace_runs/os_closed_loop/r02_g50/runs")
STORE = pathlib.Path("/dev/shm/offline_search_store")
OUT = pathlib.Path("/home/weiland/.claude/jobs/a607dd74/tmp/r03_ideation_A/tables")
OUT.mkdir(parents=True, exist_ok=True)
SUITE = {"sp": "spatial", "l10": "l10"}
MODEL = {"p": "pi05", "g": "groot"}


def lib_arrays(model, suite):
    d = STORE / "library" / f"{model}_{suite}" / "current"
    act = np.load(d / "action.npy", mmap_mode="r")
    head = np.asarray(act[:, :5, :7], np.float64)                      # (L,5,7)
    sig = head.reshape(-1, 7).std(0)                                   # sigma_d over library heads (metrics def)
    return dict(ep=np.load(d / "episode.npy"), step=np.load(d / "step.npy"), ep_len=np.load(d / "ep_len.npy"),
                nxt=np.load(d / "next.npy"), task=np.load(d / "task_id.npy"), head=head, sig=sig,
                prog=np.load(d / "progress.npy"))


def kernel_w(dt, kref=5):
    rel = dt - dt[0]
    ref = max(rel[min(kref, len(dt)) - 1], 1e-6)
    return np.exp(-(rel / ref) ** 2)


def load_arm(arm):
    parts = arm.split("_")                                             # oscl50_p_sp_cl2
    model, suite, cl = MODEL[parts[1]], SUITE[parts[2]], parts[3]
    L = lib_arrays(model, suite)
    decs, eps = [], []
    for f in sorted(glob.glob(str(RUN / arm / "server_*" / "decisions_*.jsonl"))):
        port = int(f.split("_")[-1].split(".")[0])
        with open(f) as fh:
            for line in fh:
                d = json.loads(line)
                ev = d.get("ev")
                if ev == "dec":
                    ex = d.get("extras") or {}
                    topk = np.asarray(d["topk"], np.int64)
                    sc = np.asarray(d["scores"], np.float64)
                    if cl == "cl0":
                        w = np.array([1.0]); rows = topk[:1]
                    elif cl == "cl1":
                        rows = topk[:5]; w = np.ones(len(rows))
                    elif cl == "cl3":                                  # StuckRecovery wrapper: S_i <= 0, w_i = exp(S_i - S_0)
                        rows = topk; w = np.exp(sc - sc[0])
                    else:                                              # AWM (cl2): kernel on logged top-10
                        rows = topk; w = kernel_w(-sc, 5)
                    w = w / w.sum()
                    h = np.tensordot(w, L["head"][rows], 1)             # (5,7) served head (approx for cl2)
                    hs = h / L["sig"]
                    r = dict(uid=d["uid"], task=int(d["task_id"]), init=int(d["init"]), step=int(d["step"]),
                             port=port, conn=d["conn"], top1=int(d["top1"]), conf=float(d["conf"]),
                             native_top1=(int(d["native_top1"]) if d.get("native_top1") is not None else -1),
                             agree=d.get("agree"), infer_ms=float(d["infer_ms"]), q_us=float(d["q_us"]),
                             lib_ep=int(L["ep"][topk[0]]), lib_step=int(L["step"][topk[0]]),
                             lib_eplen=int(L["ep_len"][topk[0]]), lib_prog=float(L["prog"][topk[0]]),
                             top1_next=int(L["nxt"][topk[0]]),
                             nat_lib_ep=(int(L["ep"][d["native_top1"]]) if d.get("native_top1") is not None else -1),
                             nat_lib_step=(int(L["step"][d["native_top1"]]) if d.get("native_top1") is not None else -1),
                             g0=float(np.sign(h[0, 6])), g4=float(np.sign(h[4, 6])), gmean=float(h[:, 6].mean()),
                             gabs=float(np.abs(h[:, 6]).mean()),
                             tnorm=float(np.sqrt((hs[:, :3] ** 2).sum(1)).mean()),      # translation speed (sigma units)
                             rnorm=float(np.sqrt((hs[:, 3:6] ** 2).sum(1)).mean()),     # rotation speed
                             topk_str=",".join(map(str, topk[:10])),
                             # top-1 vote fraction on gripper sign among the set used
                             gvote=float((w * np.sign(L["head"][rows][:, 0, 6])).sum()),
                             # same-episode share within the used set
                             same_ep=float((w * (L["ep"][rows] == L["ep"][rows[0]])).sum()),
                             score1=float(sc[0]), margin=float(sc[0] - sc[1]) if len(sc) > 1 else np.nan,
                             w_eff=float(1.0 / (w ** 2).sum()))
                    for k, v in ex.items():
                        if isinstance(v, (int, float)) and v is not None and not isinstance(v, bool):
                            r["x_" + k] = float(v)
                        elif isinstance(v, bool):
                            r["x_" + k] = float(v)
                    decs.append(r)
                elif ev == "episode":
                    eps.append(dict(uid=d["uid"], task=int(d["task_id"]), init=int(d["init"]), success=bool(d["success"]),
                                    n_dec=int(d["n_decisions"]), port=port, dur=float(d["t_end"] - d["t_start"])))
    D = pd.DataFrame(decs).sort_values(["uid", "step"]).reset_index(drop=True)
    E = pd.DataFrame(eps).drop_duplicates("uid").set_index("uid")
    J = pd.read_json(RUN / arm / "client" / "journal.jsonl", lines=True)
    J = J[J.accepted & J.status.isin(["done", "failed"])].drop_duplicates("task_uid").set_index("task_uid")
    E["j_success"] = J.reindex(E.index)["success"].astype(float)
    E["j_dur"] = J.reindex(E.index)["duration_s"]
    E["arm"] = arm
    D["arm"] = arm
    D["success"] = D.uid.map(E.success)
    D.to_pickle(OUT / f"{arm}_dec.pkl")
    E.to_pickle(OUT / f"{arm}_ep.pkl")
    print(arm, "decisions", len(D), "episodes", len(E), "SR", E.success.mean().round(3),
          "journal agree", float((E.success.astype(float) == E.j_success).mean()).__round__(3), "sigma", np.round(L["sig"], 3))


if __name__ == "__main__":
    arms = sys.argv[1:] or [p.name for p in sorted(RUN.iterdir()) if p.is_dir() and (RUN / p.name / "summary.json").exists()]
    for a in arms:
        load_arm(a)
