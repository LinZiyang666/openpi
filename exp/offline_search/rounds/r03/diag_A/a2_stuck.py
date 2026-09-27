"""Decision-level anatomy of deadlocks: spells of identical picks, where they start (phase, gripper), served action norm,
cancellation ratio of the synthesized set, escape mechanisms; task deep-dives (sp task 6, l10 task 2)."""
import pathlib, sys, json
import numpy as np, pandas as pd

T = pathlib.Path("/home/weiland/.claude/jobs/a607dd74/tmp/r03_ideation_A/tables")
STORE = pathlib.Path("/dev/shm/offline_search_store")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 3)
SUITE = {"sp": "spatial", "l10": "l10"}


def kernel_w(dt, kref=5):
    rel = dt - dt[0]; ref = max(rel[min(kref, len(dt)) - 1], 1e-6); return np.exp(-(rel / ref) ** 2)


def add_set_features(D, suite, cl):
    d = STORE / "library" / f"pi05_{SUITE[suite]}" / "current"
    act = np.load(d / "action.npy", mmap_mode="r"); head = np.asarray(act[:, :5, :7], np.float64)
    sig = head.reshape(-1, 7).std(0); hs = head / sig
    ep = np.load(d / "episode.npy")
    canc, gagree, t1norm, spread = [], [], [], []
    for tk, sc in zip(D.topk_str.values, D.scores_list.values):
        rows = np.array([int(x) for x in tk.split(",")]); sc = np.asarray(sc)
        if cl == "cl0": rows = rows[:1]; w = np.ones(1)
        elif cl == "cl1": rows = rows[:5]; w = np.ones(5)
        else: w = kernel_w(-sc, 5)
        w = w / w.sum(); H = hs[rows]                                     # (k,5,7)
        m = np.tensordot(w, H, 1)
        num = np.linalg.norm(m[:, :6], axis=1).mean(); den = (w[:, None] * np.linalg.norm(H[:, :, :6], axis=2)).sum(0).mean()
        canc.append(num / max(den, 1e-9))
        g = np.sign(H[:, 0, 6]); gagree.append(abs((w * g).sum()))
        t1norm.append(np.linalg.norm(hs[rows[0]][:, :3], axis=1).mean())
        spread.append(np.sqrt(((H[:, :, :6] - m[:, :6]) ** 2).sum(2).mean(1) @ w))   # weighted RMS spread of members around mean
    D["cancel"] = canc; D["gagree"] = gagree; D["t1norm"] = t1norm; D["spread"] = spread
    return D


def spells(D):
    """Mark stay (top1 == prev top1) and identical-pick spells (>= 3 consecutive identical picks)."""
    D = D.sort_values(["uid", "step"]).copy()
    prev = D.groupby("uid").top1.shift(1); D["stay"] = (D.top1 == prev).astype(int)
    prevg = D.groupby("uid").g0.shift(1); D["gchange"] = (D.g0 != prevg) & prevg.notna()
    # run id / run length
    new = (D.stay == 0).astype(int); D["run_id"] = new.groupby(D.uid).cumsum()
    D["run_len"] = D.groupby(["uid", "run_id"]).top1.transform("size")
    D["run_pos"] = D.groupby(["uid", "run_id"]).cumcount()
    D["in_spell"] = D.run_len >= 3
    return D


def main(suite):
    arms = [f"oscl50_p_{suite}_cl{i}" for i in range(3)]
    Ds = {}
    for a in arms:
        D = pd.read_pickle(T / f"{a}_dec.pkl")
        # recover the scores lists from the raw logs cheaply: re-read only scores (needed for kernel weights)
        D["scores_list"] = D.pop("score1").map(lambda s: None)  # placeholder
        Ds[a] = D
    # scores are needed for cl2 kernel weights: re-read jsonl for the scores lists
    import glob
    RUN = pathlib.Path("/home/weiland/trace_runs/os_closed_loop/r02_g50/runs")
    for a in arms:
        sc = {}
        for f in glob.glob(str(RUN / a / "server_*" / "decisions_*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    if '"ev": "dec"' not in line: continue
                    d = json.loads(line); sc[(d["uid"], int(d["step"]))] = d["scores"]
        D = Ds[a]; D["scores_list"] = [sc[(u, s)] for u, s in zip(D.uid, D.step)]
        D = add_set_features(D, suite, a[-3:]); D = spells(D); Ds[a] = D
        D.to_pickle(T / f"{a}_dec2.pkl")

    print(f"\n===== {suite}: decision-level: stuck (in_spell) vs free decisions =====")
    rows = []
    for a in arms:
        D = Ds[a]
        for lab, m in (("free", ~D.in_spell), ("spell", D.in_spell)):
            g = D[m]
            rows.append(dict(arm=a[-3:], which=lab, n=len(g), frac=len(g) / len(D), tnorm=g.tnorm.mean(), t1norm=g.t1norm.mean(),
                             cancel=g.cancel.mean(), gagree=g.gagree.mean(), gabs=g.gabs.mean(), spread=g.spread.mean(),
                             lib_prog=g.lib_prog.mean(), lib_end=(g.lib_step >= g.lib_eplen - 1).mean(), g_closed=(g.g0 > 0).mean(),
                             conf=g.conf.mean(), agree=g.agree.astype(float).mean() if g.agree.notna().any() else np.nan,
                             still=g["x_still"].mean() if "x_still" in g else np.nan, dst=g["x_dst"].mean() if "x_dst" in g else np.nan,
                             fail_share=(~g.success).mean()))
    print(pd.DataFrame(rows).set_index(["arm", "which"]).T)

    print(f"\n--- {suite}: spells per episode: count, length, phase at start, escape ---")
    rows = []
    for a in arms:
        D = Ds[a]; S = D[D.in_spell & (D.run_pos == 0)]
        # escaped = the episode continued after the spell with a different pick (spell not at episode end)
        last_step = D.groupby("uid").step.max()
        S = S.assign(end_step=S.step + S.run_len - 1)
        S["escaped"] = S.end_step < S.uid.map(last_step)
        S["term"] = S.lib_step >= S.lib_eplen - 1
        rows.append(dict(arm=a[-3:], n_spells=len(S), per_ep=len(S) / 500, per_fail_ep=len(S[~S.success]) / max((~D.groupby('uid').success.first()).sum(), 1),
                         len_mean=S.run_len.mean(), len_p90=S.run_len.quantile(.9), escaped=S.escaped.mean(),
                         start_prog_mean=S.lib_prog.mean(), start_prog_lt_p5=(S.lib_prog < .5).mean(), term_row=S.term.mean(),
                         closed_at_start=(S.g0 > 0).mean(), gchange_at_start=S.gchange.mean(),
                         tnorm_in=S.tnorm.mean(), cancel_in=S.cancel.mean(), gagree_in=S.gagree.mean(),
                         spells_in_succ_eps=S.success.mean(), start_step_mean=S.step.mean()))
    print(pd.DataFrame(rows).set_index("arm").T)

    print(f"\n--- {suite}: what changes at the escape (first decision after a spell) ---")
    for a in arms:
        D = Ds[a]
        nxt = D[(D.run_pos == 0) & (D.groupby("uid").in_spell.shift(1) == True)]
        prevrow = D.shift(1).loc[nxt.index]
        same_ep = (nxt.lib_ep.values == prevrow.lib_ep.values); fwd = np.where(same_ep, nxt.lib_step.values - prevrow.lib_step.values, np.nan)
        print(a[-3:], f"n={len(nxt)} same_lib_ep={same_ep.mean():.2f} step_adv_when_same={np.nanmean(fwd):.2f} "
              f"back_when_same={(np.nan_to_num(fwd, nan=0) < 0).mean():.2f} gflip_at_escape={(nxt.g0.values != prevrow.g0.values).mean():.2f} "
              f"tnorm_after/before={nxt.tnorm.mean() / max(prevrow.tnorm.mean(), 1e-9):.2f} later_success={nxt.success.mean():.2f}")

    print(f"\n--- {suite}: per-task deadlock share (fraction of decisions in spells) and SR ---")
    tab = pd.DataFrame({a[-3:] + "_spell": Ds[a].groupby("task").in_spell.mean() for a in arms})
    for a in arms:
        tab[a[-3:] + "_SR"] = Ds[a].groupby("uid").first().groupby("task").success.mean()
        tab[a[-3:] + "_canc"] = Ds[a].groupby("task").cancel.mean()
        tab[a[-3:] + "_gagr"] = Ds[a].groupby("task").gagree.mean()
    print(tab.round(2))
    man = json.load(open(STORE / "library" / f"pi05_{SUITE[suite]}" / "current" / "manifest.json"))["tasks"]
    for t, s in man.items(): print("  task", t, s)


if __name__ == "__main__":
    for s in (sys.argv[1:] or ["sp", "l10"]):
        main(s)
