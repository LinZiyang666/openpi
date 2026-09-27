"""Spell taxonomy, hub concentration, library pause-row proxy, P(success | spells), AWM fit gap per task."""
import pathlib, json, numpy as np, pandas as pd
T = pathlib.Path("/home/weiland/.claude/jobs/a607dd74/tmp/r03_ideation_A/tables"); OUT = T.parent
STORE = pathlib.Path("/dev/shm/offline_search_store"); SUITE = {"sp": "spatial", "l10": "l10"}
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 3)

def lib_selfmotion(suite):
    d = STORE / "library" / f"pi05_{SUITE[suite]}" / "current"
    nxt = np.load(d / "next.npy"); rs = np.load(d / "rs.npy")[:, :8].astype(np.float64)
    out = {}
    for f in ("v0", "v1"):
        K = np.load(d / f"key_{f}.npy", mmap_mode="r"); n = K.shape[0]
        cos = np.full(n, np.nan)
        mu = np.zeros(K.shape[1])
        for lo in range(0, n, 512): mu += np.asarray(K[lo:lo+512], np.float64).sum(0)
        mu /= n
        for lo in range(0, n, 512):
            hi = min(n, lo + 512); A = np.asarray(K[lo:hi], np.float64) - mu
            idx = nxt[lo:hi]; ok = idx >= 0
            B = np.asarray(K[np.where(ok, idx, 0)], np.float64) - mu
            c = (A * B).sum(1) / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1), 1e-12)
            cos[lo:hi] = np.where(ok, c, np.nan)
        out[f] = cos
    ds = np.full(len(nxt), np.nan); ok = nxt >= 0
    ds[ok] = np.linalg.norm(rs[nxt[ok]] - rs[ok], axis=1)
    act = np.asarray(np.load(d / "action.npy", mmap_mode="r")[:, :5, :6], np.float64); sig = act.reshape(-1, 6).std(0)
    tn = np.linalg.norm(act / sig, axis=2).mean(1)
    return pd.DataFrame(dict(cos_v0=out["v0"], cos_v1=out["v1"], drs=ds, tnorm=tn, ep=np.load(d / "episode.npy"), step=np.load(d / "step.npy"),
                             ep_len=np.load(d / "ep_len.npy"), term=nxt < 0))

for suite in ("sp", "l10"):
    Lm = lib_selfmotion(suite)
    # pause rows: visual self-motion in the lowest decile (both cams) or state motion lowest decile, non-terminal
    vis = (Lm.cos_v0 + Lm.cos_v1); thr_v = np.nanquantile(vis, .9); thr_s = np.nanquantile(Lm.drs, .1)
    Lm["pause"] = ((vis >= thr_v) | (Lm.drs <= thr_s)) & ~Lm.term
    print(f"\n===== {suite}: library rows {len(Lm)}, terminal {Lm.term.mean():.3f}, pause (top-decile stillness, non-terminal) {Lm.pause.mean():.3f}; pause rows tnorm {Lm.tnorm[Lm.pause].mean():.2f} vs other {Lm.tnorm[~Lm.pause & ~Lm.term].mean():.2f}; terminal rows tnorm {Lm.tnorm[Lm.term].mean():.2f}")
    print("   pause rows by library progress tercile:", np.histogram(Lm.step[Lm.pause] / Lm.ep_len[Lm.pause], bins=[0, .33, .66, 1.01])[0], " all rows:", np.histogram(Lm.step / Lm.ep_len, bins=[0, .33, .66, 1.01])[0])
    rows = []
    for i in range(3):
        D = pd.read_pickle(T / f"oscl50_p_{suite}_cl{i}_dec2.pkl")
        D["pause"] = Lm.pause.values[D.top1.values]; D["term"] = Lm.term.values[D.top1.values]
        E = D.groupby("uid").agg(success=("success", "first"), any_spell=("in_spell", "any"), n_spell=("run_pos", lambda x: 0))
        S = D[D.in_spell & (D.run_pos == 0)].copy()
        S["cls"] = np.select([S.term, S.gagree < .5, S.tnorm < .6, S.pause], ["T_terminal", "G_gripsplit", "Z_nearzero", "P_pause"], "H_hub")
        E["n_spell"] = S.groupby("uid").size().reindex(E.index).fillna(0)
        first = S.sort_values("step").groupby("uid").first()
        fail_uids = E.index[~E.success]
        rows.append(dict(arm=f"cl{i}", P_succ_no_spell=E.success[~E.any_spell].mean(), n_no_spell=(~E.any_spell).sum(),
                         P_succ_1spell=E.success[E.n_spell == 1].mean(), n_1=(E.n_spell == 1).sum(),
                         P_succ_2plus=E.success[E.n_spell >= 2].mean(), n_2p=(E.n_spell >= 2).sum(),
                         fail_with_spell=E.any_spell[~E.success].mean(),
                         # class shares of ALL spells and of the FIRST spell of failed episodes
                         **{"all_" + k: v for k, v in S.cls.value_counts(normalize=True).round(2).items()},
                         **{"firstfail_" + k: v for k, v in first.loc[first.index.intersection(fail_uids)].cls.value_counts(normalize=True).round(2).items()},
                         # served-weight on pause / terminal rows (top-1 based here) over all decisions and in free decisions
                         top1_pause=D.pause.mean(), top1_term=D.term.mean(), top1_pause_free=D.pause[~D.in_spell].mean(),
                         # hub concentration: share of spells started by the 10 most frequent rows
                         hub_top10=S.top1.value_counts().iloc[:10].sum() / len(S), n_distinct_spell_rows=S.top1.nunique(), n_spells=len(S),
                         # library self-motion of the spell rows vs all served rows
                         spellrow_vis=(Lm.cos_v0.values[S.top1] + Lm.cos_v1.values[S.top1]).mean(), served_vis=(Lm.cos_v0.values[D.top1] + Lm.cos_v1.values[D.top1]).mean(),
                         spellrow_drs=np.nanmean(Lm.drs.values[S.top1]), served_drs=np.nanmean(Lm.drs.values[D.top1])))
    print(pd.DataFrame(rows).set_index("arm").T.to_string())

# AWM fit gap per task: cur/fcur vs cur/fbig vs B0 (offline err, cache cells) + rows per task
(allrows, pertask) = pd.read_pickle(OUT / "a3_proxies.pkl")
for cell in ("pi05_spatial_cache", "pi05_l10_cache"):
    m, s = cell.split("_")[:2]; tid = np.load(STORE / "library" / f"{m}_{s}" / "current" / "task_id.npy")
    df = pd.DataFrame(dict(rows=pd.Series(tid).value_counts().sort_index(), B0=pertask[(cell, "B0")].err, AWM_fcur=pertask[(cell, "AWMkr5")].err,
                           AWM_fbig=pertask[(cell, "AWMfbig")].err, AWM_vis=pertask[(cell, "AWMvis")].err, M4=pertask[(cell, "M4k5mean")].err,
                           gsplit_fcur=pertask[(cell, "AWMkr5")].gsplit, gsplit_fbig=pertask[(cell, "AWMfbig")].gsplit, gsplit_M4=pertask[(cell, "M4k5mean")].gsplit))
    print(f"\n===== {cell}: per-task offline err (current library candidates) =====\n", df.round(3).T.to_string())
