"""Are deadlock rows at library gripper transitions? Is the trap entered right after a served gripper change?"""
import pathlib, numpy as np, pandas as pd
T = pathlib.Path("/home/weiland/.claude/jobs/a607dd74/tmp/r03_ideation_A/tables"); STORE = pathlib.Path("/dev/shm/offline_search_store")
for suite, sn in (("sp", "spatial"), ("l10", "l10")):
    d = STORE / "library" / f"pi05_{sn}" / "current"
    act = np.load(d / "action.npy", mmap_mode="r"); g = np.sign(np.asarray(act[:, 0, 6])); ep = np.load(d / "episode.npy"); st = np.load(d / "step.npy")
    # library gripper transition rows: sign differs from the previous row of the same episode
    prev_same = np.r_[False, ep[1:] == ep[:-1]]; trans = np.zeros(len(g), bool); trans[1:] = prev_same[1:] & (g[1:] != g[:-1])
    near = np.zeros(len(g), bool)
    for k in (-2, -1, 0, 1, 2):
        idx = np.where(trans)[0] + k; idx = idx[(idx >= 0) & (idx < len(g))]; near[idx[ep[idx] == ep[np.clip(idx - k, 0, len(g) - 1)]]] = True
    print(f"\n== {suite}: library rows near a gripper transition (+-2 steps): {near.mean():.3f}")
    for i in range(3):
        D = pd.read_pickle(T / f"oscl50_p_{suite}_cl{i}_dec2.pkl")
        S = D[D.in_spell & (D.run_pos == 0)]
        # served gripper changed within the 3 decisions before the spell start
        D = D.sort_values(["uid", "step"]); gc = D.groupby("uid").g0.transform(lambda x: x.ne(x.shift(1)).astype(float).rolling(3, min_periods=1).sum())
        pre = gc.loc[S.index]
        fail_first = S.sort_values("step").groupby("uid").first(); fail_first = fail_first[~fail_first.success]
        print(f"  cl{i}: spell rows near lib gripper transition {near[S.top1].mean():.2f} (all served {near[D.top1].mean():.2f}); "
              f"served gripper changed within 3 dec before spell start {(pre > 0).mean():.2f} (baseline per-3-dec {(gc > 0).mean():.2f}); "
              f"first spell of FAILED eps: near-trans {near[fail_first.top1].mean():.2f}, gripper closed {(fail_first.g0 > 0).mean():.2f}, lib_prog {fail_first.lib_prog.mean():.2f}, step {fail_first.step.mean():.1f}")
