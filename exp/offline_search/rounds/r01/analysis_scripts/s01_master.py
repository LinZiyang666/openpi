"""Master table from the scoreboard: latest run_ts per (method, cell); adds regret vs big-library oracle from the
coverage cache. Writes master.csv and prints compact per-regime tables."""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 60); pd.set_option("display.max_rows", 500)

SB = "/home/weiland/projects/openpi/exp/offline_search/results/scoreboard.csv"
OUT = "/home/weiland/.claude/jobs/a607dd74/tmp/analysis_r01"
ROOT = "/dev/shm/offline_search_store"
CELLS = [f"{m}_{s}_{a}" for m in ("pi05", "groot") for s in ("spatial", "l10") for a in ("inf", "cache")]

df = pd.read_csv(SB)
df = df.sort_values("run_ts").groupby(["method", "cell"], as_index=False).tail(1)
print("rows", len(df), "methods", df.method.nunique())
# big-library oracle per cell: coverage cache
big = {"pi05": "bpool_cs", "groot": "bpool_all"}
borc = {}
for lib in ("bpool_cs", "bpool_all", "current"):
    z = np.load(f"{ROOT}/profile_cache/coverage_{lib}.npz")
    for c in CELLS:
        k = f"{c}__oracle_err"
        if k in z.files:
            borc[(lib, c)] = float(np.nanmean(z[k]))
for c in CELLS:
    m = c.split("_")[0]
    df.loc[df.cell == c, "big_oracle"] = borc[(big[m], c)]
    df.loc[df.cell == c, "cur_oracle_cov"] = borc[("current", c)]
df["regret_big"] = df.err_mean - df.big_oracle
df.to_csv(f"{OUT}/master.csv", index=False)
print("oracle (current/big) per cell:")
for c in CELLS:
    m = c.split("_")[0]
    print(f"  {c:20s} current {borc[('current', c)]:.3f}  big {borc[(big[m], c)]:.3f}")

cols = ["err_mean", "err_median", "regret_mean", "regret_big", "aurc", "risk_c30", "risk_c50", "grip_mis", "bad_c100", "indist",
        "ms_per_query", "bytes_per_entry", "fit_s"]
for arm in ("inf", "cache"):
    print(f"\n===== {arm} cells: err_mean per cell (8-col order pi05_sp, pi05_l10, groot_sp, groot_l10) =====")
    sub = df[df.arm == arm]
    piv = sub.pivot(index="method", columns="cell", values="err_mean")
    piv = piv[[c for c in CELLS if c.endswith(arm)]]
    piv["mean"] = piv.mean(axis=1)
    print(piv.sort_values("mean").round(3).to_string())
    for v in ("aurc", "risk_c30", "grip_mis", "regret_big", "err_median"):
        print(f"\n----- {arm}: {v} -----")
        piv = sub.pivot(index="method", columns="cell", values=v)
        piv = piv[[c for c in CELLS if c.endswith(arm)]]
        piv["mean"] = piv.mean(axis=1)
        print(piv.sort_values("mean").round(3).to_string())
print("\n===== cost (mean over cells) =====")
print(df.groupby("method")[["ms_per_query", "bytes_per_entry", "fit_s", "peak_rss_mb"]].mean().round(3).to_string())
print(df.groupby("method")[["tier", "uses_nonlibrary_action", "library"]].first().to_string())
