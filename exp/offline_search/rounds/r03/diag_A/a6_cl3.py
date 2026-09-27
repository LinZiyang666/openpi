"""CL3 (StuckRecovery+insurance over AWM) vs CL2 on spatial: did recovery fire, did it escape, what changed."""
import pathlib, numpy as np, pandas as pd
T = pathlib.Path("/home/weiland/.claude/jobs/a607dd74/tmp/r03_ideation_A/tables")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 3)
def spells(D):
    D = D.sort_values(["uid", "step"]).copy()
    prev = D.groupby("uid").top1.shift(1); D["stay"] = (D.top1 == prev).astype(int)
    new = (D.stay == 0).astype(int); D["run_id"] = new.groupby(D.uid).cumsum()
    D["run_len"] = D.groupby(["uid", "run_id"]).top1.transform("size"); D["run_pos"] = D.groupby(["uid", "run_id"]).cumcount()
    D["in_spell"] = D.run_len >= 3; return D
D3 = spells(pd.read_pickle(T / "oscl50_p_sp_cl3_dec.pkl")); D2 = pd.read_pickle(T / "oscl50_p_sp_cl2_dec2.pkl")
E3 = D3.groupby("uid").agg(task=("task","first"), init=("init","first"), success=("success","first"), n=("step","size")).reset_index().set_index(["task","init"])
E2 = D2.groupby("uid").agg(task=("task","first"), init=("init","first"), success=("success","first"), n=("step","size")).reset_index().set_index(["task","init"])
j = E2.join(E3, rsuffix="_3")
print("CL2 -> CL3 paired:", pd.crosstab(j.success, j.success_3, rownames=["CL2"], colnames=["CL3"]).to_dict())
print("per task SR CL2 / CL3:", pd.DataFrame({"cl2": E2.groupby(level=0).success.mean(), "cl3": E3.groupby(level=0).success.mean()}).round(2).T.to_string())
print(f"\nCL3 decisions {len(D3)}: stuck flag rate {D3.x_stuck.mean():.3f}, still rate {D3.x_still.mean():.3f}, terminal rate {D3.x_terminal.mean():.3f}, level dist {D3.x_level.value_counts(normalize=True).round(3).to_dict()}, blend used {D3.x_blend.mean():.3f}, nx_in {D3.x_nx_in.mean():.3f}, w_nx mean {D3.x_w_nx.mean():.3f}")
print(f"  stuck decisions in failed eps {D3.x_stuck[~D3.success].mean():.3f} vs succ {D3.x_stuck[D3.success].mean():.3f}; spells: frac decisions in spell {D3.in_spell.mean():.3f} (CL2 {D2.in_spell.mean():.3f}); spells/ep {(D3.in_spell & (D3.run_pos==0)).sum()/500:.2f} (CL2 {(D2.in_spell & (D2.run_pos==0)).sum()/500:.2f})")
S3 = D3[D3.in_spell]; print(f"  within spells: stuck flag {S3.x_stuck.mean():.2f}, still {S3.x_still.mean():.2f}, level>0 {(S3.x_level>0).mean():.2f}, motion {S3.x_motion.mean():.3f}, vself {S3.x_vself.mean():.3f} | free: motion {D3[~D3.in_spell].x_motion.mean():.3f} vself {D3[~D3.in_spell].x_vself.mean():.3f}")
# what happens at recovery decisions: does the top1 change vs previous decision? does the spell end within 2 decisions?
R = D3[D3.x_level > 0]
print(f"  recovery decisions {len(R)} ({len(R)/len(D3):.3f}); top1 changed vs prev {R.stay.eq(0).mean():.2f}; in failed eps {(~R.success).mean():.2f}; level counts {R.x_level.value_counts().to_dict()}; relaxed {R.x_relaxed.mean():.2f}")
# escapes: for each spell in CL3, was a recovery attempted during it, and did the episode succeed afterwards?
sp = D3[D3.in_spell].groupby(["uid","run_id"]).agg(len=("step","size"), rec=("x_level", lambda x: (x>0).mean()), stuckflag=("x_stuck","mean"), succ=("success","first"), start=("step","min"), term=("x_terminal","mean"))
print(f"  spells {len(sp)}: with any recovery {(sp.rec>0).mean():.2f}; mean stuck-flag share {sp.stuckflag.mean():.2f}; terminal share {sp.term.mean():.2f}; P(success | spell w/ recovery) {sp.succ[sp.rec>0].mean():.2f} vs w/o {sp.succ[sp.rec==0].mean():.2f}")
# gripper hysteresis effect: flips per episode
g3 = D3.groupby("uid").apply(lambda g: (g.g0.values[1:] != g.g0.values[:-1]).sum()); g2 = D2.groupby("uid").apply(lambda g: (g.g0.values[1:] != g.g0.values[:-1]).sum())
print(f"  gripper flips/ep CL3 {g3.mean():.2f} (fail {g3[~D3.groupby('uid').success.first()].mean():.2f}) vs CL2 {g2.mean():.2f} (fail {g2[~D2.groupby('uid').success.first()].mean():.2f}); tnorm CL3 {D3.tnorm.mean():.2f} vs CL2 {D2.tnorm.mean():.2f}")
# task 6 in CL3
t6 = D3[D3.task == 6]; print(f"  task 6 CL3: SR {t6.groupby('uid').success.first().mean():.2f}, in_spell {t6.in_spell.mean():.2f}, stuck flag {t6.x_stuck.mean():.2f}, recovery decisions {(t6.x_level>0).mean():.3f}, gagree-like gvote {t6.gvote.abs().mean():.2f}, flips/ep {g3[t6.uid.unique()].mean():.2f}")
