"""Task deep-dives: spatial task 6 (AWM collapse), l10 task 2 (AWM loss), l10 tasks 2/3/9 (M4 losses)."""
import pathlib, numpy as np, pandas as pd
T = pathlib.Path("/home/weiland/.claude/jobs/a607dd74/tmp/r03_ideation_A/tables")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.precision", 3)
def load(suite): return {i: pd.read_pickle(T / f"oscl50_p_{suite}_cl{i}_dec2.pkl") for i in range(3)}
def ep_of(D): return D.groupby("uid").agg(task=("task","first"), init=("init","first"), success=("success","first"), n=("step","size")).reset_index().set_index(["task","init"])
for suite, task, arms in (("sp", 6, (0, 2)), ("sp", 6, (0, 1)), ("l10", 2, (0, 2)), ("l10", 3, (0, 1)), ("l10", 2, (0, 1)), ("l10", 9, (0, 1))):
    Ds = load(suite); a, b = arms
    Ea, Eb = ep_of(Ds[a]), ep_of(Ds[b])
    j = Ea.join(Eb, rsuffix="_b"); j = j[j.index.get_level_values(0) == task]
    sf = j[j.success & ~j.success_b]; fs = j[~j.success & j.success_b]
    print(f"\n===== {suite} task {task}: CL{a} -> CL{b}: S->F {len(sf)}  F->S {len(fs)}  SR {j.success.mean():.2f} -> {j.success_b.mean():.2f}")
    Db = Ds[b][Ds[b].task == task]; Da = Ds[a][Ds[a].task == task]
    for lab, D in ((f"CL{b}", Db), (f"CL{a}", Da)):
        S = D[D.success]; F = D[~D.success]
        print(f"  {lab}: decisions S={len(S)} F={len(F)} | in_spell S {S.in_spell.mean():.2f} F {F.in_spell.mean():.2f} | gagree S {S.gagree.mean():.2f} F {F.gagree.mean():.2f} "
              f"| gabs S {S.gabs.mean():.2f} F {F.gabs.mean():.2f} | cancel S {S.cancel.mean():.2f} F {F.cancel.mean():.2f} | tnorm S {S.tnorm.mean():.2f} F {F.tnorm.mean():.2f} "
              f"| same_ep(free) F {F[~F.in_spell].groupby('uid').lib_ep.apply(lambda x: (x.values[1:]==x.values[:-1]).mean() if len(x)>1 else np.nan).mean():.2f}")
        # gripper flips per episode and where the first spell starts
        gfl = D.groupby("uid").gchange.sum(); sp0 = D[D.in_spell & (D.run_pos == 0)].groupby("uid").first()
        print(f"      gripper flips/ep S {gfl[D.groupby('uid').success.first()].mean():.2f} F {gfl[~D.groupby('uid').success.first()].mean():.2f} | "
              f"first-spell start: step {sp0.step.mean():.1f} lib_prog {sp0.lib_prog.mean():.2f} closed {(sp0.g0>0).mean():.2f} term_row {(sp0.lib_step>=sp0.lib_eplen-1).mean():.2f} gagree {sp0.gagree.mean():.2f} tnorm {sp0.tnorm.mean():.2f}")
    # S->F inits: print 3 example sequences of CL_b (first 30 decisions): lib_ep/lib_step/g0/gagree
    for (t, init) in list(sf.index[:3]):
        g = Db[(Db.init == init)].sort_values("step")
        print(f"  --- S->F example init {init} (CL{b}, {len(g)} dec): step:libep.libstep g0 gagree tnorm")
        print("   ", " | ".join(f"{r.step}:{r.lib_ep}.{r.lib_step} {'C' if r.g0>0 else 'o'}{r.gagree:.1f} {r.tnorm:.1f}" for r in g.itertuples() if r.step < 30))
        ga = Da[(Da.init == init)].sort_values("step")
        print(f"      CL{a} same init ({len(ga)} dec):", " | ".join(f"{r.step}:{r.lib_ep}.{r.lib_step} {'C' if r.g0>0 else 'o'}" for r in ga.itertuples() if r.step < 30))
