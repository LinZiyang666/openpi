import json
cells = ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]
def load(cell, lib): return json.load(open(f"out/w_{cell}__{lib}.json"))
names = ["base", "base_mean5", "base_top1", "nn1", "nn10", "lam0p01", "lam1", "p32", "p128", "rank16", "rank32", "rank64", "chunk", "stepwin3", "succonly", "shared", "fitcur", "novis_state", "early", "st_x3",
         "track_pure", "base_track0p5", "base_track1", "base_track2", "base_antistuck"]
for lib in ("big", "current"):
    for regime, arm in (("stale", "cache"), ("step0", "cache"), ("fresh", "inf")):
        if lib == "current" and arm == "inf": continue
        R = [load(f"{c}_{arm}", lib) for c in cells]
        print(f"\n##### lib={lib} regime={regime}  L=" + " ".join(str(r["L"]) for r in R) + "  n=" + " ".join(str(r[regime]["n"]) for r in R) + "  mappable=" + " ".join(str(r["mappable"]) for r in R))
        print("oracle           " + " ".join(f"{r[regime]['oracle']:.3f}" for r in R) + "   drift frac " + " ".join(f"{r[regime]['drift_frac']:.2f}" for r in R) + "  oracle_drift " + " ".join(f"{r[regime]['oracle_drift']:.3f}" for r in R))
        print(f"{'variant':15s} {'err':28s} | {'p50':28s} | {'succ eps':28s} | {'fail eps':28s} | {'drift':28s} | nodrift")
        for n in names:
            if n not in R[0][regime]: continue
            f = lambda k: " ".join(f"{r[regime][n][k]:.3f}" if n in r[regime] else "  -  " for r in R)
            print(f"{n:15s} {f('err'):28s} | {f('p50'):28s} | {f('err_succ'):28s} | {f('err_fail'):28s} | {f('err_drift'):28s} | {f('err_nodrift')}")
        print("AURC (base kern16):")
        for k in R[0][regime]["aurc"]:
            print(f"  {k:26s} " + " ".join(f"{r[regime]['aurc'].get(k, float('nan')):.3f}" for r in R) + "   rho " + " ".join(f"{r[regime]['rho'].get(k, float('nan')):+.2f}" for r in R))
    print("timing (ms/query, single thread, concurrent run): " + " ".join(f"{c}:{load(c+'_cache', lib)['timing']['ms_p50']:.2f}(proj {load(c+'_cache', lib)['timing']['ms_projection_only']:.2f}, C={load(c+'_cache', lib)['timing']['cands']})" for c in cells))
