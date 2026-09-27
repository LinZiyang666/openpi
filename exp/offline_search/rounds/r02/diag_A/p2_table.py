import json, sys
cells = ["pi05_spatial", "pi05_l10", "groot_spatial", "groot_l10"]
names = ["state_k5", "abs32", "abs128", "abs32_st", "abs32_st_k8", "abs32_st_step", "delta128", "abs32_delta", "abs32_delta_st",
         "white_l2", "white_vis", "white_k8", "white_kern16", "lda32", "cca16", "cca16_st", "band_abs32_st", "abs32_st_gclu", "white_top1_disp"]
def load(cell, lib):
    return json.load(open(f"out/{cell}__{lib}.json"))
for lib in ("current", "big"):
    for regime, arm in (("stale", "cache"), ("fresh", "inf"), ("step0", "inf")):
        R = [load(f"{c}_{arm}", lib) for c in cells]
        print(f"\n##### lib={lib} regime={regime}  (cells: " + " ".join(f"{c}" for c in cells) + ")  L=" + " ".join(str(r["L"]) for r in R) + "  n=" + " ".join(str(r[regime]["n"]) for r in R))
        print("oracle           " + " ".join(f"{r[regime]['oracle']:.3f}" for r in R))
        print(f"{'method':17s} {'err (4 cells)':28s} | {'err1 (top-1)':28s} | {'AURC(top)':28s} | {'grip':28s} | rho")
        for n in names:
            e = " ".join(f"{r[regime][n]['err']:.3f}" for r in R); e1 = " ".join(f"{r[regime][n]['err1']:.3f}" for r in R)
            a = " ".join(f"{r[regime][n]['aurc_top']:.3f}" for r in R); g = " ".join(f"{r[regime][n]['grip']:.3f}" for r in R)
            rho = " ".join(f"{r[regime][n]['rho_top']:+.2f}" for r in R)
            print(f"{n:17s} {e:28s} | {e1:28s} | {a:28s} | {g:28s} | {rho}")
        print("aurc_opt(abs32_st) " + " ".join(f"{r[regime]['abs32_st']['aurc_opt']:.3f}" for r in R))
        print("phat MAE vis/state " + " ".join(f"{r[regime]['phat_mae_vis']:.3f}/{r[regime]['phat_mae_state']:.3f}" for r in R))
        print("drift frac / err_drift / err_nodrift " + " ".join(f"{r[regime]['drift_frac']:.2f}/{r[regime]['err_drift']:.3f}/{r[regime]['err_nodrift']:.3f}" for r in R))
        print("err succ eps / fail eps " + " ".join(f"{r[regime]['err_succ_eps']:.3f}/{r[regime]['err_fail_eps']:.3f}" for r in R))
        if regime == "stale":
            for sig in ["vis_motion", "vis_still", "hit_verify", "exp_next_gain", "phat_vs_top1phase", "neg_d1nn_state", "abs32_top1cos"]:
                print(f"  {sig:18s} rho(-err) " + " ".join(f"{r[regime]['rho_'+sig+'_negerr']:+.2f}" for r in R) + "  rho(drift) " + " ".join(f"{r[regime]['rho_'+sig+'_drift']:+.2f}" for r in R)
                      + "  AURC " + " ".join(f"{r[regime]['aurc_'+sig]:.3f}" for r in R) + "  succ/fail mean " + " ".join(f"{r[regime][sig+'_succ_mean']:.2f}/{r[regime][sig+'_fail_mean']:.2f}" for r in R))
            for k in ["aurc_abs32st_plus_still", "aurc_abs32st_plus_motion", "aurc_abs32st_plus_hitverify", "aurc_abs32st_plus_expnext"]:
                print(f"  {k:28s} " + " ".join(f"{r[regime][k]:.3f}" for r in R))
