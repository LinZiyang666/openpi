"""R12 pilot scorer: per cell, method arms vs pooled deployed replicates (same B-dev selection).

Owner IR (compute share): full look .152/.148, one camera .076/.074, completion +.076/.074, call +.848/.852.
Deployed replicates: fix1 off arms, old invalid look-half pilot arms (execution_audit=false => deployed),
r12_devlooks_* recordings (r11_dev_* arms). Knob-on arms are scored only against knob-on replicates.
"""
import glob, json, math, os, sys

PILOTS = "/home/weiland/projects/openpi/exp/offline_search/rounds/r12"
TRACE = "/home/weiland/trace_runs/os_closed_loop"
PRICE = {"pi05": (.152, .848), "groot": (.148, .852)}


def load(path):
    d = json.load(open(path))
    model = d["cost_ledger"]["model"]
    v, m = PRICE[model]
    lh = d.get("lookhalf_cost_ledger")
    c = d["cost_ledger"]
    if lh:
        dec, work, one, misses = lh["decisions"], lh["work"], lh["one_camera_looks"], lh["misses"]
        looks = lh["vision_decisions"]
    else:
        dec, looks, misses, one = c["decisions"], c["vision_decisions"], c["misses"], 0
        work = v * looks + m * misses
    eps = d["complete"]
    return dict(arm=d["arm"], model=model, n=eps, sr=d["sr"], ir=work / dec, calls_ep=misses / eps,
                cam_save=one * v / 2 / dec, call_rate=misses / dec, m=m,
                one_share=one / max(looks, 1), dec_ep=dec / eps, path=path)


def cell_of(arm):
    for key in ("pi05_l10_50_periodic_ir32", "pi05_l10_50", "pi05_spatial_50", "pi05_spatial_200",
                "groot_spatial_50", "groot_l10_50"):
        if ("_" + key + "_") in "_" + arm + "_":
            return key
    return None


rows = []
for p in [q for q in glob.glob(PILOTS + "/*/pilots/*/runs/*/summary.json") if "/ablate/" not in q]:
    rows.append(load(p))
for p in glob.glob(TRACE + "/r12_devlooks_*/runs/*/summary.json"):
    rows.append(load(p))
for r in rows:
    a = r["arm"]
    r["cell"] = cell_of(a.replace("r11_devknob_", "").replace("r11_dev_", "").replace("r12_lh_", "").replace("r12_guard_", ""))
    if "periodic_ir32" in a:
        r["cell"] = r["cell"] if r["cell"] and "periodic" in r["cell"] else (r["cell"] or "?") + "_periodic_ir32"
    old_invalid = "/lookhalf/pilots/r12_lookhalf_" in r["path"] and "_fix1_" not in r["path"]
    r["deployed"] = a.endswith("_off") or a.startswith("r11_dev") or old_invalid
    r["src"] = r["path"].split("/runs/")[0].split("/")[-1]

cells = sorted({r["cell"] for r in rows if r["cell"]})
only = sys.argv[1:] or cells
for cell in only:
    rs = [r for r in rows if r["cell"] == cell]
    dep = [r for r in rs if r["deployed"]]
    met = [r for r in rs if not r["deployed"]]
    if not dep:
        continue
    n = sum(r["n"] for r in dep)
    sr0 = sum(r["sr"] * r["n"] for r in dep) / n
    ir0 = sum(r["ir"] * r["n"] for r in dep) / n
    sd_ir = (sum((r["ir"] - ir0) ** 2 for r in dep) / max(len(dep) - 1, 1)) ** .5 if len(dep) > 1 else float("nan")
    calls0 = sum(r["calls_ep"] * r["n"] for r in dep) / n
    cr0 = sum(r["call_rate"] * r["n"] for r in dep) / n
    print(f"\n== {cell}: deployed {len(dep)} reps / {n} eps  SR {sr0:.3f}  IR {ir0:.4f} (rep SD {sd_ir:.4f})  calls/ep {calls0:.2f}")
    for r in sorted(dep, key=lambda r: r["src"]):
        print(f"   dep  {r['src'][:34]:34s} {r['arm'][:44]:44s} SR {r['sr']:.2f} IR {r['ir']:.4f} calls/ep {r['calls_ep']:.2f}")
    for r in sorted(met, key=lambda r: r["arm"]):
        se_sr = math.sqrt(sr0 * (1 - sr0) / r["n"] + sr0 * (1 - sr0) / n)
        print(f"   METH {r['arm'][:50]:50s} SR {r['sr']:.2f} (d {r['sr'] - sr0:+.3f} +-{1.96 * se_sr:.3f})  "
              f"IR {r['ir']:.4f} (d {r['ir'] - ir0:+.4f} = camera -{r['cam_save']:.4f} / calls {r['m'] * (r['call_rate'] - cr0):+.4f} / rest)  "
              f"calls/ep {r['calls_ep']:.2f}  one-cam {r['one_share']:.2f}")
