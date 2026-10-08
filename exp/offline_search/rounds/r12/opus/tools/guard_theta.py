"""R12 opus (guard task): library-only per-cell threshold for the risk guard.
theta_cell = quantile of the risk over the cell's own library LOEO look rows at 1 - g_lib, where g_lib is the deployed
no-progress guard's open-loop rate on the same library replay. Then the closed-loop dev call rate at theta_cell."""
import glob, json
import numpy as np
from exp.offline_search.rounds.r12.opus.tools import common as cm
from exp.offline_search.rounds.r12.opus.tools.guard_lib import lib_table
from exp.offline_search.rounds.r12.opus.tools.guard_final import risk_fn
from exp.offline_search.rounds.r12.opus.tools.guard_design import X_of, metrics
from exp.offline_search.rounds.r12.opus.tools.analyze_replay import load as load_replay, guard_flags

G = json.loads((cm.OUT / "guard_lib.json").read_text())
dev = {p.split("/")[-1][:-4]: dict(np.load(p)) for p in sorted(glob.glob(str(cm.OUT / "guard" / "*.npz")))}
cells = {"r11_dev_pi05_l10_50": ("pi05", "l10", 50), "r11_dev_pi05_spatial_50": ("pi05", "spatial", 50),
         "r11_dev_pi05_l10_200": ("pi05", "l10", 200), "r11_dev_pi05_spatial_200": ("pi05", "spatial", 200),
         "r11_dev_pi05_l10_400": ("pi05", "l10", 500), "r11_dev_groot_l10_50": ("groot", "l10", 50),
         "r11_dev_groot_spatial_50": ("groot", "spatial", 50), "r11_dev_groot_l10_400": ("groot", "l10", 500),
         "r11_devknob_pi05_l10_50_periodic_ir32": ("pi05", "l10", 50),
         "r11_devknob_groot_l10_50_periodic_ir32": ("groot", "l10", 50)}
out = {}
for arm, (m, s, n) in cells.items():
    coef = G[f"{m}|coef"]; f = risk_fn(coef)
    T = lib_table(m, s, n)
    Z = load_replay(m, s, n)
    C = cm.load_cell(m, s)
    fl = guard_flags(Z, "F", C["prog"], C["ep_len"])
    look = (Z["step"] % 2 == 0) & (Z["step"] >= 4)
    g_lib = float(fl[look].mean())
    rl = f(X_of(T, coef["feats"]))
    theta = float(np.quantile(rl, 1 - g_lib))
    z = dev[arm]
    g0 = z["called"] & (z["os_reason"] == 4)
    rd = f(X_of(z, coef["feats"]))
    gl = rd >= theta
    out[arm] = dict(lib_size=n, g_lib=g_lib, theta=theta, dev_g0=float(g0.mean()), dev_rate=float(gl.mean()),
                    G0=metrics(z, g0), Glib=metrics(z, gl))
    r = out[arm]
    print(f"{arm} (library {n}): g_lib {g_lib:.3f} theta {theta:.3f} | dev deployed {r['dev_g0']:.3f} vs risk guard {r['dev_rate']:.3f}"
          f" | inF {r['G0']['in_failed']:.2f}->{r['Glib']['in_failed']:.2f} succF {r['G0']['succ_flags']:.3f}->{r['Glib']['succ_flags']:.3f}"
          + (f" D {r['G0']['D_at_calls']:.3f}->{r['Glib']['D_at_calls']:.3f}" if 'D_at_calls' in r['G0'] else ""))
cm.save_json(out, cm.OUT / "guard_theta.json")
