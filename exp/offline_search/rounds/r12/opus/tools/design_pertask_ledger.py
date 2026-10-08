"""Per-task ledger of the deployed system on B-dev recordings (out/dev feature tables, deployed arms, knob off):
share of the cell's decisions per task (omega_t), non-call looks per decision (u_t), calls per decision."""
import json
import numpy as np
D = "/home/weiland/projects/openpi/exp/offline_search/rounds/r12/opus/out/dev"
ARMS = {"pi05_l10_50": "r12_devlooks_50L__r11_dev_pi05_l10_50", "groot_l10_50": "r12_devlooks_50L__r11_dev_groot_l10_50",
        "pi05_spatial_50": "r12_devlooks_50S__r11_dev_pi05_spatial_50", "groot_spatial_50": "r12_devlooks_50S__r11_dev_groot_spatial_50",
        "pi05_spatial_200": "r12_devlooks_200__r11_dev_pi05_spatial_200"}
out = {}
for cell, f in ARMS.items():
    z = np.load(f"{D}/{f}.npz", allow_pickle=False)
    uid, step, task, hit = z["uid"], z["step"], z["task"], z["hit"].astype(bool)
    eps = sorted(set(uid.tolist()))
    dec_t = np.zeros(10); look_t = np.zeros(10); call_t = np.zeros(10); ep_t = np.zeros(10)
    for u in eps:
        m = uid == u; t = int(task[m][0])
        dec_t[t] += step[m].max() + 2; look_t[t] += m.sum(); call_t[t] += (~hit[m]).sum(); ep_t[t] += 1
    omega = dec_t / dec_t.sum(); u_t = (look_t - call_t) / dec_t; m_t = call_t / dec_t
    out[cell] = dict(omega=omega.tolist(), u=u_t.tolist(), m=m_t.tolist(), dec_per_ep=(dec_t / ep_t).tolist(),
                     u_cell=float((look_t - call_t).sum() / dec_t.sum()), m_cell=float(call_t.sum() / dec_t.sum()), n_eps=len(eps))
    print(f"{cell:18s} eps {len(eps)} u_cell {out[cell]['u_cell']:.3f} m_cell {out[cell]['m_cell']:.3f}")
    print("   omega " + " ".join(f"{x:.3f}" for x in omega))
    print("   u     " + " ".join(f"{x:.3f}" for x in u_t))
    print("   dec/ep " + " ".join(f"{x:5.1f}" for x in dec_t / ep_t))
json.dump(out, open("/home/weiland/projects/openpi/exp/offline_search/rounds/r12/opus/out/design/pertask_ledger.json", "w"), indent=1)
