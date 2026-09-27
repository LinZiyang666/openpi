"""V8 T2 metric trainer (subprocess of AWMT2.fit; torch lives only here so the harness main process / forked
workers never import it).

    python t2_train.py <in.npz> <out.npz>

in.npz : Xn f32 [N, 136] (per-task z-scored fit features), tix int64 [N] (task index 0..T-1), pi/pj int64 [P]
         (global row pairs, both rows of a pair in the same task), r f32 [P] (head RMS target, sigma units),
         kind int8 [P] (0 = nearest in the initial metric, 1 = random, 2 = nearest in action space),
         H f32 [N, 35] (sigma-scaled heads), ep int64 [N] (library episode ids),
         Wl0 f32 [T, 136, 64] (alpha-scaled AWM rank-64 whitening = initial linear path), b0 f32 [T] (initial
         offset, loss off/log), val bool [N] (held-out rows, diagnostic only), cfg (json string).
cfg    : phi "lin" | "mlp", hidden, epochs, lr_lin, lr_mlp, batch, seed, huber_delta, loss "abs" | "off" | "log",
         n_nn, remine (bool), device "auto"|"cpu"|"cuda", min_free_gb, cpu_threads.
Model  : phi_t(x) = x @ Wl[t]  (+ relu(x @ W1 + b1 + c[t]) @ W2 for "mlp"; c[t] = per-task hidden bias = one-hot task
         embedding; W2 = 0 at init, so phi == the AWM linear path at step 0).
Loss   : abs  Huber(d_ij - r_ij, delta)                      (the ideation-A P4 spec), d = ||phi(x_i) - phi(x_j)||
         mlkr metric learning for kernel regression (NOT the spec; the retrieval's own objective): per anchor, the
              softmax(-d^2/s_t)-weighted mean head of its mlkr_k (32) nearest other-episode rows under the current
              metric (re-mined every epoch; s_t fixed = median d^2 to the 5th neighbour at init) regresses the
              anchor's own head, Huber(delta 1) over the 35 dims; the pair arrays are then only used for logging
         off  Huber(b_t + d_ij - r_ij, delta), b_t >= 0 learned (the head RMS of the best candidate is never 0:
              teacher noise + state mismatch; ranking / kernel weights are invariant to b)
         log  Huber(log(b_t + d_ij) - log(r_ij), delta_log = 0.25) (relative error: near pairs weigh as much as far)
         Adam, per-task mini-batches (shuffled task order). val rows (if any) are excluded from training pairs;
         val loss = pairs whose first row is a val row and second a train row.
remine : before every epoch after the first, the kind-0 pairs are re-mined as the n_nn nearest other-episode rows under
         the CURRENT phi (hard-negative mining: pairs the metric now calls near get their true action distance).
Proxy  : loeo_err = in-library retrieval error of the method's synthesis (top-16 kernel mean, kref 5, own episode
         excluded) for up to 128 fixed rows per task (val rows when val_frac > 0, else train rows), logged before
         training and after every epoch -- the quantity the harness measures, in distribution.
out.npz: Wl [T,136,64] (+ W1 [136,H], b1 [H], c [T,H], W2 [H,64]), log (json string).
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def pick_device(cfg) -> str:
    want = cfg.get("device", "auto")
    if want == "cpu" or not torch.cuda.is_available():
        return "cpu"
    try:
        free, _tot = torch.cuda.mem_get_info()
    except Exception:
        return "cpu"
    if free < float(cfg.get("min_free_gb", 8.0)) * 2 ** 30:
        return "cpu"
    return "cuda"


def main(inp, outp):
    t_start = time.time()
    z = np.load(inp)
    cfg = json.loads(str(z["cfg"]))
    dev = pick_device(cfg)
    if dev == "cpu":
        torch.set_num_threads(int(cfg.get("cpu_threads", 4)))
    torch.manual_seed(int(cfg["seed"]))
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    X = torch.tensor(z["Xn"], device=dev)
    N = int(X.shape[0])
    tix = z["tix"]
    pi, pj = z["pi"].copy(), z["pj"].copy()
    kind = z["kind"].copy() if "kind" in z.files else np.zeros(len(pi), np.int8)
    Hn = z["H"] if "H" in z.files else None
    epn = z["ep"] if "ep" in z.files else None
    Ht = torch.tensor(Hn, device=dev) if Hn is not None else None
    r_np = z["r"].copy()
    T = int(z["Wl0"].shape[0])
    D, K = int(z["Wl0"].shape[1]), int(z["Wl0"].shape[2])
    mlp = cfg["phi"] == "mlp"
    Hd = int(cfg.get("hidden", 256))
    Wl = torch.nn.Parameter(torch.tensor(z["Wl0"], device=dev))
    groups = [{"params": [Wl], "lr": float(cfg["lr_lin"])}]
    if mlp:
        lin = torch.nn.Linear(D, Hd)                    # default (kaiming-uniform) init on z-scored inputs
        W1 = torch.nn.Parameter(lin.weight.detach().T.contiguous().to(dev))
        b1 = torch.nn.Parameter(torch.zeros(Hd, device=dev))
        c = torch.nn.Parameter(torch.zeros(T, Hd, device=dev))
        W2 = torch.nn.Parameter(torch.zeros(Hd, K, device=dev))
        groups.append({"params": [W1, b1, c, W2], "lr": float(cfg["lr_mlp"])})
    loss_kind = cfg.get("loss", "abs")
    braw = None
    if loss_kind != "abs":
        b0 = np.maximum(z["b0"].astype(np.float64), 1e-3)
        braw = torch.nn.Parameter(torch.tensor(np.log(np.expm1(b0)), dtype=torch.float32, device=dev))  # softplus^-1
        groups.append({"params": [braw], "lr": 1e-3})
    opt = torch.optim.Adam(groups)
    delta = float(cfg["huber_delta"]) if loss_kind != "log" else 0.25
    B = int(cfg.get("batch", 512))
    val = z["val"] if "val" in z.files else np.zeros(N, bool)
    rows_of = [np.flatnonzero(tix == t) for t in range(T)]
    rng = np.random.default_rng(int(cfg["seed"]))
    prng = np.random.default_rng(int(cfg["seed"]) + 99)
    probe = []
    for t in range(T):
        cand = rows_of[t][val[rows_of[t]]] if val.any() else rows_of[t]
        probe.append(np.sort(prng.choice(cand, size=min(128, len(cand)), replace=False)) if len(cand) else cand)

    def phi(xb, t):
        zz = xb @ Wl[t]
        if mlp:
            zz = zz + torch.relu(xb @ W1 + b1 + c[t]) @ W2
        return zz

    def state():
        r = torch.tensor(r_np, device=dev)
        ptask = tix[pi]
        trn = ~val[pi] & ~val[pj]
        vpair = val[pi] & ~val[pj]
        return r, [np.flatnonzero((ptask == t) & trn) for t in range(T)], [np.flatnonzero((ptask == t) & vpair) for t in range(T)]

    r, by_task, by_task_val = state()

    def loss_of(idx, t):
        a = phi(X[torch.as_tensor(pi[idx], device=dev)], t)
        b = phi(X[torch.as_tensor(pj[idx], device=dev)], t)
        d = torch.sqrt(((a - b) ** 2).sum(1) + 1e-8)
        rr = r[torch.as_tensor(idx, device=dev)]
        if loss_kind in ("abs", "mlkr"):
            pred, tgt = d, rr
        elif loss_kind == "off":
            pred, tgt = torch.nn.functional.softplus(braw[t]) + d, rr
        else:
            pred, tgt = torch.log(torch.nn.functional.softplus(braw[t]) + d), torch.log(rr + 1e-3)
        return torch.nn.functional.huber_loss(pred, tgt, delta=delta, reduction="none")

    @torch.no_grad()
    def full_loss(groups_=None):
        groups_ = by_task if groups_ is None else groups_
        tot, n = 0.0, 0
        for t in range(T):
            ids = groups_[t]
            for lo in range(0, len(ids), 8192):
                l_ = loss_of(ids[lo:lo + 8192], t)
                tot += float(l_.sum())
                n += int(l_.numel())
        return tot / max(n, 1) if n else float("nan")

    @torch.no_grad()
    def codes(t):
        return phi(X[torch.as_tensor(rows_of[t], device=dev)], t)

    @torch.no_grad()
    def loeo_err():
        if Ht is None or epn is None:
            return float("nan")
        tot, n = 0.0, 0
        for t in range(T):
            rt = rows_of[t]
            if not len(probe[t]):
                continue
            Zt = codes(t)
            loc = np.searchsorted(rt, probe[t])
            Dq = torch.cdist(Zt[torch.as_tensor(loc, device=dev)], Zt)
            same = torch.as_tensor(epn[probe[t]][:, None] == epn[rt][None, :], device=dev)
            if val.any():                                 # candidates = train rows only
                same |= torch.as_tensor(val[rt][None, :], device=dev)
            Dq = Dq.masked_fill(same, float("inf"))
            k = min(16, Dq.shape[1])
            dv, o = torch.topk(Dq, k, dim=1, largest=False)
            dd = dv - dv[:, :1]
            ref = torch.clamp(dd[:, min(5, k) - 1:min(5, k)], min=1e-6)
            w = torch.exp(-(dd / ref) ** 2)
            Hc = Ht[torch.as_tensor(rt, device=dev)]
            head = (w[:, :, None] * Hc[o]).sum(1) / w.sum(1, keepdim=True)
            e = torch.sqrt(((head - Ht[torch.as_tensor(probe[t], device=dev)]) ** 2).mean(1))
            tot += float(e.sum())
            n += int(e.numel())
        return tot / max(n, 1)

    @torch.no_grad()
    def remine():
        nonlocal pi, pj, kind, r_np
        keep = kind != 0
        ni, nj = [pi[keep]], [pj[keep]]
        kk = int(cfg.get("n_nn", 16))
        for t in range(T):
            rt = rows_of[t]
            Zt = codes(t)
            Dm = torch.cdist(Zt, Zt)
            bad = torch.as_tensor((epn[rt][:, None] == epn[rt][None, :]) | val[rt][None, :], device=dev)
            Dm = Dm.masked_fill(bad, float("inf"))
            k = min(kk, len(rt) - 1)
            dv, o = torch.topk(Dm, k, dim=1, largest=False)
            ok = torch.isfinite(dv).cpu().numpy()
            o = o.cpu().numpy()
            a = np.repeat(rt, k).reshape(len(rt), k)[ok]
            b = rt[o][ok]
            anc = ~val[a]
            ni.append(a[anc])
            nj.append(b[anc])
        new_i, new_j = np.concatenate(ni), np.concatenate(nj)
        kind = np.concatenate([kind[keep], np.zeros(len(new_i) - int(keep.sum()), np.int8)])
        pi, pj = new_i, new_j
        r_np = np.sqrt(np.mean((Hn[pi] - Hn[pj]) ** 2, axis=1)).astype(np.float32)

    def mine_cands(K):
        cand = np.full((N, K), -1, np.int64)
        for t in range(T):
            rt = rows_of[t]
            Zt = codes(t)
            Dm = torch.cdist(Zt, Zt)
            bad = torch.as_tensor((epn[rt][:, None] == epn[rt][None, :]) | val[rt][None, :], device=dev)
            Dm = Dm.masked_fill(bad, float("inf"))
            k = min(K, len(rt) - 1)
            _, o = torch.topk(Dm, k, dim=1, largest=False)
            cand[rt, :k] = rt[o.cpu().numpy()]
        return cand

    log = {"device": dev, "T": T, "N": N, "P": int(len(pi)), "phi": cfg["phi"], "hidden": Hd if mlp else 0,
           "epochs": int(cfg["epochs"]), "batch": B, "huber_delta": delta, "loss": loss_kind,
           "remine": bool(cfg.get("remine", False)), "loss_init": full_loss(), "val_loss_init": full_loss(by_task_val),
           "n_val_rows": int(val.sum()), "loeo_init": loeo_err(), "epoch_loss": [], "loeo": [], "remine_overlap": []}
    t0 = time.time()
    steps = 0
    if loss_kind == "mlkr":
        # Metric learning for kernel regression (Weinberger & Tesauro 2007), the objective of the method's own
        # synthesis: for every (train) anchor, predicted head = softmax(-d^2 / s_t)-weighted mean of the heads of its
        # K nearest other-episode rows under the CURRENT metric (re-mined every epoch); loss = Huber(pred - own head).
        Km = int(cfg.get("mlkr_k", 32))
        cand = mine_cands(Km)
        s_t = []
        with torch.no_grad():
            for t in range(T):
                rt = rows_of[t]
                d2 = ((codes(t) - phi(X[torch.as_tensor(cand[rt, min(4, Km - 1)], device=dev)], t)) ** 2).sum(1)
                s_t.append(float(d2.median()) + 1e-8)
        log["mlkr_s"] = s_t
        Ba = max(64, B // 2)

        def mlkr_loss(a_idx, t):
            ca = torch.as_tensor(cand[a_idx], device=dev)
            Za = phi(X[torch.as_tensor(a_idx, device=dev)], t)
            Zc = phi(X[ca.reshape(-1)], t).reshape(len(a_idx), ca.shape[1], -1)
            d2 = ((Za[:, None] - Zc) ** 2).sum(-1)
            w = torch.softmax(-d2 / s_t[t], dim=1)
            pred = (w[..., None] * Ht[ca]).sum(1)
            return torch.nn.functional.huber_loss(pred, Ht[torch.as_tensor(a_idx, device=dev)], delta=1.0,
                                                  reduction="none").mean(1)

        for ep_ in range(int(cfg["epochs"])):
            if ep_ > 0:
                cand = mine_cands(Km)
            batches = []
            for t in range(T):
                rt = rows_of[t]
                anc = rt[~val[rt] & (cand[rt] >= 0).all(1)]
                anc = anc[rng.permutation(len(anc))]
                batches += [(t, anc[lo:lo + Ba]) for lo in range(0, len(anc), Ba)]
            order = rng.permutation(len(batches))
            tot, n = 0.0, 0
            for bi in order:
                t, idx = batches[bi]
                loss = mlkr_loss(idx, t).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                steps += 1
                tot += float(loss.detach()) * len(idx)
                n += len(idx)
            log["epoch_loss"].append(tot / max(n, 1))
            log["loeo"].append(loeo_err())
    for ep_ in range(int(cfg["epochs"]) if loss_kind != "mlkr" else 0):
        if ep_ > 0 and cfg.get("remine", False) and Hn is not None:
            old = set(zip(pi[kind == 0].tolist(), pj[kind == 0].tolist()))
            remine()
            new = list(zip(pi[kind == 0].tolist(), pj[kind == 0].tolist()))
            log["remine_overlap"].append(float(np.mean([p in old for p in new])) if new else float("nan"))
            r, by_task, by_task_val = state()
        batches = []
        for t in range(T):
            ids = by_task[t][rng.permutation(len(by_task[t]))]
            batches += [(t, ids[lo:lo + B]) for lo in range(0, len(ids), B)]
        order = rng.permutation(len(batches))
        tot, n = 0.0, 0
        for bi in order:
            t, idx = batches[bi]
            l_ = loss_of(idx, t)
            loss = l_.mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            tot += float(loss.detach()) * len(idx)
            n += len(idx)
        log["epoch_loss"].append(tot / max(n, 1))
        log["loeo"].append(loeo_err())
    if dev == "cuda":
        torch.cuda.synchronize()
    log["train_s"] = time.time() - t0
    log["steps"] = steps
    log["loss_final"] = full_loss()
    log["val_loss_final"] = full_loss(by_task_val)
    if braw is not None:
        log["b"] = [float(v) for v in torch.nn.functional.softplus(braw).detach().cpu().numpy()]
    out = {"Wl": Wl.detach().cpu().numpy().astype(np.float32)}
    if mlp:
        with torch.no_grad():
            act_any = torch.zeros(Hd, dtype=torch.bool, device=dev)
            for t in range(T):
                m = torch.as_tensor(rows_of[t], device=dev)
                act_any |= (X[m] @ W1 + b1 + c[t] > 0).any(0)
            dead = int((~act_any).sum())
            res = torch.relu(X @ W1 + b1 + c[torch.as_tensor(tix, device=dev)]) @ W2
            lin_ = torch.stack([(X[i:i + 1] @ Wl[int(tix[i])]) for i in range(0, N, max(1, N // 512))])
        log["dead_units"] = dead
        log["residual_norm_ratio"] = float(res.norm(dim=1).mean() / lin_.squeeze(1).norm(dim=1).mean())
        out.update(W1=W1.detach().cpu().numpy().astype(np.float32), b1=b1.detach().cpu().numpy().astype(np.float32),
                   c=c.detach().cpu().numpy().astype(np.float32), W2=W2.detach().cpu().numpy().astype(np.float32))
    log["wall_s"] = time.time() - t_start
    out["log"] = np.array(json.dumps(log))
    tmp = outp + f".{os.getpid()}.tmp.npz"
    np.savez(tmp, **out)
    os.replace(tmp, outp)
    print(json.dumps(log), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
