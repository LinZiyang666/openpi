"""floor_common.py -- shared pieces of the teacher noise-floor scripts (R0-D).

Pure numpy + h5py so that both the openpi .venv (pi0.5) and the GR00T island venv
can import it. Holds: trace / library paths, the harness action metric, the
per-arm file index and the stratified decision sampler used by the GPU resampling.

Metric (protocol §4.2, execution segment): on the first 5 steps x 7 dims,
    err = sqrt(mean_{t<5, d<7} (((a1 - a2) / sigma_d) ** 2))
with sigma_d the per-dim std (ddof=0) of the current library's actions over their
first 5 steps. Gripper = dim 6, whose normalized values are ~ +-1, so mismatch is a
sign test. rel_l2 = ||a - ref||_F / ||ref||_F on the same [:5, :7] block.

Valid dims (coordinator, 2026-09-26): only action dims 0..6 are real (pi0.5 7..31 are
~0.001 padding, GR00T 7..31 are noise-like padding) and only the first 5 steps are
executed. Every statistic here is on [:5, :7]; the one secondary column is the whole
chunk on valid dims [:, :7] (rel_l2_chunk7). Never all 32 dims. The constants mirror
exp/offline_search/harness/dims.py (not yet present when this was written).
"""
from __future__ import annotations

import glob
import json
import pathlib

import h5py
import numpy as np

TRACE = pathlib.Path("/home/weiland/trace_runs/dual_20260923")
LIBS = TRACE / "audit" / "libs"
STORE = pathlib.Path("/home/weiland/trace_runs/offline_search_store/floor")
SMOKE_STORE = pathlib.Path("/home/weiland/trace_runs/offline_search_store_smoke/floor")

MODELS = ("pi05", "groot")
SUITES = ("spatial", "l10")
ARM_SUITE = {"spatial": "sp", "l10": "l10"}               # run-dir suite code
LIB_SUITE = {"spatial": "libero_spatial", "l10": "libero_10"}
HORIZON = {"pi05": 10, "groot": 16}
MODEL_ID = {"pi05": 0, "groot": 1}
SUITE_ID = {"spatial": 0, "l10": 1}
T_EXEC, D_EXEC, GRIP = 5, 7, 6          # executed steps, valid action dims, gripper dim
N_TASKS, N_THIRDS = 10, 3
DEFAULT_SEED = 20260926


def arm_dir(m: str, s: str, a: str) -> pathlib.Path:
    return TRACE / "runs" / f"tr_{m}_{ARM_SUITE[s]}_{a}"


def arm_files(m: str, s: str, a: str) -> list[str]:
    return sorted(glob.glob(str(arm_dir(m, s, a) / "trace" / "**" / "*.h5"), recursive=True))


def lib_sigma(m: str, s: str) -> np.ndarray:
    """Per-dim std (d<7) of the current library's actions over their first 5 steps."""
    a = np.load(LIBS / f"{m}_{LIB_SUITE[s]}.action.npy", mmap_mode="r")
    x = np.asarray(a[:, :T_EXEC, :D_EXEC], np.float64).reshape(-1, D_EXEC)
    return x.std(0)


def _blk(a) -> np.ndarray:
    return np.asarray(a, np.float64)[..., :T_EXEC, :D_EXEC]


def err(a, b, sigma) -> np.ndarray:
    """Harness execution-segment error; broadcasts over leading dims."""
    d = (_blk(a) - _blk(b)) / np.asarray(sigma, np.float64)
    return np.sqrt(np.mean(d * d, axis=(-2, -1)))


def grip_mis(a, b) -> np.ndarray:
    """Fraction of the 5 executed steps whose gripper (dim 6) sign disagrees."""
    ga = np.asarray(a)[..., :T_EXEC, GRIP] >= 0
    gb = np.asarray(b)[..., :T_EXEC, GRIP] >= 0
    return np.mean(ga != gb, axis=-1)


def rel_l2(a, ref) -> np.ndarray:
    d = _blk(a) - _blk(ref)
    return np.sqrt(np.sum(d * d, axis=(-2, -1))) / np.maximum(
        np.sqrt(np.sum(_blk(ref) ** 2, axis=(-2, -1))), 1e-12)


def rel_l2_chunk7(a, ref) -> np.ndarray:
    """Secondary column: relative L2 over the whole horizon on the valid dims [:, :7]."""
    a = np.asarray(a, np.float64)[..., :D_EXEC]; ref = np.asarray(ref, np.float64)[..., :D_EXEC]
    return np.sqrt(np.sum((a - ref) ** 2, axis=(-2, -1))) / np.maximum(
        np.sqrt(np.sum(ref ** 2, axis=(-2, -1))), 1e-12)


QS = (0.10, 0.25, 0.50, 0.75, 0.90)


def qstats(x) -> dict:
    x = np.asarray(x, np.float64).ravel()
    if x.size == 0:
        return {"n": 0}
    q = np.quantile(x, QS)
    return {"n": int(x.size), "mean": float(x.mean()),
            **{f"p{int(round(p * 100)):02d}": float(v) for p, v in zip(QS, q)}}


def pair_stats(e, g, r) -> dict:
    """Summary block for one set of paired distances (err, grip_mis, rel_l2)."""
    g = np.asarray(g, np.float64).ravel()
    return {"err": qstats(e), "rel_l2": qstats(r),
            "grip_mis_rate": float(g.mean()) if g.size else None,           # over (pair, t<5)
            "grip_mis_any": float((g > 0).mean()) if g.size else None}      # pairs with >=1 flip


def read_attrs(path: str) -> dict:
    with h5py.File(path, "r") as f:
        return {"file": path, "task_id": int(f.attrs["task_id"]),
                "init": int(f.attrs["orig_init_state_idx"]), "num_steps": int(f.attrs["num_steps"])}


def index_arm(m: str, s: str, a: str) -> list[dict]:
    rows = [read_attrs(p) for p in arm_files(m, s, a)]
    rows.sort(key=lambda r: (r["task_id"], r["init"], r["file"]))
    return rows


def third_range(t: int, n: int) -> tuple[int, int]:
    """Steps k in [lo, hi) with (3*k)//n == t, i.e. third t of an n-step episode."""
    lo = -(-t * n // N_THIRDS)
    hi = -(-(t + 1) * n // N_THIRDS)
    return lo, hi


def step_third(k: int, n: int) -> int:
    return min(N_THIRDS - 1, (N_THIRDS * k) // n)


def sample_plan(m: str, s: str, S: int, seed: int = DEFAULT_SEED) -> list[dict]:
    """Stratified decision sample: S//2 from inf, the rest from cache; within an arm the
    budget is spread evenly over 10 tasks x 3 step-thirds (remainder cells drawn by the
    rng), and each cell draws distinct episodes of that task plus a uniform step inside
    the episode's third. Deterministic in (m, s, S, seed)."""
    rng = np.random.default_rng([seed, MODEL_ID[m], SUITE_ID[s], S])
    plan = []
    for a, n_arm in (("inf", S // 2), ("cache", S - S // 2)):
        idx = index_arm(m, s, a)
        by_task: dict[int, list[dict]] = {}
        for r in idx:
            by_task.setdefault(r["task_id"], []).append(r)
        assert sorted(by_task) == list(range(N_TASKS)), f"{m}_{s}_{a}: tasks {sorted(by_task)}"
        cells = [(t, th) for t in range(N_TASKS) for th in range(N_THIRDS)]
        n_cell = np.full(len(cells), n_arm // len(cells), int)
        rem = n_arm % len(cells)
        if rem:
            n_cell[rng.choice(len(cells), rem, replace=False)] += 1
        for (t, th), n in zip(cells, n_cell):
            if n == 0:
                continue
            eps = [r for r in by_task[t] if third_range(th, r["num_steps"])[1] > third_range(th, r["num_steps"])[0]]
            assert len(eps) >= n, f"{m}_{s}_{a} task {t} third {th}: {len(eps)} eps < {n}"
            for j in rng.choice(len(eps), n, replace=False):
                r = eps[int(j)]
                lo, hi = third_range(th, r["num_steps"])
                k = int(rng.integers(lo, hi))
                plan.append(dict(arm=a, file=r["file"], step=k, task_id=t, init=r["init"],
                                 num_steps=r["num_steps"], third=th))
    return plan


def plan_arrays(plan: list[dict]) -> dict:
    return {
        "arm": np.array([p["arm"] for p in plan]),
        "file": np.array([p["file"] for p in plan]),
        "step": np.array([p["step"] for p in plan], np.int32),
        "task_id": np.array([p["task_id"] for p in plan], np.int32),
        "init": np.array([p["init"] for p in plan], np.int32),
        "num_steps": np.array([p["num_steps"] for p in plan], np.int32),
        "third": np.array([p["third"] for p in plan], np.int32),
    }


def resample_summary(m: str, s: str, sigma: np.ndarray, arrays: dict) -> dict:
    """Summary of a resample.npz payload: parity, fresh vs recorded, fresh vs fresh."""
    rec = arrays["a_recorded"]; rep = arrays["a_replay_recorded_noise"]; fr = arrays["a_fresh"]
    S, K = fr.shape[:2]
    third = arrays["third"]; arm = arrays["arm"]
    out = {"model": m, "suite": s, "S": int(S), "K": int(K), "sigma": sigma.tolist()}
    out["parity_replay_vs_recorded"] = {
        "rel_l2_exec": qstats(rel_l2(rep, rec)), "max_rel_l2_exec": float(rel_l2(rep, rec).max()) if S else None,
        "err": qstats(err(rep, rec, sigma)), "rel_l2_chunk7": qstats(rel_l2_chunk7(rep, rec)),
        "grip_mis_rate": float(grip_mis(rep, rec).mean()) if S else None,
        "noise0_exact": f"{int(arrays['noise0_exact'].sum())}/{S}",
    }
    e_fr = err(fr, rec[:, None], sigma); g_fr = grip_mis(fr, rec[:, None]); r_fr = rel_l2(fr, rec[:, None])  # [S,K]
    c_fr = rel_l2_chunk7(fr, rec[:, None])
    jj, kk = np.triu_indices(K, 1)
    e_ff = err(fr[:, jj], fr[:, kk], sigma); g_ff = grip_mis(fr[:, jj], fr[:, kk]); r_ff = rel_l2(fr[:, jj], fr[:, kk])
    # the recorded-noise replay is itself a sample: replay vs fresh is a second view of (i)
    e_rp = err(fr, rep[:, None], sigma)

    def split(mask):
        return {
            "fresh_vs_recorded": pair_stats(e_fr[mask], g_fr[mask], r_fr[mask]),
            "fresh_vs_recorded_rel_l2_chunk7": qstats(c_fr[mask]),
            "fresh_vs_fresh": pair_stats(e_ff[mask], g_ff[mask], r_ff[mask]),
            "fresh_vs_replay_err": qstats(e_rp[mask]),
            "n_decisions": int(mask.sum()),
        }

    everything = np.ones(S, bool)
    out["all"] = split(everything)
    out["by_third"] = {name: split(third == t) for t, name in enumerate(("early", "mid", "late"))}
    out["by_arm"] = {a: split(arm == a) for a in ("inf", "cache")}
    out["by_arm_third"] = {f"{a}_{name}": split((arm == a) & (third == t))
                           for a in ("inf", "cache") for t, name in enumerate(("early", "mid", "late"))}
    return out


def save_json(path, obj) -> None:
    pathlib.Path(path).write_text(json.dumps(obj, indent=1))


# ---------------------------------------------------------------------------
# GPU resampling driver (shared by floor_resample_pi05.py / floor_resample_groot.py)
# ---------------------------------------------------------------------------

def fresh_seed(seed: int, m: str, s: str, i: int, k: int) -> list[int]:
    """Per-(decision, sample) seed; decision i is the plan index."""
    return [seed, MODEL_ID[m], SUITE_ID[s], i, k]


def savez_dict(path, d: dict) -> None:
    """np.savez without keyword-name limits (np.savez(file, **kw) rejects a 'file' key)."""
    import zipfile
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED, allowZip64=True) as z:
        for k, v in d.items():
            with z.open(k + ".npy", "w", force_zip64=True) as fh:
                np.lib.format.write_array(fh, np.asanyarray(v), allow_pickle=False)


def _same_plan(z, arrays) -> bool:
    """The partial file holds the first n_done plan rows; they must match this plan."""
    return all(np.array_equal(z[k], arrays[k][: len(z[k])]) for k in ("arm", "file", "step"))


def run_resample(m: str, s: str, S: int, K: int, seed: int, out_root: pathlib.Path, sampler,
                 save_every: int = 25, log=print) -> dict:
    """Drive `sampler(i, row, step_group) -> dict(a_replay, noise0_exact, a_fresh[K,H,32],
    noise_fresh[K,H,32])` over the stratified plan; checkpoint to resample.partial.npz every
    `save_every` decisions (resumed automatically when the plan and K match); finally write
    resample.npz + resample_summary.json."""
    import time
    H = HORIZON[m]
    cell = out_root / f"{m}_{s}"
    cell.mkdir(parents=True, exist_ok=True)
    plan = sample_plan(m, s, S, seed)
    arrays = plan_arrays(plan)
    part = cell / "resample.partial.npz"
    res = {"a_recorded": [], "a_replay_recorded_noise": [], "a_fresh": [], "noise_fresh": [], "noise0_exact": [], "sec": []}
    if part.exists():
        with np.load(part) as z:
            if int(z["S"]) == S and int(z["K"]) == K and int(z["seed"]) == seed and _same_plan(z, arrays):
                n = int(z["n_done"])
                for key in res:
                    res[key] = list(z[key][:n])
                log(f"[{m}_{s}] resume from {n}/{len(plan)} decisions")
            else:
                log(f"[{m}_{s}] partial file does not match this plan/K/seed; starting over")

    def dump(path, n):
        payload = {k: np.stack(v) if v else np.zeros((0,)) for k, v in res.items()}
        tmp = path.with_suffix(".tmp.npz")
        savez_dict(tmp, {**{k: v[:n] for k, v in arrays.items()}, **payload, "S": S, "K": K, "seed": seed,
                         "n_done": n, "sigma": lib_sigma(m, s)})
        tmp.replace(path)

    t0 = time.time(); n0 = len(res["a_recorded"])
    for i in range(n0, len(plan)):
        row = plan[i]
        ts = time.time()
        with h5py.File(row["file"], "r") as f:
            g = f[f"step_{row['step']:04d}"]
            rec = g["trace/actions/full_inference"][()]
            out = sampler(i, row, g)
        assert out["a_replay"].shape == (H, 32) and out["a_fresh"].shape == (K, H, 32), (out["a_replay"].shape, out["a_fresh"].shape)
        res["a_recorded"].append(rec.astype(np.float32))
        res["a_replay_recorded_noise"].append(np.asarray(out["a_replay"], np.float32))
        res["a_fresh"].append(np.asarray(out["a_fresh"], np.float32))
        res["noise_fresh"].append(np.asarray(out["noise_fresh"], np.float32))
        res["noise0_exact"].append(bool(out["noise0_exact"]))
        res["sec"].append(time.time() - ts)
        done = i + 1
        if done % 10 == 0 or done == len(plan):
            rate = (time.time() - t0) / (done - n0)
            par = rel_l2(res["a_replay_recorded_noise"][-1], res["a_recorded"][-1])
            log(f"[{m}_{s}] {done}/{len(plan)}  {rate:.2f}s/decision  eta {rate * (len(plan) - done) / 60:.1f} min  "
                f"last parity rel_l2_exec={float(par):.4f}")
        if done % save_every == 0 and done < len(plan):
            dump(part, done)
    final = cell / "resample.npz"
    dump(final, len(plan))
    with np.load(final) as z:
        payload = {k: z[k] for k in z.files}
    summ = resample_summary(m, s, lib_sigma(m, s), payload)
    summ["seed"] = seed
    summ["sec_per_decision"] = qstats(payload["sec"])
    save_json(cell / "resample_summary.json", summ)
    if part.exists():
        part.unlink()
    return summ


def print_resample_summary(summ: dict, log=print) -> None:
    p = summ["parity_replay_vs_recorded"]
    log(f"[{summ['model']}_{summ['suite']}] parity: noise0_exact {p['noise0_exact']}  rel_l2_exec p50/max "
        f"{p['rel_l2_exec'].get('p50', float('nan')):.4f}/{p['max_rel_l2_exec']:.4f}  err p50 {p['err'].get('p50', float('nan')):.4f}")
    for name, blk in [("all", summ["all"])] + list(summ["by_third"].items()):
        fr, ff = blk["fresh_vs_recorded"], blk["fresh_vs_fresh"]
        fe, fe2 = fr["err"], ff["err"]
        if not fe.get("n"):
            log(f"   {name:5s} (no decisions)"); continue
        log(f"   {name:5s} n={blk['n_decisions']:4d}  fresh-vs-rec err p10/p50/p90 {fe['p10']:.3f}/{fe['p50']:.3f}/{fe['p90']:.3f} "
            f"grip {fr['grip_mis_rate']:.4f} relL2 p50 {fr['rel_l2']['p50']:.3f} | fresh-vs-fresh err p50 "
            f"{fe2.get('p50', float('nan')):.3f} grip {ff['grip_mis_rate'] if ff['grip_mis_rate'] is not None else float('nan'):.4f}")
