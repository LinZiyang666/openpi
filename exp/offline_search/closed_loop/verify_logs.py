"""Replay the plugin's logged per-episode inputs through the OFFLINE harness and compare decision by decision.

For every inputs/*.npz a plugin server wrote with --os-log-inputs, a mini store is built in the harness layout
(queries/<cell>/ from the logged keys / rs / raw_state / executed chunks; library/<m>_<s> symlinked to the real store),
the logged method is fitted exactly as the harness fits it and run with harness.run.run_jobs_inprocess, i.e. the
QueryView the offline harness would construct for that trajectory. Checks:

  * offline topk / scores / confidence / library / synthesized action / scalar extras == online (bit-exact); with
    ProbeB0 / ProbeHist the extras include crc32 digests of every QueryView field (keys, rs, raw_state, hist_*,
    prev_a_exec, prev_hit, hist_hit, episode identity), so equality proves the online inputs match
  * logged executed chunk == the chunk the method selected (library row action, or the synthesized action)
  * native shadow (method mode) / native winner (native mode) vs offline B0Current on the same live keys

    taskset -c 34-37,78-81 .venv/bin/python -m exp.offline_search.closed_loop.verify_logs --log-dir <server log dir> \
        [--tag <tag>] [--work /tmp/osplug_verify] [--b0-check]
"""
from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import glob  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

import numpy as np  # noqa: E402


def load_logs(log_dir: pathlib.Path, tag: str | None):
    pat = f"{tag}_c*.npz" if tag else "*.npz"
    eps = []
    for f in sorted(glob.glob(str(log_dir / "inputs" / pat))):
        z = np.load(f, allow_pickle=False)
        m = json.loads(str(z["meta"]))
        eps.append((f, m, {k: z[k] for k in z.files if k != "meta"}))
    return eps


def build_ministore(work: pathlib.Path, root: pathlib.Path, cell: str, eps, H: int):
    from exp.offline_search.harness import store

    m, s, _ = store.parse_cell(cell)
    qd = work / "queries" / cell
    if work.exists():
        shutil.rmtree(work)
    qd.mkdir(parents=True)
    (work / "library").mkdir()
    os.symlink(root / "library" / f"{m}_{s}", work / "library" / f"{m}_{s}")
    episodes, cols = [], {k: [] for k in ("ep", "step", "key_v0", "key_v1", "rs", "raw_state", "a_exec")}
    start = 0
    for i, (_f, meta, z) in enumerate(eps):
        n = int(z["step"].shape[0])
        aex = np.full((n, H, 32), np.nan, np.float32)
        aex[: z["a_exec"].shape[0]] = z["a_exec"][:n]
        episodes.append({"uid": meta["uid"], "file": _f, "task": meta["task"], "task_id": int(meta["task_id"]),
                         "init": int(meta["init"]), "success": bool(meta.get("success") or False),
                         "num_steps": n, "start": start, "end": start + n})
        cols["ep"].append(np.full(n, i, np.int32))
        cols["step"].append(z["step"].astype(np.int16))
        for k in ("key_v0", "key_v1", "rs", "raw_state"):
            cols[k].append(z[k].astype(np.float32))
        cols["a_exec"].append(aex)
        start += n
    arr = {k: np.concatenate(v) for k, v in cols.items()}
    N = arr["ep"].shape[0]
    for k, v in arr.items():
        np.save(qd / f"{k}.npy", v)
    np.save(qd / "a_hit.npy", arr["a_exec"])                       # pure cache: executed == served library chunk
    np.save(qd / "a_inf.npy", np.nan_to_num(arr["a_exec"]) + 1.0)   # != a_exec, so exec_hit_flag = 1 (HIT)
    np.save(qd / "rec_top1.npy", np.zeros(N, np.int32))
    np.save(qd / "rec_score.npy", np.zeros(N, np.float32))
    np.save(qd / "rec_perfield.npy", np.zeros((N, 3), np.float32))
    (qd / "episodes.json").write_text(json.dumps(episodes))
    (qd / "manifest.json").write_text(json.dumps({"schema": "offline_search.queries.v1", "arm": cell,
                                                  "complete": True, "source": "closed_loop logs"}))
    return episodes


def run_offline(method_spec, kwargs, cell, work, seed, episodes):
    from exp.offline_search.harness import run, store

    cls, _ = run.load_method_class(method_spec)
    F = run._fit_cell(cls, kwargs, cell, root=work, out_dir=work / "out", seed=seed, profile=False)
    qc = store.QueryCell(work, cell)
    jobs = [(i, np.arange(e["start"], e["end"], dtype=np.int64)) for i, e in enumerate(episodes)]
    return run.run_jobs_inprocess(F["method"], qc, jobs, lib_sizes=F["lib_sizes"], seed=seed, cell=cell), F


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--work", default="/tmp/osplug_verify")
    ap.add_argument("--b0-check", action="store_true", help="also run offline B0Current and compare with the native "
                    "winners (native mode) / native shadow (method mode)")
    ap.add_argument("--root", default="/dev/shm/offline_search_store", help="store root when the logs lack it")
    ap.add_argument("--out", default="", help="write the report JSON here (default <log-dir>/verify_<tag>.json)")
    a = ap.parse_args(argv)

    from exp.offline_search.harness import run

    log_dir = pathlib.Path(a.log_dir)
    eps = load_logs(log_dir, a.tag or None)
    if not eps:
        raise SystemExit(f"no inputs/*.npz under {log_dir}")
    metas = {(m["method_spec"], json.dumps(m["kwargs"], sort_keys=True), m["cell"], m.get("run_seed", 0),
              m.get("root")) for _, m, _ in eps}
    if len(metas) != 1:
        raise SystemExit(f"logs mix several configurations {metas}; pass --tag")
    spec, kw, cell, seed, root = next(iter(metas))
    root = root or a.root
    kwargs = json.loads(kw)
    H = int(eps[0][1]["H"])
    native_mode = spec == "native"
    work = pathlib.Path(a.work) / (a.tag or "all")
    episodes = build_ministore(work, pathlib.Path(root), cell, eps, H)
    rep = {"log_dir": str(log_dir), "tag": a.tag, "method_spec": spec, "kwargs": kwargs, "cell": cell,
           "episodes": len(eps), "decisions": int(sum(e["num_steps"] for e in episodes))}

    # seeds as the harness derives them
    rep["episode_seed_equal"] = all(int(m["seed"]) == run.ep_seed(seed, m["uid"]) for _, m, _ in eps)
    from exp.offline_search.harness import store

    lib = store.LibraryView(root, store.lib_key(cell), "current")
    tables = {"current": np.asarray(lib.action, np.float32)}
    for ln in store.library_names(root, store.lib_key(cell)):
        if ln != "current":
            tables[ln] = store.LibraryView(root, store.lib_key(cell), ln).action

    # executed == selected
    ex_ok = ex_n = 0
    for _, m, z in eps:
        n_ex = z["a_exec"].shape[0]
        for s in range(n_ex):
            sel = z["synth"][s] if z["used_synth"][s] else (
                np.asarray(tables[str(z["lib"][s])][int(z["top1"][s])], np.float32) if str(z["lib"][s]) in tables
                else None)
            if sel is None:
                continue
            ex_n += 1
            ex_ok += int(np.array_equal(z["a_exec"][s], sel))
    rep["executed_equals_selected"] = {"n": ex_n, "equal": ex_ok}

    if not native_mode:
        off, F = run_offline(spec, kwargs, cell, work, seed, episodes)
        eq = dict(topk=0, scores=0, conf=0, lib=0, synth=0, extras=0)
        first = None
        j = 0
        for (_, m, z), e in zip(eps, episodes):
            for s in range(e["num_steps"]):
                k = off["topk"][j].size
                c = {"topk": np.array_equal(z["topk"][s, :k], off["topk"][j]),
                     "scores": np.array_equal(z["scores"][s, :k], off["scores"][j]),
                     "conf": bool(z["conf"][s] == off["conf"][j]),
                     "lib": str(z["lib"][s]) == off["lib"][j]}
                osyn = off["synth"][j]
                c["synth"] = (osyn is None and not z["used_synth"][s]) or (
                    osyn is not None and np.array_equal(np.nan_to_num(z["synth"][s, :5, :7]), np.nan_to_num(osyn)))
                exo = off["extras"][j] or {}
                bad = {kk: (float(z[f"x_{kk}"][s]) if f"x_{kk}" in z else None, float(np.asarray(v).reshape(-1)[0]))
                       for kk, v in exo.items() if np.asarray(v).size == 1 and
                       not (f"x_{kk}" in z and np.float64(z[f"x_{kk}"][s]) == np.float64(np.asarray(v).reshape(-1)[0]))}
                c["extras"] = not bad
                for kk, v in c.items():
                    eq[kk] += int(bool(v))
                if first is None and not all(c.values()):
                    first = {"uid": m["uid"], "step": s, **{kk: bool(v) for kk, v in c.items()}, "extras_diff": bad}
                j += 1
        n = rep["decisions"]
        rep["offline_equal"] = {kk: v / n for kk, v in eq.items()}
        rep["first_diff"] = first
        rep["fit_s_offline"] = round(F["fit_s"], 3)
    if a.b0_check or native_mode:
        off0, _ = run_offline("exp.offline_search.harness.baselines:B0Current", {}, cell, work, seed, episodes)
        native = np.concatenate([z["native_top1"] for _, _, z in eps])
        b0 = np.asarray(off0["top1"])
        valid = native >= 0
        rep["offline_b0_vs_native"] = {"n": int(valid.sum()), "agree": int((b0[valid] == native[valid]).sum())}
        if not native_mode:
            online = np.concatenate([z["top1"] for _, _, z in eps])
            libs = np.concatenate([z["lib"] for _, _, z in eps])
            rep["online_vs_offline_b0"] = {"n": int(online.size),
                                           "agree": int(((online == b0) & (libs == "current")).sum())}
    q = np.concatenate([z["q_us"] for _, _, z in eps])
    im = np.concatenate([z["infer_ms"] for _, _, z in eps])
    rep["q_us"] = {"p50": float(np.percentile(q, 50)), "p95": float(np.percentile(q, 95))}
    rep["infer_ms"] = {"p50": float(np.percentile(im, 50)), "p95": float(np.percentile(im, 95))}
    out = pathlib.Path(a.out) if a.out else log_dir / f"verify_{a.tag or 'all'}.json"
    out.write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    ok = rep["episode_seed_equal"] and ex_ok == ex_n
    if not native_mode:
        ok = ok and all(v == 1.0 for v in rep["offline_equal"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
