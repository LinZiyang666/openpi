"""Smoke test + API-contract check for one method on a few episodes of one cell (CPU, in-process).

    python -m exp.offline_search.harness.smoke --method <module_or_file>:<Class> [--kwargs JSON] \
        [--cell pi05_spatial_inf] [--episodes 5] [--root <store root>] [--subsample tok] [--no-ref] [--out DIR]

Checks (each printed PASS / FAIL / WARN):
  attrs        name / tier / fit / reset / query / bytes_per_entry present and valid
  fit+query    fit on library current, query every decision of the selected episodes; every Result is validated
               (types, shapes, rows inside the declared library, finite confidence, action shape (H, 32), extras
               <= 4 KB); a ForbiddenAccess / TokensUnavailable / ContractError is reported with its message
  bytes        bytes_per_entry() finite and >= 0
  leak         a fresh fit re-run over the same episodes in REVERSED order gives identical outputs per decision
               (no state carried across reset(), deterministic given the episode seed)
  static       source scan for forbidden names (a_inf, rec_top1, num_steps, _reference_gt, ._A, ._gt ...): WARN
  prev_hit     q.prev_hit on the selected episodes matches the arm (cache: all True, inf: all False at step >= 1)
Then prints the metrics of the run (and of B0 on the same episodes unless --no-ref). Exit code 0 iff no FAIL.
"""
from __future__ import annotations

import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import pathlib  # noqa: E402
import re  # noqa: E402
import tempfile  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402

from . import api, run, store  # noqa: E402

STATIC_PATTERNS = [r"\ba_inf\b", r"\ba_hit\b", r"\brec_top1\b", r"\brec_score\b", r"\brec_perfield\b",
                   r"\bnum_steps\b", r"_reference_gt", r"\._A\b", r"\._gt\b", r"queries/", r"a_inf\.npy",
                   r"\bsuccess\b"]


def _say(tag, what, msg=""):
    print(f"  [{tag:4s}] {what}{(': ' + msg) if msg else ''}", flush=True)


def pick_episodes(qc, n, subsample):
    jobs = run.episode_jobs(qc, subsample, None)
    idx = [i for i, _ in jobs]
    if n >= len(idx):
        return idx
    sel = np.unique(np.linspace(0, len(idx) - 1, n).round().astype(int))
    return [idx[i] for i in sel]


def _fmt_metrics(m):
    keys = ("n", "err_mean", "err_median", "oracle_err_mean", "regret_mean", "indist", "grip_mis", "phase_err_mean",
            "aurc", "eaurc", "risk_c50", "flip_rate")
    return "  ".join(f"{k}={m[k]:.4g}" if isinstance(m[k], float) else f"{k}={m[k]}" for k in keys if k in m)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", required=True)
    ap.add_argument("--kwargs", default="{}")
    ap.add_argument("--cell", default="pi05_spatial_inf")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--root", default=str(store.SMOKE_ROOT))
    ap.add_argument("--subsample", choices=["tok"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-ref", action="store_true", help="skip the B0 reference run")
    ap.add_argument("--out", default=None, help="where to write the smoke outputs (default: a temp dir)")
    a = ap.parse_args(argv)
    kw = json.loads(a.kwargs)
    fails, warns = [], []
    print(f"smoke: method={a.method} kwargs={kw} cell={a.cell} episodes={a.episodes} root={a.root}"
          f"{' subsample=tok' if a.subsample else ''}", flush=True)

    # attrs
    try:
        cls, src = run.load_method_class(a.method)
        m = run.build_method(cls, kw)
        _say("PASS", "attrs", f"name={m.name} tier={m.tier} uses_gt={getattr(m, 'uses_gt', False)} src={src}")
    except Exception as exc:
        _say("FAIL", "attrs", repr(exc))
        traceback.print_exc()
        return 1

    # static scan (class source; the module file when the class source is unavailable)
    try:
        try:
            lines, first = inspect.getsourcelines(cls)
        except (OSError, TypeError):
            lines, first = (pathlib.Path(src).read_text().splitlines(True) if pathlib.Path(src).exists() else []), 1
        hits = [(first + i, ln.strip()) for i, ln in enumerate(lines)
                if not ln.strip().startswith("#") and any(re.search(p, ln) for p in STATIC_PATTERNS)]
        if hits and not getattr(m, "uses_gt", False):
            warns.append("static")
            _say("WARN", "static", "source mentions forbidden names -- make sure no GT / outcome / recorded search is "
                                   "used:\n" + "\n".join(f"           line {n}: {t[:110]}" for n, t in hits[:6]))
        else:
            _say("PASS", "static")
    except Exception as exc:
        _say("WARN", "static", repr(exc))

    qc = store.QueryCell(a.root, a.cell)
    eps = pick_episodes(qc, a.episodes, a.subsample)
    if not eps:
        _say("FAIL", "episodes", "no episode selected (empty tok subsample?)")
        return 1
    out = pathlib.Path(a.out) if a.out else pathlib.Path(tempfile.mkdtemp(prefix="os_smoke_"))
    mdir = out / m.name
    mdir.mkdir(parents=True, exist_ok=True)

    # prev_hit (online HIT / MISS flag of the previous decision) on the selected episodes
    try:
        f = qc.exec_hit_flag
        prev = [r - 1 for i in eps for r in range(qc.episodes[i]["start"] + 1, qc.episodes[i]["end"])]
        vals = {1: 0, 0: 0, 2: 0, -1: 0}
        for r in prev:
            vals[int(f[r])] += 1
        want = 1 if qc.arm == "cache" else 0
        msg = (f"prev_hit over {len(prev)} decisions at step >= 1: True={vals[1]} False={vals[0]} "
               f"None={vals[2] + vals[-1]} (expected all {'True' if want else 'False'} in a {qc.arm} arm)")
        if vals[want] == len(prev):
            _say("PASS", "prev_hit", msg)
        else:
            warns.append("prev_hit")
            _say("WARN", "prev_hit", msg)
        if getattr(m, "uses_nonlibrary_action", False):
            _say("WARN", "nonlib", "uses_nonlibrary_action=True: Result.action may come from non-library data "
                                   "(reference rows only; recorded in the scoreboard)")
    except Exception as exc:
        warns.append("prev_hit")
        _say("WARN", "prev_hit", repr(exc))

    # fit + query (full runner path, in-process, with metrics)
    summ = None
    try:
        summ = run.run_cell(cls, kw, a.cell, root=a.root, out_dir=mdir, workers=1, subsample=a.subsample, seed=a.seed,
                            profile=True, episodes=eps, progress_path=mdir / "progress.jsonl", do_timing=True,
                            fam=run.infer_family(m, src), src=src)
        _say("PASS", "fit+query", f"{summ['n_decisions']} decisions over {summ['n_episodes']} episodes; "
                                  f"library={summ['library']} fit={summ['fit']['fit_s']:.2f}s "
                                  f"q={summ['timing']['ms_per_query']:.3f} ms/query (single thread)")
    except api.SkipCell as exc:
        _say("WARN", "fit+query", f"method skipped this cell: {exc}")
        warns.append("skip")
    except (api.ForbiddenAccess, api.TokensUnavailable, api.ContractError) as exc:
        fails.append("contract")
        _say("FAIL", "fit+query", f"{type(exc).__name__}: {exc}")
        traceback.print_exc(limit=-2)
    except Exception as exc:
        fails.append("fit+query")
        _say("FAIL", "fit+query", f"{type(exc).__name__}: {exc}")
        traceback.print_exc()

    if summ is not None:
        bpe = summ["fit"]["bytes_per_entry"]
        if isinstance(bpe, float) and (math.isnan(bpe) or bpe < 0):
            fails.append("bytes")
            _say("FAIL", "bytes", f"bytes_per_entry()={bpe}; must be a finite number >= 0")
        else:
            _say("PASS", "bytes", f"bytes_per_entry={bpe:g}")
        if summ["extras_keys"]:
            _say("PASS", "extras", ", ".join(summ["extras_keys"]))

        # leak / determinism check: fresh fit, episodes reversed, compare per row
        try:
            ref = np.load(mdir / f"{a.cell}.npz")
            m2 = run.build_method(cls, kw)
            lib = store.LibraryView(a.root, qc.lib_key, "current")
            ctx = api.Context(root=a.root, cell=a.cell, seed=a.seed, scratch=mdir / "scratch" / (a.cell + "_leak"))
            ctx.scratch.mkdir(parents=True, exist_ok=True)
            m2.fit(lib, ctx)
            sizes = dict(summ["fit"]["libraries"])
            jobs = [j for j in run.episode_jobs(qc, a.subsample, eps)][::-1]
            r2 = run.run_jobs_inprocess(m2, qc, jobs, lib_sizes=sizes, seed=a.seed, cell=a.cell)
            pos = {int(r): i for i, r in enumerate(ref["row"])}
            bad = []
            for i, r in enumerate(r2["rows"]):
                j = pos[int(r)]
                k = len(r2["topk"][i])
                same = (int(ref["top1"][j]) == r2["top1"][i]
                        and np.array_equal(ref["topk"][j][:min(k, api.TOPK_SAVE)], r2["topk"][i][:api.TOPK_SAVE])
                        and float(ref["confidence"][j]) == r2["conf"][i]
                        and (r2["synth"][i] is None or np.array_equal(ref["synth_seg"][j], r2["synth"][i])))
                if not same:
                    bad.append(int(r))
            if bad:
                fails.append("leak")
                _say("FAIL", "leak", f"{len(bad)}/{len(r2['rows'])} decisions differ when the episodes run in reversed "
                                     f"order after a fresh fit (first rows {bad[:5]}): the method carries state across "
                                     f"reset() or is not deterministic given episode.seed / episode.rng")
            else:
                _say("PASS", "leak", f"{len(r2['rows'])} decisions identical with episodes reversed")
        except Exception as exc:
            fails.append("leak")
            _say("FAIL", "leak", repr(exc))
            traceback.print_exc()

        print(f"\n  metrics [{m.name}]  {_fmt_metrics(summ['metrics'])}")
        prof = summ["profile"]["timing_pass"]
        if prof:
            print("  profile (timing pass, us/query): " +
                  ", ".join(f"{k}={v['per_query_us']:.1f}" for k, v in prof.items()))
        if not a.no_ref and m.name != "B0_current":
            try:
                from .baselines import B0Current
                rs = run.run_cell(B0Current, {}, a.cell, root=a.root, out_dir=out / "B0_current", workers=1,
                                  subsample=a.subsample, seed=a.seed, episodes=eps, do_timing=False)
                print(f"  metrics [B0_current] {_fmt_metrics(rs['metrics'])}")
            except Exception as exc:
                print(f"  (B0 reference failed: {exc!r})")
    print(f"\n  outputs: {mdir}")
    verdict = "FAIL" if fails else "PASS"
    print(f"SMOKE {verdict}" + (f" (failed: {', '.join(fails)})" if fails else "") +
          (f" (warnings: {', '.join(warns)})" if warns else ""), flush=True)
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
