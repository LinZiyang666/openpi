"""CPU prefit emission for V2: frozen-fit LOEO episode-max reference tables.

The diagnostic replays recorded library states/actions with a declared vision
stride. It excludes the source episode from candidates, not from PCA/metric or
V7 fitting. These empirical p-values are NOT a proved deployment error bound.
"""
import argparse
import copy
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

from .method import awm_of, load_fit
from .campaign import make_kwargs
from .v2 import sha


def statistics(guard, ex):
    return dict(coverage=float(ex.get("pred_err", 0)),
        stuck=float(ex.get("confirmed_stuck", ex.get("stuck_n", 0))),
        lag=float(ex.get("lag", 0)), overtime=float(ex.get("overtime", 0)),
        progress=float(getattr(guard, "_noprog_span", 0)),
        terminal=float(bool(int(ex.get("os_flags", 0)) & 2)))


def calibrate(guard, lib, stride=2):
    from exp.offline_search.rounds.r02.g3_recovery import g3_core as core
    from exp.offline_search.harness import api
    awm = awm_of(guard)
    awm.prof = guard.prof = api.NULL_PROFILER
    original = awm.tasks
    records = []
    # Lazy historical keys avoid copying a whole raw-key prefix at every query.
    class History:
        def __init__(self, data, rows):
            self.data, self.rows = data, rows
        def __getitem__(self, ix):
            return self.data[self.rows[ix]]
    class Query(core.PseudoQuery):
        hist_key_v0 = property(lambda s: History(s._L.key_v0, s._rows[:s._pos]))
        hist_key_v1 = property(lambda s: History(s._L.key_v1, s._rows[:s._pos]))
        hist_has_vision = property(lambda s: np.arange(s._pos) % stride == 0)
        has_vision = True
    row_arrays = ("rows", "Z", "z2", "HD", "h2", "RS", "rs2", "Z0", "n20", "V0", "V1")
    for ep, rows in core.episode_rows(guard.C).items():
        if not bool(np.asarray(lib.success)[rows[0]]):
            continue
        task = int(guard.C.task[rows[0]])
        T = copy.copy(original[task])
        mask = awm.lib_ep[T.rows] != ep
        if mask.sum() < 16:
            continue
        for key in row_arrays:
            value = getattr(T, key, None)
            if value is not None:
                setattr(T, key, value[mask].copy())
        awm.tasks = {**original, task: T}
        q0 = Query(lib, guard.C, rows, 0, guard.model, True)
        guard.reset(q0.episode)
        maxima, flags, residuals = {}, [], []
        for pos in range(0, len(rows), stride):
            q = Query(lib, guard.C, rows, pos, guard.model, True)
            result = guard.query(q)
            ex = result.extras
            stats = statistics(guard, ex)
            for key, value in stats.items():
                maxima[key] = max(maxima.get(key, -np.inf), value)
            flags.append(int(ex.get("os_flags", 0)))
            action = result.action if result.action is not None else awm.act[result.topk[0]]
            residuals.append(float(np.sqrt(np.mean((action[:10, :7] - lib.action[rows[pos], :10, :7]) ** 2))))
        records.append(dict(episode=int(ep), task_id=task, maxima=maxima, fired_any=any(flags),
                            fired_bits=[any(f & (1 << bit) for f in flags) for bit in range(6)],
                            loeo_chunk_rms_mean=float(np.mean(residuals)), anchors=len(flags)))
    awm.tasks = original
    tables = {}
    for task in sorted({r["task_id"] for r in records}):
        rr = [r for r in records if r["task_id"] == task]
        tables[str(task)] = {key: sorted(r["maxima"][key] for r in rr) for key in rr[0]["maxima"]}
    # Pool raw maxima for audit. No unvalidated pooled fallback is used online.
    pooled = {key: sorted(r["maxima"][key] for r in records) for key in records[0]["maxima"]}
    return dict(schema="r6p3.calibration.v2", stride=stride, records=records, per_task=tables, pooled=pooled,
        effective_alpha_successful_library_episodes=float(np.mean([r["fired_any"] for r in records])),
        effective_alpha_by_task={str(t): float(np.mean([r["fired_any"] for r in records if r["task_id"] == t])) for t in sorted({r["task_id"] for r in records})},
        quality=dict(loeo_chunk_rms_episode_mean=float(np.mean([r["loeo_chunk_rms_mean"] for r in records])),
                     label="frozen-fit imitation diagnostic; not Q1 SR probability"),
        caveat="successful-library recorded policy states, synthetic all-HIT history, vision stride; candidate LOEO only; finite sample p-values, no coverage theorem")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["pi05", "groot"], required=True)
    ap.add_argument("--cell", choices=["l10_50", "l10_500", "sp_50", "sp_500"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    from exp.offline_search.harness.store import LibraryView
    kw = make_kwargs(a.model, a.cell)
    suite = "spatial" if a.cell.startswith("sp_") else "l10"
    guard = load_fit(kw["guard_fit"], kw["guard_spec"], kw["guard_kwargs"], f"{a.model}_{suite}_cache")
    lib = LibraryView("/home/weiland/trace_runs/offline_search_store", f"{a.model}_{suite}", awm_of(guard).cand_name)
    result = calibrate(guard, lib)
    result["guard_fit_sha256"] = sha(kw["guard_fit"])
    result["library_manifest"] = dict(path=str(lib.dir / "manifest.json"), sha256=sha(lib.dir / "manifest.json"))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(dict(episodes=len(result["records"]), alpha=result["effective_alpha_successful_library_episodes"], output=str(a.out))))


if __name__ == "__main__":
    main()
