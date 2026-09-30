"""Fit each wrist/stage bank once, then serialize every SW/SF+SW arm."""
import argparse
import json
import pickle
import time
from pathlib import Path

from exp.offline_search.harness import api, store
from .method import StageWrist
from .specs import SPEC, make_specs


def save(path, method, *, kwargs, cell, spec=SPEC):
    path = Path(path)
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(dict(method=method, spec=spec, kwargs=kwargs, cell=cell, registered={}, fit_s=0.), f, protocol=4)
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("/tmp/r7_C2"))
    ap.add_argument("--cells", nargs="+", default=["l10_50", "l10_500", "sp_50", "sp_500"])
    a = ap.parse_args()
    run = a.run_root.resolve()
    if not (run == Path("/tmp/r7_C2") or Path("/tmp/r7_C2") in run.parents):
        raise ValueError("C2 prefit writes only under /tmp/r7_C2; coordinator copies fits after review")
    (run / "fits").mkdir(parents=True, exist_ok=True)
    arms = make_specs(str(run))
    records = []
    for label in a.cells:
        rows = [r for r in arms if r["name"].endswith("_" + label)]
        cell = f"pi05_{rows[0]['suite']}_cache"
        ctx = api.Context(root="/home/weiland/trace_runs/offline_search_store", cell=cell,
                          seed=0, scratch=run / "scratch" / label)
        lib = ctx.open_library("current")
        t0 = time.perf_counter()
        kw = dict(rows[0]["kwargs"])
        method = StageWrist(**{**kw, "wrist_fit": "", "stage_fit": ""})
        method.fit(lib, ctx)
        wrist_kwargs = {k: getattr(method.wrist, k) for k in (
            "features", "lib", "fit_data", "kref", "k", "codes", "lam", "state_scale", "early",
            "step0_joint", "lam_c", "norm_cap", "hyst", "nn", "serving", "budget", "gates", "residual_threshold")}
        save(kw["wrist_fit"], method.wrist, kwargs=wrist_kwargs, cell=cell,
             spec="exp.offline_search.rounds.r04.k1_blind.wrist:BlindWristAWM")
        method.stages.save(kw["stage_fit"])
        method.wrist_fit, method.stage_fit = kw["wrist_fit"], kw["stage_fit"]
        for row in rows:
            if row is not rows[0]:
                method = StageWrist(**row["kwargs"])
                method.fit(lib, ctx)
            path = run / "fits" / (row["name"] + ".pkl")
            save(path, method, kwargs=row["kwargs"], cell=cell)
            rec = dict(arm=row["name"], path=str(path), bytes=path.stat().st_size,
                       metric_width=next(iter(method.wrist.tasks.values())).Wf.shape[0],
                       stage_calibration=method.stages.calibration, elapsed_s=time.perf_counter()-t0)
            records.append(rec)
            print(json.dumps(rec), flush=True)
    (run / "prefit_report.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
