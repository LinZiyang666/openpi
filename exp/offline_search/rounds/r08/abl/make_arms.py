"""Freeze PAPER_AB replicate-1 B specs and derive exactly 24 supplementary arms."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from exp.offline_search.rounds.r08.abl.judge import BITS

HERE = Path(__file__).resolve().parent
RUNS = Path("/home/weiland/trace_runs/os_closed_loop")
RUN = RUNS / "r08_abl"
STORE = "/home/weiland/trace_runs/offline_search_store"
MODULE = "exp.offline_search.rounds.r08.abl"
PREFIX = ["taskset", "-c", "10-13,54-57", "env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1",
          "MKL_NUM_THREADS=1", "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=.:src",
          ".venv/bin/python"]
CELLS = [(m, s, n) for m in ("pi05", "groot") for s in ("l10", "spatial") for n in (50, 500)]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=1) + "\n")
    tmp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(row):
    args = row["plugin_args"]
    if args.count("--os-fit-artifact") != 1:
        raise ValueError("expected exactly one --os-fit-artifact")
    return args[args.index("--os-fit-artifact") + 1]


def references(model, suite, scale):
    """Same naming convention as r06/ops/paper_ab.py:arms (which has import side effects)."""
    cell = f"{'sp' if suite == 'spatial' else suite}_{scale}"
    if model == "groot":
        astem, bstem = f"r5x_g_{cell}_tail1u", f"r6p1_c10_g_{cell}"
        a1, b1 = f"r05_x:{astem}", f"r06_paper:{bstem}"
    else:
        astem = "r4b3_p_sp_500_tail1uc" if cell == "sp_500" else f"r5t_p_{cell}_tail1uc"
        bstem = f"r5q1_c10_p_{cell}"
        a1 = f"{'r04_blind' if cell == 'sp_500' else 'r05_ptail'}:{astem}"
        b1 = f"r05_q1:{bstem}"
    return {"A": [a1, *[f"r06_paper:{astem}_rep{k}" for k in (2, 3)]],
            "B": [b1, *[f"r06_paper:{bstem}_rep{k}" for k in (2, 3)]]}


def sources(runs=RUNS):
    out = {}
    for model, suite, scale in CELLS:
        refs = references(model, suite, scale)
        run, name = refs["B"][0].split(":")
        path = runs / run / "arms_in.json"
        row, = [r for r in json.loads(path.read_text()) if r["name"] == name]
        deployed_path = runs / run / "arms.json"
        deployed, = [r for r in json.loads(deployed_path.read_text()) if r["arm"] == name]
        resolved = json.loads(json.dumps(row).replace("<RUN>", str(runs / run)))
        for field in ("method", "kwargs", "plugin_args", "full_model", "cost_ledger", "client_overrides"):
            if resolved.get(field) != deployed.get(field):
                raise ValueError(f"B input/deployed mismatch: {name} {field}")
        fit = artifact(resolved)
        out[model, suite, scale] = dict(source=str(path), source_name=name, source_row=row,
            source_file_sha256=sha(path), deployed_file=str(deployed_path),
            deployed_row=deployed, source_artifact=fit, source_fit_sha256=sha(fit), references=refs)
    return out


def make_specs(source_rows, run=RUN):
    rows, provenance = [], []
    for model, suite, scale in CELLS:
        source = source_rows[model, suite, scale]
        variants = list(BITS) + ["onlynp"] if scale == 500 else ["onlynp"]
        for variant in variants:
            disabled = ["stuck", "terminal", "overtime"] if variant == "onlynp" else [variant]
            row = copy.deepcopy(source["source_row"])
            row["name"] = f"r8abl_{variant}_{'p' if model == 'pi05' else 'g'}_{'sp' if suite == 'spatial' else suite}_{scale}"
            row["method"] = MODULE + ".judge:" + ("TriggerCommitJudge" if model == "pi05" else "TriggerGrootCommitJudge")
            row["kwargs"]["disabled_guards"] = disabled
            args = row["plugin_args"]
            args[args.index("--os-fit-artifact") + 1] = str(run / "fits" / (row["name"] + ".pkl"))
            rows.append(row)
            provenance.append(dict(arm=row["name"], model=model, suite=suite, scale=scale,
                                   disabled_guards=disabled, **copy.deepcopy(source)))
    assert len(rows) == len({r["name"] for r in rows}) == 24
    return rows, provenance


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args(argv)
    rows, provenance = make_specs(sources(), args.run)
    write_json(args.run / "arms_in.json", rows)
    write_json(args.run / "provenance.json", provenance)
    (args.run / "guard_arm_names.txt").write_text("".join(r["name"] + "\n" for r in rows))
    write_json(args.run / "manifests/eval500.json", dict(
        selected=[dict(task=t, init=i) for t in range(10) for i in range(50)]))
    print(f"wrote {len(rows)} arms: 16 leave-one-out (500-demo) + 8 only-no-progress; 8 B source cells")


if __name__ == "__main__":
    main()
