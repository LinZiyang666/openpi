"""Paths and bounded writes for S-P1; source artifacts are read-only."""
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = Path("/home/weiland/trace_runs/os_closed_loop/r09_p1")
R8 = Path("/home/weiland/trace_runs/os_closed_loop/r08_main")
FABLE = HERE.parent / "explore_fable"
DERIVED = Path("/home/weiland/trace_runs/offline_search_store/derived/r09_fable")
CELLS = ("pi05_l10_50", "groot_l10_50", "pi05_spatial_50")
VARIANTS = ("A", "CU", "topk_cu", "topk_p10")
SPEC = "exp.offline_search.rounds.r09.p1_task_budget.methods:TaskBudgetController"


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(4 << 20):
            h.update(b)
    return h.hexdigest()


def output(path):
    path = Path(path).resolve()
    if not any(path == root or root in path.parents for root in (HERE, RUN)):
        raise ValueError("S-P1 writes only in its code directory or r09_p1 run root")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path, value):
    output(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def r8_name(cell, variant):
    return "r8_" + (cell.rsplit("_", 1)[0] if variant == "P10" else cell) + "_" + variant


def fit_path(row):
    args = row["plugin_args"]
    return Path(args[args.index("--os-fit-artifact") + 1])


def source(row):
    path = fit_path(row)
    return dict(artifact=str(path), sha256=sha(path), spec=row["method"], kwargs=row["kwargs"],
                cell=row["cell"], arm=row["arm"])
