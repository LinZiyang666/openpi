"""Aggregate gate_threshold_pareto eval output and render the Pareto figures.

Two subcommands, deliberately split by where they run:

``aggregate``
    Pure stdlib, so it runs on the client node next to the raw journals
    (per_step.jsonl is ~150 MB per suite -- aggregate there, move the summary).
    Reads ``journal.jsonl`` + ``per_step.jsonl`` for one suite, emits one JSON
    object keyed by ``yaml_id``.

``plot``
    Runs locally (needs matplotlib). Consumes the per-suite summaries, writes
    ``plot_data.json`` plus one figure per suite.

Definitions, fixed here so every consumer reads the same numbers:

teacher ratio
    ``MISS`` decisions / all decisions, from ``per_step.jsonl``. The gate decides
    once every 5 control steps, so this is a decision-rate, not a step-rate.
    Only rows whose ``attempt`` matches the accepted journal entry are counted --
    a retried episode leaves its abandoned rows behind and they must not be
    double-counted. ``client_timing`` summary rows carry no ``hit_type`` and are
    skipped.

success rate
    journal ``status == "done"`` over ``done + failed``, accepted rows only,
    500 A-pool inits per arm.
"""

from __future__ import annotations

import argparse
import collections
import json
import pathlib

SERIES = (
    ("ws", "weighted_sum pkl", "#1f77b4"),
    ("cs", "cache_size S3 pkl", "#d62728"),
)
SUITE_TAG = {"libero_spatial": "sp", "libero_10": "l10"}

#: pi0.5 three-stage latency, CUDA-Graph build (RTX 4090, batch=1). Source of
#: authority: exp/data_authority records latency_bench/libero_spatial/
#: executor_costs -> content.pi05_stage_split_ms.cuda_graph (n=100-level
#: samples, 2026-08-19). Owner-ruled inference_ratio (2026-08-21): every
#: request pays Stage1 (key build); only a MISS additionally pays Stage2+3.
STAGE1_MS = 10.26
STAGE23_MS = 27.69 + 29.57
STAGE_TOTAL_MS = STAGE1_MS + STAGE23_MS


def inference_ratio(teacher_ratio: float) -> float:
    """Owner-defined cost-normalized inference ratio (2026-08-21).

    (N_req * stage1 + N_miss * (stage2+stage3)) / (N_req * (stage1+2+3))
    == (stage1 + teacher_ratio * (stage2+stage3)) / stage_total,
    an affine map of the decision-level teacher ratio: floor stage1/total
    (~0.152, the all-hit case still pays key building), ceiling 1.0.
    """
    return (STAGE1_MS + teacher_ratio * STAGE23_MS) / STAGE_TOTAL_MS


def aggregate(data_dir: pathlib.Path) -> dict:
    accepted_attempt: dict[str, int] = {}
    episodes: dict[str, dict[str, bool]] = collections.defaultdict(dict)
    with (data_dir / "journal.jsonl").open(encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            if row.get("status") not in ("done", "failed") or not row.get("accepted"):
                continue
            accepted_attempt[row["task_uid"]] = row["attempt"]
            episodes[row["yaml_id"]][row["task_uid"]] = row["status"] == "done"

    decisions: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    with (data_dir / "per_step.jsonl").open(encoding="utf-8") as handle:
        for raw in handle:
            row = json.loads(raw)
            hit_type = row.get("hit_type")
            if hit_type is None:
                continue
            if accepted_attempt.get(row["task_uid"]) != row.get("attempt"):
                continue
            counts = decisions[row["yaml_id"]]
            counts[1] += 1
            if hit_type == "MISS":
                counts[0] += 1

    out = {}
    for yaml_id in sorted(episodes):
        eps = episodes[yaml_id]
        miss, total = decisions[yaml_id]
        out[yaml_id] = {
            "n_ep": len(eps),
            "success_rate": sum(eps.values()) / len(eps),
            "teacher_ratio": miss / total if total else None,
            "decisions": total,
        }
    return out


def pareto_front(points: list[tuple[float, float, float]]) -> list[tuple[float, float, float]]:
    """Minimise teacher ratio, maximise success rate."""
    front: list[tuple[float, float, float]] = []
    best = -1.0
    for teacher, success, fh in sorted(points, key=lambda p: (p[0], -p[1])):
        if success > best:
            front.append((teacher, success, fh))
            best = success
    return front


def plot_suite(
    suite: str,
    arms: dict,
    out_dir: pathlib.Path,
    gate_only: dict | None = None,
    x_mode: str = "teacher",
) -> pathlib.Path:
    x_of = (lambda tr: tr) if x_mode == "teacher" else inference_ratio
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tag = SUITE_TAG[suite]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for lib, label, color in SERIES:
        prefix = f"gtp_{lib}_{tag}_fh"
        points = [
            (x_of(v["teacher_ratio"]), v["success_rate"], int(k[len(prefix):]) / 100.0)
            for k, v in arms.items()
            if k.startswith(prefix) and v["teacher_ratio"] is not None
        ]
        if not points:
            continue
        ax.scatter(
            [p[0] for p in points], [p[1] for p in points],
            s=28, color=color, alpha=0.35, zorder=2,
        )
        front = pareto_front(points)
        ax.plot(
            [p[0] for p in front], [p[1] for p in front],
            marker="o", ms=6, lw=2, color=color, zorder=3,
            label=f"{label} (Pareto frontier)",
        )
        for teacher, success, fh in front:
            ax.annotate(
                f"{fh:.2f}", (teacher, success), textcoords="offset points",
                xytext=(4, -11), fontsize=7.5, color=color,
            )

    if gate_only:
        for lib, label, color in SERIES:
            yid = f"gtpgo_{lib}_{SUITE_TAG[suite] if False else ''}"
            key = [k for k in gate_only if k == f"gtpgo_{lib}_{SUITE_TAG[suite]}"]
            if not key:
                continue
            v = gate_only[key[0]]
            ax.scatter(
                [x_of(v["teacher_ratio"])], [v["success_rate"]],
                marker="*", s=220, color=color, edgecolor="black",
                linewidths=0.6, zorder=4,
                label=f"{label} gate-only (verdict off, L=8)",
            )

    if x_mode == "teacher":
        ax.set_xlabel("Teacher ratio (MISS decisions / all decisions)")
    else:
        ax.set_xlabel(
            "Inference ratio  $(s_1 + \\mathrm{tr}\\cdot(s_2{+}s_3))/(s_1{+}s_2{+}s_3)$,"
            " CUDA-Graph stage latencies"
        )
    ax.set_ylabel("Success rate (500 A-pool inits)")
    ax.set_title(
        f"gate_threshold_pareto — {suite}, d=1, Hybrid gate L=6\n"
        "16 cells $f_{FH}\\in[0.05,0.80]$ per pkl; labels = $f_{FH}$ on frontier"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    stem = out_dir / (
        f"pareto_{suite}" if x_mode == "teacher" else f"pareto_ir_{suite}"
    )
    fig.savefig(stem.with_suffix(".png"), dpi=180)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    return stem.with_suffix(".png")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    agg = sub.add_parser("aggregate", help="summarise one suite's raw eval output")
    agg.add_argument("data_dir")
    agg.add_argument("out_json")

    plot = sub.add_parser("plot", help="render figures + plot_data.json")
    plot.add_argument(
        "--suite", action="append", required=True, metavar="SUITE=SUMMARY.json",
        help="repeatable, e.g. --suite libero_spatial=/tmp/sp.json",
    )
    plot.add_argument("--out-dir", default="exp/gate_threshold_pareto/analysis")
    plot.add_argument("--status", default="", help="free-text status line for plot_data.json")
    plot.add_argument(
        "--gate-only-suite", action="append", default=[], metavar="SUITE=SUMMARY.json",
        help="optional gate-only ablation summaries, starred onto the same figure",
    )

    args = ap.parse_args()

    if args.cmd == "aggregate":
        summary = aggregate(pathlib.Path(args.data_dir))
        pathlib.Path(args.out_json).write_text(
            json.dumps(summary, indent=1), encoding="utf-8"
        )
        n_ep = sorted({v["n_ep"] for v in summary.values()})
        print(f"arms={len(summary)} n_ep={n_ep}")
        return

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suites = {}
    for spec in args.suite:
        name, _, path = spec.partition("=")
        suites[name] = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

    gate_only = {}
    for spec in args.gate_only_suite:
        name, _, path = spec.partition("=")
        gate_only[name] = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))

    payload = {
        "schema_version": 1,
        "experiment": "exp/gate_threshold_pareto",
        "status": args.status,
        "teacher_ratio_definition": (
            "MISS decisions / all decisions, from per_step.jsonl (the gate decides once "
            "every 5 control steps; hit_type in {FULL_HIT, MISS}); accepted episodes only "
            "(status done|failed, accepted=true) with attempt matched to the accepted "
            "journal entry. Owner's new inference_ratio definition is still pending; this "
            "is the direct per-step reading."
        ),
        "success_rate_definition": (
            "journal status==done over (done+failed), accepted rows only, "
            "500 A-pool inits per arm"
        ),
        "raw_source": {
            "node": "timan107",
            "dir": "/scratch/zixuans8/openpi/exp/gate_threshold_pareto/data/eval/<suite>/",
        },
        "inference_ratio_definition": (
            "(N_req*stage1 + N_miss*(stage2+stage3)) / (N_req*(stage1+stage2+stage3)) "
            "with CUDA-Graph stage latencies s1=10.26 s2=27.69 s3=29.57 ms "
            "(authority: latency_bench executor_costs, pi05_stage_split_ms.cuda_graph); "
            "owner-ruled 2026-08-21; equals 0.15195 + 0.84805*teacher_ratio"
        ),
        "suites": suites,
        "suites_gate_only": gate_only or None,
        "gate_only_definition": (
            "verdict disabled (judge threshold=-1.0, every gated probe accepted); "
            "hysteresis gate L=8 is the sole protection"
        ) if gate_only else None,
    }
    (out_dir / "plot_data.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8"
    )
    for suite, arms in suites.items():
        for mode in ("teacher", "inference"):
            print("wrote", plot_suite(suite, arms, out_dir, gate_only.get(suite), x_mode=mode))
    print("wrote", out_dir / "plot_data.json")


if __name__ == "__main__":
    main()
