"""Aggregate the reduced-step teacher rollouts into per-k success rate and cost.

Input: the ``--save-episode-results`` JSON files the LIBERO client writes, one
per (suite, k, shard). Every row is one executed episode with its
``(task_id, orig_init_state_idx)`` identity and ``success``; the shards of one
k are joined here and must tile the pool exactly -- a duplicate identity or a
missing one is a refusal, not a warning, because a shard that was launched
twice or died early would otherwise change the success rate silently.

Cost: the analytic per-decision cost of a k-step teacher is the RIT lines' own
formula, ``s1 + s2 + (k / N) * s3``, read from the same authorities the
frontier figures were priced with -- ``analytic_cost`` for Pi0.5 and the
measured GR00T ledger through ``rit_cost_rc.StageCost`` -- so a k-step teacher
and a warm start that leaves k steps land on the same x. Every decision is a
MISS with k steps, so the inference ratio of a point is a constant of k and
carries no rollout noise.

Output: ``aggregate.json`` (per k: n_ep, success rate, per-task success, IR,
the cost constants and file digests) and a ``rit_pareto.figure/v1`` spec that
``exp.rit_pareto.render_figure --new`` draws. The N-step endpoint is the
suite's existing policy-alone anchor and is passed in explicitly with its
provenance rather than re-run.

Usage:
  uv run python -m exp.nfe_baseline.aggregate_nfe --policy pi05 --suite libero_spatial \
      --runs-dir exp/nfe_baseline/data/runs/pi05_libero_spatial --ks 1-9 \
      --expect-episodes 500 --anchor-sr 0.99 --anchor-source "<where>" \
      --out exp/nfe_baseline/data/runs/pi05_libero_spatial/aggregate.json \
      --figure-out-dir exp/rit_pareto/analysis/figures
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

SCHEMA = "rit_pareto.figure/v1"
RC_POLICIES = ("pi05_rc", "groot_rc")
POLICIES = ("pi05", "groot", *RC_POLICIES)
POLICY_LABEL = {
    "pi05": "π0.5",
    "groot": "GR00T N1.5",
    "pi05_rc": "π0.5",
    "groot_rc": "GR00T N1.5",
}
DEFAULT_GROOT_COST = pathlib.Path(
    "exp/libero_groot/config/rit/cost_groot_libero_measured.json"
)
RC_COST_DEFAULT = {
    "pi05_rc": pathlib.Path("exp/nfe_baseline/config/rc_cost_pi05.json"),
    "groot_rc": pathlib.Path("exp/nfe_baseline/config/rc_cost_groot_tp.json"),
}
RC_TEACHER = {"pi05_rc": "pi05", "groot_rc": "groot_tp"}
RC_LANES = {
    "main": (
        "CloseBlenderLid",
        "CloseFridge",
        "CoffeeSetupMug",
        "OpenCabinet",
        "OpenDrawer",
        "OpenStandMixerHead",
        "SlideDishwasherRack",
        "TurnOnSinkFaucet",
    ),
    "pnp": (
        "PickPlaceCounterToCabinet",
        "PickPlaceCounterToStove",
        "PickPlaceDrawerToCounter",
        "PickPlaceSinkToCounter",
        "PickPlaceToasterToCounter",
    ),
}
STYLE = {
    "pi05": {"color": "#d1495b", "marker": "o"},
    "groot": {"color": "#8d5a97", "marker": "o"},
    "pi05_rc": {"color": "#d1495b", "marker": "o"},
    "groot_rc": {"color": "#8d5a97", "marker": "o"},
}


def load_rc_cost(path: pathlib.Path, teacher: str):
    """The RoboCasa ledger as the RIT line's ``StageCost`` (schedule by teacher)."""
    from exp.robocasa365 import rit_cost_rc as rc
    from openpi.cache.types import PI05_V1, groot_n15_schedule

    d = json.loads(pathlib.Path(path).read_text(encoding="utf-8"))
    schedule = (
        groot_n15_schedule(int(d.get("num_steps", 4)))
        if teacher == "groot_tp"
        else PI05_V1
    )
    return rc.StageCost(
        teacher=teacher,
        schedule=schedule,
        stage1_ms=float(d["stage1_ms"]),
        stage2_ms=float(d["stage2_ms"]),
        stage3_head_ms=float(d["stage3_head_ms"]),
        stage3_step_ms=float(d["stage3_step_ms"]),
        provenance=str(d.get("provenance", path)),
        linear_stage3=bool(d.get("linear_stage3", False)),
    )


_POINT_STYLE = {
    "marker_size": 40,
    "alpha": 0.9,
    "point_zorder": 4,
    "linestyle": "-",
    "linewidth": 2.2,
    "line_zorder": 3,
    "annotate_fontsize": 7,
}


# ------------------------------------------------------------------
# Cost
# ------------------------------------------------------------------


class CostModel:
    """``s1 + s2 + (k / N) * s3`` over the policy's own stage constants."""

    def __init__(self, policy: str, groot_cost: pathlib.Path | None = None):
        if policy == "pi05":
            from exp.dispatch_surface.analysis import analytic_cost as ac
            from openpi.cache.types import PI05_V1

            self.num_steps = PI05_V1.num_steps
            self.stage1 = ac.STAGE1_MS
            self.stage2 = ac.STAGE2_MS
            self._stage3 = lambda k: (k / self.num_steps) * ac.STAGE3_MS
            self.miss = ac.unit_cost("MISS", None)
            self.source = "exp.dispatch_surface.analysis.analytic_cost"
        elif policy == "groot":
            from exp.libero_groot.aggregate_rit_lg import load_cost
            from exp.robocasa365 import rit_cost_rc as rc

            path = groot_cost or DEFAULT_GROOT_COST
            cost = load_cost(path)
            self.num_steps = cost.schedule.num_steps
            self.stage1 = cost.stage1_ms
            self.stage2 = cost.stage2_ms
            self._stage3 = cost.stage3_ms
            self.miss = rc.miss_cost(cost)
            self.source = f"{path} (sha256 {_sha(path)[:12]})"
        elif policy in RC_POLICIES:
            # RoboCasa365: the RIT line's measured ledgers, priced through the
            # same StageCost the RoboCasa frontier was drawn with (stage 3 is the
            # ledger's linear fit a + b*k, not the per-k samples).
            from exp.robocasa365 import rit_cost_rc as rc

            path = groot_cost or RC_COST_DEFAULT[policy]
            cost = load_rc_cost(path, RC_TEACHER[policy])
            self.num_steps = cost.schedule.num_steps
            self.stage1 = cost.stage1_ms
            self.stage2 = cost.stage2_ms
            self._stage3 = cost.stage3_ms
            self.miss = rc.miss_cost(cost)
            self.source = f"{path} (sha256 {_sha(path)[:12]})"
        else:
            raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")

    def decision_ms(self, k: int) -> float:
        if not 1 <= k <= self.num_steps:
            raise ValueError(f"k={k} outside 1..{self.num_steps}")
        return self.stage1 + self.stage2 + self._stage3(k)

    def ir_percent(self, k: int) -> float:
        return 100.0 * self.decision_ms(k) / self.miss


# ------------------------------------------------------------------
# Rows
# ------------------------------------------------------------------


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_ks(spec: str) -> list[int]:
    """``1-9`` or ``1,3,5`` -> sorted distinct ints."""
    ks: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            ks.update(range(int(a), int(b) + 1))
        elif part:
            ks.add(int(part))
    return sorted(ks)


def join_shards(paths: list[pathlib.Path], *, suite: str, expect: int | None) -> dict:
    """Join the shard files of one k; refuse duplicates, wrong suite, wrong count."""
    seen: dict[tuple[int, int], bool] = {}
    per_task: dict[int, list[bool]] = {}
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        for r in rows:
            if r.get("task_suite_name") != suite:
                raise SystemExit(
                    f"{path}: row suite {r.get('task_suite_name')!r} != {suite!r}"
                )
            key = (int(r["task_id"]), int(r["orig_init_state_idx"]))
            if key in seen:
                raise SystemExit(
                    f"{path}: duplicate episode task={key[0]} init={key[1]}"
                )
            seen[key] = bool(r["success"])
            per_task.setdefault(key[0], []).append(bool(r["success"]))
    n = len(seen)
    if expect is not None and n != expect:
        raise SystemExit(f"{[p.name for p in paths]}: {n} episodes, expected {expect}")
    if n == 0:
        raise SystemExit(f"{[p.name for p in paths]}: no episodes")
    return {
        "n_ep": n,
        "success_rate": sum(seen.values()) / n,
        "per_task": {
            str(t): {"n": len(v), "success_rate": sum(v) / len(v)}
            for t, v in sorted(per_task.items())
        },
        "shard_files": {p.name: _sha(p) for p in paths},
    }


def shard_files(runs_dir: pathlib.Path, suite: str, k: int) -> list[pathlib.Path]:
    return sorted(runs_dir.glob(f"{suite}_k{k}_s*.json"))


def rc_summary_path(
    root: pathlib.Path, teacher: str, lane: str, k: int, prefix: str = "nfek"
) -> pathlib.Path:
    return (
        root
        / teacher
        / lane
        / f"k{k}"
        / f"summary_{prefix}{k}-teacher__l1s1_{teacher}.json"
    )


def join_rc_summaries(
    paths: dict[str, pathlib.Path], *, teacher: str, per_task: int | None
) -> dict:
    """Join the main and pnp lane summaries of one k into the 13-task macro SR.

    Every lane must be ``complete`` (no errors, nothing missing), carry exactly
    its roster and, when ``per_task`` is given, score that many episodes on
    every task -- a short task would move the macro average without failing
    anything. Macro SR is the equal-weight mean over the 13 tasks (RIT line D1),
    never the episode-pooled rate.
    """
    tasks: dict[str, dict] = {}
    files: dict[str, str] = {}
    for lane, path in paths.items():
        d = json.loads(path.read_text(encoding="utf-8"))
        if d.get("teacher") != teacher:
            raise SystemExit(f"{path}: teacher {d.get('teacher')!r} != {teacher!r}")
        if not d.get("complete"):
            raise SystemExit(
                f"{path}: lane {lane} not complete (n_err={d.get('n_err')}, n_missing={d.get('n_missing')})"
            )
        roster = set(RC_LANES[lane])
        got = set(d["tasks"])
        if got != roster:
            raise SystemExit(
                f"{path}: lane {lane} tasks {sorted(got ^ roster)} differ from the roster"
            )
        for t, v in d["tasks"].items():
            if per_task is not None and int(v["n_scored"]) != per_task:
                raise SystemExit(
                    f"{path}: task {t} scored {v['n_scored']} episodes, expected {per_task}"
                )
            tasks[t] = {"n": int(v["n_scored"]), "success_rate": float(v["sr"])}
        files[path.name] = _sha(path)
    n_ep = sum(v["n"] for v in tasks.values())
    macro = sum(v["success_rate"] for v in tasks.values()) / len(tasks)
    return {
        "n_ep": n_ep,
        "n_tasks": len(tasks),
        "success_rate": macro,
        "per_task": dict(sorted(tasks.items())),
        "summary_files": files,
    }


def aggregate_rc(
    policy: str,
    root: pathlib.Path,
    ks: list[int],
    *,
    per_task: int | None,
    cost: CostModel,
    anchor: dict | None,
    prefix: str = "nfek",
) -> dict:
    """Per-k macro SR and analytic IR of one RoboCasa teacher from its lane summaries."""
    teacher = RC_TEACHER[policy]
    per_k: dict[int, dict] = {}
    for k in ks:
        paths = {
            lane: rc_summary_path(root, teacher, lane, k, prefix) for lane in RC_LANES
        }
        missing = [str(p) for p in paths.values() if not p.exists()]
        if missing:
            raise SystemExit(f"k={k}: missing summaries {missing}")
        rec = join_rc_summaries(paths, teacher=teacher, per_task=per_task)
        rec["k"] = k
        rec["decision_ms"] = cost.decision_ms(k)
        rec["ir_percent"] = cost.ir_percent(k)
        per_k[k] = rec
    return {
        "policy": policy,
        "suite": "robocasa365",
        "teacher": teacher,
        "num_steps": cost.num_steps,
        "sr_definition": "macro over 13 tasks (equal task weight); 50 seeds per task = 1,000,000 + idx",
        "cost": {
            "source": cost.source,
            "stage1_ms": cost.stage1,
            "stage2_ms": cost.stage2,
            "stage3_full_ms": cost._stage3(cost.num_steps),
            "miss_ms": cost.miss,
            "formula": "stage1 + stage2 + stage3(k) with the ledger's linear fit a + b*k",
        },
        "anchor": anchor,
        "per_k": {str(k): per_k[k] for k in sorted(per_k)},
    }


# ------------------------------------------------------------------
# Figure spec
# ------------------------------------------------------------------


def figure_spec(
    policy: str,
    suite: str,
    per_k: dict[int, dict],
    cost: CostModel,
    anchor: dict | None,
) -> dict:
    n = cost.num_steps
    pts = []
    for k in sorted(per_k):
        rec = per_k[k]
        pts.append(
            {
                "id": f"{policy}_{suite}_k{k}",
                "x": rec["ir_percent"],
                "y": rec["success_rate"],
                "label": f"k={k}",
                "label_offset": [4, 4],
                "n_ep": rec["n_ep"],
            }
        )
    if anchor is not None:
        pts.append(
            {
                "id": f"{policy}_{suite}_k{n}_anchor",
                "x": 100.0,
                "y": anchor["success_rate"],
                "label": f"k={n} (policy alone)",
                "label_offset": [4, 4],
                "n_ep": anchor["n_ep"],
            }
        )
    key = f"{POLICY_LABEL[policy]} alone, k of {n} denoising steps"
    series = [
        {
            "key": key,
            "style": {**STYLE[policy], **_POINT_STYLE},
            "show_points": True,
            "show_frontier": True,
            "annotate": True,
            "scatter_label": f"{key}: points ({{n_arms}} x {pts[0]['n_ep']} ep)",
            "frontier_label": f"{key}: Pareto frontier ({{n_front}} non-dominated)",
            "points": pts,
        }
    ]
    return {
        "schema": SCHEMA,
        "figure_id": f"nfe_{policy}_{suite}",
        "title": f"{suite}: {POLICY_LABEL[policy]} alone with k of {n} denoising steps, no cache "
        f"({pts[0]['n_ep']} episodes per point)",
        "x_label": f"inference ratio (% of full-step teacher cost; s1 + s2 + (k/{n}) s3)",
        "y_label": "success rate",
        "figsize": [8, 5.5],
        "dpi": 160,
        "grid_alpha": 0.3,
        "legend": {"fontsize": 7, "loc": "lower right"},
        "series": series,
    }


def reference_overlay(
    source_figure: str, source_series: str, key: str, color: str
) -> dict:
    """A frontier read from an existing RIT figure spec, drawn for comparison only."""
    return {
        "key": key,
        "style": {
            "color": color,
            "marker": ".",
            "marker_size": 18,
            "alpha": 0.8,
            "point_zorder": 2,
            "linestyle": "--",
            "linewidth": 1.6,
            "line_zorder": 1,
            "annotate_fontsize": 6,
        },
        "show_points": False,
        "show_frontier": True,
        "annotate": False,
        "source_figure": source_figure,
        "source_series": source_series,
        "scatter_label": f"{key}: arms ({{n_arms}})",
        "frontier_label": f"{key} (frontier, reference)",
    }


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def aggregate(
    policy: str,
    suite: str,
    runs_dir: pathlib.Path,
    ks: list[int],
    *,
    expect: int | None,
    cost: CostModel,
    anchor: dict | None,
) -> dict:
    per_k: dict[int, dict] = {}
    for k in ks:
        files = shard_files(runs_dir, suite, k)
        if not files:
            raise SystemExit(f"no shard files for {suite} k={k} under {runs_dir}")
        rec = join_shards(files, suite=suite, expect=expect)
        rec["k"] = k
        rec["decision_ms"] = cost.decision_ms(k)
        rec["ir_percent"] = cost.ir_percent(k)
        per_k[k] = rec
    return {
        "policy": policy,
        "suite": suite,
        "num_steps": cost.num_steps,
        "cost": {
            "source": cost.source,
            "stage1_ms": cost.stage1,
            "stage2_ms": cost.stage2,
            "stage3_full_ms": cost._stage3(cost.num_steps),
            "miss_ms": cost.miss,
            "formula": "stage1 + stage2 + (k / N) * stage3",
        },
        "anchor": anchor,
        "per_k": {str(k): per_k[k] for k in sorted(per_k)},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--policy", choices=POLICIES, required=True)
    ap.add_argument("--suite", required=True)
    ap.add_argument(
        "--runs-dir",
        default="",
        help="LIBERO: directory of <suite>_k<k>_s<s>.json shard results",
    )
    ap.add_argument(
        "--summaries-root",
        default="",
        help="RoboCasa: root of <teacher>/<lane>/k<k>/summary_*.json (run_ws_search --journal-dir)",
    )
    ap.add_argument(
        "--per-task",
        type=int,
        default=50,
        help="RoboCasa: episodes every task must score; 0 disables",
    )
    ap.add_argument(
        "--run-prefix",
        default="nfek",
        help="RoboCasa: run_ws_search --run-prefix stem (k appended)",
    )
    ap.add_argument("--ks", required=True, help="e.g. 1-9 or 1,3,5")
    ap.add_argument(
        "--expect-episodes",
        type=int,
        default=500,
        help="episodes every k must have; 0 disables the check (smoke only)",
    )
    ap.add_argument(
        "--groot-cost",
        default="",
        help="cost ledger JSON (default: the LIBERO GR00T measured ledger, or the RoboCasa ledger of the policy)",
    )
    ap.add_argument(
        "--anchor-sr",
        type=float,
        default=None,
        help="policy-alone success rate at N steps (existing anchor, not re-run)",
    )
    ap.add_argument("--anchor-n-ep", type=int, default=500)
    ap.add_argument(
        "--anchor-source", default="", help="where the anchor number comes from"
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("--figure-out-dir", default="")
    ap.add_argument(
        "--reference",
        action="append",
        default=[],
        help="source_figure:source_series:legend key, drawn as a dashed reference frontier",
    )
    args = ap.parse_args()

    cost = CostModel(
        args.policy, pathlib.Path(args.groot_cost) if args.groot_cost else None
    )
    anchor = None
    if args.anchor_sr is not None:
        if not args.anchor_source:
            raise SystemExit(
                "--anchor-sr needs --anchor-source (provenance of the number)"
            )
        anchor = {
            "success_rate": args.anchor_sr,
            "n_ep": args.anchor_n_ep,
            "source": args.anchor_source,
            "k": cost.num_steps,
            "ir_percent": 100.0,
        }
    expect = args.expect_episodes if args.expect_episodes > 0 else None
    if args.policy in RC_POLICIES:
        if not args.summaries_root:
            raise SystemExit("RoboCasa policies take --summaries-root")
        agg = aggregate_rc(
            args.policy,
            pathlib.Path(args.summaries_root),
            parse_ks(args.ks),
            per_task=(args.per_task or None),
            cost=cost,
            anchor=anchor,
            prefix=args.run_prefix,
        )
    else:
        if not args.runs_dir:
            raise SystemExit("LIBERO policies take --runs-dir")
        agg = aggregate(
            args.policy,
            args.suite,
            pathlib.Path(args.runs_dir),
            parse_ks(args.ks),
            expect=expect,
            cost=cost,
            anchor=anchor,
        )
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(agg, indent=1), encoding="utf-8")
    for k, rec in agg["per_k"].items():
        print(
            f"{args.policy} {args.suite} k={k}: IR {rec['ir_percent']:.2f}%  SR {rec['success_rate']:.3f}  "
            f"(n={rec['n_ep']})"
        )
    if args.figure_out_dir:
        per_k = {int(k): v for k, v in agg["per_k"].items()}
        spec = figure_spec(args.policy, args.suite, per_k, cost, anchor)
        palette = iter(["#5b6770", "#2f6db5", "#c8102e", "#e07b39"])
        for ref in args.reference:
            fig, ser, key = ref.split(":", 2)
            spec["series"].append(reference_overlay(fig, ser, key, next(palette)))
        fdir = pathlib.Path(args.figure_out_dir)
        fdir.mkdir(parents=True, exist_ok=True)
        fpath = fdir / f"{spec['figure_id']}.json"
        fpath.write_text(json.dumps(spec, indent=1), encoding="utf-8")
        print(f"figure spec -> {fpath}")


if __name__ == "__main__":
    main()
