"""Summarise the fusion-weight ablation: per-arm success on the A pool with paired comparisons.

Inputs are the two lanes' native outputs, read as they are:

* Pi0.5: the conductor journal (``task_uid = <arm>:eval:<task>:<episode>``,
  one terminal row per uid with ``success``);
* GR00T: one per-episode JSON per cell from ``orchestrate_search``
  (``task_id`` / ``init_state_idx`` / ``success``), plus, optionally, the
  historical grid cells in the same shape so the leader can be paired in.

Per arm: n, success rate, Wilson 95% interval. Per suite: every pair of arms
compared on the shared (task, init) keys with an exact sign-flip test on the
discordant episodes (McNemar without continuity correction, two-sided).

Usage:
  python exp/weighted_sum/analysis/lcw_ablation_summary.py \
      --pi05-journal libero_spatial=<journal.jsonl> --pi05-journal libero_10=<journal.jsonl> \
      --groot-results libero_spatial=<abl_results dir> --groot-results libero_10=<abl_results dir> \
      --groot-grid libero_spatial=<r1_results/v0@3_v1@2_rs@1.json> ... --out <json>

Strict manifest entry (key-builder x LDA supplement): ``--input-manifest
<active_manifest.json>`` replaces the per-lane flags. The manifest fixes, per
suite, the journals / per-step files / launch records of the pool and LLM
rounds, the reference arm to take from the fusion-ablation journal, and the
A-pool identity. Everything is verified before any rate is printed: the arm
set equals the manifest's exactly, every arm has the full 500 accepted eval
episodes over the same (task, init) keys, no accepted record carries a worker
error, every new arm's accepted episode has a FULL_HIT-only trace from its own
run and attempt, and every run that produced an accepted episode has a launch
record binding it to the manifest's A-pool digest. The script stays runnable
by path, so the lane flags work without ``PYTHONPATH``.
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import math
import pathlib
import sys


# Direct script execution needs the repository root for the shared evidence reader.
_REPO = pathlib.Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from exp.common.conductor_evidence import Winner, merge_hits, merge_source, witness_full_hit  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided sign test on discordant counts (b wins for A, c wins for B)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def load_pi05(path: pathlib.Path) -> dict[str, dict[tuple[int, int], bool]]:
    arms: dict[str, dict[tuple[int, int], bool]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not r.get("accepted", True):
            continue
        arm, phase, task, ep = r["task_uid"].rsplit(":", 3)
        arms.setdefault(arm, {})[(int(task), int(ep))] = bool(r["success"])
    return arms


def load_groot(directory: pathlib.Path) -> dict[str, dict[tuple[int, int], bool]]:
    arms = {}
    for p in sorted(directory.glob("*.json")):
        if p.name.endswith(".partial.json"):
            continue
        rows = json.loads(p.read_text())
        arms[p.stem] = {(int(r["task_id"]), int(r["init_state_idx"])): bool(r["success"]) for r in rows}
    return arms


def load_grid_cell(path: pathlib.Path) -> dict[tuple[int, int], bool]:
    rows = json.loads(path.read_text())
    return {(int(r["task_id"]), int(r["init_state_idx"])): bool(r["success"]) for r in rows}


# ------------------------------------------------------------------
# Strict manifest entry
# ------------------------------------------------------------------
NUM_TASKS = 10
TRIALS_PER_TASK = 50


def _as_list(x) -> list:
    return list(x) if isinstance(x, (list, tuple)) else [x]


def check_launch(path: pathlib.Path, *, suite: str, arms: set[str], rollup_sha256: str, require_run_id: bool) -> dict:
    """The driver's launch record must bind the run to the suite, the frozen A pool and the full protocol.

    A run source's record must also name the run it launched (``run_id``), so
    that every accepted winner can be tied back to a launch that was checked.
    A legacy record (the reference arm, frozen before run ids were recorded on
    the launch side) is accepted only where the manifest says so.
    """
    if not path.exists():
        raise SystemExit(f"{path}: launch record missing; the run cannot be bound to an A pool")
    launch = json.loads(path.read_text())
    if launch.get("suite") != suite:
        raise SystemExit(f"{path}: launch suite {launch.get('suite')!r} != {suite!r}")
    if launch.get("smoke"):
        raise SystemExit(f"{path}: a smoke launch cannot feed the reported table")
    if int(launch.get("trials_per_task", 0)) != TRIALS_PER_TASK:
        raise SystemExit(f"{path}: trials_per_task {launch.get('trials_per_task')} != {TRIALS_PER_TASK}")
    apool = launch.get("apool") or {}
    if apool.get("rollup_sha256") != rollup_sha256 or apool.get("suite") != suite:
        raise SystemExit(f"{path}: A-pool rollup {apool.get('rollup_sha256')} / suite {apool.get('suite')} "
                         f"does not match the manifest ({rollup_sha256}, {suite})")
    missing = arms - set(launch.get("arms", []))
    if missing:
        raise SystemExit(f"{path}: launch did not include arms {sorted(missing)}")
    if require_run_id and not launch.get("run_id"):
        raise SystemExit(f"{path}: launch record carries no run_id; a run source must be bound per run "
                         "(the driver writes per_step.jsonl.launch.<run_id>.json)")
    return launch


def check_coverage(arms: dict[str, dict[str, Winner]], *, expected: int) -> None:
    keys_expected = {(t, e) for t in range(NUM_TASKS) for e in range(TRIALS_PER_TASK)}
    key_sets = {}
    for arm, recs in arms.items():
        if len(recs) != expected:
            raise SystemExit(f"{arm}: {len(recs)} accepted episodes, expected {expected}")
        keys = {(r.task_id, r.episode_idx) for r in recs.values()}
        if keys != keys_expected:
            raise SystemExit(f"{arm}: (task, init) keys are not the {NUM_TASKS}x{TRIALS_PER_TASK} A-pool protocol "
                             f"(missing e.g. {sorted(keys_expected - keys)[:5]}, extra e.g. {sorted(keys - keys_expected)[:5]})")
        key_sets[arm] = keys
    for a, b in itertools.combinations(sorted(key_sets), 2):
        if len(key_sets[a] & key_sets[b]) != expected:
            raise SystemExit(f"{a} vs {b}: paired intersection {len(key_sets[a] & key_sets[b])} != {expected}")


def load_manifest_suite(suite: str, spec: dict, *, expected: int) -> tuple[dict, dict]:
    """Return ``({arm: {(task, init): success}}, provenance)`` for one suite, or raise.

    A round's ``journal`` / ``per_step`` / ``launch`` may each be one path or a
    non-empty list (a round resumed into another output directory). A run
    source (``launch_binding: run_id``, the default) must have a checked launch
    record for every run that produced an accepted episode; the reference arm
    may declare ``launch_binding: legacy`` for its frozen pre-run-id record.
    """
    sources = [(rnd, spec["rounds"][rnd], True) for rnd in ("pool", "llm")]
    sources.append(("reference", spec["reference"], False))
    declared = [set(r["arms"]) for _, r, _ in sources]
    for i, a in enumerate(declared):
        for b in declared[i + 1:]:
            if a & b:
                raise SystemExit(f"{suite}: arm declared in two sources: {sorted(a & b)}")
    expected_arms = set().union(*declared)
    merged: dict[str, dict[str, Winner]] = {}
    provenance = {}
    for name, r, restrict in sources:
        allowed = set(r["arms"])
        # Launch records may be listed or discovered by the driver's per-run
        # naming (launch_glob); either way there must be at least one.
        launch_paths = _as_list(r.get("launch") or [])
        if not launch_paths and r.get("launch_glob"):
            launch_paths = sorted(glob.glob(r["launch_glob"]))
        for field, paths in (("journal", _as_list(r.get("journal") or [])), ("launch", launch_paths),
                             ("per_step", _as_list(r.get("per_step") or []) if restrict else ["-"])):
            if not paths:
                raise SystemExit(f"{suite}/{name}: {field} must name at least one file")
        binding = r.get("launch_binding", "run_id")
        if binding not in ("run_id", "legacy"):
            raise SystemExit(f"{suite}/{name}: unknown launch_binding {binding!r}")
        if binding == "legacy" and restrict:
            raise SystemExit(f"{suite}/{name}: a run source cannot use the legacy launch contract")
        launches = [check_launch(pathlib.Path(lp), suite=suite, arms=allowed, rollup_sha256=spec["apool_rollup_sha256"],
                                 require_run_id=(binding == "run_id")) for lp in launch_paths]
        for jp in _as_list(r["journal"]):
            merge_source(merged, name, pathlib.Path(jp), allowed, restrict=restrict)
        absent = allowed - set(merged)
        if absent:
            raise SystemExit(f"{name}: arms {sorted(absent)} have no accepted episode in {_as_list(r['journal'])}")
        run_ids = {w.run_id for arm in allowed for w in merged[arm].values()}
        if binding == "run_id":
            launched = {lc["run_id"] for lc in launches}
            unbound = run_ids - launched
            if unbound:
                raise SystemExit(f"{suite}/{name}: accepted episodes come from run(s) {sorted(unbound)} with no "
                                 f"checked launch record (launched: {sorted(launched)})")
        if restrict:
            hits = merge_hits(_as_list(r["per_step"]))
            for arm in sorted(allowed):
                provenance[arm] = {"source": name, "journal": _as_list(r["journal"]), "launches": launch_paths,
                                   "full_hit": witness_full_hit(arm, merged[arm], hits)}
        else:
            for arm in sorted(allowed):
                provenance[arm] = {"source": name, "journal": _as_list(r["journal"]), "launch_binding": binding,
                                   "launch_arms": [lc["arms"] for lc in launches]}
    if set(merged) != expected_arms:
        raise SystemExit(f"{suite}: merged arms {sorted(merged)} != manifest {sorted(expected_arms)}")
    check_coverage(merged, expected=expected)
    for arm, recs in merged.items():
        attempts = [w.attempt for w in recs.values()]
        provenance[arm]["n"] = len(recs)
        provenance[arm]["attempts"] = {str(a): attempts.count(a) for a in sorted(set(attempts))}
        provenance[arm]["run_ids"] = sorted({w.run_id for w in recs.values()})
    out = {arm: {(w.task_id, w.episode_idx): w.success for w in recs.values()} for arm, recs in merged.items()}
    return out, provenance


def summarise(suite: str, arms: dict[str, dict[tuple[int, int], bool]]) -> dict:
    out = {"suite": suite, "arms": {}, "pairs": []}
    for name, eps in arms.items():
        n, k = len(eps), sum(eps.values())
        lo, hi = wilson(k, n)
        per_task = {}
        for (t, _), s in eps.items():
            per_task.setdefault(t, [0, 0])
            per_task[t][0] += int(s)
            per_task[t][1] += 1
        out["arms"][name] = {"n": n, "successes": k, "sr": k / n if n else float("nan"),
                             "wilson95": [lo, hi],
                             "per_task_sr": {str(t): v[0] / v[1] for t, v in sorted(per_task.items())}}
        print(f"  {name:36s} n={n:4d} SR={k / n if n else float('nan'):.3f} [{lo:.3f}, {hi:.3f}]")
    for a, b in itertools.combinations(arms, 2):
        shared = set(arms[a]) & set(arms[b])
        wins_a = sum(arms[a][k] and not arms[b][k] for k in shared)
        wins_b = sum(arms[b][k] and not arms[a][k] for k in shared)
        diff = (sum(arms[a][k] for k in shared) - sum(arms[b][k] for k in shared)) / max(len(shared), 1)
        p = binom_two_sided(wins_a, wins_b)
        out["pairs"].append({"a": a, "b": b, "shared": len(shared), "diff_pp": 100 * diff,
                             "a_only": wins_a, "b_only": wins_b, "p_sign": p})
        print(f"    {a} vs {b}: shared={len(shared)} diff={100 * diff:+.1f} pp  discordant {wins_a}/{wins_b}  p={p:.3f}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pi05-journal", action="append", default=[], help="suite=path")
    ap.add_argument("--groot-results", action="append", default=[], help="suite=dir")
    ap.add_argument("--groot-grid", action="append", default=[], help="suite=path (historical leader cell json)")
    ap.add_argument("--input-manifest", type=pathlib.Path, default=None,
                    help="active_manifest.json of the key-builder supplement; strict, exclusive of the lane flags")
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()
    report = {}
    if args.input_manifest is not None:
        if args.pi05_journal or args.groot_results or args.groot_grid:
            raise SystemExit("--input-manifest is exclusive of the per-lane flags")
        manifest = json.loads(args.input_manifest.read_text())
        expected = int(manifest.get("episodes_per_arm", NUM_TASKS * TRIALS_PER_TASK))
        for suite, spec in manifest["suites"].items():
            arms, provenance = load_manifest_suite(suite, spec, expected=expected)
            print(f"== pi05 {suite} (manifest)")
            report[f"pi05_{suite}"] = summarise(suite, arms)
            report[f"pi05_{suite}"]["provenance"] = provenance
        report["manifest"] = {"path": str(args.input_manifest), "clip_deferred": manifest.get("clip_deferred"),
                              "pairs_note": "every pairwise sign test is exploratory and uncorrected for "
                                            "multiple comparisons (plan section 3.5); none is confirmatory"}
    for spec in args.pi05_journal:
        suite, path = spec.split("=", 1)
        print(f"== pi05 {suite}")
        report[f"pi05_{suite}"] = summarise(suite, load_pi05(pathlib.Path(path)))
    grids = dict(s.split("=", 1) for s in args.groot_grid)
    for spec in args.groot_results:
        suite, d = spec.split("=", 1)
        arms = load_groot(pathlib.Path(d))
        if suite in grids:
            arms["grid_leader:" + pathlib.Path(grids[suite]).stem] = load_grid_cell(pathlib.Path(grids[suite]))
        print(f"== groot {suite}")
        report[f"groot_{suite}"] = summarise(suite, arms)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
