"""Post-run artifact audit + deterministic manifest for RoboCasa365 collection.

This is the ONLY gate that can catch the collector's silent failure modes, so
it is a BLOCKING precondition of formal collection (T5), not an optional
sweep: ``EpisodeDataCollector.on_episode_end`` swallows HDF5 write errors into
a server-side log line, and a zero-inference episode writes nothing at all —
in both cases the server still acks ``episode_end`` and the journal still
records the episode as done.

Contracts implemented here (frozen in the approved plan §4.3.6):

* **Admission** — a journal record is admissible iff ``accepted is True`` and
  ``success is True`` and ``error is None``. The journal has NO
  one-terminal-row-per-uid invariant: stale attempts are journaled with
  ``accepted=False``, retry-exhausted uids are never journaled at all, and a
  killed worker's first attempt reports nothing.
* **Expected-UID source** — the union of the immutable run-plan artifacts
  passed via ``--run-plan`` (written by ``run_collect.py`` from the actual
  dispatch graph). The auditor never re-derives UIDs itself; a uid in the plan
  with no journal row at all is reported as ``missing_terminal``.
* **Provenance** — h5 attempts without an admissible row are ``orphan_attempt``
  (expected, informational); admissible rows without their h5 are
  ``missing_file`` (an error the journal alone cannot see).
* **Manifest** — per task: admissible + schema-valid successes, ordered by
  ``episode_idx`` ascending, first ``--target`` entries, each with sha256.
  Consumers build libraries from the manifest, never from directory listings.
* **Completion rule** — ``min_episodes_for_target`` computes the smallest N
  with ``P(Binom(N, sr) >= target) >= confidence`` at the SR point estimate
  (no Wilson lower bound: stacked conservatism explodes the budget; deficits
  are covered by deterministic extension batches instead). Lower tails are
  summed via ``lgamma`` — ``math.comb`` products overflow floats near N≈1000.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Any

REQUIRED_STEP_FIELDS = ("vision_0", "vision_1", "vision_2", "prompt_emb", "robot_state", "clean_action")


# ------------------------------------------------------------------
# Completion rule (frozen §4.3.6-(4))
# ------------------------------------------------------------------


def _log_pmf(n: int, k: int, p: float) -> float:
    return (
        math.lgamma(n + 1)
        - math.lgamma(k + 1)
        - math.lgamma(n - k + 1)
        + k * math.log(p)
        + (n - k) * math.log1p(-p)
    )


def prob_at_least(n: int, p: float, target: int) -> float:
    """P(Binom(n, p) >= target); stable via the (short) lower tail."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0 if n >= target else 0.0
    if n < target:
        return 0.0
    lower = sum(math.exp(_log_pmf(n, k, p)) for k in range(target))
    return 1.0 - lower


def min_episodes_for_target(sr: float, *, target: int = 20, confidence: float = 0.90, cap: int = 20000) -> int:
    """Smallest N with P(Binom(N, sr) >= target) >= confidence (point estimate)."""
    if not 0.0 < sr <= 1.0:
        raise ValueError(f"sr must be in (0, 1], got {sr}")
    n = target
    while n <= cap:
        if prob_at_least(n, sr, target) >= confidence:
            return n
        n += 1
    raise ValueError(f"no N <= {cap} reaches P(X>={target}) >= {confidence} at sr={sr}")


# ------------------------------------------------------------------
# Run-plan loading (frozen §4.3.6-(6))
# ------------------------------------------------------------------


def compute_plan_hash(payload: dict[str, Any]) -> str:
    body = {key: value for key, value in payload.items() if key != "plan_hash"}
    canonical = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_run_plan(path: str | pathlib.Path) -> dict[str, Any]:
    """Load one run-plan, re-verifying its stored hash on every read."""
    payload = json.loads(pathlib.Path(path).read_text())
    stored = payload.get("plan_hash")
    recomputed = compute_plan_hash(payload)
    if stored != recomputed:
        raise ValueError(f"run-plan {path}: stored plan_hash {stored} != recomputed {recomputed}")
    return payload


def merge_run_plans(plans: list[dict[str, Any]]) -> tuple[list[str], dict[str, str], dict[str, int], list[str]]:
    """Union of expected uids across batches; duplicate/conflicting uids are an error."""
    uids: list[str] = []
    prefixes: dict[str, str] = {}
    batches: dict[str, int] = {}
    hashes: list[str] = []
    seen: set[str] = set()
    for plan in plans:
        hashes.append(plan["plan_hash"])
        batch = int(plan["params"]["batch"])
        for uid in plan["uids"]:
            if uid in seen:
                raise ValueError(f"uid {uid!r} appears in more than one run-plan; batches must be disjoint")
            seen.add(uid)
            uids.append(uid)
            prefixes[uid] = plan["prefixes"][uid]
            batches[uid] = batch
    return uids, prefixes, batches, hashes


# ------------------------------------------------------------------
# Journal loading + admission
# ------------------------------------------------------------------


def load_journal(path: str | pathlib.Path) -> list[dict[str, Any]]:
    records = []
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def is_admissible(record: dict[str, Any]) -> bool:
    return record.get("accepted") is True and record.get("success") is True and record.get("error") is None


# ------------------------------------------------------------------
# Audit
# ------------------------------------------------------------------


def _check_h5_schema(path: pathlib.Path, expected_task: str) -> list[str]:
    """Validate ONE admitted file: attrs (values included) + EVERY step group.

    Checking only the first step would pass a file whose write died halfway; an
    admitted file's ``success`` attr must also BE true, not merely exist — the
    journal said success, and a False attr means the h5 belongs to a different
    outcome than the ledger claims.
    """
    import h5py

    problems: list[str] = []
    with h5py.File(path, "r") as f:
        for attr in ("task", "success", "num_steps"):
            if attr not in f.attrs:
                problems.append(f"missing attr {attr!r}")
        task_attr = str(f.attrs.get("task", ""))
        if task_attr != expected_task:
            problems.append(f"attr task={task_attr!r} != canonical {expected_task!r}")
        if "success" in f.attrs and not bool(f.attrs["success"]):
            problems.append("attr success=False on a journal-admitted (success) episode")
        step_groups = sorted(k for k in f.keys() if k.startswith("step_"))
        num_steps = int(f.attrs.get("num_steps", -1))
        if num_steps != len(step_groups):
            problems.append(f"num_steps={num_steps} != {len(step_groups)} step groups")
        if not step_groups:
            problems.append("zero step groups")
        for name in step_groups:
            group = f[name]
            for field in REQUIRED_STEP_FIELDS:
                if field not in group:
                    problems.append(f"{name}: step field {field!r} missing")
    return problems


def audit(
    *,
    root: str | pathlib.Path,
    journal_records: list[dict[str, Any]],
    plans: list[dict[str, Any]],
    target: int,
) -> dict[str, Any]:
    """Full audit against the run-plan uid set. Returns a JSON-able report."""
    root = pathlib.Path(root)
    expected_uids, prefixes, batches, plan_hashes = merge_run_plans(plans)

    by_uid: dict[str, list[dict[str, Any]]] = {}
    for record in journal_records:
        by_uid.setdefault(record["task_uid"], []).append(record)

    missing_terminal: list[str] = []
    failed: list[str] = []
    missing_file: list[str] = []
    schema_errors: dict[str, list[str]] = {}
    multiple_accepted: list[str] = []
    admitted: dict[str, dict[str, Any]] = {}  # uid -> {path, attempt, ...}

    for uid in expected_uids:
        rows = by_uid.get(uid, [])
        if not rows:
            # Retry exhaustion and never-ran both leave zero journal rows —
            # only the run-plan can surface them.
            missing_terminal.append(uid)
            continue
        admissible = [r for r in rows if is_admissible(r)]
        if len(admissible) > 1:
            multiple_accepted.append(uid)
            continue
        if not admissible:
            failed.append(uid)  # legitimate failure (all attempts unsuccessful)
            continue
        row = admissible[0]
        attempt = int(row.get("attempt", 1))
        h5_path = root / f"{prefixes[uid]}_a{attempt:02d}.h5"
        if not h5_path.is_file():
            missing_file.append(uid)
            continue
        task_name = prefixes[uid].split("/")[1]
        problems = _check_h5_schema(h5_path, task_name)
        if problems:
            schema_errors[uid] = problems
            continue
        admitted[uid] = {
            "path": str(h5_path.relative_to(root)),
            "attempt": attempt,
            "batch": batches[uid],
        }

    # Provenance sweep: h5 attempts on disk without an admitted row are
    # expected leftovers of retries/failures, reported but never an error.
    admitted_paths = {entry["path"] for entry in admitted.values()}
    orphan_attempts = []
    if root.is_dir():
        for h5_file in sorted(root.rglob("episode_*_a*.h5")):
            rel = str(h5_file.relative_to(root))
            if rel not in admitted_paths:
                orphan_attempts.append(rel)

    # Per-task success census vs the target.
    per_task: dict[str, int] = {}
    for uid in admitted:
        task_name = prefixes[uid].split("/")[1]
        per_task[task_name] = per_task.get(task_name, 0) + 1
    expected_tasks = sorted({prefixes[uid].split("/")[1] for uid in expected_uids})
    insufficient = {t: per_task.get(t, 0) for t in expected_tasks if per_task.get(t, 0) < target}

    ok = not (missing_terminal or missing_file or schema_errors or multiple_accepted or insufficient)
    return {
        "ok": ok,
        "plan_hashes": plan_hashes,
        "expected": len(expected_uids),
        "admitted": admitted,
        "failed": sorted(failed),
        "missing_terminal": sorted(missing_terminal),
        "missing_file": sorted(missing_file),
        "schema_errors": schema_errors,
        "multiple_accepted": sorted(multiple_accepted),
        "orphan_attempts": orphan_attempts,
        "insufficient": insufficient,
    }


# ------------------------------------------------------------------
# Manifest
# ------------------------------------------------------------------


def build_manifest(report: dict[str, Any], *, root: str | pathlib.Path, target: int) -> dict[str, Any]:
    """First ``target`` admitted successes per task, episode_idx ascending, with sha256.

    Deterministic by construction: input = the audited admission set (journal
    semantics), order = episode_idx parsed from the canonical prefix, and the
    serialization sorts keys — the same tree always yields the same bytes.
    """
    root = pathlib.Path(root)
    per_task: dict[str, list[tuple[int, str, dict[str, Any]]]] = {}
    for uid, entry in report["admitted"].items():
        task_name = entry["path"].split("/")[1]
        episode_idx = int(entry["path"].rsplit("episode_", 1)[1].split("_a")[0])
        per_task.setdefault(task_name, []).append((episode_idx, uid, entry))
    tasks: dict[str, list[dict[str, Any]]] = {}
    for task_name in sorted(per_task):
        chosen = sorted(per_task[task_name])[:target]
        rows = []
        for episode_idx, uid, entry in chosen:
            digest = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
            rows.append(
                {
                    "task_uid": uid,
                    "episode_idx": episode_idx,
                    "attempt": entry["attempt"],
                    "batch": entry["batch"],
                    "path": entry["path"],
                    "sha256": digest,
                }
            )
        tasks[task_name] = rows
    return {"target": target, "plan_hashes": report["plan_hashes"], "tasks": tasks}


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, help="scene root, e.g. /data/robocasa365_cache/build_l1s1")
    ap.add_argument("--teacher", required=True, help="recorded in the report; paths come from the run-plan")
    ap.add_argument("--journal", required=True)
    ap.add_argument(
        "--run-plan",
        action="append",
        required=True,
        help="run-plan JSON (repeatable, one per batch) — the ONLY expected-uid source",
    )
    ap.add_argument("--target", type=int, default=20)
    ap.add_argument("--report-out", default="", help="write the JSON report here (default: stdout only)")
    ap.add_argument("--manifest-out", default="", help="write the deterministic manifest here")
    args = ap.parse_args()

    run_cli(args)


def run_cli(args: argparse.Namespace) -> dict[str, Any]:
    """CLI body, callable from tests. A failed audit writes NO manifest.

    Emitting a manifest from a failing audit would hand downstream library
    builds an artifact that looks authoritative while the census behind it is
    broken — the manifest exists only on a fully passing audit.
    """
    plans = [load_run_plan(path) for path in args.run_plan]
    records = load_journal(args.journal)
    report = audit(root=args.root, journal_records=records, plans=plans, target=args.target)
    report["teacher"] = args.teacher
    rendered = json.dumps(report, sort_keys=True, indent=1)
    print(rendered)
    if args.report_out:
        pathlib.Path(args.report_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.report_out).write_text(rendered)
    if not report["ok"]:
        raise SystemExit(2)
    if args.manifest_out:
        manifest = build_manifest(report, root=args.root, target=args.target)
        out = pathlib.Path(args.manifest_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(manifest, sort_keys=True, indent=1))
    return report


if __name__ == "__main__":
    main()
