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

from exp.robocasa365.pinned_objects import compute_pin_task_id, load_pin_manifest

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
    """Union of expected uids across batches; duplicate/conflicting uids are an error.

    Batches must also agree on the object pinning: a pinned batch and an
    unpinned one describe different scene distributions, and unioning them into
    one audit would build a library out of both with nothing recording that.
    """
    pin_ids = {plan["params"].get("pin_id") for plan in plans}
    if len(pin_ids) > 1:
        raise ValueError(
            f"run-plans disagree on pin_id ({sorted(pin_ids, key=str)}); batches "
            "collected under different object pinnings cannot be audited together"
        )
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


def _check_pin_provenance(
    path: pathlib.Path,
    expected_task: str,
    pin_table: dict[str, dict[str, str]],
    pin_id: str,
) -> list[str]:
    """Prove this episode ACTUALLY ran the pinned objects, not merely claimed to.

    The declared identity is not evidence: a worker that accepted a pin table
    and then, through any bug in the plumbing, built an unpinned env would still
    stamp a perfectly correct ``pin_id``. So the judgement is made on
    ``realized_objects`` -- read back from the built scene after reset -- and the
    declared identity is only checked for agreement with it.

    Four checks, all of which must pass before the episode can be admitted:
    the slot SET matches the table exactly; every path matches byte-for-byte;
    the per-task identity re-derives from the realized values; and the global
    identity matches the auditor's own copy of the table (a correct task slice
    under a wrong global table would otherwise slip through).
    """
    import h5py

    problems: list[str] = []
    expected_slots = pin_table.get(expected_task)
    if expected_slots is None:
        return [f"pin table has no slot map for task {expected_task!r}"]
    with h5py.File(path, "r") as f:
        for attr in ("pin_id", "pin_task_id", "realized_objects"):
            if attr not in f.attrs:
                problems.append(f"missing pin attr {attr!r}")
        if problems:
            return problems
        if str(f.attrs["pin_id"]) != pin_id:
            problems.append(
                f"attr pin_id={f.attrs['pin_id']!r} != auditor's table {pin_id!r}"
            )
        try:
            realized = json.loads(str(f.attrs["realized_objects"]))
        except ValueError as exc:
            return [*problems, f"realized_objects is not JSON: {exc}"]
        if not isinstance(realized, dict):
            # Valid JSON is not enough: a scalar would make the set comparison
            # below raise out of the auditor instead of failing this episode.
            return [*problems, f"realized_objects is {type(realized).__name__}, not an object"]
        if set(realized) != set(expected_slots):
            problems.append(
                f"realized slots {sorted(realized)} != pinned slots {sorted(expected_slots)}"
            )
        else:
            for slot in sorted(expected_slots):
                if realized[slot] != expected_slots[slot]:
                    problems.append(
                        f"slot {slot!r} realized {realized[slot]!r} != pinned "
                        f"{expected_slots[slot]!r}"
                    )
        recomputed = compute_pin_task_id(expected_task, realized)
        if str(f.attrs["pin_task_id"]) != recomputed:
            problems.append(
                f"attr pin_task_id={f.attrs['pin_task_id']!r} != identity of the "
                f"realized objects {recomputed!r}"
            )
    return problems


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
    pin_id: str | None = None,
    pin_table: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Full audit against the run-plan uid set. Returns a JSON-able report."""
    if pin_table is not None and not pin_id:
        raise ValueError("pin_table given without pin_id; the global identity check needs both")
    root = pathlib.Path(root)
    expected_uids, prefixes, batches, plan_hashes = merge_run_plans(plans)

    by_uid: dict[str, list[dict[str, Any]]] = {}
    for record in journal_records:
        by_uid.setdefault(record["task_uid"], []).append(record)

    missing_terminal: list[str] = []
    failed: list[str] = []
    missing_file: list[str] = []
    schema_errors: dict[str, list[str]] = {}
    pin_errors: dict[str, list[str]] = {}
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
        if pin_table is not None:
            pin_problems = _check_pin_provenance(h5_path, task_name, pin_table, pin_id)
            if pin_problems:
                # Not admitted: an episode whose realized objects disagree with
                # the pin table is a different experiment than the one we are
                # building a library for.
                pin_errors[uid] = pin_problems
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

    ok = not (
        missing_terminal or missing_file or schema_errors or pin_errors
        or multiple_accepted or insufficient
    )
    return {
        "ok": ok,
        "plan_hashes": plan_hashes,
        "expected": len(expected_uids),
        "admitted": admitted,
        "failed": sorted(failed),
        "missing_terminal": sorted(missing_terminal),
        "missing_file": sorted(missing_file),
        "schema_errors": schema_errors,
        "pin_errors": pin_errors,
        "pin_id": pin_id,
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
    manifest = {"target": target, "plan_hashes": report["plan_hashes"], "tasks": tasks}
    if report.get("pin_id") is not None:
        # Travels with the manifest so the library builder can stamp it onto the
        # artifact without a second source of truth.
        manifest["pin_id"] = report["pin_id"]
    return manifest


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
    ap.add_argument(
        "--pinned-objects",
        default="",
        help="pin table path. Given, every admitted episode must prove -- from "
        "its recorded realized objects, not its claimed identity -- that it "
        "actually ran these exact meshes.",
    )
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
    pin_id, pin_table = (None, None)
    # getattr: existing callers build the Namespace by hand (test seam), so a
    # newly added flag must not become a required attribute.
    pin_path = getattr(args, "pinned_objects", "")
    # The run-plan is the ground truth about how these episodes were collected,
    # so batch agreement and table identity are settled here, once, with a
    # readable message. Left to the per-episode checks they would still fire --
    # but as N identical failures at the wrong level of the diagnosis.
    plan_pin_ids = {plan["params"].get("pin_id") for plan in plans}
    if len(plan_pin_ids) > 1:
        raise SystemExit(
            f"run-plans disagree on pin_id ({sorted(plan_pin_ids, key=str)}); batches "
            "collected under different object pinnings cannot be audited together"
        )
    plan_pin_id = next(iter(plan_pin_ids)) if plan_pin_ids else None
    if pin_path:
        pin_id, pin_table = load_pin_manifest(pin_path)
        if pin_id != plan_pin_id:
            raise SystemExit(
                f"--pinned-objects has pin_id {pin_id} but the run-plan records "
                f"{plan_pin_id}; this is not the table this collection ran under"
            )
    elif plan_pin_id:
        # The run-plan says these episodes were collected pinned. Auditing them
        # without the table would silently skip every provenance check and emit
        # a manifest that looks clean, so the library would be built from
        # unverified episodes.
        raise SystemExit(
            "run-plan records pin_id "
            f"{sorted({p['params'].get('pin_id') for p in plans}, key=str)} but "
            "--pinned-objects was not given; refusing to audit a pinned "
            "collection without its table"
        )
    report = audit(
        root=args.root,
        journal_records=records,
        plans=plans,
        target=args.target,
        pin_id=pin_id,
        pin_table=pin_table,
    )
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
