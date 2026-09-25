"""Join trusted dispatch, terminal and worker records to server warm reset evidence."""

from __future__ import annotations

import dataclasses
import json
from collections import Counter, defaultdict
from pathlib import Path

from exp.step_diag.envs import ENVS
from exp.warm_reset.conductor import WarmResetStrategy, manifest_digest
from exp.warm_reset.plan import config_of, read_plan
from openpi.cache.warm_reset.evidence import ExpectedEpisode, episode_problems
from openpi.cache.warm_reset.types import WarmResetSpec
from openpi.conductor.task import EpisodeTask, ServerEndpoint


def json_rows(path: Path) -> list[dict]:
    """Read JSONL strictly; corrupt tails and non-object rows cannot disappear."""
    rows = []
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{i}: invalid JSON") from exc
        if not isinstance(row, dict):
            raise TypeError(f"{path}:{i}: expected an object")
        rows.append(row)
    return rows


def trusted_expected(
    *,
    task: EpisodeTask,
    task_name: str,
    arm: dict,
    manifest: dict,
    run_id: str,
    terminals: list[dict],
    per_step: list[dict],
):
    """Derive a complete episode expectation without consulting server evidence."""
    terms = [
        r
        for r in terminals
        if r.get("run_id") == run_id
        and r.get("task_uid") == task.task_uid
        and r.get("accepted") is True
    ]
    if len(terms) != 1:
        raise ValueError("expected exactly one accepted terminal")
    term = terms[0]
    outcome, attempt = term.get("success"), term.get("attempt")
    if (
        type(outcome) is not bool
        or type(attempt) is not int
        or attempt < 1
        or term.get("status") != ("done" if outcome else "failed")
        or term.get("error")
        or term.get("yaml_id") != task.yaml_id
        or term.get("phase") != "eval"
    ):
        raise ValueError("unclean or conflicting accepted terminal")
    # A retriable result can be accepted without being terminal. Only the
    # accepted terminal's generation contributes decisions; server closure
    # still detects a missing tail in that selected generation.
    rows = [
        r
        for r in per_step
        if r.get("run_id") == run_id
        and r.get("task_uid") == task.task_uid
        and r.get("accepted") is True
        and "hit_type" in r
    ]
    if any(type(r.get("attempt")) is not int or r["attempt"] < 1 for r in rows):
        raise ValueError("invalid worker attempt")
    if any(r["attempt"] > attempt for r in rows):
        raise ValueError("worker generation is newer than the accepted terminal")
    rows = [r for r in rows if r["attempt"] == attempt]
    if not rows:
        raise ValueError("missing worker decisions")
    steps = []
    for row in rows:
        if (
            type(row.get("attempt")) is not int
            or row["attempt"] != attempt
            or row.get("yaml_id") != task.yaml_id
            or row.get("success") is not outcome
        ):
            raise ValueError("conflicting worker identity/outcome")
        step = row.get("step_idx")
        if type(step) is not int or step < 0:
            raise ValueError("invalid worker step_idx")
        if row.get("hit_type") != "WARM_START" or row.get("start_t") != arm["start_t"]:
            raise ValueError("worker did not execute the requested warm arm")
        steps.append(step)
    stride = manifest["rollout"]["replan_steps"]
    if sorted(steps) != list(range(0, len(steps) * stride, stride)):
        raise ValueError("missing or duplicate worker decision")
    cfg = config_of(arm["yaml"])
    if cfg.warm_reset is None:
        return {"attempt": attempt, "outcome": outcome, "n_decisions": len(rows)}
    spec = WarmResetSpec.from_config(cfg.warm_reset)
    identity = {
        "experiment": task.experiment,
        "task": task_name,
        "episode_id": task.episode_idx,
        "task_id": task.task_id,
        "orig_init_state_idx": task.orig_init_state_idx,
        "task_uid": task.task_uid,
        "attempt": attempt,
    }
    if ENVS[manifest["env_id"]].benchmark == "robocasa365":
        identity["seed"] = manifest["rollout"]["base_seed"] + task.orig_init_state_idx
    return ExpectedEpisode(
        task_uid=task.task_uid,
        attempt=attempt,
        outcome=outcome,
        n_decisions=len(rows),
        identity=identity,
        bundle_id=task.bundle_id,
        yaml_id=task.yaml_id,
        yaml_sha256=arm["yaml_sha256"],
        spec=spec,
        spec_digest=arm["spec_digest"],
        schedule_id=manifest["schedule_id"],
        k=manifest["k"],
        start_t=arm["start_t"],
    )


def admit(root: Path, evidence_dirs: list[Path] | None = None) -> dict:
    """Produce a fail-closed per-episode and per-arm report, including missing tasks."""
    plan = read_plan(root)
    execution = json.loads((root / "execution.json").read_text())
    if execution.get("manifest_sha256") != manifest_digest(plan):
        raise ValueError("run plan changed after execution was claimed")
    if execution.get("token") != plan["token"] or not isinstance(
        execution.get("run_id"), str
    ):
        raise ValueError("invalid execution identity")
    run_id = execution["run_id"]
    arms = {a["yaml_id"]: a for a in plan["arms"]}
    strategy = WarmResetStrategy(plan)
    strategy.plan(list(arms), dict.fromkeys(arms, ServerEndpoint(**plan["servers"][0])))
    planned = {t.task_uid: t for t in strategy.tasks}
    dispatched = [EpisodeTask(**t) for t in execution["tasks"]]
    if len(dispatched) != len(planned) or {t.task_uid for t in dispatched} != set(
        planned
    ):
        raise ValueError("execution roster differs from frozen plan")
    endpoints = {(s["host"], s["port"]) for s in plan["servers"]}
    for task in dispatched:
        want = dataclasses.replace(
            planned[task.task_uid],
            server_host=task.server_host,
            server_port=task.server_port,
        )
        if task != want or (task.server_host, task.server_port) not in endpoints:
            raise ValueError("dispatch identity differs from frozen plan")
    global_problems = []

    def read(path):
        try:
            return json_rows(path)
        except (OSError, ValueError, TypeError) as exc:
            global_problems.append(str(exc))
            return []

    terminals = read(root / "journal.jsonl")
    worker_rows = read(root / "per_step.jsonl")
    terminal_by_uid, worker_by_uid, server_by_key = (
        defaultdict(list),
        defaultdict(list),
        defaultdict(list),
    )
    for label, rows, target in (
        ("journal", terminals, terminal_by_uid),
        ("worker", worker_rows, worker_by_uid),
    ):
        for row in rows:
            uid = row.get("task_uid")
            if (
                row.get("run_id") != run_id
                or not isinstance(uid, str)
                or uid not in planned
                or type(row.get("accepted")) is not bool
            ):
                global_problems.append(f"invalid {label} run/task/accepted identity")
                continue
            target[uid].append(row)
    files = set()
    for directory in evidence_dirs or [Path(plan["evidence_dir"])]:
        files.update(path.resolve() for path in directory.rglob("*.jsonl"))
    for path in sorted(files):
        for row in read(path):
            uid, attempt = row.get("task_uid"), row.get("attempt")
            if (
                not isinstance(uid, str)
                or uid not in planned
                or type(attempt) is not int
                or attempt < 1
            ):
                global_problems.append(f"invalid server task/attempt identity: {path}")
                continue
            server_by_key[uid, attempt].append(row)
    names = {t["task_id"]: t["name"] for t in plan["tasks"]}
    reports, summary = [], {}
    for task in dispatched:
        arm = arms[task.yaml_id]
        result = {
            "task_uid": task.task_uid,
            "arm": arm["arm"],
            "problems": [],
            "total_nfe": None,
            "evidence_scope": "server+worker"
            if arm["spec_digest"]
            else "worker_reference",
        }
        try:
            expected = trusted_expected(
                task=task,
                task_name=names[task.task_id],
                arm=arm,
                manifest=plan,
                run_id=run_id,
                terminals=terminal_by_uid[task.task_uid],
                per_step=worker_by_uid[task.task_uid],
            )
            if isinstance(expected, ExpectedEpisode):
                checked = episode_problems(
                    server_by_key[task.task_uid, expected.attempt], expected=expected
                )
                result.update(
                    checked,
                    attempt=expected.attempt,
                    success=expected.outcome,
                    n_decisions=expected.n_decisions,
                )
                result["problems"] = dict(checked["problems"])
            else:
                result.update(
                    attempt=expected["attempt"],
                    success=expected["outcome"],
                    n_decisions=expected["n_decisions"],
                )
        except (ValueError, TypeError, KeyError) as exc:
            result["problems"] = [str(exc)]
        result["admitted"] = not result["problems"] and not global_problems
        if global_problems:
            result.update(total_nfe=None, continuation_nfe=None, self_start_nfe=None)
        reports.append(result)
        cell = summary.setdefault(
            arm["arm"],
            {
                "expected": 0,
                "admitted": 0,
                "successes": 0,
                "measured_total_nfe": 0 if arm["spec_digest"] else None,
            },
        )
        cell["expected"] += 1
        if result["admitted"]:
            cell["admitted"] += 1
            cell["successes"] += int(result["success"])
            if cell["measured_total_nfe"] is not None:
                cell["measured_total_nfe"] += result["total_nfe"]
    return {
        "schema": "warm_reset_admission_v1",
        "run_id": run_id,
        "ok": not global_problems and all(r["admitted"] for r in reports),
        "global_problems": dict(Counter(global_problems)),
        "arms": summary,
        "episodes": reports,
    }
