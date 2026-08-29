"""Shared identity / cost parsing for the dispatch precheck line (Rev 2 Phase 0).

Two loader families live here and must not be confused:

* ``load_accepted_episodes`` / ``load_analytic_cost`` are the Rev 1 OUTCOME
  loaders, moved verbatim from ``analyze_precheck`` (which re-imports them);
  they read and cross-validate ``status`` and ``success``.
* ``load_accepted_cells_costonly`` / ``load_cost_cells_costonly`` are the
  Phase 0 COST-ONLY loaders (G1R1-B4, G1R3-B2). They keep every identity and
  completeness discipline of the outcome loaders -- accepted terminal attempt,
  unique full grid, stale/fenced exclusion, official-init cross-check,
  client_timing/infers agreement -- but never read the ``success`` or
  ``status`` keys, which encode the same outcome bit. A static source-lock
  test walks these functions and refuses either key.

Unit costs come from ``analytic_cost`` only.
"""

from __future__ import annotations

import collections
import hashlib
import json
import pathlib

from exp.dispatch_surface.analysis.analytic_cost import (  # noqa: F401 (re-exported)
    PINNED_START_T_WS,
    STAGE1_MS,
    STAGE2_MS,
    STAGE3_MS,
    VERDICTS,
    unit_cost,
)


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def _is_json_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def parse_task_uid(task_uid: str) -> tuple[str, int, int]:
    """(yaml_id, task_id, subset_idx) from ``<arm>:<phase>:<task>:<episode>``."""
    parts = str(task_uid).rsplit(":", 3)
    if len(parts) != 4 or parts[1] != "eval":
        raise SystemExit(f"malformed task_uid {task_uid!r}")
    try:
        return parts[0], int(parts[2]), int(parts[3])
    except ValueError as exc:
        raise SystemExit(f"malformed task_uid {task_uid!r}") from exc


def load_accepted_episodes(
    journal_path: str, arms: list[str], expected_grid: set[tuple[int, int]],
) -> dict[str, dict]:
    """arm -> {(task, subset): {success, task_uid, attempt, run_id}}.

    Only ``accepted`` terminal records count. The conductor journals stale and
    fenced attempts the same way as the live one and discriminates them by
    (accepted, attempt, run_id), so an adjudicator that ignores those fields
    would mix a requeued episode's outcome with the one the scheduler actually
    took (G2 B2). Exactly one accepted record per cell is required.
    """
    out: dict[str, dict] = {a: {} for a in arms}
    for line in open(journal_path):
        row = json.loads(line)
        arm = row.get("yaml_id")
        if arm not in out:
            continue
        if not isinstance(row.get("accepted"), bool):
            raise SystemExit(f"arm {arm}: journal accepted must be a boolean")
        if row.get("accepted") is not True:
            continue
        if row.get("phase") != "eval":
            raise SystemExit(f"arm {arm}: accepted journal row is not eval phase")
        uid_arm, task, subset = parse_task_uid(row["task_uid"])
        if uid_arm != arm:
            raise SystemExit(
                f"journal row yaml_id={arm} disagrees with its task_uid {row['task_uid']}"
            )
        key = (task, subset)
        if key not in expected_grid:
            raise SystemExit(
                f"arm {arm}: accepted episode {key} is outside the pre-registered grid"
            )
        if key in out[arm]:
            raise SystemExit(
                f"arm {arm}: two accepted records for cell {key} — refusing "
                "ambiguous attribution"
            )
        if (not _is_json_int(row.get("attempt")) or row["attempt"] < 1
                or not isinstance(row.get("run_id"), str) or not row["run_id"]):
            raise SystemExit(
                f"arm {arm} cell {key}: accepted record lacks attempt/run_id — "
                "cannot bind its per-step evidence"
            )
        if row.get("status") not in ("done", "failed") or not isinstance(
            row.get("success"), bool
        ) or ((row["status"] == "done") != row["success"]):
            raise SystemExit(
                f"arm {arm} cell {key}: accepted status/success terminal schema is invalid"
            )
        out[arm][key] = {
            "success": row["success"],
            "task_uid": row["task_uid"],
            "attempt": row["attempt"],
            "run_id": row["run_id"],
        }
    for a in arms:
        got = set(out[a])
        if got != expected_grid:
            missing = sorted(expected_grid - got)[:3]
            raise SystemExit(
                f"arm {a}: {len(expected_grid - got)} cells have no accepted episode "
                f"(e.g. {missing}) — refusing incomplete/unpaired data"
            )
    return out


def load_analytic_cost(
    per_step_path: str, arms: list[str], accepted: dict[str, dict],
    officials: dict[int, list[int]],
) -> tuple[dict[str, dict], dict]:
    """arm -> {(task, subset): (cost_sum_ms, n_decisions)} plus a summary.

    The SUM and the COUNT are kept apart on purpose: the arm's cost is a
    ratio-of-sums that must be re-formed after resampling, so a per-cell mean
    computed here would already be the wrong estimand.

    Rows are joined to the ACCEPTED (task_uid, attempt, run_id) triple. Stale
    attempts, fenced reports and rows from another run are excluded and
    counted, never billed (G2 B2). Verdict rows additionally cross-check their
    ``orig_init_state_idx`` against the official index the frozen split
    manifest assigns to that subset position (G2 B3), and every accepted
    episode must contribute exactly one ``client_timing`` row whose ``infers``
    equals the number of priced decisions (G2 B1).
    """
    cost_sum: dict[str, dict] = {a: collections.defaultdict(float) for a in arms}
    n_dec: dict[str, dict] = {a: collections.defaultdict(int) for a in arms}
    verdict_counts: dict[str, collections.Counter] = {a: collections.Counter() for a in arms}
    infers_reported: dict[str, dict] = {a: {} for a in arms}
    step_indices: dict[str, dict] = {
        a: collections.defaultdict(set) for a in arms
    }
    excluded = collections.Counter()

    def _is_accepted(arm, row):
        required = ("task_uid", "task_id", "subset_init_state_idx",
                    "attempt", "run_id", "accepted", "success")
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise SystemExit(
                f"arm {arm}: per_step row lacks identity fields {missing}"
            )
        uid_arm, task, subset = parse_task_uid(row["task_uid"])
        if uid_arm != arm:
            raise SystemExit(
                f"per_step yaml_id={arm} disagrees with task_uid {row['task_uid']}"
            )
        rec = accepted[arm].get((task, subset))
        if rec is None:
            raise SystemExit(
                f"arm {arm}: per_step row for {(task, subset)} has no accepted "
                "journal episode — refusing rows outside the adjudicated grid"
            )
        if row["task_uid"] != rec["task_uid"]:
            raise SystemExit(f"arm {arm} cell {(task, subset)}: task_uid identity mismatch")
        if (not _is_json_int(row["task_id"])
                or not _is_json_int(row["subset_init_state_idx"])
                or row["task_id"] != task or row["subset_init_state_idx"] != subset):
            raise SystemExit(
                f"arm {arm} cell {(task, subset)}: duplicated task/subset fields disagree"
            )
        if (not _is_json_int(row["attempt"]) or row["attempt"] < 1
                or not isinstance(row["run_id"], str) or not row["run_id"]):
            raise SystemExit(f"arm {arm} cell {(task, subset)}: invalid attempt/run_id")
        if not isinstance(row["accepted"], bool):
            raise SystemExit(f"arm {arm} cell {(task, subset)}: accepted must be boolean")
        live = (row["attempt"] == rec["attempt"]
                and str(row["run_id"]) == rec["run_id"]
                and row["accepted"] is True)
        if live and (not isinstance(row["success"], bool)
                     or row["success"] != rec["success"]):
            raise SystemExit(
                f"arm {arm} cell {(task, subset)}: per_step success != journal outcome"
            )
        return (task, subset), live

    for line in open(per_step_path):
        row = json.loads(line)
        arm = row.get("yaml_id")
        if arm not in cost_sum:
            continue
        key, live = _is_accepted(arm, row)
        if not live:
            excluded[arm] += 1
            continue
        if row.get("_kind") == "client_timing":
            infers = row.get("infers")
            if not _is_json_int(infers) or infers < 0:
                raise SystemExit(
                    f"arm {arm} cell {key}: client_timing infers must be a nonnegative integer"
                )
            if key in infers_reported[arm]:
                raise SystemExit(f"arm {arm}: duplicate client_timing row for cell {key}")
            infers_reported[arm][key] = infers
            continue
        if row.get("_kind") is not None:
            raise SystemExit(
                f"arm {arm} cell {key}: unsupported per_step row kind "
                f"{row.get('_kind')!r}"
            )
        task, subset = key
        if row.get("phase") != "eval":
            raise SystemExit(f"arm {arm} cell {key}: verdict row is not eval phase")
        step_idx = row.get("step_idx")
        if not _is_json_int(step_idx) or step_idx < 0:
            raise SystemExit(f"arm {arm} cell {key}: invalid step_idx {step_idx!r}")
        if step_idx in step_indices[arm][key]:
            raise SystemExit(f"arm {arm} cell {key}: duplicate verdict step_idx {step_idx}")
        step_indices[arm][key].add(step_idx)
        canonical_episode = task * len(officials[task]) + subset
        if not _is_json_int(row.get("episode_id")) or row["episode_id"] != canonical_episode:
            raise SystemExit(
                f"arm {arm} cell {key}: episode_id {row.get('episode_id')!r} != "
                f"canonical {canonical_episode}"
            )
        official = row.get("orig_init_state_idx")
        if not _is_json_int(official) or official != officials[task][subset]:
            raise SystemExit(
                f"arm {arm} cell {key}: row claims official init {official!r} but the "
                f"frozen split manifest assigns {officials[task][subset]} to that "
                "subset position — identity mismatch"
            )
        hit = row.get("hit_type")
        cost_sum[arm][key] += unit_cost(hit, row.get("start_t"))
        n_dec[arm][key] += 1
        verdict_counts[arm][hit] += 1

    out: dict[str, dict] = {}
    for a in arms:
        expected = set(accepted[a])
        cells = set(n_dec[a])
        if cells != expected:
            missing = sorted(expected - cells)[:3]
            raise SystemExit(
                f"arm {a}: per_step covers {len(cells)} cells, expected {len(expected)} "
                f"(missing e.g. {missing}) — refusing incomplete cost data"
            )
        for key, n in n_dec[a].items():
            reported = infers_reported[a].get(key)
            if reported is None:
                raise SystemExit(
                    f"arm {a} cell {key}: accepted episode has no client_timing row — "
                    "the decision count cannot be verified against the episode's own "
                    "inference count"
                )
            if reported != n:
                raise SystemExit(
                    f"arm {a} cell {key}: {n} priced decisions but the episode reports "
                    f"{reported} inferences — refusing inconsistent per_step"
                )
        out[a] = {k: (cost_sum[a][k], n_dec[a][k]) for k in n_dec[a]}

    summary = {
        "per_step_sha256": _file_sha256(pathlib.Path(per_step_path)),
        "unit_cost_ms": {"stage1": STAGE1_MS, "stage2": STAGE2_MS, "stage3": STAGE3_MS,
                         "FULL_HIT": unit_cost("FULL_HIT", None),
                         "WARM_START": unit_cost("WARM_START", PINNED_START_T_WS),
                         "MISS": unit_cost("MISS", None)},
        "verdict_counts": {a: dict(verdict_counts[a]) for a in arms},
        "decisions": {a: int(sum(n for _s, n in out[a].values())) for a in arms},
        "excluded_stale_rows": dict(excluded),
    }
    return out, summary


# ------------------------------------------------------------------
# COST-ONLY loaders (Phase 0). Outcome-blind by construction: neither
# ``success`` nor ``status`` is read; every other discipline is kept.
# ------------------------------------------------------------------


def load_accepted_cells_costonly(
    journal_path: str, arms: list[str], expected_grid: set[tuple[int, int]],
) -> dict[str, dict]:
    """arm -> {(task, subset): {task_uid, attempt, run_id}} -- COST-ONLY variant.

    Only ``accepted`` terminal records count. The conductor journals stale and
    fenced attempts the same way as the live one and discriminates them by
    (accepted, attempt, run_id), so an adjudicator that ignores those fields
    would mix a requeued episode's outcome with the one the scheduler actually
    took (G2 B2). Exactly one accepted record per cell is required.
    """
    out: dict[str, dict] = {a: {} for a in arms}
    for line in open(journal_path):
        row = json.loads(line)
        arm = row.get("yaml_id")
        if arm not in out:
            continue
        if not isinstance(row.get("accepted"), bool):
            raise SystemExit(f"arm {arm}: journal accepted must be a boolean")
        if row.get("accepted") is not True:
            continue
        if row.get("phase") != "eval":
            raise SystemExit(f"arm {arm}: accepted journal row is not eval phase")
        uid_arm, task, subset = parse_task_uid(row["task_uid"])
        if uid_arm != arm:
            raise SystemExit(
                f"journal row yaml_id={arm} disagrees with its task_uid {row['task_uid']}"
            )
        key = (task, subset)
        if key not in expected_grid:
            raise SystemExit(
                f"arm {arm}: accepted episode {key} is outside the pre-registered grid"
            )
        if key in out[arm]:
            raise SystemExit(
                f"arm {arm}: two accepted records for cell {key} — refusing "
                "ambiguous attribution"
            )
        if (not _is_json_int(row.get("attempt")) or row["attempt"] < 1
                or not isinstance(row.get("run_id"), str) or not row["run_id"]):
            raise SystemExit(
                f"arm {arm} cell {key}: accepted record lacks attempt/run_id — "
                "cannot bind its per-step evidence"
            )
        out[arm][key] = {
            "task_uid": row["task_uid"],
            "attempt": row["attempt"],
            "run_id": row["run_id"],
        }
    for a in arms:
        got = set(out[a])
        if got != expected_grid:
            missing = sorted(expected_grid - got)[:3]
            raise SystemExit(
                f"arm {a}: {len(expected_grid - got)} cells have no accepted episode "
                f"(e.g. {missing}) — refusing incomplete/unpaired data"
            )
    return out


def load_cost_cells_costonly(
    per_step_path: str, arms: list[str], accepted: dict[str, dict],
    officials: dict[int, list[int]],
) -> tuple[dict[str, dict], dict]:
    """arm -> {(task, subset): (cost_sum_ms, n_decisions)} plus a summary.

    The SUM and the COUNT are kept apart on purpose: the arm's cost is a
    ratio-of-sums that must be re-formed after resampling, so a per-cell mean
    computed here would already be the wrong estimand.

    Rows are joined to the ACCEPTED (task_uid, attempt, run_id) triple. Stale
    attempts, fenced reports and rows from another run are excluded and
    counted, never billed (G2 B2). Verdict rows additionally cross-check their
    ``orig_init_state_idx`` against the official index the frozen split
    manifest assigns to that subset position (G2 B3), and every accepted
    episode must contribute exactly one ``client_timing`` row whose ``infers``
    equals the number of priced decisions (G2 B1).
    """
    cost_sum: dict[str, dict] = {a: collections.defaultdict(float) for a in arms}
    n_dec: dict[str, dict] = {a: collections.defaultdict(int) for a in arms}
    verdict_counts: dict[str, collections.Counter] = {a: collections.Counter() for a in arms}
    infers_reported: dict[str, dict] = {a: {} for a in arms}
    step_indices: dict[str, dict] = {
        a: collections.defaultdict(set) for a in arms
    }
    excluded = collections.Counter()

    def _is_accepted(arm, row):
        required = ("task_uid", "task_id", "subset_init_state_idx",
                    "attempt", "run_id", "accepted")
        missing = [key for key in required if row.get(key) is None]
        if missing:
            raise SystemExit(
                f"arm {arm}: per_step row lacks identity fields {missing}"
            )
        uid_arm, task, subset = parse_task_uid(row["task_uid"])
        if uid_arm != arm:
            raise SystemExit(
                f"per_step yaml_id={arm} disagrees with task_uid {row['task_uid']}"
            )
        rec = accepted[arm].get((task, subset))
        if rec is None:
            raise SystemExit(
                f"arm {arm}: per_step row for {(task, subset)} has no accepted "
                "journal episode — refusing rows outside the adjudicated grid"
            )
        if row["task_uid"] != rec["task_uid"]:
            raise SystemExit(f"arm {arm} cell {(task, subset)}: task_uid identity mismatch")
        if (not _is_json_int(row["task_id"])
                or not _is_json_int(row["subset_init_state_idx"])
                or row["task_id"] != task or row["subset_init_state_idx"] != subset):
            raise SystemExit(
                f"arm {arm} cell {(task, subset)}: duplicated task/subset fields disagree"
            )
        if (not _is_json_int(row["attempt"]) or row["attempt"] < 1
                or not isinstance(row["run_id"], str) or not row["run_id"]):
            raise SystemExit(f"arm {arm} cell {(task, subset)}: invalid attempt/run_id")
        if not isinstance(row["accepted"], bool):
            raise SystemExit(f"arm {arm} cell {(task, subset)}: accepted must be boolean")
        live = (row["attempt"] == rec["attempt"]
                and str(row["run_id"]) == rec["run_id"]
                and row["accepted"] is True)
        return (task, subset), live

    for line in open(per_step_path):
        row = json.loads(line)
        arm = row.get("yaml_id")
        if arm not in cost_sum:
            continue
        key, live = _is_accepted(arm, row)
        if not live:
            excluded[arm] += 1
            continue
        if row.get("_kind") == "client_timing":
            infers = row.get("infers")
            if not _is_json_int(infers) or infers < 0:
                raise SystemExit(
                    f"arm {arm} cell {key}: client_timing infers must be a nonnegative integer"
                )
            if key in infers_reported[arm]:
                raise SystemExit(f"arm {arm}: duplicate client_timing row for cell {key}")
            infers_reported[arm][key] = infers
            continue
        if row.get("_kind") is not None:
            raise SystemExit(
                f"arm {arm} cell {key}: unsupported per_step row kind "
                f"{row.get('_kind')!r}"
            )
        task, subset = key
        if row.get("phase") != "eval":
            raise SystemExit(f"arm {arm} cell {key}: verdict row is not eval phase")
        step_idx = row.get("step_idx")
        if not _is_json_int(step_idx) or step_idx < 0:
            raise SystemExit(f"arm {arm} cell {key}: invalid step_idx {step_idx!r}")
        if step_idx in step_indices[arm][key]:
            raise SystemExit(f"arm {arm} cell {key}: duplicate verdict step_idx {step_idx}")
        step_indices[arm][key].add(step_idx)
        canonical_episode = task * len(officials[task]) + subset
        if not _is_json_int(row.get("episode_id")) or row["episode_id"] != canonical_episode:
            raise SystemExit(
                f"arm {arm} cell {key}: episode_id {row.get('episode_id')!r} != "
                f"canonical {canonical_episode}"
            )
        official = row.get("orig_init_state_idx")
        if not _is_json_int(official) or official != officials[task][subset]:
            raise SystemExit(
                f"arm {arm} cell {key}: row claims official init {official!r} but the "
                f"frozen split manifest assigns {officials[task][subset]} to that "
                "subset position — identity mismatch"
            )
        hit = row.get("hit_type")
        cost_sum[arm][key] += unit_cost(hit, row.get("start_t"))
        n_dec[arm][key] += 1
        verdict_counts[arm][hit] += 1

    out: dict[str, dict] = {}
    for a in arms:
        expected = set(accepted[a])
        cells = set(n_dec[a])
        if cells != expected:
            missing = sorted(expected - cells)[:3]
            raise SystemExit(
                f"arm {a}: per_step covers {len(cells)} cells, expected {len(expected)} "
                f"(missing e.g. {missing}) — refusing incomplete cost data"
            )
        for key, n in n_dec[a].items():
            reported = infers_reported[a].get(key)
            if reported is None:
                raise SystemExit(
                    f"arm {a} cell {key}: accepted episode has no client_timing row — "
                    "the decision count cannot be verified against the episode's own "
                    "inference count"
                )
            if reported != n:
                raise SystemExit(
                    f"arm {a} cell {key}: {n} priced decisions but the episode reports "
                    f"{reported} inferences — refusing inconsistent per_step"
                )
        out[a] = {k: (cost_sum[a][k], n_dec[a][k]) for k in n_dec[a]}

    summary = {
        "per_step_sha256": _file_sha256(pathlib.Path(per_step_path)),
        "unit_cost_ms": {"stage1": STAGE1_MS, "stage2": STAGE2_MS, "stage3": STAGE3_MS,
                         "FULL_HIT": unit_cost("FULL_HIT", None),
                         "WARM_START": unit_cost("WARM_START", PINNED_START_T_WS),
                         "MISS": unit_cost("MISS", None)},
        "verdict_counts": {a: dict(verdict_counts[a]) for a in arms},
        "decisions": {a: int(sum(n for _s, n in out[a].values())) for a in arms},
        "excluded_stale_rows": dict(excluded),
    }
    return out, summary
