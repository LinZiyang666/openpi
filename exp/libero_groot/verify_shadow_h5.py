"""Accept or reject the CP2 shadow-cohort HDF5 collection, episode by episode (plan §3.9).

The collector's HDF5 is self-consistent whatever happened on the client: an
episode that raised after one inference still writes one step group and
``num_steps == 1``. Whether an episode is a *complete* teacher rollout is
therefore decided from evidence the client wrote in the same attempt
(``--save-episode-results`` rows with ``termination_reason`` and
``client_timing``), bound to the file through the stable identity the client
stamped on both: ``task_id``, ``orig_init_state_idx`` (H5 attrs from the
``episode_start`` metadata; row fields) and ``episode_id == task_id * 15 +
subset``.

Identity (G2-B1): a LIBERO task has two names. The shadow manifest's
``task_name`` is ``task.name`` -- the canonical, underscore-joined name that
locates the ``.init`` pool file; the H5 ``task`` attr is ``task.language`` --
the human instruction the model was conditioned on. They are bound through
``task_id`` with a task map written from the benchmark itself
(``emit_task_map.py``), never by rewriting one into the other. The accepted
record carries both, and every downstream consumer builds its template from
the instruction.

Acceptance, per episode:

* ``termination_reason == "success"`` with ``success == True``, or
  ``termination_reason == "step_cap"`` with ``success == False`` and
  ``client_timing.steps == max_steps + num_steps_wait`` (the suite's own cap:
  220 / 520, wait 10);
* the H5 ``success`` attr equals the client's; ``num_steps`` equals the number
  of step groups, which are exactly ``step_0000 .. step_{n-1}``; that count
  equals the client's inference count and ``ceil((steps - wait) / replan)``;
  every group carries the fields the CP2 reconstruction needs; the file is
  stamped ``groot_n15_k8_v1`` / 8 and its ``task_id`` / ``orig_init_state_idx``
  attrs agree with the client row and the manifest;
* the client's ``max_steps`` / ``num_steps_wait`` / ``replan_steps`` / ``seed``
  / ``task_suite_name`` are the frozen ones;
* anything else -- an ``exception`` termination, a missing client row or file,
  a count mismatch -- is rejected. Two client rows or two H5 files for one
  episode in one attempt make that episode *invalid in that attempt* (nothing
  is picked); a valid episode present in two attempts is a *duplicate*,
  reported, never silently picked. Every accepted task must carry one
  instruction string across its episodes.

Outputs ``accepted_shadow_manifest.json`` (``ok`` only with exactly 150
unique episodes), ``rejected.json``, ``retry_filter.json`` and
``retry/filter_task_<t>.json`` -- ``--episode-filter`` files naming only the
invalid / missing episodes for a serial re-run in a fresh ``attempt_<n>``.

Layout expected under ``--attempts-root``::

    attempt_<n>/client_task_<t>.json        (client rows, one file per task)
    attempt_<n>/srv<i>/<experiment>/*.h5    (collector output, any nesting)

Usage:
  python -m exp.libero_groot.verify_shadow_h5 --suite libero_spatial \\
      --shadow-manifest exp/libero_groot/data/rit/shadow/libero_spatial/shadow_manifest.json \\
      --task-map <task_map.json> --attempts-root /data/libero_cache/acb_shadow_h5/libero_spatial --out-dir <dir>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from typing import Any

import h5py

from exp.libero_groot.emit_task_map import load_task_map

TRIALS = 15
MAX_STEPS = {"libero_spatial": 220, "libero_10": 520}
NUM_STEPS_WAIT = 10
REPLAN_STEPS = 5
SEED = 7
SCHEDULE_ID = "groot_n15_k8_v1"
NUM_STEPS = 8
REQUIRED_STEP_DATASETS = ("vision_0", "vision_1", "prompt_emb", "robot_state", "clean_action", "noise_action_0")
REQUIRED_SNAPSHOTS = tuple(f"noise_action_{i}" for i in range(1, NUM_STEPS))
REQUIRED_H5_ATTRS = ("episode_id", "task", "success", "num_steps", "denoise_schedule_id", "denoising_num_steps",
                     "task_id", "orig_init_state_idx")
PROTOCOL = "actioncache_baseline/shadow_h5/v1"


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(4 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def expected_episodes(manifest: dict, task_map: dict[int, dict] | None = None) -> dict[tuple[int, int], dict]:
    """``(task_id, subset) -> {task_name, orig[, task_language]}`` from the shadow manifest.

    The pool file order is the manifest's ``pool_digests.shadow[task].indices``;
    the fit + cal assignment must be exactly that set. With a ``task_map`` the
    manifest's canonical ``task_name`` must be the benchmark's ``task.name`` for
    that ``task_id`` and the expected instruction (``task_language``) is filled
    in from the map.
    """
    out: dict[tuple[int, int], dict] = {}
    pools = manifest["pool_digests"]["shadow"]
    for t_str, a in manifest["assignment"].items():
        t = int(t_str)
        name = a["task_name"]
        if task_map is not None:
            if t not in task_map:
                raise SystemExit(f"task {t}: not in the task map")
            if task_map[t]["name"] != name:
                raise SystemExit(f"task {t}: manifest task_name {name!r} != benchmark task.name {task_map[t]['name']!r}")
        indices = list(pools[name]["indices"])
        if sorted(indices) != sorted(int(x) for x in list(a["fit"]) + list(a["cal"])):
            raise SystemExit(f"task {t}: pool indices {indices} != fit+cal {a['fit'] + a['cal']}")
        if len(indices) != TRIALS or len(set(indices)) != TRIALS:
            raise SystemExit(f"task {t}: pool must hold {TRIALS} distinct inits, got {indices}")
        for subset, orig in enumerate(indices):
            rec = {"task_name": name, "orig": int(orig)}
            if task_map is not None:
                rec["task_language"] = task_map[t]["language"]
            out[(t, subset)] = rec
    if len(out) != 10 * TRIALS:
        raise SystemExit(f"manifest names {len(out)} episodes, expected {10 * TRIALS}")
    return out


def episode_filter(expected: dict[tuple[int, int], dict], keys: list[tuple[int, int]] | None = None) -> list[dict]:
    """An ``examples/libero/main.py --episode-filter`` file for the given episodes."""
    keys = sorted(expected) if keys is None else sorted(keys)
    return [{"task_id": t, "subset_init_state_idx": s, "orig_init_state_idx": expected[(t, s)]["orig"]}
            for t, s in keys]


def _attr(attrs: Any, name: str):
    return attrs[name] if name in attrs else None


def _h5_identity(path: pathlib.Path) -> dict:
    """Everything the judge reads from one collector file, without loading its arrays.

    ``task_id`` / ``subset`` come from the attrs the client stamped through
    ``episode_start`` (``task_id``, ``orig_init_state_idx``) and the
    ``episode_id``; ``contiguous`` is whether the step groups are exactly
    ``step_0000 .. step_{n-1}``.
    """
    with h5py.File(path, "r") as f:
        attrs = dict(f.attrs)
        groups = sorted(n for n in f.keys() if n.startswith("step_") and n.split("_", 1)[1].isdigit())
        missing: list[str] = []
        for name in groups:
            g = f[name]
            for ds in REQUIRED_STEP_DATASETS + REQUIRED_SNAPSHOTS:
                if ds not in g:
                    missing.append(f"{name}/{ds}")
                    break
    episode_id = int(attrs.get("episode_id", -1))
    task_id_from_episode, subset = divmod(episode_id, TRIALS)
    task_id_attr = _attr(attrs, "task_id")
    orig_attr = _attr(attrs, "orig_init_state_idx")
    # Trace-mode files (plan cache_trace_mode §7.3): the version attr marks a
    # file written by the trace writer, which must also be a committed one.
    trace_version = _attr(attrs, "trace_schema_version")
    trace = None
    if trace_version is not None:
        trace = {
            "version": int(trace_version),
            "closed_ok": bool(attrs.get("trace_closed_ok", False)),
            "terminal": bool(attrs.get("trace_terminal", False)),
            "write_errors": int(attrs.get("trace_write_errors", 1)),
            "noise_recorded": bool(attrs.get("trace_noise_actions_recorded", False)),
        }
    return {
        "trace": trace,
        "path": str(path), "episode_id": episode_id, "task_id": task_id_from_episode, "subset": subset,
        "task_id_attr": None if task_id_attr is None else int(task_id_attr),
        "orig_attr": None if orig_attr is None else int(orig_attr),
        "task": str(attrs.get("task", "")), "success": attrs.get("success", None),
        "num_steps": int(attrs.get("num_steps", -1)), "n_groups": len(groups),
        "contiguous": groups == [f"step_{i:04d}" for i in range(len(groups))],
        "schedule_id": attrs.get("denoise_schedule_id", None),
        "denoising_num_steps": int(attrs.get("denoising_num_steps", -1)),
        "missing_attrs": [a for a in REQUIRED_H5_ATTRS if a not in attrs],
        "missing_datasets": missing,
    }


def _client_rows(path: pathlib.Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise SystemExit(f"{path}: client results must be a list")
    return rows


def judge_episode(suite: str, key: tuple[int, int], exp: dict, client: dict | None, h5: dict | None) -> list[str]:
    """Every reason this (client row, H5) pair is not an accepted cohort episode.

    ``exp`` is one ``expected_episodes`` value; ``h5`` one ``_h5_identity``.
    An empty list means accepted. The instruction is compared to
    ``exp["task_language"]`` when the caller bound a task map; the canonical
    ``task_name`` is never compared to the H5 instruction string.
    """
    t, subset = key
    problems: list[str] = []
    if client is None:
        problems.append("no client terminal row")
    if h5 is None:
        problems.append("no H5 file")
    if problems:
        return problems
    cap = MAX_STEPS[suite]
    reason = client.get("termination_reason")
    success = bool(client.get("success"))
    ct = client.get("client_timing") or {}
    steps, infers = int(ct.get("steps", -1)), int(ct.get("infers", -1))
    if steps > cap + NUM_STEPS_WAIT:
        problems.append(f"client steps={steps} exceeds frozen cap {cap + NUM_STEPS_WAIT}")
    if client.get("task_suite_name") != suite:
        problems.append(f"client task_suite_name {client.get('task_suite_name')!r} != {suite!r}")
    for name, want in (("max_steps", cap), ("num_steps_wait", NUM_STEPS_WAIT),
                       ("replan_steps", REPLAN_STEPS), ("seed", SEED)):
        if int(client.get(name, -10**9)) != want:
            problems.append(f"client {name}={client.get(name)!r} != frozen {want}")
    if int(client.get("task_id", -1)) != t or int(client.get("init_state_idx", -1)) != subset:
        problems.append(f"client identity ({client.get('task_id')}, {client.get('init_state_idx')}) != {key}")
    if int(client.get("episode_id", -1)) != t * TRIALS + subset:
        problems.append(f"client episode_id {client.get('episode_id')} != {t * TRIALS + subset}")
    if int(client.get("orig_init_state_idx", -1)) != exp["orig"]:
        problems.append(f"client orig_init_state_idx {client.get('orig_init_state_idx')} != manifest {exp['orig']}")
    if reason == "success":
        if not success:
            problems.append("termination success but success=False")
    elif reason == "step_cap":
        if success:
            problems.append("termination step_cap but success=True")
        if steps != cap + NUM_STEPS_WAIT:
            problems.append(f"step_cap termination with steps={steps} != {cap + NUM_STEPS_WAIT}")
    else:
        problems.append(f"termination_reason {reason!r} is not success/step_cap")
    if steps <= NUM_STEPS_WAIT or infers <= 0:
        problems.append(f"client counts steps={steps} infers={infers} not positive")
    else:
        want_infers = math.ceil((steps - NUM_STEPS_WAIT) / REPLAN_STEPS)
        if infers != want_infers:
            problems.append(f"client infers={infers} != ceil((steps-wait)/replan)={want_infers}")
    # H5 side: identity attrs, instruction, counts, stamps, fields.
    if h5["missing_attrs"]:
        problems.append(f"H5 lacks attrs {h5['missing_attrs']}")
    if h5["task_id_attr"] is not None and h5["task_id_attr"] != t:
        problems.append(f"H5 task_id attr {h5['task_id_attr']} != {t}")
    if h5["orig_attr"] is not None and h5["orig_attr"] != exp["orig"]:
        problems.append(f"H5 orig_init_state_idx attr {h5['orig_attr']} != manifest {exp['orig']}")
    if not h5["task"]:
        problems.append("H5 task (instruction) is empty")
    elif "task_language" in exp and h5["task"] != exp["task_language"]:
        problems.append(f"H5 instruction {h5['task']!r} != benchmark task.language {exp['task_language']!r}")
    if h5["success"] is None or bool(h5["success"]) != success:
        problems.append(f"H5 success {h5['success']!r} != client {success}")
    if h5["num_steps"] != h5["n_groups"]:
        problems.append(f"H5 num_steps {h5['num_steps']} != {h5['n_groups']} step groups")
    if not h5["contiguous"]:
        problems.append("H5 step groups are not step_0000..step_{n-1}")
    if h5["n_groups"] != infers:
        problems.append(f"H5 has {h5['n_groups']} step groups but the client made {infers} inferences")
    if h5["schedule_id"] != SCHEDULE_ID or h5["denoising_num_steps"] != NUM_STEPS:
        problems.append(f"H5 stamped {h5['schedule_id']!r}/{h5['denoising_num_steps']}, expected {SCHEDULE_ID}/{NUM_STEPS}")
    if h5["missing_datasets"]:
        problems.append(f"H5 missing datasets: {h5['missing_datasets'][:3]}")
    problems.extend(trace_problems(h5.get("trace")))
    return problems


KNOWN_TRACE_SCHEMA_VERSIONS = (1,)


def trace_problems(trace: dict | None) -> list[str]:
    """Commit-state checks of a trace-mode file; empty for a legacy collector file.

    The cohort needs the loop inputs (the CP2 reconstruction reads the
    snapshots), so a diagnostic trace file that recorded none is rejected
    even though its other fields are present.
    """
    if trace is None:
        return []
    problems: list[str] = []
    if trace["version"] not in KNOWN_TRACE_SCHEMA_VERSIONS:
        return [f"unknown trace_schema_version {trace['version']}"]
    if not trace["closed_ok"]:
        problems.append("trace_closed_ok is not True (episode never committed)")
    if not trace["terminal"]:
        problems.append("trace_terminal is not True (connection dropped mid-episode)")
    if trace["write_errors"] != 0:
        problems.append(f"trace_write_errors={trace['write_errors']}")
    if not trace["noise_recorded"]:
        problems.append("trace file recorded no noise_action_* (diagnostic run)")
    return problems


def _accepted_record(key, exp: dict, attempt: pathlib.Path, client: dict, h5: dict) -> dict:
    client_json = pathlib.Path(client["_json"])
    row = {k: v for k, v in client.items() if k != "_json"}
    rec = {
        "task_id": key[0], "task_name": exp["task_name"], "task_language": h5["task"],
        "subset_init_state_idx": key[1], "orig_init_state_idx": exp["orig"], "attempt": attempt.name,
        "h5": h5["path"], "h5_sha256": _sha256(pathlib.Path(h5["path"])),
        "client_json": str(client_json), "client_json_sha256": _sha256(client_json), "client_row": row,
        "success": bool(client["success"]),
        "steps": int(client["client_timing"]["steps"]), "infers": int(client["client_timing"]["infers"]),
        "termination_reason": client["termination_reason"],
    }
    return rec


def verify(suite: str, manifest_path: str | pathlib.Path, attempts_root: str | pathlib.Path,
           task_map: dict[int, dict] | None = None) -> dict[str, Any]:
    """Judge every attempt under ``attempts_root`` against the manifest.

    Returns the acceptance result: ``ok`` is True only when every expected
    episode is accepted exactly once, no episode is valid in two attempts and
    every task carries one instruction string. ``task_map`` (from
    ``emit_task_map.py``) binds the manifest's canonical names and the H5
    instructions through ``task_id``; the CLI requires it.
    """
    manifest = json.loads(pathlib.Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("suite") != suite:
        raise SystemExit(f"manifest suite {manifest.get('suite')!r} != {suite!r}")
    expected = expected_episodes(manifest, task_map)
    root = pathlib.Path(attempts_root)
    attempts = sorted(p for p in root.glob("attempt_*") if p.is_dir())
    if not attempts:
        raise SystemExit(f"no attempt_* directories under {root}")

    accepted: dict[tuple[int, int], dict] = {}
    duplicates: list[dict] = []
    rejected: list[dict] = []
    for attempt in attempts:
        clients: dict[tuple[int, int], list[dict]] = {}
        for cj in sorted(attempt.glob("client_task_*.json")):
            for row in _client_rows(cj):
                key = (int(row["task_id"]), int(row["init_state_idx"]))
                clients.setdefault(key, []).append({**row, "_json": str(cj)})
        files: dict[tuple[int, int], list[dict]] = {}
        for h5 in sorted(attempt.rglob("*.h5")):
            ident = _h5_identity(h5)
            files.setdefault((ident["task_id"], ident["subset"]), []).append(ident)
        for key in sorted(set(clients) | set(files)):
            if key not in expected:
                rejected.append({"attempt": attempt.name, "task_id": key[0], "subset": key[1],
                                 "problems": ["episode not in the shadow manifest"]})
                continue
            rows, idents = clients.get(key, []), files.get(key, [])
            conflicts: list[str] = []
            if len(rows) > 1:
                conflicts.append(f"{len(rows)} client rows for one episode in this attempt: "
                                 f"{sorted({r['_json'] for r in rows})}")
            if len(idents) > 1:
                conflicts.append(f"{len(idents)} H5 files for one episode in this attempt: "
                                 f"{[i['path'] for i in idents]}")
            if conflicts:
                # Neither first- nor last-wins: the attempt cannot vouch for this episode.
                rejected.append({"attempt": attempt.name, "task_id": key[0], "subset": key[1],
                                 "orig_init_state_idx": expected[key]["orig"], "problems": conflicts})
                continue
            client = rows[0] if rows else None
            ident = idents[0] if idents else None
            problems = judge_episode(suite, key, expected[key], client, ident)
            if problems:
                rejected.append({"attempt": attempt.name, "task_id": key[0], "subset": key[1],
                                 "orig_init_state_idx": expected[key]["orig"], "problems": problems})
                continue
            rec = _accepted_record(key, expected[key], attempt, client, ident)
            if key in accepted:
                duplicates.append({"task_id": key[0], "subset": key[1], "attempts": [accepted[key]["attempt"], attempt.name]})
            else:
                accepted[key] = rec
    languages: dict[int, set[str]] = {}
    for (t, _), rec in accepted.items():
        languages.setdefault(t, set()).add(rec["task_language"])
    inconsistent = {t: sorted(v) for t, v in languages.items() if len(v) != 1}
    missing = [k for k in sorted(expected) if k not in accepted]
    ok = len(accepted) == len(expected) and not duplicates and not inconsistent
    return {
        "protocol": PROTOCOL, "suite": suite, "trials": TRIALS, "expected": len(expected),
        "manifest": str(pathlib.Path(manifest_path).resolve()), "manifest_sha256": _sha256(pathlib.Path(manifest_path)),
        "task_map_bound": task_map is not None,
        "tasks": {str(t): {"task_name": expected[(t, 0)]["task_name"],
                           "task_language": sorted(languages.get(t, {""}))[0] if len(languages.get(t, set())) == 1 else None}
                  for t in range(10)},
        "attempts": [a.name for a in attempts], "ok": ok,
        "accepted": [accepted[k] for k in sorted(accepted)], "n_accepted": len(accepted),
        "rejected": rejected, "duplicates": duplicates, "inconsistent_languages": inconsistent,
        "missing": [{"task_id": t, "subset_init_state_idx": s, "orig_init_state_idx": expected[(t, s)]["orig"]}
                    for t, s in missing],
        "retry_filter": episode_filter(expected, sorted(set(missing))),
        "success_rate": (sum(r["success"] for r in accepted.values()) / len(accepted)) if accepted else None,
    }


def main() -> None:
    """CLI entry: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", required=True, choices=sorted(MAX_STEPS))
    ap.add_argument("--shadow-manifest", required=True)
    ap.add_argument("--task-map", required=True, help="emit_task_map.py output for this suite")
    ap.add_argument("--attempts-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--emit-full-filter", action="store_true",
                    help="also write filter_task_<t>.json for every task (the first attempt's client filters)")
    args = ap.parse_args()
    out = pathlib.Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    task_map = load_task_map(args.task_map, args.suite)
    if args.emit_full_filter:
        manifest = json.loads(pathlib.Path(args.shadow_manifest).read_text(encoding="utf-8"))
        expected = expected_episodes(manifest, task_map)
        for t in range(10):
            rows = episode_filter(expected, [k for k in expected if k[0] == t])
            (out / f"filter_task_{t}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    try:
        res = verify(args.suite, args.shadow_manifest, args.attempts_root, task_map)
    except SystemExit as exc:
        if args.emit_full_filter and "no attempt_" in str(exc):
            print("filters written; no attempts yet")
            return
        raise
    res["task_map"] = str(pathlib.Path(args.task_map).resolve())
    res["task_map_sha256"] = _sha256(pathlib.Path(args.task_map))
    (out / "accepted_shadow_manifest.json").write_text(json.dumps(
        {k: v for k, v in res.items() if k not in ("rejected", "retry_filter", "duplicates", "missing")}, indent=1),
        encoding="utf-8")
    (out / "rejected.json").write_text(json.dumps(
        {"rejected": res["rejected"], "duplicates": res["duplicates"], "missing": res["missing"],
         "inconsistent_languages": res["inconsistent_languages"]}, indent=1),
        encoding="utf-8")
    (out / "retry_filter.json").write_text(json.dumps(res["retry_filter"], indent=1), encoding="utf-8")
    # Per-task retry filters in the layout the client launcher consumes
    # (``<dir>/filter_task_<t>.json``), only for tasks with something to re-run.
    retry_dir = out / "retry"
    retry_dir.mkdir(exist_ok=True)
    for old in retry_dir.glob("filter_task_*.json"):
        old.unlink()
    for t in sorted({int(r["task_id"]) for r in res["retry_filter"]}):
        rows = [r for r in res["retry_filter"] if int(r["task_id"]) == t]
        (retry_dir / f"filter_task_{t}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"accepted={res['n_accepted']}/{res['expected']} rejected={len(res['rejected'])} "
          f"duplicates={len(res['duplicates'])} missing={len(res['missing'])} "
          f"inconsistent_languages={len(res['inconsistent_languages'])} ok={res['ok']}")
    if not res["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
