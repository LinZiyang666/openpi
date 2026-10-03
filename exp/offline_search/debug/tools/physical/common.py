"""Reader boundary, strict joins, and common machine-readable output contract."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np


RULE_VERSION = "r8.physical.v1"


class Unavailable(ValueError):
    """Evidence needed for a calculation was not captured or is invalid."""


def clean(value):
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(x) for x in value]
    return value


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(clean(value), sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def present(value):
    return value is not None and not (isinstance(value, float) and np.isnan(value))


@dataclass
class Episode:
    meta: dict
    controls: dict
    decisions: list = field(default_factory=list)
    manifest: dict = field(default_factory=dict)
    snapshots: dict = field(default_factory=dict)
    reader_arm: object = None
    error: str = ""

    @property
    def key(self):
        return str(self.meta.get("episode_key", "unknown"))

    @property
    def n(self):
        return len(self.controls.get("control_idx", []))

    @property
    def outcome(self):
        """Journal outcome with capture termination semantics, never a fake failure."""
        reason = self.meta.get("termination_reason")
        status = self.meta.get("journal_status", self.meta.get("status"))
        error = self.meta.get("journal_error") or self.meta.get("error")
        success = self.meta.get("journal_success")
        if not present(success):
            success = self.meta.get("success")
        invalid = (
            reason == "exception" or status in ("exception", "error") or bool(error)
        )
        if invalid:
            value, success = "invalid", None
        elif reason in ("step_cap", "timeout", "time_limit", "max_steps"):
            value, success = "timeout", False
        elif present(success):
            success = bool(success)
            value = "success" if success else "failure"
        else:
            value, success = "unknown", None
        return dict(
            outcome=value,
            outcome_status="invalid"
            if invalid
            else "available"
            if value != "unknown"
            else "unavailable",
            termination_reason=reason,
            success=success,
        )

    @property
    def identity(self):
        result = {
            k: self.meta.get(k)
            for k in (
                "arm",
                "episode_key",
                "task_id",
                "init",
                "attempt",
                "suite",
                "env_seed",
                "success",
                "orig_init_state_idx",
            )
        }
        result.update(self.outcome)
        return result

    def require(self, *keys):
        if self.error:
            raise Unavailable(self.error)
        if self.outcome["outcome_status"] == "invalid":
            raise Unavailable("invalid episode outcome: exception or journal error")
        if not self.n:
            raise Unavailable("no captured controls")
        idx = np.asarray(self.controls["control_idx"])
        if not np.array_equal(idx, np.arange(self.n)):
            raise Unavailable("control_idx is not contiguous from zero")
        for key in keys:
            if key not in self.controls:
                raise Unavailable("missing control field: " + key)
            arr = np.asarray(self.controls[key])
            if len(arr) != self.n:
                raise Unavailable("misaligned control field: " + key)
            if arr.dtype.kind in "fc" and not np.isfinite(arr).all():
                raise Unavailable("nonfinite control field: " + key)
        expected = self.meta.get("n_controls")
        if present(expected) and int(expected) != self.n:
            raise Unavailable("n_controls does not match capture")

    def decision_at(self, index):
        seq = int(self.controls.get("decision_seq", np.full(self.n, -1))[index])
        matches = [
            d
            for d in self.decisions
            if present(d.get("decision_seq")) and int(d["decision_seq"]) == seq
        ]
        if len(matches) > 1:
            raise Unavailable("duplicate decision_seq within accepted attempt")
        return matches[0] if matches else {"decision_seq": seq}

    def before_index(self, decision):
        """Latest measured after-state available before a decision was chosen."""
        start = decision.get("control_idx_start")
        if not present(start):
            seq = int(decision["decision_seq"])
            hits = np.flatnonzero(
                np.asarray(self.controls.get("decision_seq", [])) == seq
            )
            if not len(hits):
                raise Unavailable("decision has no applied controls or start index")
            start = int(hits[0])
        return int(start) - 1


def _records(table):
    return table.to_dict("records") if hasattr(table, "to_dict") else list(table)


def normalize_legacy(ep):
    """E3 compact extraction through the P3 reader's retained raw records.

    P3 contact naming stays invalid. This adapts offline physical quantities.
    """
    raw = ep.controls.get("legacy_records", [])
    if not raw:
        return ep
    initial = raw[0].get("before", {})
    numeric = initial.get("observation_numeric", {})
    names = sorted(
        k[:-4]
        for k in numeric
        if k.endswith("_pos")
        and not k.startswith("robot")
        and not k.endswith("_to_robot0_eef_pos")
    )
    predicates = sorted(initial.get("predicates", {}))
    ids = initial.get("entity_ids", {})
    bodies = ids.get("body_names", [])
    ep.meta["entities"] = {
        "movable": [{"name": x} for x in names],
        "predicates": predicates,
        "geom_mapping_status": "unsupported",
        "destination_names": bodies,
    }
    ep.meta["p3_contact_names_invalid"] = True
    ep.meta["env_seed"] = raw[0].get("environment_seed", ep.meta.get("env_seed"))
    ep.meta["reset_state_sha256"] = (
        fingerprint(initial.get("sim_state"))
        if initial.get("sim_state") is not None
        else None
    )
    ep.meta["suite"] = ep.meta.get("suite", ep.meta.get("experiment", "libero"))
    ep.meta["physical_adapter"] = {
        "environment": "libero",
        "obj_quat_order": "xyzw",
        "eef_quat_order": "xyzw",
    }
    ep.controls["is_settle"] = np.array([bool(x.get("wait_phase")) for x in raw])
    for field_name, source in (
        ("eef_pos", "robot0_eef_pos"),
        ("eef_quat", "robot0_eef_quat"),
        ("gripper_qpos", "robot0_gripper_qpos"),
        ("gripper_qvel", "robot0_gripper_qvel"),
    ):
        if all(
            source in x.get("after", {}).get("observation_numeric", {}) for x in raw
        ):
            ep.controls[field_name] = np.array(
                [x["after"]["observation_numeric"][source] for x in raw], float
            )
    for field_name, suffix in (("obj_pos", "_pos"), ("obj_quat", "_quat")):
        if names:
            ep.controls[field_name] = np.array(
                [
                    [x["after"]["observation_numeric"][name + suffix] for name in names]
                    for x in raw
                ],
                float,
            )
    ep.controls["predicates"] = np.array(
        [
            [
                x["after"].get("predicates", {}).get(p, np.nan)
                if x["after"].get("predicates", {}).get(p) is not None
                else np.nan
                for p in predicates
            ]
            for x in raw
        ],
        float,
    )
    if bodies and all("body_xpos" in x["after"] for x in raw):
        ep.controls["destination_pos"] = np.array(
            [x["after"]["body_xpos"] for x in raw], float
        )
    if all("qpos" in x.get("before", {}) for x in raw):
        ep.controls["before_qpos"] = np.array([x["before"]["qpos"] for x in raw], float)
    for d in ep.decisions:
        if "src" not in d and "source" in d:
            d["src"] = d["source"]
        if d.get("src") == "cache_blind":
            d["legacy_src"] = "cache_blind"
            d["src"] = "cache_tail"
    ep.controls.pop(
        "legacy_records", None
    )  # discard redundant expanded JSON after lossless numeric extraction
    return ep


def load_episodes(run_root, arms, p3v2=False, limit=None):
    from exp.offline_search.debug import reader

    episodes = []
    for name in arms:
        arm = (reader.p3v2_adapter if p3v2 else reader.open_arm)(run_root, name)
        arm.cache_enabled = False  # every physical consumer is read-only
        decisions = clean(_records(arm.decisions(accepted_only=True)))
        by_key = {}
        for d in decisions:
            key = d.get("episode_key")
            if not present(key) and present(d.get("decision_id")):
                key = str(d["decision_id"]).split(":")[0]
            by_key.setdefault(str(key), []).append(d)
        for row in _records(arm.episodes(accepted_only=True))[:limit]:
            row = dict(clean(row), arm=name)
            key = str(row["episode_key"])
            selected = (
                [
                    arm.record_meta(d)
                    for d in by_key.get(key, [])
                    if present(d.get("_server_dir"))
                ]
                if hasattr(arm, "record_meta")
                else []
            )
            meta = dict(next((m for m in selected if m), arm.server_meta))
            meta["server_process_metas"] = dict(getattr(arm, "server_metas", {}))
            meta.update(arm.manifest)
            try:
                controls = arm.controls(key)
                snapshots = arm.snapshots(key)
                ep = Episode(row, controls, by_key.get(key, []), meta, snapshots, arm)
            except (ValueError, OSError, KeyError) as exc:
                ep = Episode(
                    row,
                    {},
                    by_key.get(key, []),
                    meta,
                    reader_arm=arm,
                    error=type(exc).__name__ + ": " + str(exc),
                )
            episodes.append(normalize_legacy(ep) if p3v2 else ep)
    return episodes


def load_config(args, episodes):
    from .adapters import calibrate

    config = (
        json.loads(args.adapter_config.read_text()) if args.adapter_config else None
    )
    result = calibrate(episodes, config)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "adapter_calibration.json").write_text(
        json.dumps(clean(result), indent=2, allow_nan=False) + "\n"
    )
    return result


def safe_rows(function, episode, *args):
    try:
        return function(episode, *args)
    except (Unavailable, KeyError, ValueError, IndexError) as exc:
        return [
            dict(
                episode.identity,
                status="unavailable",
                reason=str(exc),
                rule_version=RULE_VERSION,
            )
        ]


def parallel_map(function, values, procs):
    """Bounded reader threads: a CLI remains one Python process even at default 4."""
    if procs < 1:
        raise ValueError("--procs must be positive")
    if procs == 1:
        return [function(x) for x in values]
    with ThreadPoolExecutor(max_workers=procs) as pool:
        return list(pool.map(function, values))


def cluster_interval(rows, value, draws=1000, seed=0):
    """Resample task/init clusters, keeping arms/replicates in the same cluster."""
    groups = {}
    for row in rows:
        if row.get("status", "available") == "available" and present(row.get(value)):
            key = (row.get("suite"), row.get("task_id"), row.get("init"))
            groups.setdefault(key, []).append(float(row[value]))
    if len(groups) < 2:
        return {
            "status": "unavailable",
            "reason": "fewer than two task/init clusters",
            "clusters": len(groups),
        }
    groups = list(groups.values())
    sums = np.array([sum(x) for x in groups])
    counts = np.array([len(x) for x in groups])
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(draws):
        sample = rng.integers(0, len(groups), len(groups))
        estimates.append(float(sums[sample].sum() / counts[sample].sum()))
    return {
        "status": "available",
        "clusters": len(groups),
        "draws": draws,
        "mean": float(sums.sum() / counts.sum()),
        "ci95": np.quantile(estimates, [0.025, 0.975]).tolist(),
        "unit": "task/init cluster",
        "seed": seed,
    }


def auroc(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels, bool)
    if not labels.any() or labels.all():
        return None
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), float)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    npos = int(labels.sum())
    return float(
        (ranks[labels].sum() - npos * (npos + 1) / 2.0) / (npos * (~labels).sum())
    )


def input_summary(episodes):
    """Expose shared-reader tail skips and per-process capture provenance."""
    from exp.offline_search.debug import reader

    arms = {}
    for ep in episodes:
        name = ep.meta.get("arm")
        arm = ep.reader_arm
        if name in arms or arm is None:
            continue
        dirs = getattr(arm, "server_dirs", [])
        arms[name] = dict(
            read_issues=list(getattr(arm, "read_issues", [])),
            server_meta_files=sorted(getattr(arm, "server_metas", {})),
            writer_stats={
                str(p): reader.read_json(p)
                for d in dirs
                for p in sorted(d.glob("writer_stats*.json"))
            },
        )
    return dict(
        arms=arms,
        outcomes={
            name: sum(ep.outcome["outcome"] == name for ep in episodes)
            for name in ("success", "failure", "timeout", "invalid", "unknown")
        },
        episodes_with_snapshots=sum(bool(ep.snapshots) for ep in episodes),
    )


def report(out, tool, tables, summary=None, episodes=None):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    denominators = {}
    for name, rows in tables.items():
        rows = clean(rows)
        keys = sorted({key for row in rows for key in row}) or ["status", "reason"]
        with (out / (name + ".csv")).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        k: json.dumps(v, allow_nan=False)
                        if isinstance(v, (dict, list))
                        else v
                        for k, v in row.items()
                    }
                )
        denominators[name] = {
            "rows": len(rows),
            "available": sum(r.get("status", "available") == "available" for r in rows),
            "unavailable": sum(r.get("status") == "unavailable" for r in rows),
        }
    summary = dict(summary or {})
    if episodes is not None:
        summary["input_capture"] = input_summary(episodes)
    payload = clean(
        {
            "tool": tool,
            "rule_version": RULE_VERSION,
            "denominators": denominators,
            "summary": summary or {},
            "tables": tables,
        }
    )
    (out / (tool + ".json")).write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n"
    )
    lines = [
        tool + " (" + RULE_VERSION + ")",
        "",
        "Accepted attempts only. Automatic physical labels require human validation.",
        "",
    ]
    for name, counts in denominators.items():
        lines.append(
            "- %s: %d rows; %d available; %d unavailable."
            % (name, counts["rows"], counts["available"], counts["unavailable"])
        )
    lines += [
        "",
        "Summary: `" + json.dumps(clean(summary or {}), sort_keys=True) + "`",
        "",
    ]
    (out / (tool + ".md")).write_text("\n".join(lines))
    return payload


def parser(tool):
    ap = argparse.ArgumentParser(description="Offline " + tool + " physical profile")
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--procs",
        type=int,
        default=4,
        help="bounded analysis threads; one Python process",
    )
    ap.add_argument("--p3v2", action="store_true", help="use reader.p3v2_adapter")
    ap.add_argument(
        "--limit", type=int, help="maximum accepted episodes per arm, for development"
    )
    ap.add_argument(
        "--adapter-config",
        type=Path,
        help="JSON physical conventions/scales; overrides manifest adapter",
    )
    return ap
