"""E1 CLI: boundary grids, prefix replay and next-20-control cache risk.

All fit parameters/exposure thresholds use init 0-29. Init 30-49 is a locked
analysis holdout. Risk is descriptive enrichment, never a causal call effect.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from pathlib import Path
import numpy as np

from .common import (
    load_config,
    Unavailable,
    fingerprint,
    load_episodes,
    parser,
    report,
    cluster_interval,
)
from .forensics import analyse
from .pairs import model
from .segmentation import (
    features,
    fit,
    candidates,
    retrospective,
    causal_scores,
    boundary_quality,
    prefix_audit,
    reconstruction,
    baseline_commands,
)


def split(ep):
    init = int(ep.meta.get("init", -1))
    return (
        "discovery"
        if 0 <= init < 30
        else "holdout"
        if 30 <= init < 50
        else "outside_analysis_split"
    )


def tie_coin(ep, control, candidate):
    # Independent, reproducible tie-breaking; all arms share task/init/control.
    return int(
        fingerprint(
            [
                ep.meta.get("suite"),
                ep.meta["task_id"],
                ep.meta["init"],
                control,
                candidate,
            ]
        )[:13],
        16,
    ) / float(16**13)


def pr_metrics(rows):
    available = [r for r in rows if r.get("status") == "available"]
    if not available:
        return {
            "status": "unavailable",
            "reason": "no fully observed risk windows",
            "denominator": 0,
        }
    y = np.array([r["bad_event_next20"] for r in available], bool)
    score = np.array([r["score"] for r in available], float)
    flagged = np.array([r["flagged"] for r in available], bool)
    calibrated = [r for r in available if r.get("risk_probability") is not None]
    order = np.argsort(-score, kind="stable")
    # Stepwise precision-recall at distinct thresholds, including tied scores.
    sy, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]) + 1, len(ss)]
    cum = np.cumsum(sy)
    recall, precision = cum[ends - 1] / max(int(y.sum()), 1), cum[ends - 1] / ends
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision)) if y.any() else None
    return {
        "status": "available",
        "denominator": len(available),
        "events": int(y.sum()),
        "prevalence": float(y.mean()),
        "auprc": ap,
        "flagged_exposure": float(flagged.mean()),
        "target_exposure": 0.20,
        "onset_sensitivity": float((flagged & y).sum() / y.sum()) if y.any() else None,
        "precision_at_frozen_exposure": float(y[flagged].mean())
        if flagged.any()
        else None,
        "median_actionable_lead": float(
            np.median(
                [
                    r["lead_controls"]
                    for r in available
                    if r["flagged"] and r["bad_event_next20"]
                ]
            )
        )
        if (flagged & y).any()
        else None,
        "brier_score": float(
            np.mean(
                [
                    (float(r["bad_event_next20"]) - r["risk_probability"]) ** 2
                    for r in calibrated
                ]
            )
        )
        if calibrated
        else None,
        "brier_denominator": len(calibrated),
        "calibration_source": "discovery event rates in frozen flagged/unflagged strata",
        "clusters": len({(r["suite"], r["task_id"], r["init"]) for r in available}),
    }


def build(episodes, config=None, library_root=None, independent_onsets=None):
    groups = defaultdict(list)
    for ep in episodes:
        groups[(model(ep), ep.meta.get("suite"))].append(ep)
    tables = {
        "segments": [],
        "boundary_quality": [],
        "causal_decisions": [],
        "risk_windows": [],
        "risk_metrics": [],
        "fits": [],
        "unavailable_episodes": [],
        "prefix_replay": [],
        "reconstruction": [],
    }
    for cell, eps in groups.items():
        try:
            fitted = fit(eps, config, library_root)
        except (Unavailable, KeyError, ValueError) as exc:
            tables["unavailable_episodes"].extend(
                dict(ep.identity, status="unavailable", reason=str(exc)) for ep in eps
            )
            continue
        names = candidates(fitted)
        frozen_thresholds = {}
        cached = []
        for ep in eps:
            try:
                f = baseline_commands(ep, features(ep, config), fitted)
                forensic, truth = analyse(ep, config)
                if independent_onsets is not None:
                    key = (ep.meta["arm"], ep.key)
                    if key not in independent_onsets:
                        raise Unavailable(
                            "independent onset file lacks this accepted attempt"
                        )
                    forensic = dict(forensic, **independent_onsets[key])
                scores = {name: causal_scores(f, fitted, name) for name in names}
                cached.append((ep, f, forensic, truth, scores))
            except (Unavailable, KeyError, ValueError) as exc:
                tables["unavailable_episodes"].append(
                    dict(ep.identity, status="unavailable", reason=str(exc))
                )
        for name in names:
            train = []
            for ep, _, onset, _, scores in cached:
                if split(ep) != "discovery":
                    continue
                for d in ep.decisions:
                    if d.get("src") not in ("cache", "cache_tail", "follow"):
                        continue
                    start = ep.before_index(d) + 1
                    if 0 <= start < ep.n and (
                        onset.get("onset_control") is None
                        or start <= int(onset["onset_control"])
                    ):
                        train.append(
                            (float(scores[name][start]), tie_coin(ep, start, name))
                        )
            frozen_thresholds[name] = (
                list(
                    sorted(train)[
                        min(int(np.ceil(0.8 * len(train))) - 1, len(train) - 1)
                    ]
                )
                if train
                else None
            )
        fitted["exposure_thresholds"] = frozen_thresholds
        fitted["fit_hash"] = fingerprint(
            {k: v for k, v in fitted.items() if k != "fit_hash"}
        )
        tables["fits"].append(
            {"model": cell[0], "suite": cell[1], "status": "available", **fitted}
        )
        for ep, f, onset, truth, scores in cached:
            active = np.flatnonzero(f["active"])
            start, end = int(active[0]), int(active[-1]) + 1
            gold = (
                np.flatnonzero(
                    truth["stage"][start + 1 : end] != truth["stage"][start : end - 1]
                )
                + start
                + 1
            ).tolist()
            for name in names:
                boundaries = retrospective(ep, f, fitted, name)
                knots = sorted(set([start] + boundaries + [end]))
                identity = dict(
                    ep.identity,
                    model=cell[0],
                    candidate=name,
                    split=split(ep),
                    fit_hash=fitted["fit_hash"],
                )
                for i, (left, right) in enumerate(zip(knots[:-1], knots[1:])):
                    tables["segments"].append(
                        dict(
                            identity,
                            status="available",
                            control_start=left,
                            control_end_exclusive=right,
                            parent_segment=ep.key + ":interaction",
                            segment_id=ep.key + ":" + name + ":" + str(i),
                            boundary_type=name,
                            confidence="descriptive",
                            support=right - left,
                            feature_tier="portable",
                            retrospective=True,
                            duration_controls=right - left,
                        )
                    )
                for tolerance in (1, 5, 10):
                    tables["boundary_quality"].append(
                        dict(
                            identity,
                            status="available",
                            baseline="candidate",
                            **boundary_quality(boundaries, gold, tolerance),
                        )
                    )
                    uniform = (
                        np.linspace(start, end, len(boundaries) + 2)[1:-1]
                        .round()
                        .astype(int)
                        .tolist()
                    )
                    tables["boundary_quality"].append(
                        dict(
                            identity,
                            status="available",
                            baseline="uniform_matched_boundary_count",
                            **boundary_quality(uniform, gold, tolerance),
                        )
                    )
                tables["prefix_replay"].append(
                    dict(
                        identity,
                        status="available",
                        passed=prefix_audit(f, fitted, name),
                        forbidden_features_used=[],
                        replay="prefix truncation; fixed fit",
                    )
                )
                if name.startswith("waypoint_"):
                    for resolution in (1, 5, 10):
                        points = sorted(
                            set(
                                [start]
                                + retrospective(ep, f, fitted, name, resolution)
                                + [end - 1]
                            )
                        )
                        tables["reconstruction"].append(
                            dict(
                                identity,
                                status="available",
                                resolution_controls=resolution,
                                knots=len(points),
                                **reconstruction(f, points, start, end),
                            )
                        )
                for d in ep.decisions:
                    t = ep.before_index(d) + 1
                    if not 0 <= t < ep.n or split(ep) == "outside_analysis_split":
                        continue
                    threshold = frozen_thresholds[name]
                    score = float(scores[name][t])
                    coin = tie_coin(ep, t, name)
                    flagged = (
                        (score, coin) > tuple(threshold)
                        if threshold is not None
                        else None
                    )
                    rec = dict(
                        identity,
                        decision_id=d.get("decision_id"),
                        decision_seq=d.get("decision_seq"),
                        status="available",
                        control_start=t,
                        latest_observed_control=t - 1,
                        score=score,
                        flagged=flagged,
                        frozen_threshold=threshold,
                        tie_coin=coin,
                        threshold_status="available"
                        if threshold is not None
                        else "unavailable",
                        stage_posterior=None,
                        posterior_status="unavailable",
                        unknown_mass=1.0,
                        entropy=None,
                        posterior_reason="candidate event scores are not calibrated semantic stage probabilities",
                        feature_tier="portable",
                        causal=True,
                        src=d.get("src"),
                    )
                    tables["causal_decisions"].append(rec)
                    if d.get("src") not in ("cache", "cache_tail", "follow"):
                        continue
                    onset_control = onset.get("onset_control")
                    if onset_control is not None and t > int(onset_control):
                        continue  # first-event risk; no post-onset exposure
                    bad = onset_control is not None and t <= int(onset_control) < t + 20
                    observed = ep.n - t >= 20 or bad or bool(ep.outcome["success"])
                    risk = dict(
                        rec,
                        bad_event_next20=bool(bad) if observed else None,
                        status="available"
                        if observed and threshold is not None
                        else "unavailable",
                        reason=""
                        if observed and threshold is not None
                        else "right-censored horizon"
                        if not observed
                        else "no frozen exposure threshold",
                        observed_controls=min(20, ep.n - t),
                        horizon_controls=20,
                        competing_success=bool(ep.outcome["success"]) and ep.n - t < 20,
                        lead_controls=int(onset_control) - t if bad else None,
                        onset_validated=False,
                        estimand="first heuristic onset within next 20 actual controls",
                    )
                    tables["risk_windows"].append(risk)
        for name in names:
            all_rows = [
                r
                for r in tables["risk_windows"]
                if r["model"] == cell[0]
                and r["suite"] == cell[1]
                and r["candidate"] == name
            ]
            train = [
                r
                for r in all_rows
                if r["split"] == "discovery" and r["status"] == "available"
            ]
            calibration = {}
            for flag in (False, True):
                stratum = [r for r in train if r["flagged"] == flag]
                calibration[flag] = (
                    float(np.mean([r["bad_event_next20"] for r in stratum]))
                    if stratum
                    else None
                )
            for r in all_rows:
                r["risk_probability"] = calibration.get(r["flagged"])
                r["calibration_status"] = (
                    "available" if r["risk_probability"] is not None else "unavailable"
                )
            for part in ("discovery", "holdout"):
                for arm in sorted({ep.meta["arm"] for ep in eps}):
                    rows = [
                        r for r in all_rows if r["split"] == part and r["arm"] == arm
                    ]
                    task_metrics = [
                        pr_metrics([r for r in rows if r["task_id"] == task])
                        for task in sorted({r["task_id"] for r in rows})
                    ]
                    sensitivities = [
                        x["onset_sensitivity"]
                        for x in task_metrics
                        if x.get("onset_sensitivity") is not None
                    ]
                    tables["risk_metrics"].append(
                        {
                            "model": cell[0],
                            "suite": cell[1],
                            "arm": arm,
                            "candidate": name,
                            "split": part,
                            "equal_task_onset_sensitivity": float(
                                np.mean(sensitivities)
                            )
                            if sensitivities
                            else None,
                            "contributing_tasks": len(sensitivities),
                            "risk_interval": cluster_interval(rows, "bad_event_next20"),
                            **pr_metrics(rows),
                        }
                    )
    return tables


def main(argv=None):
    ap = parser("segmentation_bench")
    ap.add_argument(
        "--library-root",
        type=Path,
        help="optional frozen library for exact G0 command centers",
    )
    ap.add_argument(
        "--onsets-file",
        type=Path,
        help="independent T1 episodes_forensics.csv, matched by arm/episode_key",
    )
    args = ap.parse_args(argv)
    onsets = None
    if args.onsets_file:
        onsets = {}
        with args.onsets_file.open() as handle:
            for row in csv.DictReader(handle):
                if row.get("status") == "available":
                    onsets[(row["arm"], row["episode_key"])] = {
                        "onset_control": int(row["onset_control"])
                        if row.get("onset_control")
                        else None
                    }
    episodes = load_episodes(args.run_root, args.arms, args.p3v2, args.limit)
    config = load_config(args, episodes)
    tables = build(episodes, config, args.library_root, onsets)
    report(
        args.out,
        "segmentation_bench",
        tables,
        {
            "accepted_episodes": len(episodes),
            "discovery_episodes": sum(split(ep) == "discovery" for ep in episodes),
            "holdout_episodes": sum(split(ep) == "holdout" for ep in episodes),
            "holdout_inits": "30-49 per task",
            "labels_validated": False,
            "causal_routing_claim": False,
            "fit_hashes": [r["fit_hash"] for r in tables["fits"]],
            "onset_source": str(args.onsets_file)
            if args.onsets_file
            else "independent physical T1 heuristics",
            "exposure_matching": "discovery p80 (score, independent hash tie coin) frozen on holdout; report realized exposure",
        },
        episodes=episodes,
    )


if __name__ == "__main__":
    main()
