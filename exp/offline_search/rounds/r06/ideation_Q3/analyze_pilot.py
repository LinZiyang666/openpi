"""Q3 P3-v2 offline analysis. See PREREG_FROZEN.md for estimands and limits.

Never launches a simulator. Optional read_v2 subprocess only reads finished logs.
All output paths must be below this script's directory. No pilot polling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
P3 = HERE.parent / "p3_profiling"
COHORTS = ["A", "dose125", "dose25", "dose50", "P10", "B", "factorial", "window", "dose_mix"]
EPKEY = ["arm", "uid", "attempt"]
CLUSTER = ["task_id", "init"]
PAIR = ["cell", "task_id", "init", "block"]
GUARDS = ["coverage", "stuck", "lag", "overtime", "progress", "terminal"]
GATES = ["coverage_high", "neighbour_high", "early", "nonadvance", "never_called", "G_alpha20"]
METRICS = ["Y", "future_misses", "future_controls", "future_cost"]
PRIMARY = ["factorial_call", "factorial_at_guard", "factorial_pre_guard", "coverage_enrichment",
           "guard_enrichment", "duration10_vs5", "hold3_vs1", "window_delay0_vs2"]


def require(test, message):
    if not bool(test):
        raise ValueError(message)


def safe_out(path):
    path = Path(path).resolve()
    require(path.is_relative_to(HERE) and path != HERE, "outputs must be in a child of ideation_Q3")
    return path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, np.ndarray)):
        return [clean(v) for v in x]
    if isinstance(x, (np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (float, np.floating)) and not np.isfinite(x):
        return None
    return x


def dump(path, value):
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + "\n")


def decode(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, str):
        if x in ("True", "False"):
            return x == "True"
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return x
    return x.item() if isinstance(x, np.generic) else x


def bseries(x):
    require(not x.isna().any(), f"missing boolean {x.name}")
    require(x.isin([True, False, "True", "False", 0, 1]).all(), f"invalid boolean {x.name}")
    return x.isin([True, "True", 1])


def number(df, col):
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def kish(weights):
    w = np.asarray(weights, float)
    return float(w.sum() ** 2 / (w @ w)) if w @ w > 0 else 0.


def cluster_uncertainty(influence):
    """Task-stratified init-cluster sandwich; all repeated seed blocks stay together."""
    k = len(influence)
    tasks = list(influence.groupby(level="task_id"))
    df = k - len(tasks)
    singleton = any(len(v) < 2 for _, v in tasks)
    variance = sum(len(v) / (len(v) - 1) * float(((v - v.mean()) ** 2).sum())
                   for _, v in tasks if len(v) > 1)
    se = math.sqrt(max(0., variance)) if df > 0 and not singleton else np.nan
    totals = np.array([float(v.sum()) for _, v in tasks])
    task_se = float(np.sqrt(len(totals) / (len(totals) - 1) * np.sum((totals - totals.mean()) ** 2))) if len(totals) > 1 else np.nan
    return dict(clusters=k, tasks=len(tasks), df=df, singleton_task=singleton, se=se,
                task_se=task_se, task_df=len(tasks)-1)


def estimate(frame, numerator, denominator=None, side_a=None, side_b=None):
    """Ratio of HT totals. Never inverse-weight by observed trajectory length."""
    if not len(frame):
        return dict(n=0, estimate=np.nan, status="empty"), pd.Series(dtype=float)
    w = frame["episode_weight"].to_numpy(float)
    num = np.asarray(numerator, float)
    den = np.ones(len(frame)) if denominator is None else np.asarray(denominator, float)
    require(np.isfinite(num).all() and np.isfinite(den).all(), "nonfinite estimator inputs")
    total = float(w @ den)
    require(total > 0, "empty risk-set denominator")
    point = float(w @ num / total)
    tmp = frame[CLUSTER].copy()
    tmp["influence"] = w * (num - point * den) / total
    influence = tmp.groupby(CLUSTER)["influence"].sum()
    rec = dict(n=len(frame), episodes=frame[EPKEY].drop_duplicates().shape[0],
               estimate=point, denominator=total, **cluster_uncertainty(influence))
    for label, s in (("a", side_a), ("b", side_b)):
        s = np.ones(len(frame)) if s is None else np.asarray(s, float)
        sw = w * s
        tmp["side"] = sw
        cw = tmp.groupby(CLUSTER)["side"].sum()
        rec.update({f"n_{label}": int(np.count_nonzero(s)), f"clusters_{label}": int((cw > 0).sum()),
                    f"row_ess_{label}": kish(sw), f"cluster_ess_{label}": kish(cw),
                    f"max_inverse_weight_{label}": float(s.max(initial=0))})
    return finish_interval(rec), influence


def finish_interval(rec):
    point, se, df = rec["estimate"], rec.get("se", np.nan), rec.get("df", 0)
    two_sided = rec.get("n_a", 0) > 0 and rec.get("n_b", 0) > 0
    for family, multiplicity in (("95", 1), ("primary", 64), ("selection", 48)):
        critical = student_t.ppf(1 - .05 / (2 * multiplicity), df) if df > 0 else np.nan
        rec[f"lo_{family}"] = point - critical * se if se > 0 and two_sided else np.nan
        rec[f"hi_{family}"] = point + critical * se if se > 0 and two_sided else np.nan
    ts, td = rec.get("task_se", np.nan), rec.get("task_df", 0)
    tc = student_t.ppf(.975, td) if td > 0 else np.nan
    rec.update(task_lo95=point-tc*ts if ts > 0 and two_sided else np.nan, task_hi95=point+tc*ts if ts > 0 and two_sided else np.nan,
               p_value=float(2*student_t.sf(abs(point/se), df)) if se > 0 and two_sided else np.nan)
    rec["nomination_support"] = bool(df >= 4 and se > 0 and all(
        rec.get(f"clusters_{s}", 0) >= 10 and rec.get(f"cluster_ess_{s}", 0) >= 10 for s in ("a", "b")))
    rec["status"] = ("point_only_no_realized_comparison" if not two_sided else
                     "estimated" if se > 0 else "point_only_no_estimable_fixed_task_variance")
    return rec


def interaction(high, low):
    rh, ih = high
    rl, il = low
    if rh.get("n", 0) == 0 or rl.get("n", 0) == 0:
        return dict(n=0, estimate=np.nan, status="missing_feature_side"), pd.Series(dtype=float)
    ih, il = ih.align(il, fill_value=0)
    infl = ih - il
    rec = dict(n=rh["n"]+rl["n"], estimate=rh["estimate"]-rl["estimate"], **cluster_uncertainty(infl))
    for key in ("n_a", "n_b", "clusters_a", "clusters_b", "cluster_ess_a", "cluster_ess_b"):
        rec[key] = min(rh[key], rl[key])
    rec["support_convention"] = "minimum across high/low strata; both required"
    return finish_interval(rec), infl


def weighted_quantile(values, weights, q=.9):
    order = np.argsort(values, kind="stable")
    x, w = np.asarray(values)[order], np.asarray(weights)[order]
    return float(x[min(np.searchsorted(np.cumsum(w), q*w.sum(), side="left"), len(x)-1)])


def cell_for(row):
    # Library aliases are not numeric; cell scale is declared by generated arm ID.
    arm = row["arm"]
    for model in ("pi05", "groot"):
        for suite in ("l10", "sp"):
            for lib in (50, 500):
                cell = f"{model}_{suite}_{lib}"
                if f"_{cell}_" in arm:
                    return cell
    raise ValueError(f"unrecognized cell ID {arm}; provide declared P3 v2 arm names")


def read_tables(directories, arms=None):
    required = {"Y", "arm", "uid", "attempt", "task_id", "init", "model", "suite", "lib", "step",
                "actual_commit_controls", "future_controls", "future_misses", "commit_controls", "commit_truncated",
                "distance.commit10_rms", "retrieval.d1", "retrieval.d1_loeo_quantile", "retrieval.loeo_n",
                "retrieval.dispersion_rms", "retrieval.progress", "resampling.selected", "resampling.selection_p",
                "resampling.dispersion_per_step", "resampling.seeds", "physical_transitions_available", "environment_seed_verified",
                "client_environment_seed", "provenance.split", "provenance.run_block", "catalog_sha256", "calibration.calibration_sha256",
                "guards.inputs_outputs.os_force_miss"}
    dcols = set(EPKEY + ["step", "Y", "vision", "actual_controls", "source", "shadow_available", "controller_observes_shadow",
                         "full_policy_forwards", "stage_invocations.stage1", "stage_invocations.stage2", "stage_invocations.stage3",
                         "client.snapshot.selected", "client.snapshot.restore_certified", "client.snapshot.probability"])
    tables = {key: [] for key in ("anchors", "decisions", "episodes")}
    inputs = []
    for directory in directories:
        directory = Path(directory)
        audit = json.loads((directory / "audit.json").read_text())
        source = dict(path=str(directory.resolve()), audit=audit, hashes={})
        for key in tables:
            use = (lambda c: c in required or c.startswith("assignment.") or c.startswith("calibration.p_values.")
                   or c.startswith("calibration.statistics.") or c.startswith("episode_assignment.")) if key == "anchors" else (
                   (lambda c: c in dcols) if key == "decisions" else None)
            f = directory / f"{key}.csv"
            g = pd.read_csv(f, usecols=use, low_memory=False)
            require(len(g) == audit[key], f"audit count mismatch: {f}")
            source["hashes"][key] = sha(f)
            if arms:
                g = g[g.arm.isin(arms)]
            tables[key].append(g)
        inputs.append(source)
    return {k: pd.concat(v, ignore_index=True) for k, v in tables.items()}, inputs


def replay_assignments(a):
    from exp.offline_search.rounds.r06.p3_profiling.assignment import assign
    from exp.offline_search.rounds.r06.p3_profiling.design import Design
    fields = ["actual_propensity", "nominal_propensity", "coin_call", "executed_policy", "override", "eligible",
              "anchor_index", "calls_before", "hold_before", "due_before", "last_call_anchor", "anchors_since_call",
              "duration_choice", "duration_probability", "hold_choice", "hold_probability", "delay_choice", "delay_probability",
              "scheduled_trigger", "commit_controls", "joint_choice_probability", "treatment_probability", "realized_source"]
    count = 0
    for _, g in a.groupby(EPKEY, sort=False):
        g = g.sort_values("step")
        first = g.iloc[0]
        for c in g:
            if c.startswith("assignment.future_controller."):
                require(g[c].nunique(dropna=False) == 1, f"within-episode design mutation: {c}")
        config = {c.removeprefix("assignment.future_controller."): decode(first[c]) for c in g if c.startswith("assignment.future_controller.")}
        for key in ("cap", "cooldown", "start_anchor", "split_modulus"):
            if config.get(key) is not None:
                config[key] = int(config[key])
        design = Design(config)
        for row in g.to_dict("records"):
            initial = assign(*(int(row["assignment."+k]) for k in ("seed", "task_id", "init", "replicate", "step")),
                             float(row["assignment.propensity"]))
            for key in ("policy_seed", "assigned_call", "uniform"):
                observed = decode(row["assignment."+key])
                require(np.isclose(observed, initial[key], rtol=0, atol=1e-14) if key == "uniform" else observed == initial[key],
                        f"source RNG replay mismatch {key}")
            expected = design.resolve(initial, bool(row["assignment.baseline_hit"]))
            for key in fields:
                observed, target = decode(row["assignment."+key]), expected[key]
                equal = np.isclose(observed, target, rtol=0, atol=1e-14) if isinstance(target, float) else observed == target
                require(equal, f"assignment replay mismatch {row['arm']} {row['uid']} {row['step']} {key}: {observed} != {target}")
            count += 1
    return count


def expected_slots(block, stage):
    upper = (25, 15, 10)[int(block)]
    inits = range(2) if stage == "pilot" else (range(2, upper) if stage == "continuation" else range(upper))
    return {(t, i) for t in range(10) for i in inits}


def validate(tables, args):
    a, e, d = (tables[k] for k in ("anchors", "episodes", "decisions"))
    for key, df in ((EPKEY, e), (EPKEY+["step"], a), (EPKEY+["step"], d)):
        require(not df.duplicated(key).any(), f"duplicate accepted identities {key}")
    require(len(e) > 0, "no episodes")
    for df in (a, e):
        for col in ("physical_transitions_available", "environment_seed_verified"):
            require(bseries(df[col]).all(), f"client verification required: {col}")
    require(bseries(e.controls_verified).all(), "controls unverified")
    require(bseries(d.shadow_available).all(), "missing policy shadow")
    require(not bseries(d.controller_observes_shadow).any(), "shadow leaked into controller")
    require(e.Y.isin([0, 1]).all() and a.Y.isin([0, 1]).all(), "invalid terminal Y")
    metadata = a.groupby(EPKEY).agg(cohort=("assignment.cohort", "first"), block=("assignment.replicate", "first"))
    e = e.merge(metadata, on=EPKEY, validate="one_to_one")
    require(len(e) == len(tables["episodes"]), "episode without anchors")
    e["cell"] = e.apply(cell_for, axis=1)
    e["split"] = np.where(e.init % 5 == 0, "calibration", "validation")
    e["episode_weight"] = 1. / (e.groupby(["cell", "cohort"]).task_id.transform("nunique") *
                                     e.groupby(["cell", "cohort", "task_id"]).uid.transform("size"))
    e["IR"] = e.deployment_IR_per_actual_5_controls
    require(np.isfinite(e.IR).all() and (e.active_controls > 0).all(), "invalid deployment denominator")
    require(not e.duplicated(PAIR+["cohort"]).any(), "duplicate task/init/block in cohort")
    a = a.merge(e[EPKEY+["cell", "cohort", "block", "split", "episode_weight", "active_controls"]], on=EPKEY, validate="many_to_one")
    require(len(a) == len(tables["anchors"]), "orphan anchors")
    ey = e[EPKEY+["Y", "task_id", "init"]]
    ay = a.merge(ey, on=EPKEY, suffixes=("", "_episode"), validate="many_to_one")
    require(all((ay[c] == ay[c+"_episode"]).all() for c in ("Y", "task_id", "init")), "anchor/episode identity or outcome mismatch")
    require((a["provenance.run_block"] == a.block).all() and (a["provenance.split"] == a.split).all(), "block/split provenance mismatch")
    require((a["assignment.cohort_probability"] == 1).all(), "fixed cohort probability must be one")
    require((a["assignment.anchor_index"] >= 0).all(), "bad anchor index")
    join = d.merge(e[EPKEY+["Y"]], on=EPKEY, suffixes=("", "_episode"), validate="many_to_one")
    require(len(join) == len(d) and (join.Y == join.Y_episode).all(), "decision outcome/episode join failure")
    for key, g in d.groupby(EPKEY, sort=False):
        require(sorted(g.step.tolist()) == list(range(len(g))), f"missing decision: {key}")
    dc = d.groupby(EPKEY).agg(actual=("actual_controls", "sum"), decisions=("step", "size"))
    er = e.set_index(EPKEY).reindex(dc.index)
    require((dc.actual == er.active_controls).all() and (dc.decisions == er.decisions).all(), "control/decision totals mismatch")
    require((d["stage_invocations.stage1"] == 1).all() and (d["stage_invocations.stage2"] == 1).all()
            and (d.full_policy_forwards == 1).all(), "measured stage counts missing")
    st3 = a[EPKEY+["step", "resampling.selected"]].copy()
    st3["extra"] = 3*bseries(st3["resampling.selected"]).astype(int)
    dd = d.merge(st3[EPKEY+["step", "extra"]], how="left", on=EPKEY+["step"], validate="one_to_one")
    require((dd["stage_invocations.stage3"] == 1 + dd.extra.fillna(0)).all(), "K4 dispatch mismatch")
    require(int(bseries(d.vision).sum()) == len(a), "anchor/vision count mismatch")
    c1 = np.where(e.model == "pi05", .152, .148)
    computed_ir = (c1*e.anchors + (1-c1)*e.misses)/(e.active_controls/5)
    require(np.allclose(computed_ir, e.IR, rtol=1e-12), "IR accounting mismatch")
    e["modeled_cost"] = c1*e.anchors + (1-c1)*e.misses
    replayed = replay_assignments(a)
    if not args.smoke:
        require((e.client_environment_seed == 603+e.block).all(), "environment seed does not match scheduled block")
        for arm, g in e.groupby("arm"):
            slots = set(map(tuple, g[["task_id", "init"]].to_numpy()))
            require(slots == expected_slots(g.block.iloc[0], args.stage), f"{arm}: incomplete/unexpected scheduled {args.stage} slots")
        for cell, g in e.groupby("cell"):
            if not args.allow_partial:
                require(set(g.cohort) == set(COHORTS) and all(set(h.block) == {0, 1, 2} for _, h in g.groupby("cohort")),
                        f"{cell}: complete cell needs all nine cohorts/three blocks; partial analysis must be explicit")
        f = a[a.cohort == "factorial"]
        expect_p = np.array([.125, .25, .5])[np.searchsorted([1/3, 2/3], f["retrieval.d1_loeo_quantile"], side="right")]
        require(np.allclose(f["assignment.propensity"], expect_p), "factorial p differs from preregistered design")
    a = a.sort_values(EPKEY+["step"]).reset_index(drop=True)
    a["future_anchors"] = a.groupby(EPKEY).cumcount(ascending=False)+1
    c1 = np.where(a.model == "pi05", .152, .148)
    a["future_cost"] = c1*a.future_anchors+(1-c1)*a.future_misses
    for _, g in a.groupby(EPKEY, sort=False):
        require(int(g.iloc[0].future_controls) == int(g.iloc[0].active_controls), "future control horizon mismatch")
        require(np.array_equal(g.future_misses.to_numpy(), np.cumsum(g["assignment.executed_policy"].to_numpy(int)[::-1])[::-1]),
                "future MISS count mismatch")
    audit = dict(episodes=len(e), anchors=len(a), decisions=len(d), arms=e.arm.nunique(), cells=e.cell.unique().tolist(),
                 assignments_replayed=replayed, actual_controls=int(e.active_controls.sum()),
                 selected_K4=int(bseries(a["resampling.selected"]).sum()),
                 measured_stage1=int(d["stage_invocations.stage1"].sum()), measured_stage2=int(d["stage_invocations.stage2"].sum()),
                 measured_stage3=int(d["stage_invocations.stage3"].sum()),
                 snapshot_restore_certification="not inferred from logging; P3 snapshots not certified for branch recovery",
                 smoke=args.smoke, partial=args.allow_partial, stage=args.stage)
    return a, e, d, audit


def features(a, e):
    a = a.copy()
    a["coverage"] = number(a, "retrieval.d1_loeo_quantile")
    a["coverage_high"] = (a.coverage >= 2/3).astype(float).where(a.coverage.notna())
    a["neighbour"] = number(a, "retrieval.dispersion_rms")
    a["shadow"] = number(a, "distance.commit10_rms")
    a["phase"] = a["retrieval.progress"].map(lambda v: float(decode(v)[0]) if decode(v) else np.nan)
    a["early"] = (a.phase < 1/3).astype(float).where(a.phase.notna())
    a["late"] = (a.phase >= 2/3).astype(float).where(a.phase.notna())
    a["k4"] = a["resampling.dispersion_per_step"].map(lambda v: float(np.mean(decode(v)[:10])) if decode(v) else np.nan)
    a["k4_sampling_weight"] = np.where(a.k4.notna(), 1/number(a, "resampling.selection_p"), np.nan)
    a["shadow_over_k4"] = (a.shadow > a.k4).astype(float).where(a.k4.notna() & a.shadow.notna())
    a["at_guard"] = bseries(a["assignment.pre_guard_call"]).astype(float)
    diagnostic = number(a, "guards.inputs_outputs.os_force_miss")
    a["diagnostic_guard"] = (diagnostic == 1).astype(float).where(diagnostic.notna())
    a["never_called"] = (number(a, "assignment.calls_before") == 0).astype(float)
    a["still"] = (number(a, "calibration.statistics.stuck") > 0).astype(float).where(number(a, "calibration.statistics.stuck").notna())
    a["nonadvance"] = (number(a, "calibration.statistics.progress") > 0).astype(float).where(number(a, "calibration.statistics.progress").notna())
    a["previous_guard"] = a.groupby(EPKEY).diagnostic_guard.transform(
        lambda s: s.cumsum().shift(fill_value=0).gt(0).astype(float).where(s.notna().cummin()))
    pv = pd.concat([number(a, "calibration.p_values."+g) for g in GUARDS], axis=1)
    a["G_score"] = (6*pv.min(axis=1)).where(pv.notna().all(axis=1))
    for alpha in (.05, .10, .20):
        a[f"G_alpha{int(100*alpha):02}"] = (a.G_score <= alpha).astype(float).where(a.G_score.notna())
    for g in GUARDS:
        a["p_"+g] = number(a, "calibration.p_values."+g)
        a["p_"+g+"_low"] = (a["p_"+g] <= .20).astype(float).where(a["p_"+g].notna())
    references = []
    for signal in ("neighbour", "shadow", "k4"):
        a[signal+"_high"] = np.nan
    a["long_since_call"] = np.nan
    for (cell, task), whole in a.groupby(["cell", "task_id"]):
        ref = whole[(whole.cohort == "A") & (whole.split == "calibration")]
        for signal in ("neighbour", "shadow", "k4"):
            r = ref[ref[signal].notna()].copy()
            threshold = np.nan
            nep = r[EPKEY].drop_duplicates().shape[0]
            if len(r) >= 20 and nep >= 2:
                r["sw"] = r.k4_sampling_weight if signal == "k4" else 1.
                r["rw"] = r.sw / r.groupby(EPKEY).sw.transform("sum")
                threshold = weighted_quantile(r[signal], r.rw)
                a.loc[whole.index, signal+"_high"] = (whole[signal] > threshold).astype(float).where(whole[signal].notna())
            references.append(dict(cell=cell, task_id=int(task), signal=signal, q=.9, threshold=threshold,
                                   reference="outcome-blind A init%5==0, episode equal", n=len(r), episodes=nep,
                                   samples=r[EPKEY+["step", signal]].to_dict("records")))
        er = e[(e.cell == cell) & (e.task_id == task) & (e.cohort == "A") & (e.split == "calibration")]
        if len(er) >= 2:
            med = float(er.anchors.median())
            age = number(whole, "assignment.anchors_since_call")/med
            a.loc[whole.index, "long_since_call"] = (age >= 1/3).astype(float).where(age.notna())
        else:
            med = np.nan
        references.append(dict(cell=cell, task_id=int(task), signal="age_reference_anchors", threshold=med, episodes=len(er)))
    return a, references


def subsets(df):
    yield "all", df
    yield "calibration", df[df.split == "calibration"]
    yield "validation", df[df.split == "validation"]
    if (df.init >= 2).any():
        yield "continuation_validation", df[(df.init >= 2) & (df.split == "validation")]


def contrast(frame, event_a, pa, event_b, pb, outcome, sampling=None):
    if not len(frame):
        return estimate(frame, [])
    ea, eb = np.asarray(event_a, bool), np.asarray(event_b, bool)
    pa, pb = np.broadcast_to(np.asarray(pa, float), (len(frame),)), np.broadcast_to(np.asarray(pb, float), (len(frame),))
    require(((pa > 0) & (pa <= 1) & (pb > 0) & (pb <= 1)).all(), "unsupported contrast propensity")
    sw = np.ones(len(frame)) if sampling is None else np.asarray(sampling, float)
    sa, sb = ea/pa*sw, eb/pb*sw
    return estimate(frame, (sa-sb)*frame[outcome].to_numpy(float), sw, sa, sb)


def call_effect(frame, outcome, sampling=None):
    z = bseries(frame["assignment.coin_call"]).to_numpy()
    p = frame["assignment.nominal_propensity"].to_numpy(float)
    return contrast(frame, z, p, ~z, 1-p, outcome, sampling)


def randomized(a):
    rows, ratios = [], []
    feature_names = GATES+["at_guard", "diagnostic_guard", "still", "late", "previous_guard", "long_since_call", "shadow_high",
                         "k4_high", "shadow_over_k4", "G_alpha05", "G_alpha10"]+["p_"+g+"_low" for g in GUARDS]
    for cell, ac in a.groupby("cell"):
        for split, asc in subsets(ac):
            for cohort in ("factorial", "dose125", "dose25", "dose50"):
                g = asc[(asc.cohort == cohort) & (asc["assignment.override"] == "coin") &
                        asc["assignment.nominal_propensity"].between(0, 1, inclusive="neither")].copy()
                if not len(g):
                    continue
                require(bseries(g["assignment.available_cache"]).all() and bseries(g["assignment.available_policy"]).all(), "unavailable source")
                if cohort == "factorial":
                    require(bseries(g["assignment.pre_guard_randomization"]).all(), "factorial not pre-guard")
                cache = {}
                def emit(name, outcome, result, feature="", level="", **extra):
                    rec, infl = result
                    row = dict(cell=cell, split=split, cohort=cohort, contrast=name, outcome=outcome,
                               primary=name in PRIMARY and split == "all", feature=feature, level=level, **rec, **extra)
                    rows.append(row)
                    cache[name, outcome] = (row, infl)
                for outcome in METRICS:
                    emit(cohort+"_call", outcome, call_effect(g, outcome))
                    first = g.sort_values("step").drop_duplicates(EPKEY)
                    emit(cohort+"_first_root", outcome, call_effect(first, outcome))
                    for feature in feature_names:
                        valid = g[g[feature].notna()]
                        high, low = valid[valid[feature] == 1], valid[valid[feature] == 0]
                        sampling = feature in ("k4_high", "shadow_over_k4")
                        rh = call_effect(high, outcome, high.k4_sampling_weight if sampling else None)
                        rl = call_effect(low, outcome, low.k4_sampling_weight if sampling else None)
                        emit(f"{cohort}:{feature}:high", outcome, rh, feature, "high", missing=int(g[feature].isna().sum()))
                        emit(f"{cohort}:{feature}:low", outcome, rl, feature, "low", missing=int(g[feature].isna().sum()))
                        name = ("coverage_enrichment" if feature == "coverage_high" else "guard_enrichment" if feature == "at_guard" else
                                f"{cohort}:{feature}:enrichment") if cohort == "factorial" else f"{cohort}:{feature}:enrichment"
                        emit(name, outcome, interaction(rh, rl), feature, "high_minus_low")
                        if cohort == "factorial" and feature == "at_guard":
                            emit("factorial_at_guard", outcome, rh)
                            emit("factorial_pre_guard", outcome, rl)
                    if cohort == "factorial":
                        calls = g[bseries(g["assignment.coin_call"])]
                        emit("duration10_vs5", outcome, contrast(calls, calls["assignment.duration_choice"] == 10, calls["assignment.duration_probability"],
                             calls["assignment.duration_choice"] == 5, calls["assignment.duration_probability"], outcome))
                        emit("hold3_vs1", outcome, contrast(calls, calls["assignment.hold_choice"] == 3, calls["assignment.hold_probability"],
                             calls["assignment.hold_choice"] == 1, calls["assignment.hold_probability"], outcome))
                # Gain per incremental remaining call, with covariance, only a stable positive denominator.
                for name, outcome in list(cache):
                    if outcome != "Y" or (name, "future_misses") not in cache:
                        continue
                    ry, iy = cache[name, "Y"]
                    rm, im = cache[name, "future_misses"]
                    rec = dict(cell=cell, split=split, cohort=cohort, contrast=name, status="unstable_delta_M")
                    if rm.get("lo_95", np.nan) > 0:
                        value = ry["estimate"]/rm["estimate"]
                        iy, im = iy.align(im, fill_value=0)
                        inf = (iy-value*im)/rm["estimate"]
                        un = cluster_uncertainty(inf)
                        se, df = un["se"], un["df"]
                        c = student_t.ppf(.975, df)
                        rec.update(estimate=value, lo95=value-c*se, hi95=value+c*se, status="delta_method_positive_denominator")
                    ratios.append(rec)
                # Seed-block stability, unchanged excursion and no new pooled policy claim.
                for block, gb in g.groupby("block"):
                    emit(cohort+f"_block{block}", "Y", call_effect(gb, "Y"), block=int(block))
            win = asc[(asc.cohort == "window") & bseries(asc["assignment.scheduled_trigger"])].copy()
            require(not win.duplicated(EPKEY).any(), "window has multiple scheduled triggers")
            for x, y in ((0, 2), (0, 1), (1, 2)):
                for outcome in METRICS:
                    rec, _ = contrast(win, win["assignment.delay_choice"] == x, win["assignment.delay_probability"],
                                      win["assignment.delay_choice"] == y, win["assignment.delay_probability"], outcome)
                    name = f"window_delay{x}_vs{y}"
                    rows.append(dict(cell=cell, split=split, cohort="window", contrast=name, outcome=outcome, primary=name in PRIMARY and split == "all", **rec))
    return pd.DataFrame(rows), pd.DataFrame(ratios)


def episode_results(e):
    from exp.offline_search.rounds.r06.p3_profiling.pilot_stats import icc
    means, contrasts, planning = [], [], []
    comparisons = [("B", "A"), ("B", "P10"), ("factorial", "B"), ("window", "A"),
                   ("dose125", "B"), ("dose25", "B"), ("dose50", "B")]
    for cell, ec in e.groupby("cell"):
        for split, es in subsets(ec):
            for cohort, g in es.groupby("cohort"):
                for outcome in ("Y", "IR", "misses", "active_controls"):
                    rec, _ = estimate(g, g[outcome])
                    means.append(dict(cell=cell, split=split, cohort=cohort, outcome=outcome, **rec))
                rec, _ = estimate(g, g.modeled_cost, g.active_controls/5)
                means.append(dict(cell=cell, split=split, cohort=cohort, outcome="IR_pooled", **rec))
            for left, right in comparisons:
                l, r = es[es.cohort == left], es[es.cohort == right]
                if not len(l) or not len(r):
                    contrasts.append(dict(cell=cell, split=split, contrast=f"{left}-{right}", status="missing_cohort"))
                    continue
                require(set(map(tuple, l[PAIR].to_numpy())) == set(map(tuple, r[PAIR].to_numpy())), "unmatched cohort slots; no complete-case dropping")
                pair = l.merge(r[PAIR+["Y", "IR", "misses", "active_controls"]], on=PAIR, suffixes=("", "_ref"), validate="one_to_one")
                for outcome in ("Y", "IR", "misses", "active_controls"):
                    difference = pair[outcome]-pair[outcome+"_ref"]
                    rec, _ = estimate(pair, difference)
                    contrasts.append(dict(cell=cell, split=split, contrast=f"{left}-{right}", outcome=outcome, **rec))
                    if outcome == "Y":
                        pair["difference"] = difference
                        groups = list(pair.groupby(CLUSTER))
                        ir = icc([v.difference.to_numpy() for _, v in groups], [k[0] for k, _ in groups])
                        precision = dict(cell=cell, split=split, contrast=f"{left}-{right}", paired_difference_icc=ir,
                                         n_current=len(pair), fixed_task_se=rec["se"], target="pointwise SR halfwidth .05")
                        if rec["se"] > 0:
                            precision["scaled_total_episodes_per_cohort"] = math.ceil(len(pair)*(1.96*rec["se"]/.05)**2)
                        planning.append(precision)
                rl = estimate(l, l.modeled_cost, l.active_controls/5)
                rr = estimate(r, r.modeled_cost, r.active_controls/5)
                rec, _ = interaction(rl, rr)
                contrasts.append(dict(cell=cell, split=split, contrast=f"{left}-{right}", outcome="IR_pooled", **rec))
    contrasts = pd.DataFrame(contrasts)
    if "outcome" not in contrasts:
        contrasts["outcome"] = np.nan
    return pd.DataFrame(means), contrasts, planning


def support_tables(a, e):
    support = a.groupby(["cell", "cohort", "split", "assignment.override", "assignment.nominal_propensity", "assignment.actual_propensity"], dropna=False).agg(
        anchors=("step", "size"), episodes=("uid", "nunique"), calls=("assignment.executed_policy", "sum")).reset_index()
    missing = []
    featurecols = GATES+["coverage", "shadow", "k4", "phase", "diagnostic_guard", "long_since_call", "shadow_high", "k4_high"]+["p_"+g for g in GUARDS]
    for (cell, cohort), g in a.groupby(["cell", "cohort"]):
        for feat in featurecols:
            x = g[feat]
            missing.append(dict(cell=cell, cohort=cohort, feature=feat, anchors=len(g), finite=int(x.notna().sum()),
                                missing=int(x.isna().sum()), min=x.min(), max=x.max(), positive=int((x > 0).sum())))
    window = []
    for cell, g in a[a.cohort == "window"].groupby("cell"):
        for split, gs in subsets(g):
            es = e[(e.cell == cell) & (e.cohort == "window")]
            if split != "all":
                es = es[es.split == split] if split != "continuation_validation" else es[(es.split == "validation") & (es.init >= 2)]
            triggers = gs[bseries(gs["assignment.scheduled_trigger"])]
            for delay in (0, 1, 2):
                chosen = triggers[triggers["assignment.delay_choice"] == delay]
                executed = chosen.future_misses > 0
                window.append(dict(cell=cell, split=split, delay=delay, episodes=len(es), triggers_all=len(triggers),
                                   no_trigger=len(es)-len(triggers), assigned=len(chosen), executed=int(executed.sum()),
                                   terminal_before_call=int((~executed).sum())))
    return support, pd.DataFrame(missing), pd.DataFrame(window)


def nominations(effects, smoke):
    rows = []
    for cell in effects.cell.unique():
        options = []
        for gate in GATES:
            names = [f"factorial:{gate}:high", "coverage_enrichment" if gate == "coverage_high" else f"factorial:{gate}:enrichment"]
            pair = [effects[(effects.cell == cell) & (effects.split == "calibration") & (effects.outcome == "Y") & (effects.contrast == name)] for name in names]
            supported = all(len(g) == 1 and bool(g.iloc[0].get("nomination_support", False)) and g.iloc[0].get("lo_selection", np.nan) > 0 for g in pair)
            enrichment = float(pair[1].iloc[0].estimate) if len(pair[1]) else np.nan
            rows.append(dict(cell=cell, gate=gate, support_and_positive_simultaneous_bounds=supported, enrichment=enrichment,
                             chosen=False, status="SMOKE_PLUMBING_ONLY" if smoke else "requires_prospective_SR_IR_validation"))
            if supported and not smoke:
                options.append((enrichment, gate))
        if options:
            winner = max(options, key=lambda x: (x[0], -GATES.index(x[1])))[1]
            for row in rows:
                if row["cell"] == cell and row["gate"] == winner:
                    row["chosen"] = True
    return pd.DataFrame(rows)


def calibration_resolution(a, directory):
    """Read only a hash-matched library artifact; never infer n from observed min p."""
    rows = []
    for cell, g in a.groupby("cell"):
        path = Path(directory) / (cell+".json")
        hashes = set(g["calibration.calibration_sha256"].dropna())
        matched = path.is_file() and hashes == {sha(path)}
        reference = json.loads(path.read_text()) if matched else None
        for task, gt in g.groupby("task_id"):
            for test in GUARDS:
                values = reference.get("per_task", {}).get(str(task), {}).get(test, []) if reference else []
                n = len(values)
                rows.append(dict(cell=cell, task_id=int(task), test=test, artifact=str(path), hash_matched=matched,
                                 n=n if matched else None, minimum_attainable_p=1/(n+1) if n else None,
                                 alpha05_attainable=bool(n and 1/(n+1) <= .05/6) if matched else None,
                                 alpha10_attainable=bool(n and 1/(n+1) <= .10/6) if matched else None,
                                 alpha20_attainable=bool(n and 1/(n+1) <= .20/6) if matched else None,
                                 observed_min_p=gt["p_"+test].min()))
    return pd.DataFrame(rows)


def dimension_screen(effects, comparisons, nomination, smoke):
    """Frozen primary screens; no claim that a combined policy has been evaluated."""
    records = []
    for cell in effects.cell.unique():
        f = effects[(effects.cell == cell) & (effects.split == "all")]
        def one(name, outcome="Y"):
            z = f[(f.contrast == name) & (f.outcome == outcome)]
            return z.iloc[0].to_dict() if len(z) == 1 else {}
        dy, dc = one("duration10_vs5"), one("duration10_vs5", "future_cost")
        choose_short = bool(dy.get("nomination_support", False) and dc.get("nomination_support", False)
                            and dy.get("hi_primary", np.inf) < .01 and dc.get("lo_primary", -np.inf) > 0)
        h = one("hold3_vs1")
        c = comparisons[(comparisons.cell == cell) & (comparisons.split == "all") &
                        (comparisons.contrast == "factorial-B") & (comparisons.outcome == "IR_pooled")]
        choose_hold = bool(h.get("nomination_support", False) and h.get("lo_primary", -np.inf) > 0 and
                           len(c) == 1 and c.iloc[0].get("hi_95", np.inf) <= 0)
        delayed, before, at = one("window_delay0_vs2"), one("factorial_pre_guard"), one("factorial_at_guard")
        chosen = nomination[(nomination.cell == cell) & nomination.chosen]
        records.append(dict(cell=cell, smoke=smoke, nominated_gate=chosen.gate.tolist(),
                            starting_controls_proposal=5 if choose_short and not smoke else 10,
                            hold_proposal=3 if choose_hold and not smoke else 1, cooldown_anchors=1,
                            earliest_delay_supported=bool(delayed.get("nomination_support", False) and delayed.get("lo_primary", -np.inf) > 0) and not smoke,
                            pre_guard_benefit_supported=bool(before.get("nomination_support", False) and before.get("lo_primary", -np.inf) > 0) and not smoke,
                            at_guard_benefit_supported=bool(at.get("nomination_support", False) and at.get("lo_primary", -np.inf) > 0) and not smoke,
                            deployment_status="NOT_VALIDATED; q budget replay and new matched deployment comparison required"))
    return records


def score_precision(effects):
    rows = []
    for _, r in effects[(effects.primary == True) & (effects.outcome == "Y")].iterrows():
        row = dict(cell=r.cell, contrast=r.contrast, clusters=r.get("clusters"), se=r.get("se"),
                   cluster_ess_a=r.get("cluster_ess_a"), cluster_ess_b=r.get("cluster_ess_b"),
                   status=r.status, halfwidth95=(r.get("hi_95", np.nan)-r.get("lo_95", np.nan))/2,
                   planning_assumption="independent new init bundles with the same repeats, risk-set mix and score variance")
        if r.status == "estimated" and r.get("clusters", 0) > 0:
            row["bundles_for_pointwise_halfwidth05"] = math.ceil(r.clusters*(1.96*r.se/.05)**2)
            row["bundles_for_MDE05_power80"] = math.ceil(r.clusters*(2.801621*r.se/.05)**2)
        rows.append(row)
    return rows


def scheduled_precision():
    rows = []
    for stage in ("pilot", "full", "continuation"):
        repeats = {}
        for block in (0, 1, 2):
            for key in expected_slots(block, stage):
                repeats[key] = repeats.get(key, 0)+1
        for split in ("all", "calibration", "validation"):
            r = np.array([n for (task, init), n in repeats.items() if split == "all" or
                          (init % 5 == 0 if split == "calibration" else init % 5 != 0)])
            n, k = int(r.sum()), len(r)
            deff = 1+.3*float((r*(r-1)).sum())/n
            neff = n/deff
            rows.append(dict(stage=stage, split=split, episodes=n, init_clusters=k, rho_assumed=.3,
                             n_eff=neff, paired_halfwidth95=1.96*math.sqrt(.2/neff),
                             paired_mde80=2.801621*math.sqrt(.2/neff), fixed_task_cluster_df=k-10))
    return rows


def plots(effects, means, out, smoke):
    os.environ["MPLCONFIGDIR"] = str(HERE / ".mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    label = "SMOKE — plumbing only; no scientific conclusions" if smoke else "Q3 pilot — pointwise 95% intervals; see simultaneous tables"
    f = effects[(effects.split == "all") & (effects.outcome == "Y") & effects.primary.fillna(False)].copy()
    f = f[np.isfinite(f.estimate) & (f.status != "point_only_no_realized_comparison")]
    fig, ax = plt.subplots(figsize=(10, max(4, len(f)*.32)))
    y = np.arange(len(f))
    for j, (_, r) in enumerate(f.iterrows()):
        ax.plot(r.estimate, j, "o", color="C0")
        if np.isfinite(r.get("lo_95", np.nan)):
            ax.plot([r.lo_95, r.hi_95], [j, j], color="C0")
    ax.set_yticks(y, [f"{r.cell}: {r.contrast}" for _, r in f.iterrows()], fontsize=8)
    ax.axvline(0, color="0.5", linewidth=1)
    ax.set_xlabel("SR excursion difference (not a whole-policy SR prediction)")
    ax.set_title(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "primary_effects.png", dpi=150)
    plt.close(fig)
    m = means[means.split == "all"].pivot(index=["cell", "cohort"], columns="outcome", values="estimate").reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for cell, g in m.groupby("cell"):
        ax.scatter(g.IR_pooled, g.Y, label=cell)
        for _, r in g.iterrows():
            ax.annotate(r.cohort, (r.IR_pooled, r.Y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.set(xlabel="Pooled deployment IR / actual five controls", ylabel="SR", title=label)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "observed_arms.png", dpi=150)
    plt.close(fig)


def obtain_tables(args, out):
    arms = args.arms
    if args.arms_file:
        require(not arms, "use either --arms or --arms-file")
        arms = Path(args.arms_file).read_text().split()
    if args.tables:
        return args.tables, arms
    require(args.run_root is not None, "supply --tables or --run-root")
    require(bool(arms) != bool(args.cell), "run-root requires exactly one of --arms/--arms-file or --cell")
    if args.cell:
        specifications = json.loads((P3 / "arms_v2.json").read_text())
        arms = [r["name"] for r in specifications if f"_{args.cell}_" in r["name"]]
        require(len(arms) == 27, "cell must match one of eight P3 v2 cells")
    root = Path(args.run_root).resolve()
    # One read of each requested journal. Incomplete manifests fail, never wait/poll.
    for arm in arms:
        journal = root / "runs" / arm / "client/journal.jsonl"
        accepted = {}
        with journal.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
                    require(r["task_uid"] not in accepted or accepted[r["task_uid"]] == r, "conflicting accepted attempts")
                    accepted[r["task_uid"]] = r
        block = int(arm.rsplit("_r", 1)[1])
        require(len(accepted) == len(expected_slots(block, args.stage)), f"{arm}: unfinished/unexpected {args.stage} manifest; refusing extraction")
    destination = out / "tables"
    command = [sys.executable, "-m", "exp.offline_search.rounds.r06.p3_profiling.read_v2", "--run-root", str(root),
               "--arms", *arms, "--client-root", str(root / "runs"), "--require-stage-counts", "--require-snapshots", "--out", str(destination)]
    # Explicit affinity even for the only child process. read_v2 does no simulation.
    command = ["taskset", "-c", "22-25,66-69", *command]
    dump(out / "reader_command.json", command)
    with (out / "reader.log").open("w") as log:
        subprocess.run(command, cwd=REPO, env=os.environ.copy(), stdout=log, stderr=subprocess.STDOUT, check=True)
    return [destination], arms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--tables", nargs="+", type=Path)
    mode.add_argument("--run-root", type=Path)
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--arms-file", type=Path)
    parser.add_argument("--cell", help="e.g. pi05_l10_50; selects all 27 manifest arms")
    parser.add_argument("--calibration-dir", type=Path, help="hash-matched P3 library calibration artifacts; otherwise unavailable")
    parser.add_argument("--stage", choices=["pilot", "continuation", "full"], default="pilot")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        require(os.environ.get(name) == "1", f"set {name}=1")
    require(os.environ.get("CUDA_VISIBLE_DEVICES") == "", "CUDA must be disabled")
    require(set(os.sched_getaffinity(0)) <= {22, 23, 24, 25, 66, 67, 68, 69}, "CPU affinity outside Q3 allocation")
    require(not args.smoke or args.allow_partial, "smoke requires --allow-partial")
    out = safe_out(args.out)
    out.mkdir(parents=True, exist_ok=False)
    directories, arms = obtain_tables(args, out)
    tables, sources = read_tables(directories, arms)
    if args.cell and args.tables:
        for key in tables:
            tables[key] = tables[key][tables[key].arm.str.contains("_"+args.cell+"_", regex=False)]
    if arms:
        require(set(tables["episodes"].arm) == set(arms), "requested arms missing from tables")
    a, e, d, audit = validate(tables, args)
    a, references = features(a, e)
    effects, ratios = randomized(a)
    means, comparisons, planning = episode_results(e)
    support, missing, window = support_tables(a, e)
    nomination = nominations(effects, args.smoke)
    calibration_dir = args.calibration_dir or (Path(args.run_root)/"calibration" if args.run_root else P3/"calibration_v2")
    resolution = calibration_resolution(a, calibration_dir)
    for name, frame in (("effects", effects), ("gain_per_call", ratios), ("episode_means", means),
                        ("episode_contrasts", comparisons), ("support", support), ("feature_missingness", missing),
                        ("window_reach", window), ("nominations", nomination), ("calibration_resolution", resolution)):
        frame["smoke"] = args.smoke
        frame.to_csv(out / f"{name}.csv", index=False)
    featurecols = list(dict.fromkeys(EPKEY+["cell", "cohort", "block", "split", "task_id", "init", "step"]+GATES+
                                    ["coverage", "neighbour", "shadow", "k4", "phase", "at_guard", "diagnostic_guard", "previous_guard", "long_since_call"]))
    a[featurecols].to_csv(out / "anchor_features.csv", index=False)
    dump(out / "feature_references.json", references)
    precision = dict(paired_variance_assumption=.20,
                     effective_pairs_halfwidth_05=math.ceil(1.96**2*.2/.05**2),
                     effective_pairs_halfwidth_01=math.ceil(1.96**2*.2/.01**2),
                     effective_pairs_NI_01_power80=math.ceil((1.644854+.841621)**2*.2/.01**2),
                     observed_paired_planning=planning,
                     observed_primary_cluster_score_planning=score_precision(effects),
                     scheduled_split_planning=scheduled_precision(),
                     warning="SE scaling assumes stable visitation, discordance and cluster structure; not an enrollment guarantee")
    dump(out / "precision.json", precision)
    dump(out / "method_screen.json", dimension_screen(effects, comparisons, nomination, args.smoke))
    audit.update(sources=sources, analysis_sha256=sha(__file__), preregistered_sha256=sha(HERE/"PREREG_FROZEN.md"),
                 estimate_rows=len(effects), episode_contrast_rows=len(comparisons),
                 eligible_nominations=int(nomination.chosen.sum()),
                 validation_boundary="table mode checks joins, assignment replay and accounting; trusts upstream read_v2 for raw input/snapshot verification",
                 scientific_use="PROHIBITED: smoke fixture" if args.smoke else "preregistered pilot screening, no deployment safety certification")
    dump(out / "audit.json", audit)
    plots(effects, means, out, args.smoke)
    (out / "README.md").write_text(
        ("# SMOKE: plumbing only\n\nNo scientific conclusions may be drawn.\n" if args.smoke else "# Q3 analysis output\n")+
        "\nSee ../PREREG.md for estimands, continuation limits, calibration and multiplicity. "
        "effects.csv contains HT excursion/package contrasts, not whole-policy SR predictions. "
        "lo_primary/hi_primary reserve 64 comparisons; lo_selection/hi_selection reserve48. "
        "Empty intervals indicate insufficient fixed-task init replication or zero variance. "
        "Task intervals target between-task variation and are secondary. Missing cohorts and feature references stay explicit.\n")
    print(json.dumps({k: audit[k] for k in ("episodes", "anchors", "decisions", "arms", "assignments_replayed", "estimate_rows", "smoke", "eligible_nominations")}))


if __name__ == "__main__":
    main()
