"""Served-action vs policy-shadow gap: per decision, per episode, per task.

The deferred ``policy_shadow`` gives, at every decision of every R8 arm, the chunk
the policy would have produced from the very observation the controller saw.
This module turns that into

* per-decision gap metrics on the executed block (5 controls x 6 motion dims in
  library-sigma units, and gripper-sign disagreement over the 5 executed
  controls), plus a noise floor from the independent ``policy_draws``;
* per-episode summaries (gap over the first K decisions / first K looks, over
  all decisions) that can be joined to the paired closed-loop outcomes;
* per-task means on discovery inits, to test whether a label-free calibration
  signal ranks tasks by how much the cache loses against the policy.

Nothing here is a success estimate: a large gap says the cache disagrees with the
policy, not that the episode fails. The value of the gap as a predictor is what
``analyze`` measures (AUROC against the real outcome of the same episode).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from exp.offline_search.debug import reader

from . import common


def library_sigma(model, suite, store=common.STORE):
    act = np.load(Path(store) / "library" / f"{model}_{suite}" / "current" / "action.npy", mmap_mode="r")
    return np.asarray(act[:, :5, :7]).std(axis=(0, 1)).astype(np.float32)


def gap_metrics(a, b, sigma):
    """Executed-block metrics between two (n, >=5, >=7) chunks: motion RMS (sigma units) and gripper disagreement."""
    a, b = np.asarray(a, np.float32), np.asarray(b, np.float32)
    motion = np.sqrt(np.mean(((a[:, :5, :6] - b[:, :5, :6]) / sigma[:6]) ** 2, axis=(1, 2)))
    grip = np.mean(np.sign(a[:, :5, 6]) != np.sign(b[:, :5, 6]), axis=1)
    commit = np.sqrt(np.mean(((a[:, :10, :6] - b[:, :10, :6]) / sigma[:6]) ** 2, axis=(1, 2)))
    return motion, grip, commit


def auroc(score, label):
    """Rank-based AUROC (ties averaged); label 1 = positive."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(score)
    score, label = score[ok], label[ok]
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(score).rank(method="average").values
    return float((ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_arm(arm, derived=common.DERIVED):
    dec = pd.read_parquet(Path(derived) / "decisions" / f"{arm}.parquet")
    arr = np.load(Path(derived) / "arrays" / f"{arm}.npz")
    if not (arr["decision_id"] == dec.decision_id.values).all():
        raise ValueError("decision order mismatch for %s" % arm)
    return dec, arr


def noise_floor(arm, sigma, run_root=common.RUN_ROOT, derived=common.DERIVED):
    """Pairwise metrics among the 3 independent draws (+ the shadow) on the 1/32 sampled decisions."""
    data = reader.open_arm(run_root, arm)
    data.cache_enabled = False
    draws = data.aug("policy_draws")
    ids = draws["decision_id"].astype(str)
    dec, arr = load_arm(arm, derived)
    pos = pd.Series(np.arange(len(dec)), index=dec.decision_id.values).reindex(ids)
    keep = pos.notna().values
    pos = pos[keep].astype(int).values
    chunks = np.concatenate([draws["chunks"][keep][:, :, :10, :7], arr["shadow"][pos][:, None]], axis=1)  # (n, 4, 10, 7)
    m, g, c = [], [], []
    for i in range(chunks.shape[1]):
        for j in range(i + 1, chunks.shape[1]):
            mi, gi, ci = gap_metrics(chunks[:, i], chunks[:, j], sigma)
            m.append(mi), g.append(gi), c.append(ci)
    return dict(n=int(len(pos)), motion=float(np.mean(m)), grip=float(np.mean(g)), commit=float(np.mean(c)),
                motion_p50=float(np.median(m)), grip_p50=float(np.median(g)))


def per_decision(arm, derived=common.DERIVED):
    model, suite, lib, variant = common.parse_arm(arm)
    sigma = library_sigma(model, suite)
    dec, arr = load_arm(arm, derived)
    motion, grip, commit = gap_metrics(arr["served"], arr["shadow"], sigma)
    out = dec[["decision_id", "episode_key", "task_id", "init", "decision_seq", "is_look", "is_call", "src", "x_d1", "x_d1_rel",
               "x_disp5", "x_dst", "x_w_eff", "conf", "journal_success"]].copy()
    out["gap_motion"], out["gap_grip"], out["gap_commit"] = motion, grip, commit
    out["shadow_ok"] = np.isfinite(motion)
    return out, sigma


def per_episode(table, ks=(1, 2, 3, 5, 10)):
    """Episode summaries of the gap: first-K decisions, first-K looks, all; plus step-0 retrieval scalars."""
    rows = []
    for ek, g in table.groupby("episode_key", sort=False):
        g = g.sort_values("decision_seq")
        looks = g[g.is_look]
        row = dict(episode_key=ek, task_id=int(g.task_id.iloc[0]), init=int(g["init"].iloc[0]),
                   success=bool(g.journal_success.iloc[0]), n_dec=int(len(g)),
                   gap_all=float(g.gap_motion.mean()), grip_all=float(g.gap_grip.mean()),
                   gap_looks=float(looks.gap_motion.mean()) if len(looks) else np.nan,
                   d1_0=float(g.x_d1.iloc[0]) if np.isfinite(g.x_d1.iloc[0]) else np.nan,
                   conf_0=float(g.conf.iloc[0]), dst_0=float(g.x_dst.iloc[0]), disp5_0=float(g.x_disp5.iloc[0]),
                   d1_mean_looks=float(looks.x_d1.mean()) if len(looks) else np.nan,
                   conf_mean_looks=float(looks.conf.mean()) if len(looks) else np.nan)
        for k in ks:
            row[f"gap_first{k}"] = float(g.gap_motion.iloc[:k].mean())
            row[f"grip_first{k}"] = float(g.gap_grip.iloc[:k].mean())
            row[f"gaplook_first{k}"] = float(looks.gap_motion.iloc[:k].mean()) if len(looks) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def analyze_arm(arm, out_dir, derived=common.DERIVED, with_noise=True):
    table, sigma = per_decision(arm, derived)
    epi = per_episode(table)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_parquet(out_dir / f"decisions_{arm}.parquet", index=False)
    epi.to_parquet(out_dir / f"episodes_{arm}.parquet", index=False)
    disc = epi[common.discovery_mask(epi["init"])]
    fail = ~disc.success.values
    summary = dict(arm=arm, sigma=sigma.tolist(), n_decisions=int(len(table)), shadow_ok=float(table.shadow_ok.mean()),
                   gap_motion_look=float(table.gap_motion[table.is_look].mean()),
                   gap_motion_blind=float(table.gap_motion[~table.is_look].mean()),
                   gap_grip_look=float(table.gap_grip[table.is_look].mean()),
                   gap_grip_blind=float(table.gap_grip[~table.is_look].mean()),
                   gap_commit_look=float(table.gap_commit[table.is_look].mean()),
                   fail_rate_discovery=float(fail.mean()),
                   auroc_fail={c: auroc(disc[c].values if not c.startswith("neg_") else -disc[c[4:]].values, fail)
                               for c in ["gap_first1", "gap_first2", "gap_first3", "gap_first5", "gap_first10", "gap_all",
                                         "grip_first5", "grip_all", "gaplook_first1", "gaplook_first3", "gaplook_first5",
                                         "d1_0", "neg_conf_0", "dst_0", "disp5_0", "d1_mean_looks", "neg_conf_mean_looks"]})
    if with_noise:
        try:
            summary["noise_floor"] = noise_floor(arm, sigma)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            summary["noise_floor"] = dict(error=repr(exc)[:200])
    per_task = disc.groupby("task_id").agg(gap_all=("gap_all", "mean"), gap_first5=("gap_first5", "mean"),
                                           grip_all=("grip_all", "mean"), sr=("success", "mean"), n=("success", "size"))
    summary["per_task_discovery"] = per_task.reset_index().to_dict("records")
    common.write_json(out_dir / f"summary_{arm}.json", summary)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("arms", nargs="*")
    ap.add_argument("--out", default=str(common.OUT / "shadow"))
    ap.add_argument("--no-noise", action="store_true")
    a = ap.parse_args(argv)
    arms = a.arms or [common.arm_name(m, s, l, "A") for m, s, l in common.CELLS]
    for arm in arms:
        s = analyze_arm(arm, a.out, with_noise=not a.no_noise)
        print(json.dumps({k: v for k, v in s.items() if k not in ("per_task_discovery", "sigma")}, default=float))
    return 0


if __name__ == "__main__":
    main()
