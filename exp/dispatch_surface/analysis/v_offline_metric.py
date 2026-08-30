"""Offline mechanism check for ``v``: out-of-fold risk prediction, SV vs S0
(confirmation plan 3.5, G1R1-B5 / G1R2-B5). Descriptive only; no gate.

Re-uses the archived SV fit record's development membership, fold map and
quantile level, refits fold-local surfaces with the frozen grid ladders
(``GRID_LADDER_SV`` for SV, ``GRID_LADDER_S_ONLY`` for S0 -- one v bin), and
scores the held-out fold rows. Two parallel readouts (horizon 7 = ``y7``,
horizon 10 = ``y10``), each: pinball loss at the fitted quantile level,
row -> episode mean -> task mean -> global mean; SV - S0 with a paired
episode bootstrap (task-stratified, R = 10000, PCG64(20260831)); empirical
coverage P(y <= q_hat) and sharpness mean(q_hat) with the same bootstrap.
Every quantity is labelled ``oof``; nothing here is a held-out coverage
certificate and the phrase is deliberately absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

from exp.dispatch_surface import rev1_package as pkgmod
from exp.dispatch_surface.fit_surface import (
    GRID_LADDER_S_ONLY,
    GRID_LADDER_SV,
    _digest_obj,
    assign_folds,
    fit_fold_models,
    load_table,
    oof_predictions,
)

PROTOCOL = "dispatch_surface_rev2_v_offline_metric"
R = 10000
SEED = 20260831
LABEL = "oof"
FORBIDDEN_PHRASE = "held-out coverage"


def pinball(y: np.ndarray, q: np.ndarray, tau: float) -> np.ndarray:
    diff = y - q
    return tau * np.maximum(diff, 0.0) + (1.0 - tau) * np.maximum(-diff, 0.0)


def _aggregate(values_by_episode: dict[str, float], task_by_episode: dict[str, int]) -> float:
    """episode mean (already) -> task mean over episodes -> mean over tasks."""
    by_task: dict[int, list[float]] = {}
    for ep, v in values_by_episode.items():
        by_task.setdefault(task_by_episode[ep], []).append(v)
    return float(np.mean([np.mean(v) for _t, v in sorted(by_task.items())]))


def episode_means(rows_value: np.ndarray, episodes: np.ndarray) -> dict[str, float]:
    out: dict[str, list[float]] = {}
    for v, ep in zip(rows_value, episodes):
        out.setdefault(str(ep), []).append(float(v))
    return {ep: float(np.mean(v)) for ep, v in out.items()}


def bootstrap_diff(ep_a: dict[str, float], ep_b: dict[str, float], task_by_episode: dict[str, int],
                   *, seed: int = SEED, reps: int = R) -> dict:
    eps = sorted(ep_a)
    by_task: dict[int, list[str]] = {}
    for ep in eps:
        by_task.setdefault(task_by_episode[ep], []).append(ep)
    rng = np.random.Generator(np.random.PCG64(seed))
    vals = np.empty(reps)
    for r in range(reps):
        acc_a: dict[str, float] = {}
        acc_b: dict[str, float] = {}
        tb: dict[str, int] = {}
        for t, lst in sorted(by_task.items()):
            idx = rng.integers(0, len(lst), len(lst))
            for j, i in enumerate(idx):
                key = f"{lst[i]}#{j}"
                acc_a[key] = ep_a[lst[i]]
                acc_b[key] = ep_b[lst[i]]
                tb[key] = t
        vals[r] = _aggregate(acc_a, tb) - _aggregate(acc_b, tb)
    return {"mean": float(vals.mean()), "q05": float(np.quantile(vals, 0.05)), "q95": float(np.quantile(vals, 0.95)),
            "R": reps, "seed": seed, "unit": "episode (task-stratified, paired)"}


def compute(package_manifest: str, table_path: str) -> dict:
    manifest, pkg, manifest_sha = pkgmod.load_manifest(package_manifest)
    pkgmod.verify_package(package_manifest)
    rec = pkgmod.load_json_member(manifest, pkg, "fit.sv")
    table_sha = hashlib.sha256(pathlib.Path(table_path).read_bytes()).hexdigest()
    if table_sha != (rec.get("input_digests") or {}).get("table"):
        raise SystemExit("--table is not the table the archived SV fit record binds")
    table = load_table(table_path, ref_mode="fresh")
    dev_eps = {str(m[0]) for m in rec["dev_membership"]}
    dev_mask = np.array([str(e) in dev_eps for e in table.episode])
    if int(dev_mask.sum()) == 0:
        raise SystemExit("no development rows found for the archived membership")
    folds = assign_folds(table, dev_mask)
    recorded = {str(ep): int(f) for ep, f in rec["fold_map"]}
    computed = {}
    for ep in np.unique(table.episode[dev_mask]):
        fs = np.unique(folds[dev_mask & (table.episode == ep)])
        if len(fs) != 1:
            raise SystemExit(f"episode {ep} spans several folds")
        computed[str(ep)] = int(fs[0])
    if computed != recorded:
        raise SystemExit("recomputed fold assignment != the archived fold map; refusing")
    if _digest_obj([[ep, f] for ep, f in rec["fold_map"]]) != rec.get("fold_map_sha256"):
        raise SystemExit("archived fold map digest does not match its own list")
    miscoverage = float(rec["quantile_alpha"])
    tau = 1.0 - miscoverage
    models_sv = fit_fold_models(table, dev_mask, folds, GRID_LADDER_SV, miscoverage)
    models_s0 = fit_fold_models(table, dev_mask, folds, GRID_LADDER_S_ONLY, miscoverage)
    if models_sv is None or models_s0 is None:
        raise SystemExit("fold-local grid descent exhausted the ladder (stop-loss); no OOF surface")
    for m in models_sv + models_s0:
        # fold-local training must exclude the held-out rows
        if m.heldout_local.sum() == 0 or (~m.heldout_local).sum() == 0:
            raise SystemExit("degenerate fold split")
    q_sv7, q_sv10 = oof_predictions(models_sv, table, dev_mask)
    q_s07, q_s010 = oof_predictions(models_s0, table, dev_mask)
    dev_idx = np.where(dev_mask)[0]
    episodes = table.episode[dev_idx]
    task_by_episode = {str(e): int(t) for e, t in zip(episodes, table.task[dev_idx])}
    readouts = {}
    for horizon, y, q_sv, q_s0 in (("h7", table.y7[dev_idx], q_sv7, q_s07), ("h10", table.y10[dev_idx], q_sv10, q_s010)):
        block = {"label": LABEL, "quantile_level": tau, "miscoverage_alpha": miscoverage}
        for metric, fa, fb in (("pinball", pinball(y, q_sv, tau), pinball(y, q_s0, tau)),
                               ("coverage", (y <= q_sv).astype(float), (y <= q_s0).astype(float)),
                               ("sharpness_mean_qhat", q_sv, q_s0)):
            ea, eb = episode_means(fa, episodes), episode_means(fb, episodes)
            block[metric] = {"sv": _aggregate(ea, task_by_episode), "s0": _aggregate(eb, task_by_episode),
                             "sv_minus_s0": bootstrap_diff(ea, eb, task_by_episode),
                             "direction": {"pinball": "negative = SV better", "coverage": "closer to quantile_level = better calibrated",
                                           "sharpness_mean_qhat": "smaller = sharper (only comparable at similar coverage)"}[metric]}
        block["per_task_pinball"] = {}
        for t in sorted(set(task_by_episode.values())):
            eps_t = [e for e, tt in task_by_episode.items() if tt == t]
            ea, eb = episode_means(pinball(y, q_sv, tau), episodes), episode_means(pinball(y, q_s0, tau), episodes)
            block["per_task_pinball"][str(t)] = {"sv": float(np.mean([ea[e] for e in eps_t])), "s0": float(np.mean([eb[e] for e in eps_t]))}
        readouts[horizon] = block
    out = {"protocol": PROTOCOL, "descriptive": True, "gating": False, "label": LABEL,
           "suite": manifest["suite"], "rev1_package_manifest_sha256": manifest_sha, "table_sha256": table_sha,
           "fit_record_sha256": pkgmod.member_sha(manifest, "fit.sv"), "fold_map_sha256": rec.get("fold_map_sha256"),
           "n_dev_rows": int(dev_mask.sum()), "n_dev_episodes": len(dev_eps),
           "ladders": {"sv": [list(x) for x in GRID_LADDER_SV], "s0": [list(x) for x in GRID_LADDER_S_ONLY]},
           "readouts": readouts,
           "note": "out-of-fold (5-fold by init within task) risk prediction; answers whether v improves risk prediction; "
                   "does not replace closed-loop H1 and does not restore H2 gating"}
    if FORBIDDEN_PHRASE in json.dumps(out):
        raise SystemExit("output must not claim a held-out coverage certificate")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rev1-package-manifest", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = compute(args.rev1_package_manifest, args.table)
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    print(json.dumps({h: {"pinball_sv_minus_s0": b["pinball"]["sv_minus_s0"]} for h, b in out["readouts"].items()}, indent=2))


if __name__ == "__main__":
    main()
