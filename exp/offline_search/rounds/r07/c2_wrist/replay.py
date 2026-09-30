"""Strict recorded-input replay: disabled identity and enabled camera routing.

No outcome tuning or counterfactual success claim. Histories use the recorded
executed chunks and states. The methods choose blind versus vision online.
"""
import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from exp.offline_search.closed_loop.blind import BlindQueryView, BlindResult, LookReason
from exp.offline_search.closed_loop.plugin import clone_method
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import load_base, sources
from .method import StageWrist


def equal(left, right):
    if isinstance(left, LookReason):
        assert left == right
        return
    assert type(left) is type(right)
    for key in (["action", "rows", "weights"] if isinstance(left, BlindResult) else ["action", "topk", "scores"]):
        assert np.array_equal(getattr(left, key), getattr(right, key), equal_nan=True), key
    assert left.library == right.library
    if not isinstance(left, BlindResult):
        assert left.confidence == right.confidence
    assert left.extras.keys() == right.extras.keys()
    for key in left.extras:
        assert np.array_equal(np.asarray(left.extras[key]), np.asarray(right.extras[key]), equal_nan=True), key


def episodes(dataset, label):
    root = Path("/home/weiland/trace_runs/os_closed_loop") / dataset / "tables"
    directory = next(p for name in ("pi05_" + label, "pi05_" + label.replace("sp_", "spatial_"))
                     if (p := root / name / "decisions.csv").exists()).parent
    groups = defaultdict(list)
    with (directory / "decisions.csv").open() as f:
        for row in csv.DictReader(f):
            if dataset == "r06_p3_pilot" and not row["arm"].endswith("_A_r0"):
                continue
            groups[(row["uid"], row["attempt"])].append(row)
    for group in groups.values():
        group.sort(key=lambda row: int(row["step"]))
        assert [int(row["step"]) for row in group] == list(range(len(group)))
        yield group


def run(label, dataset, fits):
    source = sources()["pi05_" + label.replace("sp_", "spatial_")]
    baseline, _ = load_base(source)
    off = StageWrist(enabled=False)
    off.base, _ = clone_method(baseline, strict=True)
    enabled = {}
    for variant in ("sw", "sf_sw"):
        with (fits / f"r7_{variant}_pi05_{label}.pkl").open("rb") as f:
            enabled[variant] = pickle.load(f)["method"]
    counts = dict(cell="pi05_"+label, dataset=dataset, episodes=0, decisions=0, identical_actions=0,
                  identical_verdicts=0, identical_vision=0, identical_extras=0)
    routing = {key: dict(full=0, wrist=0, blind=0, wrist_metric_exact=0, proposed_cost=0.) for key in enabled}
    for group in episodes(dataset, label):
        ep = SimpleNamespace(uid=group[0]["uid"], task_id=int(group[0]["task_id"]), init=int(group[0].get("init", 0)), seed=0)
        baseline.reset(ep); off.reset(ep)
        for method in enabled.values():
            method.reset(ep)
        states, raw, executed, hit, k0, k1 = [], [], [], [], [], []
        visions, age = [], 0
        variants = {key: {"visions": [], "age": 0, "k0": [], "k1": []} for key in enabled}
        for step, row in enumerate(group):
            with np.load(row["absolute_input_archive"]) as archive:
                z = {key: archive[key].copy() for key in archive.files}
            hist_a = np.asarray(executed, np.float32).reshape((-1, *z["executed_chunk"].shape))
            hist_rs = np.asarray(states, np.float32).reshape((-1, len(z["robot_state"])))
            prev = bool(hit[-1]) if hit else None
            q = SimpleNamespace(step=step, task_id=ep.task_id, episode=ep, rs=z["robot_state"], raw_state=z["raw_state"],
                key_v0=z["vision_0"], key_v1=z["vision_1"],
                hist_key_v0=np.asarray(k0), hist_key_v1=np.asarray(k1), hist_rs=hist_rs,
                hist_a_exec=hist_a, prev_a_exec=executed[-1] if executed else None, prev_hit=prev)
            bq = BlindQueryView(step, ep.task_id, ep, q.rs, q.raw_state, prev, q.prev_a_exec, hist_a,
                                np.asarray(hit, np.int8), hist_rs, np.asarray(visions), age)
            a, b = baseline.blind_step(bq), off.blind_step(bq)
            equal(a, b)
            vision = isinstance(a, LookReason)
            if vision:
                a, b = baseline.query(q), off.query(q)
                equal(a, b)
            # The unchanged judge sees the same confidence and flags; recorded
            # policy chunks remain identical when the input stream supplies MISS.
            counts["identical_actions"] += 1
            counts["identical_verdicts"] += 1
            counts["identical_vision"] += 1
            counts["identical_extras"] += 1
            counts["decisions"] += 1
            visions.append(vision); age = 0 if vision else age+1
            k0.append(z["vision_0"] if vision else np.full_like(z["vision_0"], np.nan))
            k1.append(z["vision_1"] if vision else np.full_like(z["vision_1"], np.nan))
            for key, method in enabled.items():
                history, out = variants[key], routing[key]
                vq = SimpleNamespace(**vars(q))
                vq.hist_key_v0, vq.hist_key_v1 = np.asarray(history["k0"]), np.asarray(history["k1"])
                vbq = dataclass_replace(bq, hist_has_vision=np.asarray(history["visions"]), blind_age=history["age"])
                proposal = method.blind_step(vbq)
                vv = isinstance(proposal, LookReason)
                if vv:
                    mode = "full" if step == 0 or proposal.code != 1 else method.next_camera_mode
                    method.set_camera_mode(mode)
                    if mode == "wrist_only":
                        # Fabricated base keys cannot enter wrist distance.
                        vq.key_v0 = np.full_like(vq.key_v0, np.nan)
                    proposal = method.query(vq)
                    out["wrist" if mode == "wrist_only" else "full"] += 1
                    out["proposed_cost"] += .055198 if mode == "wrist_only" else .152
                    if mode == "wrist_only":
                        reference = method.wrist.query(vq)
                        assert np.array_equal(reference.action, proposal.action)
                        assert np.array_equal(reference.topk, proposal.topk)
                        assert np.array_equal(reference.scores, proposal.scores)
                        assert reference.confidence == proposal.confidence
                        out["wrist_metric_exact"] += 1
                else:
                    out["blind"] += 1
                history["visions"].append(vv); history["age"] = 0 if vv else history["age"]+1
                history["k0"].append(vq.key_v0 if vv else np.full_like(vq.key_v0, np.nan))
                history["k1"].append(vq.key_v1 if vv else np.full_like(vq.key_v1, np.nan))
            states.append(z["robot_state"]); raw.append(z["raw_state"]); executed.append(z["executed_chunk"])
            hit.append(row.get("hit", "True").lower() == "true")
        counts["episodes"] += 1
    counts["enabled_fixed_stream"] = routing
    counts["PASS"] = True
    return counts


def dataclass_replace(value, **kwargs):
    import dataclasses
    return dataclasses.replace(value, **kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", type=Path, default=Path("/tmp/r7_C2/fits"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/r7_C2/replay_report.json"))
    a = ap.parse_args()
    reports = []
    for label in ("l10_50", "l10_500", "sp_50", "sp_500"):
        for dataset in ("r06_c_cal", "r06_p3_pilot"):
            report = run(label, dataset, a.fits)
            reports.append(report)
            print(json.dumps(report), flush=True)
    a.out.write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
