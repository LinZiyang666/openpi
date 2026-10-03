"""Exercise fresh fits on recorded query keys with identical supplied histories.

The deterministic sample uses init 0 of every task in both cache/inf streams,
first 24 decisions, with all-vision and alternating vision histories. It is an
offline method check, not a reconstruction of the live B experiment.
"""
import argparse
import collections
import json
from pathlib import Path
import pickle

import numpy as np

from exp.offline_search.closed_loop.blind import BlindResult, LookReason
from exp.offline_search.harness import api, store
from exp.offline_search.rounds.r04.k1_blind.checks import blind_view, view
from exp.offline_search.rounds.r06.p2_ablations.judge import TriggerCommitJudge as R6Pi
from exp.offline_search.rounds.r06.p2_ablations.judge import TriggerGrootCommitJudge as R6Groot
from exp.offline_search.rounds.r08.abl.logic_check import flags_from_diagnostics
from exp.offline_search.rounds.r08.abl.make_arms import RUN, STORE, artifact, write_json


class VisionView:
    def __init__(self, q, vision):
        self.q, self.hist_has_vision = q, vision

    def __getattr__(self, field):
        if field in ("hist_key_v0", "hist_key_v1"):
            keys = np.array(getattr(self.q, field), copy=True)
            keys[~self.hist_has_vision] = np.nan
            return keys
        return getattr(self.q, field)


def load(path):
    with Path(path).open("rb") as stream:
        return pickle.load(stream)["method"]


def equal_payload(ref, got):
    for field in ("topk", "scores", "action"):
        assert np.array_equal(getattr(ref, field), getattr(got, field)), field
    assert np.float64(ref.confidence).tobytes() == np.float64(got.confidence).tobytes()
    assert ref.library == got.library


def equal_blind(ref, got):
    assert type(ref) is type(got), (ref, got)
    if isinstance(ref, LookReason):
        assert ref == got
    else:
        for field in ("action", "rows", "weights"):
            assert np.array_equal(getattr(ref, field), getattr(got, field)), field
        assert ref.library == got.library and ref.extras == got.extras


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--steps", type=int, default=24)
    args = parser.parse_args(argv)
    if args.steps < 4:
        parser.error("--steps must be at least 4")
    rows = json.loads((args.run / "arms_in.json").read_text())
    provs = {p["arm"]: p for p in json.loads((args.run / "provenance.json").read_text())}
    grouped = collections.defaultdict(list)
    for row in rows:
        p = provs[row["name"]]
        grouped[p["model"], p["suite"], p["scale"]].append(row)
    reports, totals = [], collections.Counter()
    for (model, suite, scale), variants in grouped.items():
        source = load(provs[variants[0]["name"]]["source_artifact"])
        methods = [(row["name"], load(artifact(row))) for row in variants]
        r6_methods = []
        for name, m in methods:
            if len(m.disabled_guards) == 1:
                cls = R6Pi if model == "pi05" else R6Groot
                old = load(artifact(next(r for r in variants if r["name"] == name)))
                old.__class__ = cls
                old.disabled_guard = m.disabled_guards[0]
                r6_methods.append((name, old))
        counts, flag_counts = collections.Counter(), collections.Counter()
        for stream in ("cache", "inf"):
            qc = store.QueryCell(STORE, f"{model}_{suite}_{stream}")
            arrays = api.QueryArrays(qc)
            episodes = [e for e in qc.episodes if int(e["init"]) == 0]
            assert {int(e["task_id"]) for e in episodes} == set(range(10)) and len(episodes) == 10
            for schedule in ("all_vision", "alternating"):
                for ep in episodes:
                    q0 = view(qc, arrays, ep["start"])
                    source.reset(q0.episode)
                    for _, m in methods + r6_methods:
                        m.reset(q0.episode)
                    indexes = range(ep["start"], min(ep["end"], ep["start"] + args.steps))
                    for index in indexes:
                        q = view(qc, arrays, index)
                        if schedule == "alternating" and q.step % 2:
                            continue
                        hv = np.ones(q.step, bool) if schedule == "all_vision" else np.arange(q.step) % 2 == 0
                        q = VisionView(q, hv)
                        ref = source.query(q)
                        flags = int(ref.extras["os_flags"])
                        assert flags == flags_from_diagnostics(model, ref.extras, source)
                        flag_counts[flags] += 1
                        counts[schedule + "_queries"] += 1
                        got_by_name = {}
                        for name, m in methods:
                            got = m.query(q)
                            equal_payload(ref, got)
                            want = flags & ~m.disabled_mask
                            assert got.extras == {**ref.extras, "os_flags": float(want),
                                "os_reason": float((want & -want).bit_length()), "os_force_miss": float(bool(want))}
                            assert m._s["stuck_n"] == source._s["stuck_n"]
                            assert m._noprog_span == source._noprog_span
                            assert m._vision_progress == source._vision_progress
                            assert m._s["flag"][-1] == int(bool(want))
                            got_by_name[name] = got
                            counts["variant_query_comparisons"] += 1
                        for name, old in r6_methods:
                            got = old.query(q)
                            equal_payload(got_by_name[name], got)
                            assert got.extras == got_by_name[name].extras
                            counts["R6_single_guard_query_comparisons"] += 1
            # Explicit positive-span blind veto and inherited committed-policy hook.
            q0, q1 = view(qc, arrays, 0), view(qc, arrays, 1)
            for _, m in [("B", source)] + methods:
                m.reset(q0.episode)
                m.query(q0)
                m._noprog_span = 1
            bq = blind_view(q1, prev_hit=True)
            ref = source.blind_step(bq)
            assert isinstance(ref, LookReason) and ref.name == "noprog_span"
            for _, m in methods:
                got = m.blind_step(bq)
                if m.disabled_mask & 8:
                    assert isinstance(got, BlindResult)
                else:
                    equal_blind(ref, got)
                assert m._noprog_span == 1
                counts["explicit_blind_veto_comparisons"] += 1
            # Lifecycle hook sees the same recorded previous policy action in all variants.
            bq = blind_view(q1, prev_hit=False, hist_hit=np.zeros(1, np.int8))
            ref = source.policy_tail_step(bq)
            assert isinstance(ref, BlindResult)
            for _, m in methods:
                equal_blind(ref, m.policy_tail_step(bq))
                counts["policy_tail_comparisons"] += 1
        totals.update(counts)
        reports.append(dict(model=model, suite=suite, scale=scale, counts=dict(counts), flag_counts=dict(flag_counts)))
        print(f"{model}/{suite}/{scale}: {dict(counts)}, PASS", flush=True)
    out = dict(PASS=True, cells=8, **totals, mismatches=0, cells_report=reports,
        protocol="init 0 of all 10 tasks; cache and inf keys; all-vision and alternating supplied histories; fixed observations/actions",
        limitations="sampled offline method check; no live B sensory reconstruction or counterfactual SR estimate")
    write_json(args.run / "validation/query_replay.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "cells_report"}), flush=True)


if __name__ == "__main__":
    main()
