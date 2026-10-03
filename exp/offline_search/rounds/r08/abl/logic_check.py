"""Replay the mask on all available logged PAPER_AB B verdicts, without a simulator.

These standard logs contain diagnostics and returned payloads, not visual keys.
This proves the verdict transformation on the recorded B history; it does not
reconstruct B's sensory query or assert equal future closed-loop trajectories.
"""
import argparse
import collections
import itertools
import json
from pathlib import Path

import numpy as np

from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.p2_ablations.judge import _TriggerMask
from exp.offline_search.rounds.r08.abl.judge import BITS, TriggerCommitJudge, TriggerGrootCommitJudge
from exp.offline_search.rounds.r08.abl.make_arms import RUN, RUNS, sha, write_json


class RecordedParent:
    def query(self, result):
        return result


class RecordedMask(_TriggerMask, RecordedParent):
    """Run the exact inherited R6/R8 post-query hook on a recorded B Result."""


def masked_result(ref, mask):
    method = object.__new__(RecordedMask)
    method.disabled_mask = mask
    method._s = dict(burst_end=0, ret_end=0, flag=[int(bool(ref.extras["os_flags"]))])
    got = method.query(ref)
    flags = int(ref.extras["os_flags"])
    want = flags & ~mask
    assert got.extras == {**ref.extras, "os_flags": float(want),
        "os_reason": float((want & -want).bit_length()), "os_force_miss": float(bool(want))}
    assert method._s["flag"] == [int(bool(want))]
    for key in ("topk", "scores", "action"):
        assert getattr(got, key) is getattr(ref, key)
    assert got.confidence == ref.confidence and got.library == ref.library
    for bit in BITS.values():
        assert not (int(got.extras["os_flags"]) & mask)
        if not mask & bit:
            assert (want & bit) == (flags & bit)
    return got


def flags_from_diagnostics(model, extras, method):
    n = extras["stuck_n"]
    closed = extras.get("gexec", 0.) * (-1 if model == "groot" else 1) > 0
    span = extras.get("noprog_span", extras.get("noprog_n", 0.))
    return (int(n >= method.stuck_thr) | (int(bool(extras["term1"]) and closed) << 1)
        | (int(extras.get("overtime", 0.) > 1 and extras.get("lag", 0.) > method.lag_thr and n >= 1) << 2)
        | (int(span >= method.noprog_n - 1) << 3))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    args = parser.parse_args(argv)
    rows = json.loads((args.run / "arms_in.json").read_text())
    provenance = json.loads((args.run / "provenance.json").read_text())
    provs = {p["arm"]: p for p in provenance}
    grouped = collections.defaultdict(list)
    exhaustive = 0
    for cls in (TriggerCommitJudge, TriggerGrootCommitJudge):
        model = "pi05" if cls is TriggerCommitJudge else "groot"
        kwargs = next(r["kwargs"] for r in rows if r["model"] == model).copy()
        kwargs.pop("disabled_guards")
        for size in range(5):
            for guards in itertools.combinations(BITS, size):
                m = cls(disabled_guards=guards, **kwargs)
                mask = sum(BITS[g] for g in guards)
                assert m.disabled_mask == mask
                for flags in range(16):
                    ref = api.Result(np.arange(16), np.arange(16.), .5, action=np.zeros((10, 32)),
                        extras=dict(os_flags=float(flags), os_reason=float((flags & -flags).bit_length()),
                                    os_force_miss=float(bool(flags)), stuck_n=3., noprog_span=5.))
                    masked_result(ref, mask)
                    exhaustive += 1
    for row in rows:
        p = provs[row["name"]]
        cls = TriggerCommitJudge if row["model"] == "pi05" else TriggerGrootCommitJudge
        m = cls(**row["kwargs"])
        for spec in p["references"]["B"]:
            grouped[spec].append((row, m))
    reports, changes = [], {row["name"]: collections.Counter() for row in rows}
    decisions = comparisons = 0
    for spec, variants in grouped.items():
        run, name = spec.split(":")
        directory = RUNS / run / "runs" / name
        paths = sorted(directory.glob("server_*/decisions_*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"no B decision logs for {spec}")
        count, seen_flags, files = 0, collections.Counter(), []
        for path in paths:
            malformed, parsed, verdicts = 0, 0, 0
            with path.open() as stream:
                for line in stream:
                    try:
                        record = json.loads(line)
                    except ValueError:
                        malformed += 1
                        continue
                    parsed += 1
                    extras = record.get("extras", {})
                    if record.get("ev") != "dec" or "os_flags" not in extras:
                        continue
                    flags = int(extras["os_flags"])
                    assert 0 <= flags <= 15 and record["vision"], (spec, record.get("step"), flags)
                    model = variants[0][0]["model"]
                    assert flags == flags_from_diagnostics(model, extras, variants[0][1]), (spec, record.get("uid"), extras)
                    assert extras["os_reason"] == (flags & -flags).bit_length()
                    assert extras["os_force_miss"] == float(bool(flags))
                    ref = api.Result(np.asarray(record["topk"]), np.asarray(record["scores"]), record["conf"],
                                     action=np.asarray(record["served_head"]), library=record["lib"], extras=extras)
                    for row, m in variants:
                        got = masked_result(ref, m.disabled_mask)
                        new = int(got.extras["os_flags"])
                        c = changes[row["name"]]
                        c["vision_verdicts"] += 1
                        c["baseline_misses"] += bool(flags)
                        c["retained_misses"] += bool(new)
                        c["removed_misses"] += bool(flags) and not new
                        c["changed_reason_with_miss_retained"] += bool(new) and got.extras["os_reason"] != extras["os_reason"]
                        for guard, bit in BITS.items():
                            c[f"B_{guard}_fires"] += bool(flags & bit)
                            c[f"retained_{guard}_fires"] += bool(new & bit)
                        comparisons += 1
                    verdicts += 1
                    count += 1
                    seen_flags[flags] += 1
            assert malformed == 0, (path, malformed)
            files.append(dict(path=str(path), sha256=sha(path), parsed_records=parsed,
                              vision_verdicts=verdicts, malformed_lines=malformed))
        assert count > 0, spec
        decisions += count
        reports.append(dict(reference=spec, vision_verdicts=count,
                            observed_flag_counts=dict(seen_flags), files=files))
        print(f"{spec}: {count} B vision verdicts, {len(variants)} variants, PASS", flush=True)
    out = dict(PASS=True, exhaustive_subset_flag_cases=exhaustive, B_references=len(reports),
        logged_B_vision_verdicts=decisions, variant_comparisons=comparisons, mismatches=0,
        protocol="all logged B verdicts including historical repair attempts; fixed B history; post-query hook only",
        limitations="no visual-key reconstruction, counterfactual trajectories or closed-loop SR claims",
        variants={k: dict(v) for k, v in changes.items()}, references=reports)
    write_json(args.run / "validation/logged_verdict_replay.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("variants", "references")}), flush=True)


if __name__ == "__main__":
    main()
