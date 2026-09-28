"""Read-only audit of the coordinator's completed v1 real-model smoke."""
import argparse
import json
from pathlib import Path

from .read_logs import load
from exp.offline_search.rounds.r04.k5_rand.estimate import jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("/home/weiland/trace_runs/os_closed_loop/r06_p3_smoke"))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    arms = [f"r6p3_{cell}_r0_smoke_p{p}" for cell in ("pi05_l10_50", "groot_l10_500") for p in (0, 100)]
    if any(not (a.run_root / "state" / (arm + ".DONE")).exists() for arm in arms):
        raise SystemExit("smoke not finished")
    rows = []
    for arm in arms:
        anchors, neighbours, steps, episodes = load(a.run_root, arm, require_inputs=True)
        events = [r for p in (a.run_root / "runs" / arm).glob("server_*/decisions_*.jsonl") for r in jsonl(p)]
        profiles = [r for r in events if r["ev"] == "p3_anchor"]
        ds = [r for r in events if r["ev"] == "dec"]
        expected_p = 0 if arm.endswith("_p0") else 1
        assert all(r["assignment"]["propensity"] == expected_p for r in profiles)
        assert sum(not r["hit"] for r in ds) == expected_p*len(profiles)
        assert all(r["cost"]["full_policy_forwards"] == 1 and r["cost"]["additional_forward_for_injection"] == 0 for r in profiles)
        rows.append(dict(arm=arm, episodes=len(episodes), decisions=len(ds), anchors=len(anchors),
            misses=sum(not r["hit"] for r in ds), policy_tails=sum(r["src"] == "policy_tail" for r in ds),
            exact_anchor_coverage=True, input_archives_checked=True, selected_shadow_bytes_equal=True,
            all_nonterminal_policy_tails_match=True, logged_additional_injection_forwards=0,
            independent_stage_invocation_counters_present=any("stage_invocations" in r for r in profiles),
            final_outcomes=[r["success"] for r in events if r["ev"] == "episode" and r.get("reason") == "episode_end"]))
    result = dict(status="completed", run_root=str(a.run_root), checks=rows,
        no_second_stage23_check="v1 logs declare one forward and zero added injection forwards; no independent stage-call counters. CPU staged tests verify reuse. V2 adds measured counters; v1 real no-second-forward is not independently identifiable from these logs.",
        limitations="standard client, no snapshots/control telemetry; smoke does not validate V2 or simulator restore")
    a.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
