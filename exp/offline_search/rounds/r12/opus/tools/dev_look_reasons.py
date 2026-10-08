"""R12 opus: look-reason census of B-pool dev closed-loop decision logs (R11 dev roots; non-test inits only).

Which looks would the plugin even allow to be half looks? The request-camera path forces a full look at step 0 and
for look reasons 6 (lifecycle, e.g. after a MISS), 7 and 8 (no-progress span veto). Only reason-1 ("budget") looks
are eligible. Also reports the look-to-look gap distribution (for lagged-camera designs).
"""
import collections, glob, json, sys
from pathlib import Path

ROOTS = ["r11_dev_current", "r11_dev_size50", "r11_dev_size200", "r11_devknob_50"]
BASE = Path("/home/weiland/trace_runs/os_closed_loop")


def main():
    out = {}
    for root in ROOTS:
        for arm in sorted((BASE / root / "runs").glob("r11_*")):
            rows = []
            for f in arm.glob("server_*/decisions_*.jsonl"):
                for line in f.open():
                    if '"ev": "dec"' not in line:
                        continue
                    d = json.loads(line)
                    rows.append(d)
            # accepted attempts only: keep latest attempt per uid
            last = {}
            for d in rows:
                last[d["uid"]] = max(last.get(d["uid"], 0), d.get("attempt", 1))
            rows = [d for d in rows if d.get("attempt", 1) == last[d["uid"]]]
            seen = set(); ded = []
            for d in sorted(rows, key=lambda d: (d["uid"], d["step"], d.get("ts", 0))):
                k = (d["uid"], d["step"])
                if k in seen:
                    continue
                seen.add(k); ded.append(d)
            rows = ded
            N = len(rows)
            looks = [d for d in rows if d.get("vision")]
            rc = collections.Counter(); miss_by = collections.Counter()
            for d in looks:
                r = d.get("look_reason")
                key = "step0" if d["step"] == 0 else str(r)
                rc[key] += 1
                if not d.get("hit", True):
                    miss_by[key] += 1
            # gaps between consecutive looks within an episode
            by_ep = collections.defaultdict(list)
            for d in looks:
                by_ep[d["uid"]].append(d["step"])
            gaps = collections.Counter()
            for s in by_ep.values():
                s.sort()
                for a, b in zip(s[:-1], s[1:]):
                    gaps[b - a] += 1
            # srcs of the decision before a look
            out[f"{root}/{arm.name}"] = dict(N=N, V=len(looks), reasons=dict(rc), miss_by_reason=dict(miss_by),
                                             gaps=dict(sorted(gaps.items())))
            v = len(looks)
            elig = rc.get("1", 0)
            print(f"{root}/{arm.name}: N={N} V={v} v={v/max(N,1):.3f} reasons={dict(rc)} miss={dict(miss_by)} "
                  f"eligible(budget)={elig/max(v,1):.3f} gaps={dict(sorted(gaps.items()))}", flush=True)
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
